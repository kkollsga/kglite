import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { kglite, freshGraph } from './helpers.mjs';

const { graph, cleanup } = await freshGraph();
test.after(cleanup);

test('every rejection is a KgliteError with a string code', async () => {
  await assert.rejects(graph.executeRead('MATCH (n RETURN n'), (e) => {
    assert.ok(e instanceof Error);
    assert.equal(e.name, 'KgliteError');
    assert.equal(typeof e.code, 'string');
    return true;
  });
});

test('syntax errors carry the engine code and a positioned message', async () => {
  await assert.rejects(graph.executeRead('MATCH (n RETURN n'), (e) => {
    assert.equal(e.code, 'CypherSyntax');
    assert.match(e.message, /syntax error at line 1/i);
    return true;
  });
});

test('a mutating statement through executeRead rejects InvalidArgument and changes nothing', async () => {
  await assert.rejects(graph.executeRead('CREATE (:Ghost)'), (e) => e.code === 'InvalidArgument');
  const r = await graph.executeRead('MATCH (n:Ghost) RETURN count(n) AS c');
  assert.equal(r.rows[0].c, 0);
});

test('a missing parameter is reported by name', async () => {
  await assert.rejects(graph.executeRead('RETURN $nope'), (e) => /Missing parameter: \$nope/.test(e.message));
});

test('deadline overrun rejects CypherTimeout and the graph stays usable', async () => {
  const started = Date.now();
  await assert.rejects(
    graph.executeRead(
      'UNWIND range(1, 100000) AS a UNWIND range(1, 100000) AS b RETURN count(*) AS c',
      null,
      { timeoutMs: 200 },
    ),
    (e) => e.code === 'CypherTimeout' && /limit 200ms/.test(e.message),
  );
  assert.ok(Date.now() - started < 5000, 'the deadline is honoured, not waited out');
  assert.equal((await graph.executeRead('RETURN 1 AS one')).rows[0].one, 1);
});

test('a default timeout set at open applies to every query', async () => {
  const { graph: g, cleanup: done } = await freshGraph({ timeoutMs: 200 });
  try {
    await assert.rejects(
      g.executeRead('UNWIND range(1, 100000) AS a UNWIND range(1, 100000) AS b RETURN count(*) AS c'),
      (e) => e.code === 'CypherTimeout',
    );
  } finally {
    done();
  }
});

test('a work budget overrun rejects instead of truncating', async () => {
  await assert.rejects(
    graph.executeRead('UNWIND range(1, 1000) AS a RETURN a', null, { maxWorkUnits: 10 }),
    (e) => /max_work_units/.test(e.message),
  );
});

test('declared constraints reject with ConstraintViolation', async () => {
  await graph.executeWrite('CREATE CONSTRAINT FOR (n:Uniq) REQUIRE n.k IS UNIQUE');
  await graph.executeWrite('CREATE (:Uniq {k: 1})');
  await assert.rejects(graph.executeWrite('CREATE (:Uniq {k: 1})'), (e) => e.code === 'ConstraintViolation');
  // The failed statement published nothing.
  const r = await graph.executeRead('MATCH (n:Uniq) RETURN count(n) AS c');
  assert.equal(r.rows[0].c, 1);
});

test('opening a held path rejects WriterLeaseHeld', async () => {
  const { graph: held, dir, cleanup: done } = await freshGraph();
  try {
    // freshGraph opened `durability: off`, which still takes the writer lease.
    await assert.rejects(
      kglite.open(join(dir, 'g.kgl'), { durability: 'off' }),
      (e) => e.code === 'WriterLeaseHeld',
    );
    assert.equal(resolve(held.path), resolve(dir, 'g.kgl'));
  } finally {
    done();
  }
});

test('panics become Internal errors and the process keeps running', async () => {
  assert.equal(typeof kglite.__panic, 'function', 'build with --features test-hooks (make test-node)');
  // On the JS thread (the export boundary).
  assert.throws(() => kglite.__panic(), (e) => e.code === 'Internal' && /deliberate test panic/.test(e.message));
  // On a pool thread.
  await assert.rejects(kglite.__panicInWorker(), (e) => e.code === 'Internal' && /deliberate worker panic/.test(e.message));
  // While building the result on the JS thread.
  await assert.rejects(kglite.__panicInSettle(), (e) => e.code === 'Internal' && /deliberate settle panic/.test(e.message));
  // Every pool thread survived: more panics than workers, then real queries.
  await Promise.allSettled(Array.from({ length: 16 }, () => kglite.__panicInWorker()));
  const rows = await Promise.all(Array.from({ length: 16 }, (_, i) => graph.executeRead('RETURN $i AS i', { i })));
  assert.deepEqual(rows.map((r) => r.rows[0].i), Array.from({ length: 16 }, (_, i) => i));
});

test('queries near the nesting ceilings complete; past them they are refused, not fatal', async () => {
  const nested = (n) => `RETURN ${'['.repeat(n)}1${']'.repeat(n)} AS v`;
  const depthOf = (v) => {
    let d = 0;
    while (Array.isArray(v)) {
      v = v[0];
      d++;
    }
    return d;
  };
  // A built value is bounded at 256 levels; the parser accepts 512.
  for (const n of [200, 256]) {
    const r = await graph.executeRead(nested(n));
    assert.equal(depthOf(r.rows[0].v), n);
  }
  for (const n of [257, 511]) {
    await assert.rejects(graph.executeRead(nested(n)), (e) => e.code !== 'Internal' && /256/.test(e.message));
  }
  await assert.rejects(graph.executeRead(nested(513)), (e) => e.code === 'CypherSyntax' && /nesting exceeds/.test(e.message));
  await assert.rejects(graph.executeRead(nested(100000)), (e) => e.code === 'CypherSyntax');
  // The same ceiling in an OR chain, which the executor walks recursively.
  const chain = Array.from({ length: 510 }, (_, i) => `n.k = ${i}`).join(' OR ');
  const r = await graph.executeRead(`MATCH (n:Uniq) WHERE ${chain} RETURN count(n) AS c`);
  assert.equal(typeof r.rows[0].c, 'number');
});

test('warnings reach the result and are never echoed to stderr', () => {
  const addon = JSON.stringify(fileURLToPath(new URL('../index.js', import.meta.url)));
  const script = `
    const k = require(${addon});
    (async () => {
      const g = await k.open(require('os').tmpdir() + '/kglite-node-warn-' + process.pid + '.kgl', { durability: 'off' });
      await g.executeWrite('CREATE (:Real)');
      const r = await g.executeRead('MATCH (n:NoSuchLabel) RETURN n');
      process.stdout.write(JSON.stringify(r.warnings));
    })();
  `;
  const r = spawnSync(process.execPath, ['-e', script], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  const warnings = JSON.parse(r.stdout);
  assert.ok(warnings.length >= 1 && /NoSuchLabel/.test(warnings[0]));
  assert.equal(r.stderr, '', 'the engine echo is silenced on pool threads');
});
