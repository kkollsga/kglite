import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const { kglite } = await import('./helpers.mjs');

async function open(options = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'kglite-node-group-'));
  const graph = await kglite.open(join(dir, 'g.kgl'), { durability: 'full', ...options });
  test.after(async () => {
    await graph.close().catch(() => {});
    rmSync(dir, { recursive: true, force: true });
  });
  return graph;
}

const count = async (graph, label) =>
  Number((await graph.executeRead(`MATCH (n:${label}) RETURN count(n) AS c`)).rows[0].c);

test('concurrent writes at full share log barriers', async () => {
  assert.equal(typeof (await open()).__walBarriers, 'function', 'build with --features test-hooks (make test-node)');
  const graph = await open();
  await graph.executeWrite('CREATE (:Warm)');
  const writers = 64;
  const before = graph.__walBarriers();
  const results = await Promise.all(
    Array.from({ length: writers }, (_, i) => graph.executeWrite('CREATE (:Shared {i: $i})', { i })),
  );
  const barriers = graph.__walBarriers() - before;
  assert.equal(results.length, writers);
  assert.equal(await count(graph, 'Shared'), writers);
  assert.ok(barriers < writers, `${writers} commits took ${barriers} barriers; none were shared`);
});

test('writes awaited one after another commit in call order', async () => {
  const graph = await open();
  for (let i = 0; i < 20; i++) await graph.executeWrite('CREATE (:Seq {i: $i})', { i });
  const rows = (await graph.executeRead('MATCH (n:Seq) RETURN n.i AS i')).rows.map((r) => Number(r.i));
  assert.deepEqual(rows, Array.from({ length: 20 }, (_, i) => i));
});

test('schema statements and explicit transactions interleave with grouped writes without losing a commit', async () => {
  const graph = await open();
  const calls = [];
  for (let i = 0; i < 60; i++) calls.push(graph.executeWrite('CREATE (:Mixed {i: $i})', { i }));
  for (let k = 0; k < 5; k++) calls.push(graph.executeWrite(`CREATE INDEX FOR (n:Mixed) ON (n.p${k})`));
  const tx = await graph.begin();
  await tx.run('CREATE (:ViaTx)');
  calls.push(tx.commit());
  const settled = await Promise.allSettled(calls);
  const failed = settled.filter((r) => r.status === 'rejected');
  assert.equal(failed.length, 0, failed[0] && String(failed[0].reason));
  assert.equal(await count(graph, 'Mixed'), 60);
  assert.equal(await count(graph, 'ViaTx'), 1);
});

test('a write aborted before it starts never runs, and close waits for writes in flight', async () => {
  const graph = await open();
  const ac = new AbortController();
  ac.abort();
  await assert.rejects(graph.executeWrite('CREATE (:Never)', null, { signal: ac.signal }), (e) => e.code === 'Cancelled');
  const inflight = Array.from({ length: 32 }, (_, i) => graph.executeWrite('CREATE (:Late {i: $i})', { i }));
  const closing = graph.close();
  const settled = await Promise.allSettled(inflight);
  await closing;
  // Each write either committed before close or was refused because the graph was closed.
  assert.ok(settled.every((r) => r.status === 'fulfilled' || r.reason.code === 'Closed'), JSON.stringify(settled));
});
