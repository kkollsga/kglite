/**
 * Official Neo4j JavaScript driver conformance suite for kglite-bolt-server.
 *
 * Mirrors, check for check, the Java suite in `../java` — the two exist to
 * answer one question the Python suite cannot: does a *different* official
 * driver, with its own PackStream implementation, its own retry machinery, and
 * its own exception hierarchy, agree with what this server sends?
 *
 * The server is started by the caller (`tests/test_bolt_driver_conformance.py`)
 * and its URI arrives as argv[2] or $KGLITE_BOLT_URI. It serves the standard
 * bolt fixture graph: 4 `:Person` nodes (Alice/Bob/Carol/Dave, each with a
 * `city`) and 3 `:KNOWS` edges 1->2, 2->3, 3->4.
 *
 * Exit code 0 = every check passed. Non-zero = the count of failures, with
 * each failure printed as `FAIL <name>: <detail>`.
 */

import neo4j from "neo4j-driver";

const URI = process.argv[2] ?? process.env.KGLITE_BOLT_URI;
if (!URI) {
  console.error("usage: node conformance.mjs bolt://host:port");
  process.exit(64);
}
const AUTH = neo4j.auth.basic("neo4j", "password");

const results = [];

async function check(name, fn) {
  try {
    await fn();
    results.push({ name, ok: true });
    console.log(`PASS ${name}`);
  } catch (err) {
    const detail = err && err.message ? err.message : String(err);
    results.push({ name, ok: false, detail });
    console.log(`FAIL ${name}: ${detail}`);
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertEqual(actual, expected, message) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  assert(a === e, `${message}: expected ${e}, got ${a}`);
}

/** Read every record of a read query through an auto-commit session. */
async function read(driver, query, params = {}) {
  const session = driver.session();
  try {
    const result = await session.run(query, params);
    return result.records;
  } finally {
    await session.close();
  }
}

async function main() {
  const driver = neo4j.driver(URI, AUTH);

  // ── Connectivity ──────────────────────────────────────────────────────
  await check("connectivity.verify", async () => {
    // The driver's own handshake + protocol negotiation.
    await driver.verifyConnectivity();
  });

  await check("session.scalar_return", async () => {
    const records = await read(driver, "RETURN 1 AS one");
    assertEqual(records.length, 1, "row count");
    assertEqual(records[0].get("one").toNumber(), 1, "value");
  });

  await check("session.parameters", async () => {
    const records = await read(
      driver,
      "MATCH (p:Person) WHERE p.city = $city RETURN p.title AS name ORDER BY name",
      { city: "Oslo" },
    );
    assert(records.length > 0, "expected at least one Oslo person");
    for (const record of records) {
      assert(typeof record.get("name") === "string", "name should be a string");
    }
  });

  // ── Type round-trips ──────────────────────────────────────────────────
  // A parameter goes out over PackStream and comes back through RETURN, so a
  // mismatch is a wire bug on one side or the other.
  await check("types.integer", async () => {
    const records = await read(driver, "RETURN $v AS v", { v: neo4j.int(42) });
    assertEqual(records[0].get("v").toNumber(), 42, "integer");
  });

  await check("types.float", async () => {
    const records = await read(driver, "RETURN $v AS v", { v: 1.5 });
    assertEqual(records[0].get("v"), 1.5, "float");
  });

  await check("types.string", async () => {
    const records = await read(driver, "RETURN $v AS v", { v: "hei på deg" });
    assertEqual(records[0].get("v"), "hei på deg", "string (non-ASCII)");
  });

  await check("types.boolean", async () => {
    const records = await read(driver, "RETURN $v AS v", { v: true });
    assertEqual(records[0].get("v"), true, "boolean");
  });

  await check("types.null", async () => {
    const records = await read(driver, "RETURN $v AS v", { v: null });
    assertEqual(records[0].get("v"), null, "null");
  });

  await check("types.list", async () => {
    const records = await read(driver, "RETURN $v AS v", { v: ["a", "b"] });
    assertEqual(records[0].get("v"), ["a", "b"], "list");
  });

  await check("types.map", async () => {
    const records = await read(driver, "RETURN $v.k AS k", { v: { k: "v" } });
    assertEqual(records[0].get("k"), "v", "map member");
  });

  // ── Graph types ───────────────────────────────────────────────────────
  await check("graph.node", async () => {
    const records = await read(driver, "MATCH (p:Person) RETURN p ORDER BY p.title LIMIT 1");
    const node = records[0].get("p");
    assert(node instanceof neo4j.types.Node, "expected a Node instance");
    assert(node.labels.includes("Person"), `labels were ${JSON.stringify(node.labels)}`);
    assert(node.properties.title !== undefined, "expected a title property");
  });

  await check("graph.relationship", async () => {
    const records = await read(driver, "MATCH ()-[r:KNOWS]->() RETURN r LIMIT 1");
    const rel = records[0].get("r");
    assert(rel instanceof neo4j.types.Relationship, "expected a Relationship instance");
    assertEqual(rel.type, "KNOWS", "relationship type");
  });

  await check("graph.path", async () => {
    const records = await read(driver, "MATCH p = ()-[:KNOWS]->() RETURN p LIMIT 1");
    const path = records[0].get("p");
    assert(path instanceof neo4j.types.Path, "expected a Path instance");
    assertEqual(path.segments.length, 1, "segment count");
  });

  // ── Transactions ──────────────────────────────────────────────────────
  await check("tx.explicit_write_commits", async () => {
    const session = driver.session();
    try {
      const tx = session.beginTransaction();
      await tx.run("CREATE (:JsProbe {id: 1, title: 'committed'})");
      await tx.commit();
      const records = await read(driver, "MATCH (n:JsProbe {id: 1}) RETURN count(n) AS n");
      assertEqual(records[0].get("n").toNumber(), 1, "committed node count");
    } finally {
      await session.close();
    }
  });

  await check("tx.rollback_discards", async () => {
    const session = driver.session();
    try {
      const tx = session.beginTransaction();
      await tx.run("CREATE (:JsProbe {id: 2, title: 'rolled back'})");
      await tx.rollback();
      const records = await read(driver, "MATCH (n:JsProbe {id: 2}) RETURN count(n) AS n");
      assertEqual(records[0].get("n").toNumber(), 0, "rolled-back node count");
    } finally {
      await session.close();
    }
  });

  await check("tx.executeWrite_managed_retry", async () => {
    // The driver's managed-transaction API — the shape a ported app actually
    // uses. Exercises BEGIN/RUN/COMMIT driven by the driver itself.
    const session = driver.session();
    try {
      const written = await session.executeWrite(async (tx) => {
        const result = await tx.run("CREATE (n:JsProbe {id: 3}) RETURN n.id AS id");
        return result.records[0].get("id").toNumber();
      });
      assertEqual(written, 3, "managed write return value");
    } finally {
      await session.close();
    }
  });

  await check("tx.autocommit_mutation_is_rejected", async () => {
    // Documented kglite limitation: writes need an explicit transaction. The
    // point of asserting it is that the client gets a *clear* refusal rather
    // than a silent no-op.
    const session = driver.session();
    let raised = null;
    try {
      await session.run("CREATE (:JsProbe {id: 99})");
    } catch (err) {
      raised = err;
    } finally {
      await session.close();
    }
    assert(raised !== null, "expected auto-commit CREATE to be rejected");
    assert(
      /auto-commit/i.test(raised.message),
      `expected the message to mention auto-commit, got: ${raised.message}`,
    );
  });

  await check("tx.occ_conflict_code", async () => {
    // Two stale-vs-fresh committers. The loser must report the documented
    // status code, because branching on the code is how a retry loop is
    // written.
    const sessionA = driver.session();
    const sessionB = driver.session();
    try {
      const txA = sessionA.beginTransaction();
      const txB = sessionB.beginTransaction();
      await txA.run("CREATE (:JsProbe {id: 10, title: 'A'})");
      await txB.run("CREATE (:JsProbe {id: 11, title: 'B'})");
      await txA.commit();
      let raised = null;
      try {
        await txB.commit();
      } catch (err) {
        raised = err;
      }
      assert(raised !== null, "expected the stale commit to conflict");
      assertEqual(raised.code, "Neo.ClientError.Transaction.ConflictDetected", "conflict code");
    } finally {
      await sessionA.close();
      await sessionB.close();
    }
  });

  // ── Error codes ───────────────────────────────────────────────────────
  await check("errors.syntax_error_code", async () => {
    let raised = null;
    try {
      await read(driver, "MATCH (((");
    } catch (err) {
      raised = err;
    }
    assert(raised !== null, "expected a syntax error");
    assertEqual(raised.code, "Neo.ClientError.Statement.SyntaxError", "syntax error code");
  });

  await check("errors.codes_are_neo4j_shaped", async () => {
    let raised = null;
    try {
      await read(driver, "RETURN $missing AS v");
    } catch (err) {
      raised = err;
    }
    if (raised !== null) {
      const parts = String(raised.code).split(".");
      assertEqual(parts.length, 4, `code should have 4 dotted segments: ${raised.code}`);
      assertEqual(parts[0], "Neo", "code should start with Neo");
    }
  });

  // ── Procedures + capability gate ──────────────────────────────────────
  await check("procedures.db_labels", async () => {
    const records = await read(driver, "CALL db.labels() YIELD label RETURN label ORDER BY label");
    const labels = records.map((r) => r.get("label"));
    assert(labels.includes("Person"), `expected Person in ${JSON.stringify(labels)}`);
  });

  await check("capability.load_csv_denied_for_remote_clients", async () => {
    // The server under test is started without --allow-csv-import, so a
    // remote LOAD CSV must be refused. This is a security assertion, not a
    // feature one.
    let raised = null;
    try {
      await read(driver, "LOAD CSV FROM 'file:///etc/hosts' AS row RETURN row[0] AS line");
    } catch (err) {
      raised = err;
    }
    assert(raised !== null, "expected LOAD CSV to be refused");
    assert(
      /not enabled for this connection/.test(raised.message),
      `expected the capability refusal, got: ${raised.message}`,
    );
  });

  await driver.close();

  const failures = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failures.length}/${results.length} checks passed`);
  if (failures.length > 0) {
    console.log("\nfailures:");
    for (const f of failures) console.log(`  ${f.name}: ${f.detail}`);
  }
  process.exit(failures.length === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error(`harness error: ${err && err.stack ? err.stack : err}`);
  process.exit(70);
});
