package dev.kglite.conformance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Record;
import org.neo4j.driver.Session;
import org.neo4j.driver.Transaction;
import org.neo4j.driver.Value;
import org.neo4j.driver.Values;
import org.neo4j.driver.exceptions.Neo4jException;
import org.neo4j.driver.types.Node;
import org.neo4j.driver.types.Path;
import org.neo4j.driver.types.Relationship;

/**
 * Official Neo4j Java driver conformance suite for {@code kglite-bolt-server}.
 *
 * <p>Mirrors {@code ../js/conformance.mjs} check for check. The two suites exist to answer one
 * question the Python suite cannot: does a <em>different</em> official driver — its own PackStream
 * implementation, its own managed-transaction retry machinery, its own exception hierarchy — agree
 * with what this server sends?
 *
 * <p>The JVM matters specifically here: there is no in-process route to a kglite graph from the
 * JVM, so this server is the entire integration surface for JVM consumers, and it only works if the
 * official driver works unmodified.
 *
 * <p>The server is started by {@code tests/test_bolt_driver_conformance.py} and its URI arrives as
 * the {@code kglite.bolt.uri} system property. It serves the standard bolt fixture graph: four
 * {@code :Person} nodes (Alice/Bob/Carol/Dave, each with a {@code city}) joined by three {@code
 * :KNOWS} edges.
 */
@DisplayName("kglite-bolt-server / official Neo4j Java driver")
class BoltConformanceTest {

  private static final String URI_PROPERTY = "kglite.bolt.uri";
  private static Driver driver;

  @BeforeAll
  static void connect() {
    String uri = System.getProperty(URI_PROPERTY);
    assumeTrue(
        uri != null && !uri.isBlank(),
        "set -D" + URI_PROPERTY + "=bolt://host:port to point the suite at a running server");
    driver = GraphDatabase.driver(uri, AuthTokens.basic("neo4j", "password"));
  }

  @AfterAll
  static void disconnect() {
    if (driver != null) {
      driver.close();
    }
  }

  /** Run a read query on an auto-commit session and collect every record. */
  private static List<Record> read(String query, Value parameters) {
    try (Session session = driver.session()) {
      return new ArrayList<>(session.run(query, parameters).list());
    }
  }

  private static List<Record> read(String query) {
    return read(query, Values.parameters());
  }

  private static long count(String query) {
    return read(query).get(0).get("n").asLong();
  }

  // ── Connectivity ────────────────────────────────────────────────────────

  @Test
  @DisplayName("connectivity.verify — handshake and protocol negotiation")
  void connectivityVerify() {
    driver.verifyConnectivity();
  }

  @Test
  @DisplayName("session.scalar_return")
  void sessionScalarReturn() {
    List<Record> records = read("RETURN 1 AS one");
    assertEquals(1, records.size());
    assertEquals(1, records.get(0).get("one").asInt());
  }

  @Test
  @DisplayName("session.parameters")
  void sessionParameters() {
    List<Record> records =
        read(
            "MATCH (p:Person) WHERE p.city = $city RETURN p.title AS name ORDER BY name",
            Values.parameters("city", "Oslo"));
    assertTrue(records.size() > 0, "expected at least one Oslo person");
    for (Record record : records) {
      assertNotNull(record.get("name").asString());
    }
  }

  // ── Type round-trips ────────────────────────────────────────────────────
  // Each value goes out over PackStream as a parameter and comes back through
  // RETURN, so a mismatch is a wire bug on one side or the other.

  @Test
  @DisplayName("types.integer")
  void typesInteger() {
    assertEquals(42L, read("RETURN $v AS v", Values.parameters("v", 42L)).get(0).get("v").asLong());
  }

  @Test
  @DisplayName("types.float")
  void typesFloat() {
    assertEquals(
        1.5d, read("RETURN $v AS v", Values.parameters("v", 1.5d)).get(0).get("v").asDouble());
  }

  @Test
  @DisplayName("types.string — non-ASCII survives the round trip")
  void typesString() {
    assertEquals(
        "hei på deg",
        read("RETURN $v AS v", Values.parameters("v", "hei på deg")).get(0).get("v").asString());
  }

  @Test
  @DisplayName("types.boolean")
  void typesBoolean() {
    assertTrue(read("RETURN $v AS v", Values.parameters("v", true)).get(0).get("v").asBoolean());
  }

  @Test
  @DisplayName("types.null")
  void typesNull() {
    assertTrue(read("RETURN $v AS v", Values.parameters("v", (Object) null)).get(0).get("v").isNull());
  }

  @Test
  @DisplayName("types.list")
  void typesList() {
    List<Object> value = read("RETURN $v AS v", Values.parameters("v", List.of("a", "b")))
        .get(0)
        .get("v")
        .asList();
    assertEquals(List.of("a", "b"), value);
  }

  @Test
  @DisplayName("types.map — dotted access on a map parameter")
  void typesMap() {
    assertEquals(
        "v",
        read("RETURN $v.k AS k", Values.parameters("v", Map.of("k", "v"))).get(0).get("k").asString());
  }

  // ── Graph types ─────────────────────────────────────────────────────────

  @Test
  @DisplayName("graph.node")
  void graphNode() {
    Node node = read("MATCH (p:Person) RETURN p ORDER BY p.title LIMIT 1").get(0).get("p").asNode();
    List<String> labels = new ArrayList<>();
    node.labels().forEach(labels::add);
    assertTrue(labels.contains("Person"), "labels were " + labels);
    assertNotNull(node.get("title").asString());
  }

  @Test
  @DisplayName("graph.relationship")
  void graphRelationship() {
    Relationship rel =
        read("MATCH ()-[r:KNOWS]->() RETURN r LIMIT 1").get(0).get("r").asRelationship();
    assertEquals("KNOWS", rel.type());
  }

  @Test
  @DisplayName("graph.path")
  void graphPath() {
    Path path = read("MATCH p = ()-[:KNOWS]->() RETURN p LIMIT 1").get(0).get("p").asPath();
    assertEquals(1, path.length());
  }

  // ── Transactions ────────────────────────────────────────────────────────

  @Test
  @DisplayName("tx.explicit_write_commits")
  void txExplicitWriteCommits() {
    try (Session session = driver.session()) {
      try (Transaction tx = session.beginTransaction()) {
        tx.run("CREATE (:JavaProbe {id: 1, title: 'committed'})");
        tx.commit();
      }
    }
    assertEquals(1L, count("MATCH (n:JavaProbe {id: 1}) RETURN count(n) AS n"));
  }

  @Test
  @DisplayName("tx.rollback_discards")
  void txRollbackDiscards() {
    try (Session session = driver.session()) {
      try (Transaction tx = session.beginTransaction()) {
        tx.run("CREATE (:JavaProbe {id: 2, title: 'rolled back'})");
        tx.rollback();
      }
    }
    assertEquals(0L, count("MATCH (n:JavaProbe {id: 2}) RETURN count(n) AS n"));
  }

  @Test
  @DisplayName("tx.executeWrite_managed_retry — the shape a ported app uses")
  void txExecuteWriteManagedRetry() {
    try (Session session = driver.session()) {
      long written =
          session.executeWrite(
              tx -> tx.run("CREATE (n:JavaProbe {id: 3}) RETURN n.id AS id").single().get("id").asLong());
      assertEquals(3L, written);
    }
  }

  @Test
  @DisplayName("tx.autocommit_mutation_is_rejected — documented, and refused clearly")
  void txAutocommitMutationIsRejected() {
    try (Session session = driver.session()) {
      Neo4jException raised =
          assertThrows(
              Neo4jException.class,
              () -> session.run("CREATE (:JavaProbe {id: 99})").consume(),
              "expected auto-commit CREATE to be rejected");
      assertTrue(
          raised.getMessage().toLowerCase().contains("auto-commit"),
          "expected the message to mention auto-commit, got: " + raised.getMessage());
    }
  }

  @Test
  @DisplayName("tx.occ_conflict_code — the code a retry loop branches on")
  void txOccConflictCode() {
    try (Session sessionA = driver.session();
        Session sessionB = driver.session()) {
      Transaction txA = sessionA.beginTransaction();
      Transaction txB = sessionB.beginTransaction();
      txA.run("CREATE (:JavaProbe {id: 10, title: 'A'})");
      txB.run("CREATE (:JavaProbe {id: 11, title: 'B'})");
      txA.commit();
      Neo4jException raised =
          assertThrows(Neo4jException.class, txB::commit, "expected the stale commit to conflict");
      // The `Neo.TransientError.*` class is what makes the driver's
      // managed-transaction machinery retry instead of throwing through.
      assertEquals("Neo.TransientError.Transaction.Outdated", raised.code());
    }
  }

  // ── Error codes ─────────────────────────────────────────────────────────

  @Test
  @DisplayName("errors.syntax_error_code")
  void errorsSyntaxErrorCode() {
    Neo4jException raised = assertThrows(Neo4jException.class, () -> read("MATCH ((("));
    assertEquals("Neo.ClientError.Statement.SyntaxError", raised.code());
  }

  @Test
  @DisplayName("errors.codes_are_neo4j_shaped")
  void errorsCodesAreNeo4jShaped() {
    try {
      read("RETURN $missing AS v");
    } catch (Neo4jException raised) {
      String[] parts = raised.code().split("\\.");
      assertEquals(4, parts.length, "code should have 4 dotted segments: " + raised.code());
      assertEquals("Neo", parts[0]);
    }
  }

  // ── Procedures + capability gate ────────────────────────────────────────

  @Test
  @DisplayName("procedures.db_labels")
  void proceduresDbLabels() {
    List<String> labels = new ArrayList<>();
    for (Record record : read("CALL db.labels() YIELD label RETURN label ORDER BY label")) {
      labels.add(record.get("label").asString());
    }
    assertTrue(labels.contains("Person"), "expected Person in " + labels);
  }

  @Test
  @DisplayName("capability.load_csv_denied_for_remote_clients")
  void capabilityLoadCsvDeniedForRemoteClients() {
    // The server under test is started without --allow-csv-import, so a remote
    // LOAD CSV must be refused. A security assertion, not a feature one.
    Neo4jException raised =
        assertThrows(
            Neo4jException.class,
            () -> read("LOAD CSV FROM 'file:///etc/hosts' AS row RETURN row[0] AS line"),
            "expected LOAD CSV to be refused");
    assertTrue(
        raised.getMessage().contains("not enabled for this connection"),
        "expected the capability refusal, got: " + raised.getMessage());
  }
}
