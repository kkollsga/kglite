# Pointing Neo4j Browser at kglite-bolt-server

`kglite-bolt-server` speaks the Neo4j Bolt v5 wire protocol, so the
**Neo4j Browser** — the standard graph GUI — connects to it unchanged. This
walkthrough builds a small graph, serves it, and explores it in the browser.

## 1. Build a graph and save it

```python
import pandas as pd
import kglite

g = kglite.KnowledgeGraph()
people = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "title": ["Alice", "Bob", "Carol", "Dave"],
        "city": ["Oslo", "Bergen", "Oslo", "Trondheim"],
    }
)
g.add_nodes(people, "Person", "id", "title")
edges = pd.DataFrame({"src": [1, 2, 3], "dst": [2, 3, 4]})
g.add_connections(edges, "KNOWS", "Person", "src", "Person", "dst")
g.save("people.kgl")
```

## 2. Start the server

```bash
# Build the binary first (once):
cargo build -p kglite-bolt-server --release

# Serve the graph on the default Bolt port (7687):
./target/release/kglite-bolt-server --graph people.kgl
```

The server listens on `127.0.0.1:7687` with authentication **disabled** by
default. Useful flags:

| Flag | Purpose |
|---|---|
| `--port 7687` | Bolt port (default `7687`, the Neo4j default). |
| `--bind 0.0.0.0` | Listen on all interfaces (default `127.0.0.1`). |
| `--readonly` | Reject all writes — recommended for browsing. |
| `--auth basic --auth-user neo4j --auth-pass secret` | Require credentials. |
| `--advertise-addr host:port` | Address returned to routing-aware drivers. |
| `--tls-cert cert.pem --tls-key key.pem` | Serve over TLS. |

For read-only exploration, start it as:

```bash
./target/release/kglite-bolt-server --graph people.kgl --readonly
```

## 3. Connect Neo4j Browser

Open Neo4j Browser (the desktop app, or the web build at
<https://browser.neo4j.io>) and connect with:

- **Connect URL:** `bolt://localhost:7687`
- **Authentication type:** *No authentication* (or *Username / Password*
  with any values if you started with `--auth none`; with `--auth basic`,
  use the credentials you set).

Click **Connect**. The browser performs the Bolt handshake against
kglite-bolt-server exactly as it would against Neo4j.

## 4. Explore

Run Cypher in the browser's query bar. The graph visualisation, table view,
and result export all work, because nodes and relationships round-trip as
real PackStream `Node` / `Relationship` structs.

```cypher
// Everyone, drawn as a graph
MATCH (p:Person) RETURN p

// People in Oslo
MATCH (p:Person) WHERE p.city = 'Oslo' RETURN p.title AS name

// Who Alice knows, 1-2 hops out
MATCH (a:Person {title: 'Alice'})-[:KNOWS*1..2]->(b:Person)
RETURN DISTINCT b.title AS friend

// The relationship graph (renders as a visual)
MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b
```

## Notes

- **Single graph, single database.** kglite is embedded — there's no
  multi-database concept. The browser's database selector is cosmetic;
  everything runs against the one loaded `.kgl`.
- **Writes** require `BEGIN`/`COMMIT` (the driver does this automatically)
  and a server started without `--readonly`. Mutations land in memory; they
  are not written back to the `.kgl` file.
- **`db.*` procedures.** `CALL db.labels()` and `CALL db.relationshipTypes()`
  work and return Neo4j-aligned column names (`label`, `relationshipType`),
  so browser sidebar introspection populates correctly.
- For a programmatic client instead of the GUI, see
  [`bolt_client_neo4j_python.py`](bolt_client_neo4j_python.py).
