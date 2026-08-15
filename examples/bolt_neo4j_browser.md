# Pointing Neo4j Browser at kglite-bolt-server

`kglite-bolt-server` speaks the Neo4j Bolt v5 wire protocol, so the
**Neo4j Browser** — the standard graph GUI — can connect to it when the
server is started with `--neo4j-compat` (the Browser reads
`dbms.components()` for its version banner, and the flag makes that row —
and the handshake agent — report a Neo4j identity). This walkthrough builds
a small graph, serves it, and explores it in the browser.

> Browser's connect sequence varies across its releases; if your version
> shows a connection error or an empty sidebar, run the server with
> `RUST_LOG=debug` — every incoming query is logged, so the unmet call is
> visible immediately.

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
./target/release/kglite-bolt-server --graph people.kgl --neo4j-compat
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
./target/release/kglite-bolt-server --graph people.kgl --readonly --neo4j-compat
```

## 3. Connect Neo4j Browser

Open Neo4j Browser (the desktop app, or the web build at
<https://browser.neo4j.io>) and connect with:

- **Connect URL:** `bolt://localhost:7687`
- **Authentication type:** *No authentication* (or *Username / Password*
  with any values if you started with `--auth none`; with `--auth basic`,
  use the credentials you set).

Click **Connect**. The browser performs the Bolt handshake and reads the
server's identity from `dbms.components()`; under `--neo4j-compat` both
report a Neo4j 5.26 identity (with the real product kept in the agent
string).

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
- **Introspection.** The sidebar's label / relationship-type / property
  lists come from `db.labels()`, `db.relationshipTypes()` and
  `db.propertyKeys()`; the schema tab renders
  `db.schema.visualization()`; `SHOW DATABASES`, `dbms.components()` and
  `dbms.showCurrentUser()` are answered by the server. These are the calls
  Browser's connect sequence makes — verified against the Bolt server with
  the official Python driver; if your Browser version sends something more,
  the `RUST_LOG=debug` query log will show it.
- For a programmatic client instead of the GUI, see
  [`bolt_client_neo4j_python.py`](bolt_client_neo4j_python.py).
