# Cat G-N test fixtures

Minimal graph fixtures for the forward-looking categories of the
pre-release test suite spec (categories G–N from the audit message
on 2026-05-12).

Each `.kgl` is built synthetically by `build_fixtures.py` (see cover
message) — deterministic, < 3 KB each, no dependency on real-world
data. The committed binaries use the current RGF5/Postcard format and
must be regenerated after a persistence-format change. Each is paired
with a tiny `_mcp.yaml` declaring `name`,
`instructions`, and the target category in `overview_prefix`.

## Files

| File | Pair YAML | Size | Target tests |
|---|---|---:|---|
| `spatial_graph.kgl` | `spatial_graph_mcp.yaml` | 1.4 KB | Cat J — contains, centroid, distance |
| `timeseries_graph.kgl` | `timeseries_graph_mcp.yaml` | 2.3 KB | Cat K — ts_sum, ts_at, ts_series |
| `graph_with_orphans.kgl` | `graph_with_orphans_mcp.yaml` | 1.2 KB | Cat L — CALL orphan_node |
| `graph_with_duplicates.kgl` | `graph_with_duplicates_mcp.yaml` | 0.9 KB | Cat L — CALL duplicate_title |

## Sanity-check assertions (already verified at build time)

| Query | Expected |
|---|---|
| `MATCH (a:Area), (w:Well) WHERE contains(a, point(w.latitude, w.longitude)) RETURN count(*)` | `3` |
| `MATCH (a:Area {id: 'north'}) RETURN centroid(a)` | `{latitude: ~61.0, longitude: ~5.0}` |
| `MATCH (f:Field {title:'TROLL'}) RETURN ts_sum(f.oil_col, '2019')` | positive float |
| `MATCH (f:Field {title:'TROLL'}) RETURN ts_at(f.oil_col, '2019-3')` | positive float |
| `CALL orphan_node({type:'Wellbore'}) YIELD node RETURN count(node)` | `3` |
| `CALL duplicate_title({type:'Prospect'}) YIELD node RETURN count(node)` | `4` |

## Use in the pre-release suite

Suggested test invocation pattern (matches the spec we sent):

```python
@pytest.fixture
def spatial_graph(): return BINARY_ROOT / "tests/fixtures/spatial_graph.kgl"

def test_spatial_contains_point(mcp_server, spatial_graph):
    proc = mcp_server("--graph", str(spatial_graph))
    body = call_tool(proc, "cypher_query", {
        "query": "MATCH (a:Area) WHERE contains(a, point(61.0, 5.0)) RETURN a.title"
    })
    proc.kill()
    assert "NORTH_BLOCK" in body
```

## Regeneration

`build_fixtures.py` is in the cover message. Re-running it produces
byte-identical fixtures (seeded random for the timeseries values).
