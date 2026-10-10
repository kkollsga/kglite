# Timeseries

Attach time-indexed numeric data directly to nodes. You do not create separate nodes per data point. Data is stored as compact columnar arrays. You query it with date strings, through the Cypher `ts_*()` functions, at the resolution the data has.

## Configuration

Configure timeseries metadata per node type: resolution, channel names, units, and bin type.

```python
graph.set_timeseries("Project",
    resolution="month",                         # "year", "month" or "day"
    channels=["output", "flow"],                    # channel names
    units={"output": "MU", "flow": "BU"},      # optional: per-channel units
    bin_type="total",                            # optional: "total", "mean", or "sample"
)

graph.timeseries_config("Project")
# {'resolution': 'month', 'channels': ['output', 'flow'],
#  'units': {'output': 'MU', 'flow': 'BU'}, 'bin_type': 'total'}
```

## Loading data

```python
# Bulk load from a DataFrame (most common)
graph.add_timeseries(
    "Project",
    data=production_df,
    fk="person_id",                              # FK column → matches node.id
    time_key=["year", "month"],              # composite time key columns
    channels={"output": "outOutputCol", "flow": "outFlowCol"},  # channel → column
    resolution="month",                       # required if set_timeseries() wasn't called
    units={"output": "MU"},                   # optional, merged into config
)

# Or manually per node
graph.set_time_index(node_id, [[2020,1], [2020,2], [2020,3]])
graph.add_ts_channel(node_id, "output", [1.23, 1.18, 1.25])
graph.add_ts_channel(node_id, "flow", [0.45, 0.42, 0.48])
```

**Validation:** `time_key` column count must match resolution depth (1 for year, 2 for month, 3 for day).

## Inline loading via `add_nodes`

If your DataFrame has one row per time step per entity, use the `timeseries` parameter on `add_nodes`. It loads nodes and timeseries in a single call:

```python
prod_df = pd.DataFrame({
    'field_id': ['Tundra']*3 + ['Delta']*3,
    'field_name': ['Tundra']*3 + ['Delta']*3,
    'date': ['2020-01', '2020-02', '2020-03']*2,
    'output': [100, 110, 120, 200, 210, 220],
    'flow': [50, 55, 60, 80, 85, 90],
})

# Single call — creates 2 nodes with 3 time steps each
graph.add_nodes(prod_df, 'Production', 'field_id', 'field_name',
    timeseries={
        'time': 'date',                   # date string column
        'channels': ['output', 'flow'],       # value columns
    }
)
```

The `timeseries` dict accepts:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `time` | `str` or `dict` | Yes | Date string column name, or dict mapping resolution levels to column names |
| `channels` | `list[str]` | Yes | Column names containing numeric time-varying data |
| `resolution` | `str` | No | `"year"`, `"month"`, `"day"` — auto-detected if omitted |
| `units` | `dict[str, str]` | No | Per-channel unit labels |

**Separate time columns.** If time is split across multiple columns, map each resolution level to its column:

```python
graph.add_nodes(df, 'Production', 'field_id', 'field_name',
    timeseries={
        'time': {'year': 'ar', 'month': 'maned'},
        'channels': ['output', 'flow'],
    }
)
```

## Querying via Cypher

All `ts_*()` functions take **date strings** (`'2020'`, `'2020-2'`, `'2020-2-15'`, etc.). The precision of a date string is validated against the data resolution.

```python
# Aggregate monthly data by year
graph.cypher("MATCH (f:Project) RETURN f.title, ts_sum(f.output, '2020') AS prod")

# Top 10 fields by production
graph.cypher("""
    MATCH (f:Project)
    RETURN f.title, ts_sum(f.output, '2020') AS prod
    ORDER BY prod DESC LIMIT 10
""")

# Month-level range
graph.cypher("MATCH (f:Project) RETURN ts_avg(f.output, '2020-1', '2020-6') AS h1_avg")

# Multi-year range
graph.cypher("MATCH (f:Project) RETURN ts_sum(f.output, '2018', '2023') AS total")

# Exact month lookup
graph.cypher("MATCH (f:Project) RETURN ts_at(f.output, '2020-3') AS march")

# Change between periods
graph.cypher("MATCH (f:Project) RETURN ts_delta(f.output, '2019', '2021') AS change")

# Latest sensor reading
graph.cypher("MATCH (s:Sensor) RETURN s.title, ts_last(s.temperature)")

# Extract full series for plotting
graph.cypher("MATCH (f:Project {title: 'TUNDRA'}) RETURN ts_series(f.output, '2015', '2020')")
```

## Retrieval

```python
# All channels
graph.timeseries(node_id)
# {'keys': [[2020,1], [2020,2], ...], 'channels': {'output': [...], 'flow': [...]}}

# Single channel
graph.timeseries(node_id, channel="output")
# {'keys': [...], 'values': [...]}

# Date-string range filter
graph.timeseries(node_id, start='2020', end='2020')
```

**Available functions:** `ts_at`, `ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_count`, `ts_first`, `ts_last`, `ts_series`, `ts_delta`.

See the [Cypher reference](../../reference/cypher-reference.md) for the full documentation.

---

## Validity intervals

Timeseries attaches numeric channels to a node. *Validity* is a different axis.
Nodes and relationships each hold a period, such as a role from its start to its
end date, or a team membership from one transfer to the next.

You ask a validity question as of an instant, with `FOR VALID_TIME AS OF`,
`cypher(valid_at=…)` or the fluent `date()` context. See {doc}`valid-time`.
