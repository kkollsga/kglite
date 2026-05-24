# Timeseries

Attach time-indexed numeric data directly to nodes — no need to create separate nodes per data point. Data is stored as compact columnar arrays with resolution-aware date-string queries through Cypher `ts_*()` functions.

## Configuration

Configure timeseries metadata per node type: resolution, channel names, units, and bin type.

```python
graph.set_timeseries("Field",
    resolution="month",                         # "year", "month", "day", "hour", "minute"
    channels=["oil", "gas"],                    # channel names
    units={"oil": "MSm3", "gas": "BSm3"},      # optional: per-channel units
    bin_type="total",                            # optional: "total", "mean", or "sample"
)

graph.timeseries_config("Field")
# {'resolution': 'month', 'channels': ['oil', 'gas'],
#  'units': {'oil': 'MSm3', 'gas': 'BSm3'}, 'bin_type': 'total'}
```

## Loading Data

```python
# Bulk load from a DataFrame (most common)
graph.add_timeseries(
    "Field",
    data=production_df,
    fk="npdid",                              # FK column → matches node.id
    time_key=["year", "month"],              # composite time key columns
    channels={"oil": "prfOilCol", "gas": "prfGasCol"},  # channel → column
    resolution="month",                       # required if set_timeseries() wasn't called
    units={"oil": "MSm3"},                   # optional, merged into config
)

# Or manually per node
graph.set_time_index(node_id, [[2020,1], [2020,2], [2020,3]])
graph.add_ts_channel(node_id, "oil", [1.23, 1.18, 1.25])
graph.add_ts_channel(node_id, "gas", [0.45, 0.42, 0.48])
```

**Validation:** `time_key` column count must match resolution depth (1 for year, 2 for month, 3 for day, 4 for hour, 5 for minute).

## Inline Loading via `add_nodes`

When your DataFrame has one row per time step per entity, use the `timeseries` parameter on `add_nodes` to load nodes and timeseries in a single call:

```python
prod_df = pd.DataFrame({
    'field_id': ['Troll']*3 + ['Draugen']*3,
    'field_name': ['Troll']*3 + ['Draugen']*3,
    'date': ['2020-01', '2020-02', '2020-03']*2,
    'oil': [100, 110, 120, 200, 210, 220],
    'gas': [50, 55, 60, 80, 85, 90],
})

# Single call — creates 2 nodes with 3 time steps each
graph.add_nodes(prod_df, 'Production', 'field_id', 'field_name',
    timeseries={
        'time': 'date',                   # date string column
        'channels': ['oil', 'gas'],       # value columns
    }
)
```

The `timeseries` dict accepts:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `time` | `str` or `dict` | Yes | Date string column name, or dict mapping resolution levels to column names |
| `channels` | `list[str]` | Yes | Column names containing numeric time-varying data |
| `resolution` | `str` | No | `"year"`, `"month"`, `"day"`, `"hour"`, `"minute"` — auto-detected if omitted |
| `units` | `dict[str, str]` | No | Per-channel unit labels |

**Separate time columns** — when time is split across multiple columns:

```python
graph.add_nodes(df, 'Production', 'field_id', 'field_name',
    timeseries={
        'time': {'year': 'ar', 'month': 'maned'},
        'channels': ['oil', 'gas'],
    }
)
```

## Querying via Cypher

All `ts_*()` functions use **date strings** (`'2020'`, `'2020-2'`, `'2020-2-15'`, etc.). Precision is validated against the data resolution.

```python
# Aggregate monthly data by year
graph.cypher("MATCH (f:Field) RETURN f.title, ts_sum(f.oil, '2020') AS prod")

# Top 10 fields by production
graph.cypher("""
    MATCH (f:Field)
    RETURN f.title, ts_sum(f.oil, '2020') AS prod
    ORDER BY prod DESC LIMIT 10
""")

# Month-level range
graph.cypher("MATCH (f:Field) RETURN ts_avg(f.oil, '2020-1', '2020-6') AS h1_avg")

# Multi-year range
graph.cypher("MATCH (f:Field) RETURN ts_sum(f.oil, '2018', '2023') AS total")

# Exact month lookup
graph.cypher("MATCH (f:Field) RETURN ts_at(f.oil, '2020-3') AS march")

# Change between periods
graph.cypher("MATCH (f:Field) RETURN ts_delta(f.oil, '2019', '2021') AS change")

# Latest sensor reading
graph.cypher("MATCH (s:Sensor) RETURN s.title, ts_last(s.temperature)")

# Extract full series for plotting
graph.cypher("MATCH (f:Field {title: 'TROLL'}) RETURN ts_series(f.oil, '2015', '2020')")
```

## Retrieval

```python
# All channels
graph.timeseries(node_id)
# {'keys': [[2020,1], [2020,2], ...], 'channels': {'oil': [...], 'gas': [...]}}

# Single channel
graph.timeseries(node_id, channel="oil")
# {'keys': [...], 'values': [...]}

# Date-string range filter
graph.timeseries(node_id, start='2020', end='2020')
```

**Available functions:** `ts_at`, `ts_sum`, `ts_avg`, `ts_min`, `ts_max`, `ts_count`, `ts_first`, `ts_last`, `ts_series`, `ts_delta`. See the [Cypher reference](../reference/cypher-reference.md) for the full documentation.
