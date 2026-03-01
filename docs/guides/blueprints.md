# Blueprints

Build a complete knowledge graph from CSV files using a declarative JSON blueprint. Instead of writing `add_nodes` / `add_connections` calls, describe your data in JSON — `from_blueprint()` handles the rest.

```python
import kglite

graph = kglite.from_blueprint("blueprint.json")
```

This guide walks through building a blueprint from scratch, starting simple and adding features incrementally.

## Your First Blueprint

Suppose you have a file `employees.csv`:

| employee_id | name    | department | salary |
|-------------|---------|------------|--------|
| 1           | Alice   | Engineering| 95000  |
| 2           | Bob     | Sales      | 72000  |
| 3           | Charlie | Engineering| 88000  |

The blueprint to load this is:

```json
{
  "settings": {
    "root": "./data"
  },
  "nodes": {
    "Employee": {
      "csv": "employees.csv",
      "pk": "employee_id",
      "title": "name"
    }
  }
}
```

That's it. Three decisions:

1. **`root`** — where the CSV files live (relative paths in the blueprint resolve from here)
2. **`pk`** — which column uniquely identifies each row (becomes the node's `id`)
3. **`title`** — which column is the display name

All other columns (`department`, `salary`) are auto-detected and stored as properties.

```python
graph = kglite.from_blueprint("blueprint.json")
graph.cypher("MATCH (e:Employee) RETURN e.name, e.salary ORDER BY e.salary DESC")
```

## Property Types

By default, column types are auto-detected from the CSV. Use `properties` to override when auto-detection isn't enough:

```json
{
  "Employee": {
    "csv": "employees.csv",
    "pk": "employee_id",
    "title": "name",
    "properties": {
      "salary": "float",
      "hired": "date",
      "department": "string"
    }
  }
}
```

Available types:

| Type | Stored as | Notes |
|------|-----------|-------|
| `"string"` | text | Default for text columns |
| `"int"` | integer | Whole numbers |
| `"float"` | float | Decimal numbers |
| `"date"` | datetime | Expects epoch milliseconds in CSV; converts to datetime |

Columns not listed in `properties` are still loaded — they just use auto-detection. You only need to specify types when auto-detection gets it wrong.

### Skipping Columns

Use `skipped` to exclude columns you don't want stored as properties:

```json
{
  "Employee": {
    "csv": "employees.csv",
    "pk": "employee_id",
    "title": "name",
    "skipped": ["internal_code", "etl_timestamp"]
  }
}
```

### Filtering Rows

Use `filter` to load only a subset of rows from the CSV:

```json
{
  "Employee": {
    "csv": "employees.csv",
    "pk": "employee_id",
    "title": "name",
    "filter": {
      "status": "Active",
      "salary": {">": 50000}
    }
  }
}
```

Simple values mean equality (`"status": "Active"` keeps only rows where status equals "Active"). Operator dicts support: `=`, `!=`, `>`, `<`, `>=`, `<=`.

## Adding Connections

### FK Edges (One-to-Many)

If `employees.csv` has a `company_id` column referencing another node type:

| employee_id | name  | company_id |
|-------------|-------|------------|
| 1           | Alice | ACME       |
| 2           | Bob   | ACME       |
| 3           | Charlie | GLOBEX   |

And you have `companies.csv`:

| company_id | company_name | industry    |
|------------|-------------|-------------|
| ACME       | Acme Corp   | Manufacturing |
| GLOBEX     | Globex Inc  | Technology    |

```json
{
  "settings": { "root": "./data" },
  "nodes": {
    "Employee": {
      "csv": "employees.csv",
      "pk": "employee_id",
      "title": "name",
      "skipped": ["company_id"],
      "connections": {
        "fk_edges": {
          "WORKS_AT": {
            "target": "Company",
            "fk": "company_id"
          }
        }
      }
    },
    "Company": {
      "csv": "companies.csv",
      "pk": "company_id",
      "title": "company_name"
    }
  }
}
```

This creates `(Employee)-[:WORKS_AT]->(Company)` edges. The `fk` column in the source CSV must match the `pk` values of the target node type.

> **Tip:** Add FK columns to `skipped` if you don't want them stored as node properties — the edge already captures the relationship.

### Manual Nodes (No CSV)

If you don't have a separate CSV for the target type, omit the `csv` field. The loader will automatically create nodes from the distinct FK values it finds:

```json
{
  "nodes": {
    "Employee": {
      "csv": "employees.csv",
      "pk": "employee_id",
      "title": "name",
      "connections": {
        "fk_edges": {
          "IN_DEPARTMENT": {
            "target": "Department",
            "fk": "department"
          }
        }
      }
    },
    "Department": {
      "pk": "name",
      "title": "name"
    }
  }
}
```

The loader scans all FK edges targeting `Department`, collects the distinct values (`"Engineering"`, `"Sales"`), and creates nodes from them.

### Junction Edges (Many-to-Many)

For many-to-many relationships, use a separate lookup CSV. Suppose `project_assignments.csv`:

| employee_id | project_id | role      | assigned_date |
|-------------|------------|-----------|---------------|
| 1           | P100       | Lead      | 1672531200000 |
| 1           | P200       | Member    | 1675209600000 |
| 2           | P100       | Member    | 1672531200000 |

```json
{
  "Employee": {
    "csv": "employees.csv",
    "pk": "employee_id",
    "title": "name",
    "connections": {
      "junction_edges": {
        "ASSIGNED_TO": {
          "csv": "project_assignments.csv",
          "source_fk": "employee_id",
          "target": "Project",
          "target_fk": "project_id",
          "properties": ["role", "assigned_date"],
          "property_types": {
            "assigned_date": "date"
          }
        }
      }
    }
  }
}
```

Junction edges can carry properties — list them in `properties` and use `property_types` for type hints. This creates `(Employee)-[:ASSIGNED_TO {role: "Lead", assigned_date: ...}]->(Project)` edges.

## Sub-Nodes

Sub-nodes are hierarchical children of a parent node type. They live in a separate CSV and link to the parent via a foreign key.

Suppose each employee has performance reviews in `reviews.csv`:

| review_id | employee_id | year | rating | summary           |
|-----------|-------------|------|--------|-------------------|
| R1        | 1           | 2024 | 5      | Excellent work    |
| R2        | 1           | 2023 | 4      | Strong performer  |
| R3        | 2           | 2024 | 3      | Meets expectations|

```json
{
  "Employee": {
    "csv": "employees.csv",
    "pk": "employee_id",
    "title": "name",
    "sub_nodes": {
      "Review": {
        "csv": "reviews.csv",
        "pk": "review_id",
        "title": "summary",
        "parent_fk": "employee_id",
        "properties": {
          "rating": "int",
          "year": "int"
        },
        "skipped": ["employee_id"]
      }
    }
  }
}
```

This creates `Review` nodes linked to their parent `Employee` via an `OF_EMPLOYEE` edge (auto-generated from the parent type name). The `parent_fk` column must match the parent's `pk` values.

> Use `"pk": "auto"` if your sub-node CSV doesn't have a natural primary key — the loader generates sequential IDs (1, 2, 3, ...).

Sub-nodes can also have their own `connections` (FK edges and junction edges), using the same syntax as core nodes.

## Timeseries

Attach time-indexed numeric data directly to nodes. This is ideal for metrics like monthly production, daily sales, or hourly sensor readings.

Suppose `monthly_sales.csv` contains per-employee sales data:

| employee_id | name  | department | yr   | mo | units_sold | revenue |
|-------------|-------|------------|------|----|------------|---------|
| 1           | Alice | Engineering| 2024 | 1  | 15         | 45000   |
| 1           | Alice | Engineering| 2024 | 2  | 22         | 66000   |
| 2           | Bob   | Sales      | 2024 | 1  | 30         | 90000   |

```json
{
  "Employee": {
    "csv": "monthly_sales.csv",
    "pk": "employee_id",
    "title": "name",
    "timeseries": {
      "time_key": {"year": "yr", "month": "mo"},
      "resolution": "month",
      "channels": {
        "units": "units_sold",
        "revenue": "revenue"
      },
      "units": {
        "units": "count",
        "revenue": "USD"
      }
    }
  }
}
```

Key points:

- **`time_key`** — a single column name (`"date_col"`) or a composite dict (`{"year": "yr", "month": "mo"}`). Composite keys support `year`, `month`, `day`, `hour`.
- **`resolution`** — `"year"`, `"month"`, `"day"`, or `"hour"`.
- **`channels`** — maps channel names (what you want to call them) to CSV column names (what they're called in the file). Format: `{"channel_name": "csv_column_name"}`.
- **`units`** — optional per-channel units.

Aggregate rows where time components are zero (e.g., `month=0` for annual totals) are automatically dropped.

After loading, query timeseries with Cypher `ts_*()` functions — see the [Timeseries guide](timeseries.md) for details.

## Spatial Data

Use special property types to enable spatial indexing and queries.

| Type | Purpose |
|------|---------|
| `"location.lat"` | Latitude coordinate column |
| `"location.lon"` | Longitude coordinate column |
| `"geometry"` | WKT geometry column (converted from GeoJSON `_geometry` column in CSV) |

```json
{
  "Office": {
    "csv": "offices.csv",
    "pk": "office_id",
    "title": "name",
    "properties": {
      "latitude": "location.lat",
      "longitude": "location.lon",
      "boundary": "geometry"
    }
  }
}
```

For `"geometry"`, the CSV must have a `_geometry` column containing GeoJSON strings. The loader converts these to WKT format and computes centroid lat/lon automatically. Requires the `shapely` package (`pip install shapely`).

After loading, use spatial queries like `distance()`, `near_point_m()`, and `contains()` — see the [Spatial guide](spatial.md) for details.

## Temporal Properties

Use `"validFrom"` and `"validTo"` types to enable temporal filtering:

```json
{
  "Contract": {
    "csv": "contracts.csv",
    "pk": "contract_id",
    "title": "name",
    "properties": {
      "start_date": "validFrom",
      "end_date": "validTo",
      "value": "float"
    }
  }
}
```

After loading, query with temporal methods:

```python
graph.select("Contract").valid_at("2024-06-15")
graph.select("Contract").valid_during("2024-01-01", "2024-12-31")
```

## Settings Reference

```json
{
  "settings": {
    "root": "./data",
    "output": "output/graph.kgl"
  }
}
```

| Key | Description |
|-----|-------------|
| `root` (or `input_root`) | Base directory for resolving CSV paths. Defaults to `"."`. |
| `output` | Path to auto-save the graph after loading (when `save=True`). |
| `output_path` | Alternative: output directory (combined with `output_file`). |
| `output_file` | Alternative: output filename (combined with `output_path`). |

## Loading Options

```python
# Basic load
graph = kglite.from_blueprint("blueprint.json")

# Verbose output — prints progress for every node/edge type
graph = kglite.from_blueprint("blueprint.json", verbose=True)

# Skip auto-save (just build in memory)
graph = kglite.from_blueprint("blueprint.json", save=False)
```

## How Loading Works

`from_blueprint()` processes nodes in dependency order across five phases:

1. **Manual nodes** — types without `csv` (created from distinct FK values found across all CSVs)
2. **Core nodes** — types with CSV files
3. **Sub-nodes** — hierarchical children, linked to parents via `parent_fk`
4. **FK edges** — direct foreign key relationships
5. **Junction edges** — many-to-many via lookup tables

Each phase depends on the previous ones completing. For example, FK edges are only created after all nodes exist.

## Complete Example

Here's a full blueprint that uses most features — a company directory with employees, departments, projects, and monthly metrics:

**`data/employees.csv`**

| employee_id | name    | department | hired         | status |
|-------------|---------|------------|---------------|--------|
| 1           | Alice   | Engineering| 1577836800000 | Active |
| 2           | Bob     | Sales      | 1609459200000 | Active |
| 3           | Charlie | Engineering| 1640995200000 | Inactive |

**`data/projects.csv`**

| project_id | project_name | budget |
|------------|-------------|--------|
| P100       | Atlas       | 500000 |
| P200       | Beacon      | 250000 |

**`data/assignments.csv`**

| employee_id | project_id | role   |
|-------------|------------|--------|
| 1           | P100       | Lead   |
| 1           | P200       | Member |
| 2           | P100       | Member |

**`data/reviews.csv`**

| employee_id | year | rating | summary           |
|-------------|------|--------|-------------------|
| 1           | 2024 | 5      | Excellent work    |
| 2           | 2024 | 4      | Strong performer  |

**`blueprint.json`**

```json
{
  "settings": {
    "root": "./data",
    "output": "output/company.kgl"
  },
  "nodes": {
    "Employee": {
      "csv": "employees.csv",
      "pk": "employee_id",
      "title": "name",
      "properties": {
        "hired": "date"
      },
      "skipped": ["department"],
      "filter": {"status": "Active"},
      "connections": {
        "fk_edges": {
          "IN_DEPARTMENT": {
            "target": "Department",
            "fk": "department"
          }
        },
        "junction_edges": {
          "ASSIGNED_TO": {
            "csv": "assignments.csv",
            "source_fk": "employee_id",
            "target": "Project",
            "target_fk": "project_id",
            "properties": ["role"]
          }
        }
      },
      "sub_nodes": {
        "Review": {
          "csv": "reviews.csv",
          "pk": "auto",
          "title": "summary",
          "parent_fk": "employee_id",
          "properties": {"rating": "int", "year": "int"},
          "skipped": ["employee_id"]
        }
      }
    },
    "Department": {
      "pk": "name",
      "title": "name"
    },
    "Project": {
      "csv": "projects.csv",
      "pk": "project_id",
      "title": "project_name",
      "properties": {"budget": "float"}
    }
  }
}
```

```python
graph = kglite.from_blueprint("blueprint.json", verbose=True)

# Query the loaded graph
graph.cypher("MATCH (e:Employee)-[:IN_DEPARTMENT]->(d) RETURN d.title, count(e)")
graph.cypher("MATCH (e:Employee)-[:ASSIGNED_TO]->(p:Project) RETURN e.name, p.title")
graph.cypher("MATCH (e:Employee)<-[:OF_EMPLOYEE]-(r:Review) RETURN e.name, r.rating")
```

## Troubleshooting

### Missing CSV files

Non-fatal. The loader logs an error and continues — the graph is created with whatever data is available. Check the console output for `error(s)` at the end of loading.

### FK column has NaN or missing values

Rows with NaN in a foreign key column are silently skipped when creating edges. The nodes are still created — only the edge for that row is omitted.

### Float IDs (e.g., `260.0` instead of `260`)

Pandas reads integer columns with NaN as `float64`. The loader automatically coerces whole-number floats back to int for ID matching. No action needed.

### Filter not working

Filters compare values exactly — `{"status": "Active"}` won't match `"active"` or `" Active"` (leading space). Check for case and whitespace in your CSV.

### Timeseries aggregate rows

If your CSV has aggregate rows (e.g., `month=0` for annual totals), they are automatically dropped. Only rows with non-zero time components are loaded.

### Geometry requires shapely

If your blueprint uses `"geometry"`, `"location.lat"`, or `"location.lon"` types, install shapely:

```bash
pip install shapely
```

The CSV must have a `_geometry` column containing GeoJSON strings for geometry conversion.
