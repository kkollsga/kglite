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
| `"bool"` / `"boolean"` | boolean | Accepts true/false and common 1/0/yes/no forms |
| `"date"` / `"datetime"` | date | Accepts `YYYY-MM-DD`, timestamp text, or epoch milliseconds; stores the date |
| `"list"` / `"array"` | list | Cell is a JSON array, e.g. `["a","b"]` — see below |
| `"validFrom"` / `"validTo"` | date | Date column plus temporal-role metadata |
| `"geometry"` | WKT string | Uses existing WKT or converts `_geometry` GeoJSON in Rust |
| `"location.lat"` / `"location.lon"` | float | Coordinates; may receive GeoJSON centroids |

Columns not listed in `properties` are still loaded — they just use auto-detection. You only need to specify types when auto-detection gets it wrong.

### List Columns

A CSV cell holding several values is loaded as a list when you declare the column
`"list"` (or `"array"`, the same keyword). The cell must be a **JSON array**:

```csv
gene_id,name,synonyms
1,adhE,"[""adhC"",""ADHE""]"
2,pfkA,"[""pfk1""]"
```

```json
{"Gene": {"csv": "genes.csv", "pk": "gene_id", "properties": {"synonyms": "list"}}}
```

```cypher
MATCH (g:Gene) WHERE 'ADHE' IN g.synonyms RETURN g.name
```

There is deliberately no delimiter option — no `"list:|"`, no `sep` key. A
delimited column is ambiguous the moment a value contains the delimiter, and the
blueprint has no way to say which values were escaped. Split the column into a
JSON array in your export step instead.

A cell that is not a JSON array becomes a **one-element** list holding the cell
whole. That is right for a lone value and wrong for `adhC|ADHE`, so the build
report warns when a non-array cell contains `|`, `;` or `,`, naming the column,
how many cells are affected, and the first offending row and its text:

```
node 'Gene': column 'synonyms' is declared list but 1 cell(s) are not a JSON
array and contain a separator ('|', ';' or ','); each was kept whole as a
one-element list. First at row 1: 'adhC|ADHE'. Write list cells as JSON
arrays, e.g. ["a","b"].
```

Junction-edge properties take the same keyword through `property_types`.

CSV **export** is the one place lists do not round-trip exactly: `to_csv` writes
a list as its JSON text and the generated blueprint declares it `"string"`, so
re-importing that output gives you the text back, not a list. Declare `"list"`
yourself in the re-import blueprint if you need one.

### Labels

`labels` stamps secondary labels on every node of a type, so a query can name
one label instead of enumerating the types under it:

```json
{
  "Disease":   {"csv": "diseases.csv",   "pk": "id", "labels": ["Condition"]},
  "Phenotype": {"csv": "phenotypes.csv", "pk": "id", "labels": ["Condition"]}
}
```

```cypher
MATCH (c:Condition) RETURN count(c)
```

The rules:

- The node type is the primary label and is stamped for you. Listing it in
  `labels` is a no-op, not a duplicate.
- **A blueprint owns every node of the types it declares.** Labels are stamped
  after all node *and* edge phases, so a provisional stub — an endpoint some
  edge referenced and no CSV supplied — carries them too. Without that,
  `MATCH (:Condition)` would silently miss exactly the nodes that arrived via
  an edge rather than a row.
- `sub_nodes` entries take the key on the same terms.
- If you merge blueprints with a deep-merge helper, note that arrays are
  replaced wholesale rather than concatenated: the last `labels` array wins.

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

### Properties on an FK Edge

An `fk_edges` entry reads the same three property keys a junction edge does.
The columns come from the source node's own row — the one that carried the FK
value — so the edge can record *how* the two are related:

```json
{
  "Employee": {
    "csv": "employees.csv",
    "pk": "employee_id",
    "title": "name",
    "skipped": ["company_id"],
    "connections": {
      "fk_edges": {
        "WORKS_AT": {
          "target": "Company",
          "fk": "company_id",
          "properties": ["role", "hired_on"],
          "property_types": {"hired_on": "date"},
          "rename": {"hired_on": "validFrom"}
        }
      }
    }
  }
}
```

That builds `(Employee)-[:WORKS_AT {role: "Lead", validFrom: "2023-01-01"}]->(Company)`.
The rules are the junction ones: `property_types` stays keyed by the **CSV**
spelling, `rename` keys must be listed in `properties`, and neither the `fk`
nor the `pk` column is renamable or declarable as a property — both are build
errors that skip the edge. A property column the CSV does not have is reported
and the edge is built without it.

```{note}
Declaring a column as an edge property never implicitly skips it from the
node. The two landings are independent: `properties` copies the column onto
the edge, `skipped` is what keeps it off the node. List the column in both if
you want it only on the edge.
```

Rows whose FK cell is empty produce no edge, and their property values go with
them — an edge's properties always come from the row that created it.

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

### A Junction Over a Union of Target Types

When a relationship's range is an abstract class — `ASSOCIATED_WITH` pointing
at a `Disease`, a `Phenotype` **or** an `Exposure` — `target` takes a list
instead of a string:

```json
{
  "ASSOCIATED_WITH": {
    "csv": "associations.csv",
    "source_fk": "microbe_id",
    "target": ["Disease", "Phenotype", "Exposure"],
    "target_type_column": "target_type",
    "target_fk": "target_id"
  }
}
```

One relationship name, three target types. Without it the same data needs
`ASSOCIATED_WITH_DISEASE` / `_PHENOTYPE` / `_EXPOSURE`, which no query and no
ontology `range` declaration can put back together.

Each row picks its target type one of two ways:

- **`target_type_column`** names a CSV column holding the type per row. Its
  values must be among the declared `target` types; a row naming anything else
  builds no edge and the build report says how many rows named which value.
  The column is routing only — it becomes an edge property the same way any
  other column does, by being listed in `properties`.
- **Without it**, the declared types are probed in order and the first that
  already has a node with the row's target id wins. An id no declared type has
  takes the *first* declared type, where the usual missing-endpoint handling
  vivifies its stub — the edge is never dropped.

Declare the union in the ontology as one relationship whose `range` is the
abstract class, and `ontology_audit()` reports it as one rule:

```json
{"classes": {"Outcome": {"abstract": true}, "Disease": {"is_a": "Outcome"}},
 "relationships": {"ASSOCIATED_WITH": {"domain": "Microbe", "range": "Outcome"}}}
```

```{note}
Union targets are a junction-edge feature. An `fk_edges` entry still points at
exactly one `target`: its rows come from the source node's own CSV, where a
per-row target type would be a column of that node's table rather than of the
relationship.
```

### Renaming Junction Properties

To store a column under a different property name, add a `rename` map. All
three keys can appear on one edge — `properties` selects the columns,
`property_types` declares their types, `rename` decides the property name
each one lands under:

```json
{
  "ASSIGNED_TO": {
    "csv": "project_assignments.csv",
    "source_fk": "employee_id",
    "target": "Project",
    "target_fk": "project_id",
    "properties": ["role", "assigned_date"],
    "property_types": {"assigned_date": "date"},
    "rename": {"assigned_date": "validFrom"}
  }
}
```

That builds `(Employee)-[:ASSIGNED_TO {role: "Lead", validFrom: "2023-01-01"}]->(Project)`:
the column is *typed* as `assigned_date` and *stored* as `validFrom`. The old
name is gone — `r.assigned_date` is null afterwards.

```{warning}
`property_types` stays keyed by the **CSV spelling**, never the renamed one.
`rename` runs after typing, so `"property_types": {"validFrom": "date"}`
matches no column: the type declaration is silently skipped and the value
falls through to inference — the epoch integer `1672531200000` instead of the
date `"2023-01-01"`. Nothing warns about it, because an unknown *key* is
indistinguishable from a column you chose not to type. This mistake has been
made twice in production loaders; check the keys against the CSV header, not
against the property names you expect to query.
```

`rename` keys must be columns listed in `properties`, and the fk columns are
not renamable — both are build errors that skip the junction. And note that
`property_types` itself never renames anything: it declares column types,
and an unrecognized *value* there (`"property_types": {"from": "renamedTo"}`)
is ignored with a build warning.

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

If `_geometry` contains GeoJSON, the Rust loader converts it to WKT and can
populate centroid latitude/longitude. Existing WKT passes through unchanged.
Plain lat/lon needs no `_geometry`, and blueprint conversion needs no Shapely.

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

## Declaring Inputs

`"csv": "diseases.csv"` on a node spec or a junction edge names an input
inline. When several specs read the same file — a node type and the junction
that links it, or two node types carved out of one table — the path is repeated
at every one of them, and moving the file means editing each.

A `files` section declares each input once by name; specs then reference it
with `"file"`:

```json
{
  "settings": { "root": "./data" },
  "files": {
    "diseases": { "path": "disease.csv", "format": "csv" },
    "links":    { "path": "disease_gene.csv" }
  },
  "nodes": {
    "Disease": {
      "file": "diseases",
      "pk": "id",
      "connections": {
        "junction_edges": {
          "ASSOCIATED_WITH": {
            "file": "links",
            "source_fk": "disease_id",
            "target": "Gene",
            "target_fk": "gene_id"
          }
        }
      }
    }
  }
}
```

| Key | Description |
|-----|-------------|
| `path` | The file this input reads, resolved against `settings.root`. Required for a file-backed format. |
| `format` | How to read it: `"csv"` (the default), `"delimited"` (below), or `"frame"` — an in-memory table passed to `from_blueprint(..., frames={...})`, which takes no `path`. Each format brings its own keys. |

`"csv": "x.csv"` remains valid and is exactly shorthand for a `files` entry
`{ "path": "x.csv", "format": "csv" }` named `x.csv`, so the two spellings
build the same graph and the two styles mix freely in one blueprint. Two specs
naming the same input — by `file` or by the same `csv` string — read one input,
not two.

The build refuses, rather than guessing, when:

- a spec sets both `csv` and `file`;
- `file` names an entry `files` does not declare (the error lists the ones it
  does);
- a `files` entry has no `path`;
- a `files` entry's `format` is not one this build reads (the error lists
  those);
- a `files` entry is named after a `csv` shorthand that means a different file
  — both would claim the same input name.

A stray key inside a `files` entry is a warning, like stray keys elsewhere in a
blueprint, and names the accepted keys for that entry's format.

The `compute:` pipeline reads and rewrites CSV files directly, so a compute op
whose source type reads a non-CSV input is refused at load time.

### `format: "delimited"` — separators, preambles and headerless files

Public bulk data is full of tables a CSV reader cannot open. A `delimited`
entry names the separator itself, so those files are read where they land
instead of being pre-processed into CSV first.

NCBI's taxonomy dump separates fields with `\t|\t` and closes every line with
`\t|`, and has no header row:

```json
{
  "files": {
    "taxa": {
      "path": "nodes.dmp",
      "format": "delimited",
      "delimiter": "\t|\t",
      "line_suffix": "\t|",
      "header": false,
      "columns": ["tax_id", "parent_tax_id", "rank", "embl_code", "division_id"]
    }
  },
  "nodes": {
    "Taxon": {
      "file": "taxa",
      "pk": "tax_id",
      "properties": { "rank": "string" },
      "connections": {
        "fk_edges": { "HAS_PARENT": { "target": "Taxon", "fk": "parent_tax_id" } }
      }
    }
  }
}
```

BugSigDB's export puts a licence line above the header. Count it, or mark it:

```json
{
  "files": {
    "studies": { "path": "full_dump.csv", "format": "delimited", "delimiter": ",", "skip_lines": 1 },
    "same":    { "path": "full_dump.csv", "format": "delimited", "delimiter": ",", "comment_prefix": "#" }
  }
}
```

| Key | Description |
|-----|-------------|
| `delimiter` | The text between two fields. Required, any length — `","`, `"\t"`, `"\t|\t"`. |
| `quote` | Quote character, a single ASCII character. Defaults to `"`. Only for a single-character `delimiter` (see below). |
| `header` | `true` (default): the first surviving line names the columns. `false`: `columns` names them. |
| `columns` | The column names, in order. Required with `"header": false`, and refused with `"header": true` — the two would name the columns twice. |
| `skip_lines` | Physical lines dropped before anything else looks at the file. How a licence preamble goes. |
| `comment_prefix` | Lines starting with this are dropped wherever they occur. |
| `line_suffix` | Removed once from the end of every line, before splitting — so a `\t|` trailer never becomes a phantom last column. |
| `encoding` | `"utf-8"` (default) or `"latin-1"`. Any other name is refused rather than mojibaked. |
| `prefix_strip` | `{ "column": "prefix" }` — removed from the start of that column's cells, before typing. `cpd:C00022` becomes `C00022`. A cell without the prefix keeps its value, and a column the file does not have is ignored. |

**One knob picks the engine.** A single-character `delimiter` is read by the
same reader the `csv` format uses, so quoting, escapes and newlines inside
quoted fields behave exactly as they do there. A longer one is read line by
line with **no quoting at all** — no such convention exists for those files —
and a `quote` declared beside it is refused rather than silently ignored.
Everything else is shared: a UTF-8 BOM is stripped either way, and rows land
rectangular exactly as a CSV's do — a short row is null-padded, fields past the
header's width are dropped, and an empty cell is null.

`skip_lines`, `comment_prefix` and `line_suffix` are applied line by line,
before quoting, so a value spanning several lines inside quotes is not exempt
from them.

**Row numbers count data rows.** A warning saying "row 12" means the twelfth
row of data — after `skip_lines`, comment lines, blank lines and the header are
gone — the same thing it counts for a CSV, not the physical line number. Read
errors, which have no data row to attribute yet, name the physical line
instead.

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
| `root` (or `input_root`) | Base directory for resolving input paths — `files` entries and `csv` shorthands alike. Defaults to `"."`. |
| `output` | Path to auto-save the graph to after loading. |
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

### Where the graph gets saved

A build has a **save destination** when either of these is true:

- the blueprint declares `output` (or `output_path` + `output_file`), or
- `storage="disk"` was given a `path`.

The disk case matters because in disk mode the directory *is* the graph, and
building alone leaves it unpublished — a directory that looks like a graph but
that `kglite.load()` refuses with *"missing disk_graph_meta.json"*. Saving is
what publishes it:

```python
kglite.from_blueprint("blueprint.json", storage="disk", path="graph/")
reopened = kglite.load("graph/")     # works — the build published the directory
```

The `save` argument then selects the policy:

| `save` | Behaviour |
|--------|-----------|
| omitted (default) | Save if a destination exists; build in memory if not. |
| `True` | Save; raise `ValueError` if there is no destination. |
| `False` | Never save. |

Passing `save=True` on a blueprint with no `output` and no disk `path` is an
error rather than a silent no-op, so a pipeline that believes it is persisting
its output finds out at the first run.

## How Loading Works

`from_blueprint()` first applies the ordered top-level `compute` pipeline, then
processes graph construction in dependency order. Compute operations are
`derive` (row properties), `filter` (in-place or into a new type), `chain`
(ordered group edges), `calendar` (Date hierarchy/linking), and `aggregate`
(summary nodes/edges). Later operations can consume earlier outputs.

Graph construction then has five steps:

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

### Geometry inputs

Blueprint GeoJSON → WKT/centroid conversion runs in Rust and needs no Shapely.
Supply `_geometry` only for GeoJSON conversion; existing WKT and plain lat/lon
columns are accepted directly. Shapely remains optional for Python-side
geometry objects and GeoDataFrame helpers outside the blueprint loader.
