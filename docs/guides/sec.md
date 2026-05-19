# SEC EDGAR dataset

KGLite ships a built-in loader for SEC EDGAR filings — US public-company
financial disclosures — at `kglite.datasets.sec.SEC`. CC0 source data,
no licensing constraints on redistributing a `.kgl` graph.

## Quick start

```python
from kglite.datasets.sec import SEC

g = SEC.open(
    "./sec_workdir",
    years=10,                            # historical Filing index
    detailed=2,                          # full payload window
    mode="mapped",                       # or "memory" / "disk"
    user_agent="Acme Research contact@acme.com",  # REQUIRED
)
```

The SEC's fair-access policy mandates the `user_agent` header on every
request. Missing or generic UA → 403. Use your name + email.

## Workdir layout

Three-tier strict cache:

```
sec_workdir/
  raw/                       # immutable byte-for-byte SEC cache
    index/                   #   quarterly master.idx files
    submissions/             #   bulk submissions.zip + extracts
    insider/                 #   Form 4 JSONL (when fetched)
    form13f/                 #   13F-HR XML/TSV (when fetched)
    financials/              #   FSNDS num.tsv (when fetched)
    filings/                 #   per-filing payloads (Exhibit 21, 8-K)
    company_tickers.json
  processed/                 # parsed CSVs (shared across modes)
    company.csv  filing.csv
    person.csv  transaction.csv  has_insider.csv
    institutional_manager.csv  security.csv  holds.csv
  graph/
    memory/sec.kgl           # built graph per storage mode
    mapped/sec.kgl           # …each mode lives in its own subdir
    disk/                    # …and coexists with the others
```

Reopening `SEC.open(path, mode=X)` loads the cached graph for `X` if
it exists. Different modes coexist freely; opening one never touches
the others.

## Schema

| Node | Source | Notes |
|---|---|---|
| `Company` | submissions.zip | nid = CIK (integer) |
| `Filing` | submissions + master.idx | nid = accession_number (string) |
| `Person` | Form 4 XML | nid = reporter CIK (integer) |
| `Transaction` (sub of Person) | Form 4 XML | one per insider transaction |
| `InstitutionalManager` | 13F-HR XML | nid = manager CIK (integer) |
| `Security` | 13F-HR + Form 4 | nid = CUSIP |

| Edge | From → To | Notes |
|---|---|---|
| `FILED_BY` | Filing → Company | every filing's filer |
| `HAS_INSIDER` | Company → Person | junction with role flags + title |
| `OF_PERSON` | Transaction → Person | sub-node parent link |
| `INVOLVES_ISSUER` | Transaction → Company | the issuer (often = filer) |
| `REPORTED_IN_FILING` | Transaction → Filing | source filing |
| `HOLDS` | InstitutionalManager → Security | junction with shares/value/voting |

## Storage modes

| `mode` | When |
|---|---|
| `"memory"` | Small slices, fastest cold queries. Heap-resident graph. |
| `"mapped"` | Default. mmap-backed columnar — survives Python session restarts cheaply. |
| `"disk"` | Very large graphs (full XBRL ingest, multi-year deep windows). CSR + mmap. |

## Sizing

Approximate sizes by configuration (workdir cache):

| `years` | `detailed` | Detail parsers | Workdir |
|---|---|---|---|
| 0 | 2 | none | ~12 GB |
| 10 | 0 | n/a | ~7 GB |
| 10 | 2 | all on | ~22 GB |
| `"all"` | 2 | all on | ~30 GB |
| 10 | 5 | all on | ~50 GB |

## Caveats

- **CIK is stored as an integer**, not the zero-padded display form.
  Query with `MATCH (c:Company {cik: 320193})`. Reconstruct the
  zero-padded form via `lpad(toString(c.cik), 10, '0')` when needed
  for SEC URLs.
- **All-digit CUSIPs** auto-type to integer too. CUSIPs with letters
  (~5% of the universe) round-trip as strings.
- **Form 4 / 13F XML fetchers** are per-filing (no SEC bulk dataset
  exists). Per-filing fetches are rate-limited at 10 req/s.

## MCP exposure

The built `.kgl` works as a generic KGLite graph for the MCP server:

```bash
kglite-mcp-server --graph sec_workdir/graph/mapped/sec.kgl
```

…and agents get `cypher_query` + `graph_overview` against the SEC
graph without any SEC-specific server code.
