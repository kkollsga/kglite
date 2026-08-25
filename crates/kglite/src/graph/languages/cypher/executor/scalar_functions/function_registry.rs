//! The single source of truth for the *function* surface, the way
//! `procedure_registry` is for procedures.
//!
//! Two consumers read this table: the `SHOW FUNCTIONS` statement (Neo4j
//! clients — G.V(), Browser — send it for function autocomplete on connect)
//! and the drift gate below.
//!
//! **Why the gate exists.** A hand-maintained function list is worthless the
//! day someone adds a `match` arm to one of the category modules and forgets
//! this file — the listing then advertises a surface that is not the surface.
//! The dispatcher lives in nine `match name { … }` blocks and cannot be
//! enumerated from Rust, so the table is written by hand *and every entry is
//! executed against the real dispatcher* by
//! [`tests::every_registry_name_dispatches`]: an entry naming a function the
//! engine does not have fails the build. That direction — registry → engine —
//! is the hard one, and it is closed.
//!
//! The reverse direction (engine → registry: a new arm nobody registered) is
//! soft; it is a doc-comment on the dispatcher chain in
//! [`super::CypherExecutor::evaluate_scalar_function`]. Aggregates are the
//! exception: they are a single enumerable list in
//! `ast::is_aggregate_function_name`, so [`tests::aggregate_names_round_trip`]
//! closes both directions for that category.
//!
//! Window functions (`row_number`, `rank`, `dense_rank`) are deliberately
//! absent: they are not scalar functions, are rejected outside an `OVER`
//! clause, and have no dispatcher this gate could probe.

/// One function: canonical (display) spelling, accepted alternate spellings,
/// a category, a one-line description, and a Neo4j-style signature.
///
/// `name` and `aliases` are display spellings — the parser lowercases every
/// function name before dispatch, so `toUpper` here is matched as `toupper`.
pub(crate) struct FunctionSpec {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub category: &'static str,
    pub description: &'static str,
    pub signature: &'static str,
}

/// The category name used for entries dispatched through the aggregation
/// engine rather than [`super::CypherExecutor::evaluate_scalar_function`].
pub(crate) const AGGREGATE_CATEGORY: &str = "aggregate";

pub(crate) const FUNCTIONS: &[FunctionSpec] = &[
    // ── string ────────────────────────────────────────────────────────────
    FunctionSpec {
        name: "toUpper",
        aliases: &["toUpperCase"],
        category: "string",
        description: "Uppercase a string; null for any non-string input",
        signature: "toUpper(input :: STRING) :: STRING?",
    },
    FunctionSpec {
        name: "toLower",
        aliases: &["toLowerCase"],
        category: "string",
        description: "Lowercase a string; null for any non-string input",
        signature: "toLower(input :: STRING) :: STRING?",
    },
    FunctionSpec {
        name: "toString",
        aliases: &[],
        category: "string",
        description: "Render any value in its compact string form",
        signature: "toString(input :: ANY) :: STRING",
    },
    FunctionSpec {
        name: "text_edit_distance",
        aliases: &[],
        category: "string",
        description: "Levenshtein edit distance between two strings",
        signature: "text_edit_distance(a :: STRING, b :: STRING) :: INTEGER?",
    },
    FunctionSpec {
        name: "text_normalize",
        aliases: &[],
        category: "string",
        description: "Lowercase, drop punctuation, collapse whitespace runs, trim",
        signature: "text_normalize(input :: STRING) :: STRING?",
    },
    FunctionSpec {
        name: "text_jaccard",
        aliases: &[],
        category: "string",
        description: "Jaccard similarity of two token sets (default separator: whitespace)",
        signature: "text_jaccard(a :: STRING, b :: STRING, separator :: STRING?) :: FLOAT?",
    },
    FunctionSpec {
        name: "text_ngrams",
        aliases: &[],
        category: "string",
        description: "Character n-grams of a string as a list",
        signature: "text_ngrams(input :: STRING, n :: INTEGER) :: LIST<STRING>?",
    },
    FunctionSpec {
        name: "text_contains_any",
        aliases: &[],
        category: "string",
        description: "True when the text contains any of the given needles",
        signature: "text_contains_any(text :: STRING, needles :: LIST<STRING> | STRING...) :: BOOLEAN?",
    },
    FunctionSpec {
        name: "text_starts_with_any",
        aliases: &[],
        category: "string",
        description: "True when the text starts with any of the given prefixes",
        signature: "text_starts_with_any(text :: STRING, prefixes :: LIST<STRING> | STRING...) :: BOOLEAN?",
    },
    FunctionSpec {
        name: "text_match_regex",
        aliases: &[],
        category: "string",
        description: "Regex match with cached compilation; flags from imsxU",
        signature: "text_match_regex(text :: STRING, pattern :: STRING, flags :: STRING?) :: BOOLEAN?",
    },
    FunctionSpec {
        name: "split",
        aliases: &[],
        category: "string",
        description: "Split a string on a delimiter into a list",
        signature: "split(input :: STRING, delimiter :: STRING) :: LIST<STRING>?",
    },
    FunctionSpec {
        name: "replace",
        aliases: &[],
        category: "string",
        description: "Replace every occurrence of a substring",
        signature: "replace(input :: STRING, search :: STRING, replacement :: STRING) :: STRING?",
    },
    FunctionSpec {
        name: "substring",
        aliases: &[],
        category: "string",
        description: "Character-indexed substring from start, optionally limited to length",
        signature: "substring(input :: STRING, start :: INTEGER, length :: INTEGER?) :: STRING?",
    },
    FunctionSpec {
        name: "left",
        aliases: &[],
        category: "string",
        description: "The leftmost n characters",
        signature: "left(input :: STRING, length :: INTEGER) :: STRING?",
    },
    FunctionSpec {
        name: "right",
        aliases: &[],
        category: "string",
        description: "The rightmost n characters",
        signature: "right(input :: STRING, length :: INTEGER) :: STRING?",
    },
    FunctionSpec {
        name: "trim",
        aliases: &["btrim"],
        category: "string",
        description: "Strip leading and trailing whitespace",
        signature: "trim(input :: STRING) :: STRING?",
    },
    FunctionSpec {
        name: "ltrim",
        aliases: &[],
        category: "string",
        description: "Strip leading whitespace",
        signature: "ltrim(input :: STRING) :: STRING?",
    },
    FunctionSpec {
        name: "rtrim",
        aliases: &[],
        category: "string",
        description: "Strip trailing whitespace",
        signature: "rtrim(input :: STRING) :: STRING?",
    },
    // ── numeric ───────────────────────────────────────────────────────────
    FunctionSpec {
        name: "toInteger",
        aliases: &["toInt"],
        category: "numeric",
        description: "Coerce to an integer (strings parsed, booleans 1/0); null when impossible",
        signature: "toInteger(input :: ANY) :: INTEGER?",
    },
    FunctionSpec {
        name: "toFloat",
        aliases: &[],
        category: "numeric",
        description: "Coerce to a float (strings parsed); null when impossible",
        signature: "toFloat(input :: ANY) :: FLOAT?",
    },
    FunctionSpec {
        name: "abs",
        aliases: &[],
        category: "numeric",
        description: "Absolute value; keeps the integer type for integer input",
        signature: "abs(value :: NUMBER) :: NUMBER?",
    },
    FunctionSpec {
        name: "ceil",
        aliases: &["ceiling"],
        category: "numeric",
        description: "Round up to the nearest whole number",
        signature: "ceil(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "floor",
        aliases: &[],
        category: "numeric",
        description: "Round down to the nearest whole number",
        signature: "floor(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "round",
        aliases: &[],
        category: "numeric",
        description: "Round to the given number of decimal places (default 0)",
        signature: "round(value :: NUMBER, precision :: INTEGER?) :: FLOAT?",
    },
    FunctionSpec {
        name: "sqrt",
        aliases: &[],
        category: "numeric",
        description: "Square root; null for a negative argument",
        signature: "sqrt(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "sign",
        aliases: &[],
        category: "numeric",
        description: "-1, 0 or 1 according to the sign of the argument",
        signature: "sign(value :: NUMBER) :: INTEGER?",
    },
    FunctionSpec {
        name: "log",
        aliases: &["ln"],
        category: "numeric",
        description: "Natural logarithm; null for a non-positive argument",
        signature: "log(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "log10",
        aliases: &[],
        category: "numeric",
        description: "Base-10 logarithm; null for a non-positive argument",
        signature: "log10(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "exp",
        aliases: &[],
        category: "numeric",
        description: "e raised to the given power",
        signature: "exp(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "pow",
        aliases: &["power"],
        category: "numeric",
        description: "Base raised to the given exponent",
        signature: "pow(base :: NUMBER, exponent :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "pi",
        aliases: &[],
        category: "numeric",
        description: "The constant pi",
        signature: "pi() :: FLOAT",
    },
    FunctionSpec {
        name: "sin",
        aliases: &[],
        category: "numeric",
        description: "Sine of an angle in radians",
        signature: "sin(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "cos",
        aliases: &[],
        category: "numeric",
        description: "Cosine of an angle in radians",
        signature: "cos(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "tan",
        aliases: &[],
        category: "numeric",
        description: "Tangent of an angle in radians",
        signature: "tan(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "asin",
        aliases: &[],
        category: "numeric",
        description: "Arcsine, in radians",
        signature: "asin(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "acos",
        aliases: &[],
        category: "numeric",
        description: "Arccosine, in radians",
        signature: "acos(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "atan",
        aliases: &[],
        category: "numeric",
        description: "Arctangent, in radians",
        signature: "atan(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "atan2",
        aliases: &[],
        category: "numeric",
        description: "Quadrant-aware arctangent of y/x, in radians",
        signature: "atan2(y :: NUMBER, x :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "cot",
        aliases: &[],
        category: "numeric",
        description: "Cotangent of an angle in radians",
        signature: "cot(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "haversin",
        aliases: &[],
        category: "numeric",
        description: "Half the versed sine, (1 - cos(x)) / 2 — the great-circle term",
        signature: "haversin(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "degrees",
        aliases: &[],
        category: "numeric",
        description: "Convert radians to degrees",
        signature: "degrees(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "radians",
        aliases: &[],
        category: "numeric",
        description: "Convert degrees to radians",
        signature: "radians(value :: NUMBER) :: FLOAT?",
    },
    // ── temporal ──────────────────────────────────────────────────────────
    FunctionSpec {
        name: "date",
        aliases: &[],
        category: "temporal",
        description: "Parse a date; an unparseable string is null, not an error",
        signature: "date(input :: STRING) :: DATE?",
    },
    FunctionSpec {
        name: "datetime",
        aliases: &[],
        category: "temporal",
        description: "Now, or a parsed ISO-8601 datetime normalised to UTC",
        signature: "datetime(input :: STRING?) :: DATETIME",
    },
    FunctionSpec {
        name: "date_diff",
        aliases: &["datediff"],
        category: "temporal",
        description: "Whole days between two dates or datetimes (a - b)",
        signature: "date_diff(a :: DATE | DATETIME, b :: DATE | DATETIME) :: INTEGER?",
    },
    FunctionSpec {
        name: "duration",
        aliases: &[],
        category: "temporal",
        description: "Build a duration from a literal component map (years/months/weeks/days/hours/minutes/seconds)",
        signature: "duration(components :: MAP) :: DURATION",
    },
    FunctionSpec {
        name: "duration.between",
        aliases: &[],
        category: "temporal",
        description: "The duration from one date/datetime to another",
        signature: "duration.between(start :: DATE | DATETIME, end :: DATE | DATETIME) :: DURATION?",
    },
    FunctionSpec {
        name: "add_days",
        aliases: &[],
        category: "temporal",
        description: "Shift a date by a whole number of days",
        signature: "add_days(date :: DATE, days :: INTEGER) :: DATE?",
    },
    FunctionSpec {
        name: "add_months",
        aliases: &[],
        category: "temporal",
        description: "Shift a date by whole months, clamping the day-of-month",
        signature: "add_months(date :: DATE, months :: INTEGER) :: DATE?",
    },
    FunctionSpec {
        name: "add_years",
        aliases: &[],
        category: "temporal",
        description: "Shift a date by whole years",
        signature: "add_years(date :: DATE, years :: INTEGER) :: DATE?",
    },
    FunctionSpec {
        name: "date_truncate",
        aliases: &[],
        category: "temporal",
        description: "Round a date down to the start of a year/month/week/day",
        signature: "date_truncate(date :: DATE, unit :: STRING) :: DATE?",
    },
    FunctionSpec {
        name: "localdatetime",
        aliases: &[],
        category: "temporal",
        description: "Local wall-clock now, or a parsed datetime keeping its wall clock",
        signature: "localdatetime(input :: STRING?) :: DATETIME",
    },
    FunctionSpec {
        name: "localtime",
        aliases: &[],
        category: "temporal",
        description: "Local time of day as HH:MM:SS — there is no time-of-day value type",
        signature: "localtime(input :: STRING?) :: STRING",
    },
    FunctionSpec {
        name: "time",
        aliases: &[],
        category: "temporal",
        description: "Local time of day as HH:MM:SS; KGLite has no zoned time type",
        signature: "time(input :: STRING?) :: STRING",
    },
    // ── graph ─────────────────────────────────────────────────────────────
    FunctionSpec {
        name: "nodes",
        aliases: &[],
        category: "graph",
        description: "The nodes of a bound path, in traversal order",
        signature: "nodes(path :: PATH) :: LIST<NODE>?",
    },
    FunctionSpec {
        name: "relationships",
        aliases: &["rels"],
        category: "graph",
        description: "The relationships of a bound path, in traversal order",
        signature: "relationships(path :: PATH) :: LIST<RELATIONSHIP>?",
    },
    FunctionSpec {
        name: "type",
        aliases: &[],
        category: "graph",
        description: "The type of a bound relationship",
        signature: "type(rel :: RELATIONSHIP) :: STRING?",
    },
    FunctionSpec {
        name: "elementId",
        aliases: &[],
        category: "graph",
        description: "Neo4j 5 element identity: an opaque string keyed off the graph slot",
        signature: "elementId(entity :: NODE | RELATIONSHIP) :: STRING?",
    },
    FunctionSpec {
        name: "id",
        aliases: &[],
        category: "graph",
        description: "Logical node identity (the id property) or stable relationship identity",
        signature: "id(entity :: NODE | RELATIONSHIP) :: ANY?",
    },
    FunctionSpec {
        name: "shortest_path_length",
        aliases: &[],
        category: "graph",
        description: "Undirected BFS hop count between two bound nodes; null when disconnected",
        signature: "shortest_path_length(a :: NODE, b :: NODE) :: INTEGER?",
    },
    FunctionSpec {
        name: "degree",
        aliases: &[],
        category: "graph",
        description: "Edge count in both directions (a self-loop counts twice)",
        signature: "degree(node :: NODE) :: INTEGER?",
    },
    FunctionSpec {
        name: "inDegree",
        aliases: &[],
        category: "graph",
        description: "Incoming edge count",
        signature: "inDegree(node :: NODE) :: INTEGER?",
    },
    FunctionSpec {
        name: "outDegree",
        aliases: &[],
        category: "graph",
        description: "Outgoing edge count",
        signature: "outDegree(node :: NODE) :: INTEGER?",
    },
    FunctionSpec {
        name: "labels",
        aliases: &[],
        category: "graph",
        description: "A node's labels, primary type first then secondaries",
        signature: "labels(node :: NODE) :: LIST<STRING>?",
    },
    FunctionSpec {
        name: "keys",
        aliases: &[],
        category: "graph",
        description: "Sorted property keys of a node or relationship",
        signature: "keys(entity :: NODE | RELATIONSHIP) :: LIST<STRING>?",
    },
    FunctionSpec {
        name: "properties",
        aliases: &[],
        category: "graph",
        description: "A node's or relationship's properties as a map",
        signature: "properties(entity :: NODE | RELATIONSHIP) :: MAP?",
    },
    FunctionSpec {
        name: "startNode",
        aliases: &["start_node"],
        category: "graph",
        description: "The source node of a bound relationship",
        signature: "startNode(rel :: RELATIONSHIP) :: NODE?",
    },
    FunctionSpec {
        name: "endNode",
        aliases: &["end_node"],
        category: "graph",
        description: "The target node of a bound relationship",
        signature: "endNode(rel :: RELATIONSHIP) :: NODE?",
    },
    // ── collection ────────────────────────────────────────────────────────
    FunctionSpec {
        name: "size",
        aliases: &[],
        category: "collection",
        description: "Element count of a list or map, or character count of a string",
        signature: "size(value :: LIST | MAP | STRING) :: INTEGER?",
    },
    FunctionSpec {
        name: "length",
        aliases: &[],
        category: "collection",
        description: "Hop count of a path, or element/character count of a list or string",
        signature: "length(value :: PATH | LIST | MAP | STRING) :: INTEGER?",
    },
    FunctionSpec {
        name: "coalesce",
        aliases: &[],
        category: "collection",
        description: "The first non-null argument",
        signature: "coalesce(values :: ANY?...) :: ANY?",
    },
    FunctionSpec {
        name: "reverse",
        aliases: &[],
        category: "collection",
        description: "Reverse a list or a string",
        signature: "reverse(value :: LIST | STRING) :: ANY?",
    },
    FunctionSpec {
        name: "head",
        aliases: &[],
        category: "collection",
        description: "The first element of a list",
        signature: "head(list :: LIST) :: ANY?",
    },
    FunctionSpec {
        name: "last",
        aliases: &[],
        category: "collection",
        description: "The last element of a list",
        signature: "last(list :: LIST) :: ANY?",
    },
    FunctionSpec {
        name: "range",
        aliases: &[],
        category: "collection",
        description: "Inclusive integer range with an optional step",
        signature: "range(start :: INTEGER, end :: INTEGER, step :: INTEGER?) :: LIST<INTEGER>",
    },
    // ── spatial ───────────────────────────────────────────────────────────
    FunctionSpec {
        name: "point",
        aliases: &[],
        category: "spatial",
        description: "A geographic point from positional latitude and longitude",
        signature: "point(latitude :: FLOAT, longitude :: FLOAT) :: POINT",
    },
    FunctionSpec {
        name: "distance",
        aliases: &[],
        category: "spatial",
        description: "Geodesic distance in metres, from two spatial values or four coordinates",
        signature: "distance(a :: POINT | GEOMETRY, b :: POINT | GEOMETRY) :: FLOAT?",
    },
    FunctionSpec {
        name: "contains",
        aliases: &[],
        category: "spatial",
        description: "True when the first geometry contains the second value",
        signature: "contains(container :: GEOMETRY, item :: POINT | GEOMETRY) :: BOOLEAN",
    },
    FunctionSpec {
        name: "intersects",
        aliases: &[],
        category: "spatial",
        description: "True when two geometries intersect",
        signature: "intersects(a :: POINT | GEOMETRY, b :: POINT | GEOMETRY) :: BOOLEAN",
    },
    FunctionSpec {
        name: "centroid",
        aliases: &[],
        category: "spatial",
        description: "The centroid of a geometry",
        signature: "centroid(geometry :: POINT | GEOMETRY) :: POINT?",
    },
    FunctionSpec {
        name: "area",
        aliases: &[],
        category: "spatial",
        description: "Area of a geometry in square metres",
        signature: "area(geometry :: GEOMETRY) :: FLOAT?",
    },
    FunctionSpec {
        name: "perimeter",
        aliases: &[],
        category: "spatial",
        description: "Perimeter of a geometry in metres",
        signature: "perimeter(geometry :: GEOMETRY) :: FLOAT?",
    },
    FunctionSpec {
        name: "latitude",
        aliases: &[],
        category: "spatial",
        description: "Latitude of a point",
        signature: "latitude(point :: POINT) :: FLOAT",
    },
    FunctionSpec {
        name: "longitude",
        aliases: &[],
        category: "spatial",
        description: "Longitude of a point",
        signature: "longitude(point :: POINT) :: FLOAT",
    },
    FunctionSpec {
        name: "geom_buffer",
        aliases: &[],
        category: "spatial",
        description: "Buffer a geometry by a distance in metres, as WKT",
        signature: "geom_buffer(geometry :: GEOMETRY, meters :: FLOAT) :: STRING?",
    },
    FunctionSpec {
        name: "geom_convex_hull",
        aliases: &[],
        category: "spatial",
        description: "Convex hull of a list of geometries, as WKT",
        signature: "geom_convex_hull(geometries :: LIST<STRING> | GEOMETRY...) :: STRING?",
    },
    FunctionSpec {
        name: "geom_union",
        aliases: &[],
        category: "spatial",
        description: "Union of two geometries, as WKT",
        signature: "geom_union(a :: GEOMETRY, b :: GEOMETRY) :: STRING?",
    },
    FunctionSpec {
        name: "geom_intersection",
        aliases: &[],
        category: "spatial",
        description: "Intersection of two geometries, as WKT",
        signature: "geom_intersection(a :: GEOMETRY, b :: GEOMETRY) :: STRING?",
    },
    FunctionSpec {
        name: "geom_difference",
        aliases: &[],
        category: "spatial",
        description: "Difference of two geometries, as WKT",
        signature: "geom_difference(a :: GEOMETRY, b :: GEOMETRY) :: STRING?",
    },
    FunctionSpec {
        name: "geom_is_valid",
        aliases: &[],
        category: "spatial",
        description: "Whether a geometry is topologically valid",
        signature: "geom_is_valid(geometry :: GEOMETRY) :: BOOLEAN?",
    },
    FunctionSpec {
        name: "geom_length",
        aliases: &[],
        category: "spatial",
        description: "Length of a geometry in metres",
        signature: "geom_length(geometry :: GEOMETRY) :: FLOAT?",
    },
    // ── timeseries ────────────────────────────────────────────────────────
    // Every ts_* function takes a timeseries channel as a property access
    // (`n.channel`), not an ordinary value.
    FunctionSpec {
        name: "ts_at",
        aliases: &[],
        category: "timeseries",
        description: "Channel value at one date key",
        signature: "ts_at(channel :: PROPERTY, date :: STRING?) :: FLOAT?",
    },
    FunctionSpec {
        name: "ts_sum",
        aliases: &[],
        category: "timeseries",
        description: "Sum of a channel over an optional date range",
        signature: "ts_sum(channel :: PROPERTY, start :: STRING?, end :: STRING?) :: FLOAT",
    },
    FunctionSpec {
        name: "ts_avg",
        aliases: &[],
        category: "timeseries",
        description: "Mean of a channel over an optional date range",
        signature: "ts_avg(channel :: PROPERTY, start :: STRING?, end :: STRING?) :: FLOAT?",
    },
    FunctionSpec {
        name: "ts_min",
        aliases: &[],
        category: "timeseries",
        description: "Minimum of a channel over an optional date range",
        signature: "ts_min(channel :: PROPERTY, start :: STRING?, end :: STRING?) :: FLOAT?",
    },
    FunctionSpec {
        name: "ts_max",
        aliases: &[],
        category: "timeseries",
        description: "Maximum of a channel over an optional date range",
        signature: "ts_max(channel :: PROPERTY, start :: STRING?, end :: STRING?) :: FLOAT?",
    },
    FunctionSpec {
        name: "ts_count",
        aliases: &[],
        category: "timeseries",
        description: "Count of finite observations over an optional date range",
        signature: "ts_count(channel :: PROPERTY, start :: STRING?, end :: STRING?) :: INTEGER",
    },
    FunctionSpec {
        name: "ts_first",
        aliases: &[],
        category: "timeseries",
        description: "First finite value of a channel",
        signature: "ts_first(channel :: PROPERTY) :: FLOAT?",
    },
    FunctionSpec {
        name: "ts_last",
        aliases: &[],
        category: "timeseries",
        description: "Last finite value of a channel",
        signature: "ts_last(channel :: PROPERTY) :: FLOAT?",
    },
    FunctionSpec {
        name: "ts_delta",
        aliases: &[],
        category: "timeseries",
        description: "Change in a channel between two dates",
        signature: "ts_delta(channel :: PROPERTY, from :: STRING?, to :: STRING?) :: FLOAT?",
    },
    FunctionSpec {
        name: "ts_series",
        aliases: &[],
        category: "timeseries",
        description: "Channel as a list of {time, value} maps over an optional date range",
        signature: "ts_series(channel :: PROPERTY, start :: STRING?, end :: STRING?) :: LIST<MAP>",
    },
    // ── vector ────────────────────────────────────────────────────────────
    FunctionSpec {
        name: "dot",
        aliases: &[],
        category: "vector",
        description: "Dot product of two numeric lists",
        signature: "dot(a :: LIST<NUMBER>, b :: LIST<NUMBER>) :: FLOAT?",
    },
    FunctionSpec {
        name: "cosine",
        aliases: &[],
        category: "vector",
        description: "Cosine similarity of two numeric lists",
        signature: "cosine(a :: LIST<NUMBER>, b :: LIST<NUMBER>) :: FLOAT?",
    },
    FunctionSpec {
        name: "norm",
        aliases: &[],
        category: "vector",
        description: "L2 norm of a numeric list",
        signature: "norm(a :: LIST<NUMBER>) :: FLOAT?",
    },
    // ── utility ───────────────────────────────────────────────────────────
    FunctionSpec {
        name: "vector_score",
        aliases: &[],
        category: "utility",
        description: "Similarity of a node's stored embedding to a query vector",
        signature: "vector_score(node :: NODE, property :: STRING, queryVector :: LIST<FLOAT>, metric :: STRING?) :: FLOAT?",
    },
    FunctionSpec {
        name: "embedding_norm",
        aliases: &[],
        category: "utility",
        description: "L2 norm of a node's stored embedding (Poincare depth proxy)",
        signature: "embedding_norm(node :: NODE, property :: STRING) :: FLOAT?",
    },
    FunctionSpec {
        name: "text_bm25",
        aliases: &[],
        category: "utility",
        description: "BM25 relevance of a node's indexed text against a query; null when the node has no document",
        signature: "text_bm25(node :: NODE, property :: STRING, query :: STRING) :: FLOAT?",
    },
    FunctionSpec {
        name: "text_score",
        aliases: &[],
        category: "utility",
        description: "Similarity of a node's embedding to an embedded query string; requires set_embedder()",
        signature: "text_score(node :: NODE, property :: STRING, query :: STRING) :: FLOAT?",
    },
    FunctionSpec {
        name: "randomUUID",
        aliases: &[],
        category: "utility",
        description: "A random RFC-4122 version-4 UUID string",
        signature: "randomUUID() :: STRING",
    },
    FunctionSpec {
        name: "rand",
        aliases: &["random"],
        category: "utility",
        description: "A random float in [0, 1)",
        signature: "rand() :: FLOAT",
    },
    FunctionSpec {
        name: "valid_at",
        aliases: &[],
        category: "utility",
        description: "Whether an entity's validity interval covers a date; null bounds are open",
        signature: "valid_at(entity :: NODE | RELATIONSHIP, date :: ANY, fromField :: STRING, toField :: STRING) :: BOOLEAN",
    },
    FunctionSpec {
        name: "valid_during",
        aliases: &[],
        category: "utility",
        description: "Whether an entity's validity interval overlaps a date range",
        signature: "valid_during(entity :: NODE | RELATIONSHIP, start :: ANY, end :: ANY, fromField :: STRING, toField :: STRING) :: BOOLEAN",
    },
    FunctionSpec {
        name: "parse_json",
        aliases: &["from_json"],
        category: "utility",
        description: "Parse a JSON string into structured values; bad input is null, not an error",
        signature: "parse_json(text :: STRING?) :: ANY?",
    },
    // ── aggregate ─────────────────────────────────────────────────────────
    // Dispatched by the aggregation engine, not `evaluate_scalar_function`,
    // and legal only in RETURN / WITH. Gated against
    // `ast::is_aggregate_function_name` in both directions.
    FunctionSpec {
        name: "count",
        aliases: &[],
        category: AGGREGATE_CATEGORY,
        description: "Count rows, or non-null values of an expression",
        signature: "count(value :: ANY?) :: INTEGER",
    },
    FunctionSpec {
        name: "sum",
        aliases: &[],
        category: AGGREGATE_CATEGORY,
        description: "Sum of numeric values",
        signature: "sum(value :: NUMBER) :: NUMBER",
    },
    FunctionSpec {
        name: "avg",
        aliases: &["mean", "average"],
        category: AGGREGATE_CATEGORY,
        description: "Arithmetic mean of numeric values",
        signature: "avg(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "min",
        aliases: &[],
        category: AGGREGATE_CATEGORY,
        description: "Smallest value",
        signature: "min(value :: ANY) :: ANY?",
    },
    FunctionSpec {
        name: "max",
        aliases: &[],
        category: AGGREGATE_CATEGORY,
        description: "Largest value",
        signature: "max(value :: ANY) :: ANY?",
    },
    FunctionSpec {
        name: "collect",
        aliases: &[],
        category: AGGREGATE_CATEGORY,
        description: "Gather values into a list, skipping nulls",
        signature: "collect(value :: ANY) :: LIST<ANY>",
    },
    FunctionSpec {
        name: "stdev",
        aliases: &["std"],
        category: AGGREGATE_CATEGORY,
        description: "Sample standard deviation",
        signature: "stdev(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "variance",
        aliases: &["var_samp"],
        category: AGGREGATE_CATEGORY,
        description: "Sample variance",
        signature: "variance(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "median",
        aliases: &[],
        category: AGGREGATE_CATEGORY,
        description: "Median of numeric values",
        signature: "median(value :: NUMBER) :: FLOAT?",
    },
    FunctionSpec {
        name: "mode",
        aliases: &[],
        category: AGGREGATE_CATEGORY,
        description: "Most frequent value",
        signature: "mode(value :: ANY) :: ANY?",
    },
    FunctionSpec {
        name: "percentile_cont",
        aliases: &[],
        category: AGGREGATE_CATEGORY,
        description: "Interpolated percentile of numeric values",
        signature: "percentile_cont(value :: NUMBER, percentile :: FLOAT) :: FLOAT?",
    },
    FunctionSpec {
        name: "percentile_disc",
        aliases: &[],
        category: AGGREGATE_CATEGORY,
        description: "Nearest-observed percentile of numeric values",
        signature: "percentile_disc(value :: NUMBER, percentile :: FLOAT) :: ANY?",
    },
];

#[cfg(test)]
mod tests {
    // `super::super::super` is the `executor` module, whose glob imports
    // already carry `CypherExecutor`, `DirGraph`, `ResultRow`, `Expression`,
    // `is_aggregate_function_name` and `HashMap`.
    use super::super::super::*;
    use super::*;
    use crate::datatypes::values::Value;

    /// Every spelling this table advertises — canonical names and aliases —
    /// lowercased the way the parser hands them to the dispatcher.
    fn all_spellings(spec: &FunctionSpec) -> Vec<String> {
        std::iter::once(spec.name)
            .chain(spec.aliases.iter().copied())
            .map(|n| n.to_ascii_lowercase())
            .collect()
    }

    /// **The drift gate.** Every non-aggregate spelling in the table is pushed
    /// through the *real* dispatcher chain
    /// ([`super::super::CypherExecutor::evaluate_scalar_function`]) and must
    /// not come back "Unknown function". A wrong-arity or wrong-type error is
    /// fine and expected — the probe passes five nulls, which is one more
    /// argument than the widest function accepts, so no arm can index out of
    /// bounds. What the gate rejects is a name that no `match` arm owns, i.e.
    /// a registry entry describing a function this engine does not have.
    ///
    /// Verified able to fail: adding a `FunctionSpec { name: "not_a_function",
    /// … }` makes this test report `not_a_function: registry entry does not
    /// dispatch`.
    #[test]
    fn every_registry_name_dispatches() {
        let graph = DirGraph::new();
        let params = HashMap::new();
        let executor = CypherExecutor::with_params(&graph, &params, None);
        let row = ResultRow::new();
        let args: Vec<Expression> = (0..5).map(|_| Expression::Literal(Value::Null)).collect();

        let mut missing = Vec::new();
        for spec in FUNCTIONS {
            if spec.category == AGGREGATE_CATEGORY {
                continue;
            }
            for spelling in all_spellings(spec) {
                if let Err(err) = executor.test_evaluate_scalar_function(&spelling, &args, &row) {
                    if err.starts_with("Unknown function") {
                        missing.push(format!("{spelling}: registry entry does not dispatch"));
                    }
                }
            }
        }
        assert!(missing.is_empty(), "{}", missing.join("\n"));
    }

    /// The aggregate half of the gate, closed in *both* directions: every
    /// aggregate spelling in the table is one the parser classifies as an
    /// aggregate, and the table's aggregate set is exactly the set
    /// `ast::is_aggregate_function_name` recognises. That list is a `matches!`
    /// and cannot be enumerated from code, so it is restated here — an
    /// aggregate added to `ast.rs` and not to this table fails on the second
    /// assertion.
    #[test]
    fn aggregate_names_round_trip() {
        const AST_AGGREGATES: [&str; 16] = [
            "count",
            "sum",
            "avg",
            "mean",
            "average",
            "min",
            "max",
            "collect",
            "std",
            "stdev",
            "variance",
            "var_samp",
            "median",
            "mode",
            "percentile_cont",
            "percentile_disc",
        ];
        // The restated list must itself agree with `ast.rs` — this is what
        // makes the set comparison below trustworthy.
        for name in AST_AGGREGATES {
            assert!(
                is_aggregate_function_name(name),
                "{name} is no longer an aggregate in ast.rs"
            );
        }

        let registered: std::collections::BTreeSet<String> = FUNCTIONS
            .iter()
            .filter(|spec| spec.category == AGGREGATE_CATEGORY)
            .flat_map(all_spellings)
            .collect();
        let expected: std::collections::BTreeSet<String> =
            AST_AGGREGATES.iter().map(|n| n.to_string()).collect();
        assert_eq!(
            registered, expected,
            "the aggregate registry and ast::is_aggregate_function_name disagree"
        );
    }

    /// A duplicate spelling would make `SHOW FUNCTIONS` list the same callable
    /// twice under two descriptions.
    #[test]
    fn spellings_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for spec in FUNCTIONS {
            for spelling in all_spellings(spec) {
                assert!(
                    seen.insert(spelling.clone()),
                    "duplicate spelling: {spelling}"
                );
            }
        }
    }

    /// Every entry carries the three things `SHOW FUNCTIONS` projects, and the
    /// signature names the function it belongs to.
    #[test]
    fn every_spec_is_complete() {
        for spec in FUNCTIONS {
            assert!(
                !spec.description.is_empty(),
                "{} has no description",
                spec.name
            );
            assert!(!spec.category.is_empty(), "{} has no category", spec.name);
            assert!(
                spec.signature.starts_with(&format!("{}(", spec.name)),
                "{} signature does not open with its own name: {}",
                spec.name,
                spec.signature
            );
        }
    }

    /// Functions and procedures are separate namespaces read from separate
    /// registries. `degree` is the one name in both — `RETURN degree(n)` is a
    /// scalar function, `CALL degree()` is the centrality procedure — and it
    /// is pinned here so a third collision has to be a deliberate act.
    #[test]
    fn function_and_procedure_namespaces_overlap_only_at_degree() {
        let procedures: std::collections::HashSet<String> =
            super::super::super::procedure_registry::PROCEDURES
                .iter()
                .map(|spec| spec.name.to_ascii_lowercase())
                .collect();
        let overlap: std::collections::BTreeSet<String> = FUNCTIONS
            .iter()
            .map(|spec| spec.name.to_ascii_lowercase())
            .filter(|name| procedures.contains(name))
            .collect();
        assert_eq!(
            overlap.into_iter().collect::<Vec<_>>(),
            vec!["degree".to_string()]
        );
    }
}
