//! Tier-3 topic detail writers — deep docs for each Cypher clause
//! (MATCH/WHERE/RETURN/...), each algorithm procedure, and each Fluent
//! API subsystem. Rendered by describe() when the user asks for a
//! specific topic.

// ── Cypher tier 3: topic detail functions ──────────────────────────────────

const CYPHER_TOPIC_LIST: &str = "MATCH, WHERE, RETURN, WITH, HAVING, ORDER BY, UNWIND, UNION, \
    CALL_SUBQUERY, CASE, CREATE, SET, DELETE, MERGE, EXPLAIN, PROFILE, operators, functions, patterns, spatial, \
    temporal, pagerank, betweenness, degree, closeness, louvain, leiden, \
    label_propagation, connected_components, k_core, clustering_coefficient, cluster, orphan_node, self_loop, \
    cycle_2step, missing_required_edge, missing_inbound_edge, duplicate_title, \
    null_property, inverse_violation, transitivity_violation, cardinality_violation, \
    type_domain_violation, type_range_violation, parallel_edges";

/// Tier 3: detailed Cypher docs for specific topics with params and examples.
pub(super) fn write_cypher_topics(xml: &mut String, topics: &[String]) -> Result<(), String> {
    // Empty list → tier 2 overview
    if topics.is_empty() {
        write_cypher_overview(xml);
        return Ok(());
    }

    xml.push_str("<cypher>\n");
    for topic in topics {
        let key = topic.to_uppercase();
        match key.as_str() {
            "MATCH" => write_topic_match(xml),
            "WHERE" => write_topic_where(xml),
            "RETURN" => write_topic_return(xml),
            "WITH" => write_topic_with(xml),
            "HAVING" => write_topic_having(xml),
            "ORDER BY" | "ORDERBY" | "ORDER_BY" => write_topic_order_by(xml),
            "UNWIND" => write_topic_unwind(xml),
            "UNION" => write_topic_union(xml),
            "CASE" => write_topic_case(xml),
            "CREATE" => write_topic_create(xml),
            "SET" => write_topic_set(xml),
            "DELETE" | "REMOVE" => write_topic_delete(xml),
            "MERGE" => write_topic_merge(xml),
            "CALL_SUBQUERY" | "CALLSUBQUERY" | "CALL {}" | "CALL { }" => {
                write_topic_call_subquery(xml);
            }
            "OPERATORS" => write_topic_operators(xml),
            "FUNCTIONS" => write_topic_functions(xml),
            "PATTERNS" => write_topic_patterns(xml),
            "PAGERANK" => write_topic_pagerank(xml),
            "BETWEENNESS" => write_topic_betweenness(xml),
            "DEGREE" => write_topic_degree(xml),
            "CLOSENESS" => write_topic_closeness(xml),
            "LOUVAIN" => write_topic_louvain(xml),
            "LEIDEN" => write_topic_leiden(xml),
            "LABEL_PROPAGATION" | "LABELPROPAGATION" => write_topic_label_propagation(xml),
            "CONNECTED_COMPONENTS" | "CONNECTEDCOMPONENTS" => {
                write_topic_connected_components(xml);
            }
            "K_CORE" | "KCORE" | "CORENESS" => write_topic_k_core(xml),
            "CLUSTERING_COEFFICIENT" | "CLUSTERINGCOEFFICIENT" => {
                write_topic_clustering_coefficient(xml);
            }
            "CLUSTER" => write_topic_cluster(xml),
            "ORPHAN_NODE" => write_topic_orphan_node(xml),
            "SELF_LOOP" => write_topic_self_loop(xml),
            "CYCLE_2STEP" => write_topic_cycle_2step(xml),
            "MISSING_REQUIRED_EDGE" => write_topic_missing_required_edge(xml),
            "MISSING_INBOUND_EDGE" => write_topic_missing_inbound_edge(xml),
            "DUPLICATE_TITLE" => write_topic_duplicate_title(xml),
            "NULL_PROPERTY" => write_topic_null_property(xml),
            "INVERSE_VIOLATION" => write_topic_inverse_violation(xml),
            "TRANSITIVITY_VIOLATION" => write_topic_transitivity_violation(xml),
            "CARDINALITY_VIOLATION" => write_topic_cardinality_violation(xml),
            "TYPE_DOMAIN_VIOLATION" => write_topic_type_domain_violation(xml),
            "TYPE_RANGE_VIOLATION" => write_topic_type_range_violation(xml),
            "PARALLEL_EDGES" => write_topic_parallel_edges(xml),
            "SPATIAL" => write_topic_spatial(xml),
            "TEMPORAL" => write_topic_temporal(xml),
            "EXPLAIN" => write_topic_explain(xml),
            "PROFILE" => write_topic_profile(xml),
            _ => {
                return Err(format!(
                    "Unknown Cypher topic '{}'. Available: {}",
                    topic, CYPHER_TOPIC_LIST
                ));
            }
        }
    }
    xml.push_str("</cypher>\n");
    Ok(())
}

pub(super) fn write_topic_match(xml: &mut String) {
    xml.push_str("  <MATCH>\n");
    xml.push_str("    <desc>Pattern-match nodes and relationships. OPTIONAL MATCH returns nulls for non-matching patterns (left join).</desc>\n");
    xml.push_str("    <syntax>MATCH (n:Label {prop: val})-[r:TYPE]-&gt;(m)</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"all nodes of type\">MATCH (n:Field) RETURN n.name</ex>\n");
    xml.push_str("      <ex desc=\"with relationship\">MATCH (a:Person)-[:KNOWS]-&gt;(b) RETURN a.name, b.name</ex>\n");
    xml.push_str("      <ex desc=\"variable-length path\">MATCH (a)-[:KNOWS*1..3]-&gt;(b) RETURN a, b</ex>\n");
    xml.push_str("      <ex desc=\"inline property filter\">MATCH (n:Field {status: 'active'}) RETURN n</ex>\n");
    xml.push_str("      <ex desc=\"optional match\">MATCH (a:Field) OPTIONAL MATCH (a)-[:HAS]-&gt;(b:Well) RETURN a.name, b.name</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("    <pitfall name=\"cartesian product from multiple OPTIONAL MATCH\">\n");
    xml.push_str(
        "      Multiple OPTIONAL MATCH clauses create a cross-product of all matched paths.\n",
    );
    xml.push_str(
        "      If a node connects to 10 prospects × 5 plays × 3 licences = 150 rows per node.\n",
    );
    xml.push_str("      Fix: break with WITH to collapse dimensions before expanding the next.\n");
    xml.push_str("      <bad>MATCH (w:Well) OPTIONAL MATCH (w)-[:A]-&gt;(a) OPTIONAL MATCH (w)-[:B]-&gt;(b) OPTIONAL MATCH (w)-[:C]-&gt;(c) RETURN w, collect(a), collect(b), collect(c)</bad>\n");
    xml.push_str("      <good>MATCH (w:Well) OPTIONAL MATCH (w)-[:A]-&gt;(a) WITH w, collect(DISTINCT a.title) AS as_ OPTIONAL MATCH (w)-[:B]-&gt;(b) WITH w, as_, collect(DISTINCT b.title) AS bs OPTIONAL MATCH (w)-[:C]-&gt;(c) RETURN w.title, as_, bs, collect(DISTINCT c.title) AS cs</good>\n");
    xml.push_str("    </pitfall>\n");
    xml.push_str("  </MATCH>\n");
}

pub(super) fn write_topic_where(xml: &mut String) {
    xml.push_str("  <WHERE>\n");
    xml.push_str("    <desc>Filter results by predicate. Supports comparison, null checks, regex, string predicates, boolean logic.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"comparison\">WHERE n.depth &gt; 3000</ex>\n");
    xml.push_str("      <ex desc=\"string contains\">WHERE n.name CONTAINS 'oil'</ex>\n");
    xml.push_str("      <ex desc=\"starts/ends with\">WHERE n.name STARTS WITH '35/'</ex>\n");
    xml.push_str("      <ex desc=\"regex\">WHERE n.name =~ '35/9-.*'</ex>\n");
    xml.push_str("      <ex desc=\"null check\">WHERE n.depth IS NOT NULL</ex>\n");
    xml.push_str("      <ex desc=\"IN list\">WHERE n.status IN ['active', 'planned']</ex>\n");
    xml.push_str("      <ex desc=\"boolean\">WHERE n.depth &gt; 1000 AND n.temp &lt; 100</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </WHERE>\n");
}

pub(super) fn write_topic_return(xml: &mut String) {
    xml.push_str("  <RETURN>\n");
    xml.push_str("    <desc>Project columns to output. Supports DISTINCT, aliases (AS), expressions, aggregations.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">RETURN n.name, n.depth</ex>\n");
    xml.push_str("      <ex desc=\"alias\">RETURN n.name AS field_name</ex>\n");
    xml.push_str("      <ex desc=\"distinct\">RETURN DISTINCT n.status</ex>\n");
    xml.push_str(
        "      <ex desc=\"expression\">RETURN n.name || ' (' || n.status || ')' AS label</ex>\n",
    );
    xml.push_str("      <ex desc=\"aggregation\">RETURN n.status, count(*) AS n, collect(n.name) AS names</ex>\n");
    xml.push_str("      <ex desc=\"having\">RETURN n.type, count(*) AS cnt HAVING cnt > 5</ex>\n");
    xml.push_str("      <ex desc=\"window\">RETURN n.name, row_number() OVER (ORDER BY n.score DESC) AS rn</ex>\n");
    xml.push_str("      <ex desc=\"window-partition\">RETURN n.name, rank() OVER (PARTITION BY n.dept ORDER BY n.score DESC) AS r</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </RETURN>\n");
}

pub(super) fn write_topic_with(xml: &mut String) {
    xml.push_str("  <WITH>\n");
    xml.push_str("    <desc>Intermediate projection and aggregation. Creates a new scope — only variables listed in WITH are available in subsequent clauses.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"filter after aggregation\">MATCH (n:Field) WITH n.area AS area, count(*) AS c WHERE c &gt; 5 RETURN area, c</ex>\n");
    xml.push_str("      <ex desc=\"pipe between matches\">MATCH (a:Field) WITH a MATCH (a)-[:HAS]-&gt;(b) RETURN a.name, b.name</ex>\n");
    xml.push_str("      <ex desc=\"limit intermediate\">MATCH (n:Field) WITH n ORDER BY n.name LIMIT 10 RETURN n.name</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </WITH>\n");
}

pub(super) fn write_topic_having(xml: &mut String) {
    xml.push_str("  <HAVING>\n");
    xml.push_str("    <desc>Post-aggregation filter. Applies after grouping/aggregation in RETURN or WITH. Equivalent to WHERE but for aggregated results.</desc>\n");
    xml.push_str("    <syntax>RETURN group_expr, agg_func() AS alias HAVING predicate</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"filter by count\">MATCH (n:Person) RETURN n.city, count(*) AS pop HAVING pop > 1000</ex>\n");
    xml.push_str("      <ex desc=\"with WITH\">MATCH (n) WITH n.type AS t, count(*) AS c HAVING c >= 5 RETURN t, c</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </HAVING>\n");
}

pub(super) fn write_topic_order_by(xml: &mut String) {
    xml.push_str("  <ORDER_BY>\n");
    xml.push_str("    <desc>Sort results. Default ascending; append DESC for descending. Combine with SKIP and LIMIT for pagination.</desc>\n");
    xml.push_str("    <syntax>ORDER BY expr [DESC] [SKIP n] [LIMIT n]</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"ascending\">ORDER BY n.name</ex>\n");
    xml.push_str("      <ex desc=\"descending\">ORDER BY n.depth DESC</ex>\n");
    xml.push_str("      <ex desc=\"pagination\">ORDER BY n.name SKIP 20 LIMIT 10</ex>\n");
    xml.push_str("      <ex desc=\"multi-key\">ORDER BY n.status, n.name DESC</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </ORDER_BY>\n");
}

pub(super) fn write_topic_unwind(xml: &mut String) {
    xml.push_str("  <UNWIND>\n");
    xml.push_str("    <desc>Expand a list expression into individual rows. Each element becomes a new row bound to the alias.</desc>\n");
    xml.push_str("    <syntax>UNWIND expression AS variable</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"literal list\">UNWIND ['A','B','C'] AS x MATCH (n {code: x}) RETURN n</ex>\n");
    xml.push_str("      <ex desc=\"collected list\">MATCH (n:Field) WITH collect(n.name) AS names UNWIND names AS name RETURN name</ex>\n");
    xml.push_str("      <ex desc=\"range\">UNWIND range(1, 10) AS i RETURN i</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </UNWIND>\n");
}

pub(super) fn write_topic_union(xml: &mut String) {
    xml.push_str("  <UNION>\n");
    xml.push_str("    <desc>Combine result sets from two queries. UNION removes duplicates; UNION ALL keeps all rows. Column names must match.</desc>\n");
    xml.push_str("    <syntax>query1 UNION [ALL] query2</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic union\">MATCH (a:Field) RETURN a.name AS name UNION MATCH (b:Discovery) RETURN b.name AS name</ex>\n");
    xml.push_str("      <ex desc=\"union all\">MATCH (a:Field) RETURN a.name AS name UNION ALL MATCH (b:Field) RETURN b.name AS name</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </UNION>\n");
}

pub(super) fn write_topic_case(xml: &mut String) {
    xml.push_str("  <CASE>\n");
    xml.push_str("    <desc>Conditional expression. Two forms: simple (CASE expr WHEN val THEN ...) and generic (CASE WHEN cond THEN ...).</desc>\n");
    xml.push_str("    <syntax>CASE WHEN condition THEN value [WHEN ... THEN ...] [ELSE default] END</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"generic\">RETURN CASE WHEN n.depth &gt; 3000 THEN 'deep' WHEN n.depth &gt; 1000 THEN 'medium' ELSE 'shallow' END AS category</ex>\n");
    xml.push_str("      <ex desc=\"simple\">RETURN CASE n.status WHEN 'PRODUCING' THEN 'active' WHEN 'SHUT DOWN' THEN 'closed' ELSE 'other' END</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </CASE>\n");
}

pub(super) fn write_topic_create(xml: &mut String) {
    xml.push_str("  <CREATE>\n");
    xml.push_str("    <desc>Create new nodes and relationships with properties.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"node\">CREATE (:Field {name: 'Troll', status: 'PRODUCING'})</ex>\n",
    );
    xml.push_str("      <ex desc=\"relationship\">MATCH (a:Field {name: 'Troll'}), (b:Company {name: 'Equinor'}) CREATE (a)-[:OPERATED_BY]-&gt;(b)</ex>\n");
    xml.push_str("      <ex desc=\"with properties\">MATCH (a:Field), (b:Well) WHERE a.name = b.field CREATE (b)-[:BELONGS_TO {since: 2020}]-&gt;(a)</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </CREATE>\n");
}

pub(super) fn write_topic_set(xml: &mut String) {
    xml.push_str("  <SET>\n");
    xml.push_str("    <desc>Set or update properties on existing nodes/relationships.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"set property\">MATCH (n:Field {name: 'Troll'}) SET n.status = 'SHUT DOWN'</ex>\n");
    xml.push_str("      <ex desc=\"set multiple\">MATCH (n:Field {name: 'Troll'}) SET n.status = 'SHUT DOWN', n.end_year = 2025</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </SET>\n");
}

pub(super) fn write_topic_delete(xml: &mut String) {
    xml.push_str("  <DELETE>\n");
    xml.push_str("    <desc>Delete nodes or relationships. REMOVE drops individual properties from a node.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"delete node\">MATCH (n:Field {name: 'Test'}) DELETE n</ex>\n");
    xml.push_str(
        "      <ex desc=\"delete relationship\">MATCH (a)-[r:OLD_REL]-&gt;(b) DELETE r</ex>\n",
    );
    xml.push_str("      <ex desc=\"remove property\">MATCH (n:Field {name: 'Troll'}) REMOVE n.temp_flag</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </DELETE>\n");
}

pub(super) fn write_topic_merge(xml: &mut String) {
    xml.push_str("  <MERGE>\n");
    xml.push_str("    <desc>Match existing node/relationship or create if it doesn't exist (upsert). ON CREATE SET and ON MATCH SET for conditional property updates.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">MERGE (n:Field {name: 'Troll'})</ex>\n");
    xml.push_str("      <ex desc=\"on create\">MERGE (n:Field {name: 'Troll'}) ON CREATE SET n.created = 2025</ex>\n");
    xml.push_str("      <ex desc=\"on match\">MERGE (n:Field {name: 'Troll'}) ON MATCH SET n.updated = 2025</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </MERGE>\n");
}

pub(super) fn write_topic_call_subquery(xml: &mut String) {
    xml.push_str("  <CALL_SUBQUERY>\n");
    xml.push_str("    <desc>Nested read subquery. Uncorrelated CALL { MATCH ... RETURN ... } runs once and its rows cartesian-combine with the outer stream. Correlated CALL { WITH x ... RETURN ... } runs once per outer row with the imported variables bound. The importing WITH lists bare variables only (no aliasing/projection/aggregation). Aggregating bodies preserve the outer row with a zero value; non-aggregating bodies inner-join (zero matches drops the row). Only RETURN columns escape to the outer scope. v1 excludes writes, UNION, and unit (no-RETURN) subqueries in the body.</desc>\n");
    xml.push_str("    <syntax>CALL { [WITH vars] &lt;body clauses&gt; RETURN ... }</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"uncorrelated\">CALL { MATCH (n:Person) RETURN count(n) AS total } RETURN total</ex>\n");
    xml.push_str("      <ex desc=\"correlated per-row aggregate\">MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]-&gt;(f) RETURN count(f) AS c } RETURN p.name, c</ex>\n");
    xml.push_str("      <ex desc=\"per-row top-K\">MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]-&gt;(f) RETURN f.name AS oldest ORDER BY f.age DESC LIMIT 1 } RETURN p.name, oldest</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </CALL_SUBQUERY>\n");
}

pub(super) fn write_topic_operators(xml: &mut String) {
    xml.push_str("  <operators>\n");
    xml.push_str("    <desc>All supported operators with semantics.</desc>\n");
    xml.push_str("    <group name=\"math\" desc=\"Arithmetic\">+ (add), - (subtract), * (multiply), / (divide)</group>\n");
    xml.push_str("    <group name=\"string\" desc=\"String concatenation\">|| — null propagates: 'a' || null = null. Auto-converts numbers: 'v' || 42 = 'v42'.</group>\n");
    xml.push_str("    <group name=\"comparison\" desc=\"Comparison\">= (equal), &lt;&gt; (not equal), &lt;, &gt;, &lt;=, &gt;=, IN (list membership)</group>\n");
    xml.push_str("    <group name=\"logical\" desc=\"Boolean\">AND, OR, NOT, XOR</group>\n");
    xml.push_str("    <group name=\"null\" desc=\"Null checks\">IS NULL, IS NOT NULL</group>\n");
    xml.push_str("    <group name=\"regex\" desc=\"Regex match\">=~ 'pattern' — Java-style regex, case-sensitive by default. Use (?i) for case-insensitive.</group>\n");
    xml.push_str("    <group name=\"predicates\" desc=\"String predicates\">CONTAINS, STARTS WITH, ENDS WITH — case-sensitive substring checks.</group>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"concat with number\">RETURN n.name || '-' || n.block AS label</ex>\n",
    );
    xml.push_str("      <ex desc=\"regex case-insensitive\">WHERE n.name =~ '(?i)troll.*'</ex>\n");
    xml.push_str("      <ex desc=\"IN list\">WHERE n.status IN ['PRODUCING', 'SHUT DOWN']</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </operators>\n");
}

pub(super) fn write_topic_functions(xml: &mut String) {
    xml.push_str("  <functions>\n");
    xml.push_str("    <desc>All built-in functions grouped by category.</desc>\n");
    xml.push_str("    <group name=\"math\">abs(x), ceil(x)/ceiling(x), floor(x), round(x [,decimals]), sqrt(x), sign(x), log(x)/ln(x), log10(x), exp(x), pow(x,y), pi(), rand(), randomUUID(), toInteger(x)/toInt(x), toFloat(x)</group>\n");
    xml.push_str("    <group name=\"trig\">sin(x), cos(x), tan(x), asin(x), acos(x), atan(x), atan2(y,x), cot(x), haversin(x), degrees(x), radians(x). Angles in radians; NULL/non-numeric → NULL.</group>\n");
    xml.push_str("    <group name=\"string\">toString(x), toUpper(s), toLower(s), trim(s), lTrim(s), rTrim(s), replace(s,from,to), substring(s,start[,len]), left(s,n), right(s,n), split(s,delim), reverse(s), size(s)</group>\n");
    xml.push_str("    <group name=\"text_predicates\">text_edit_distance(a,b) — Levenshtein; text_normalize(s) — lowercase + strip punct + collapse whitespace; text_jaccard(a,b[,sep]) — token Jaccard; text_ngrams(s,n) — char n-grams; text_contains_any(s, needles) / text_starts_with_any(s, prefixes) — variadic or list arg</group>\n");
    xml.push_str("    <group name=\"geometry\">geom_buffer(geom, meters); geom_convex_hull(geoms); geom_union/intersection/difference(g1, g2); geom_is_valid(geom); geom_length(geom). Accept WKT strings, node variables, or Points; return WKT strings.</group>\n");
    xml.push_str("    <group name=\"aggregate\">count(*)/count(expr), sum(expr), avg(expr), min(expr), max(expr), collect(expr), stDev(expr)/std(expr), variance(expr)/var_samp(expr), median(expr), percentile_cont(expr,p), percentile_disc(expr,p)</group>\n");
    xml.push_str("    <group name=\"graph\">size(list), length(path), id(node), labels(node), type(rel), coalesce(expr,...) — first non-null, range(start,end[,step]), keys(node), properties(node)/properties(rel) — full property map, start_node(rel)/end_node(rel) — endpoints</group>\n");
    xml.push_str("    <group name=\"list\">reduce(acc = init, x IN list | body) — fold accumulator over list; any/all/none/single(x IN list WHERE pred); [x IN list WHERE pred | map_expr] — comprehension</group>\n");
    xml.push_str("    <group name=\"json\">parse_json(s)/from_json(s) — parse a JSON string into a structured map/list/scalar (null on invalid). Use it to predicate over properties stored as JSON, e.g. code-graph Function.parameters / Class.fields: any(p IN parse_json(f.parameters) WHERE p.type_annotation = 'Dataset')</group>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"round precision\">RETURN round(n.depth / 1000.0, 1) AS depth_km</ex>\n",
    );
    xml.push_str("      <ex desc=\"coalesce\">RETURN coalesce(n.nickname, n.name) AS label</ex>\n");
    xml.push_str("      <ex desc=\"string\">RETURN toLower(n.name) AS lower_name</ex>\n");
    xml.push_str("      <ex desc=\"aggregate\">RETURN n.status, count(*) AS n, avg(n.depth) AS avg_depth</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("    <group name=\"temporal\">date(str)/datetime(str), localdatetime()/localtime()/time() (wall-clock now as ISO strings; 1-arg parse form), date_diff(d1,d2), date ± N (add/sub days), date - date → days (int), d.year/d.month/d.day</group>\n");
    xml.push_str("    <group name=\"window\">row_number() OVER (...), rank() OVER (...), dense_rank() OVER (...). Syntax: func() OVER (PARTITION BY expr ORDER BY expr [DESC]). PARTITION BY optional.</group>\n");
    xml.push_str("    <group name=\"semantic\">text_score(n, 'col', 'query' [, metric]) — similarity score (metrics: 'cosine', 'poincare', 'dot_product', 'euclidean'); embedding_norm(n, 'col') — L2 norm of embedding vector (hierarchy depth in Poincaré space, 0=root, ~1=leaf)</group>\n");
    xml.push_str("  </functions>\n");
}

pub(super) fn write_topic_patterns(xml: &mut String) {
    xml.push_str("  <patterns>\n");
    xml.push_str("    <desc>Pattern syntax for matching graph structures.</desc>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"labeled node\">(n:Field)</ex>\n");
    xml.push_str("      <ex desc=\"inline properties\">(n:Field {status: 'active'})</ex>\n");
    xml.push_str("      <ex desc=\"directed relationship\">(a)-[:BELONGS_TO]-&gt;(b)</ex>\n");
    xml.push_str(
        "      <ex desc=\"variable-length\">(a)-[:KNOWS*1..3]-&gt;(b) — path length 1 to 3</ex>\n",
    );
    xml.push_str("      <ex desc=\"any relationship\">(a)--&gt;(b) or (a)-[r]-&gt;(b)</ex>\n");
    xml.push_str("      <ex desc=\"list comprehension\">[x IN collect(n.name) WHERE x STARTS WITH '35']</ex>\n");
    xml.push_str("      <ex desc=\"map projection\">n {.name, .status} — returns {name: ..., status: ...}</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </patterns>\n");
}

// ── Procedure deep-dive functions ──────────────────────────────────────────

pub(super) fn write_topic_pagerank(xml: &mut String) {
    xml.push_str("  <pagerank>\n");
    xml.push_str("    <desc>Compute PageRank centrality for all nodes. Higher score = more influential.</desc>\n");
    xml.push_str("    <syntax>CALL pagerank({params}) YIELD node, score</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"damping_factor\" type=\"float\" default=\"0.85\">Probability of following a link vs random jump.</param>\n");
    xml.push_str("      <param name=\"max_iterations\" type=\"int\" default=\"100\">Convergence iteration limit.</param>\n");
    xml.push_str("      <param name=\"tolerance\" type=\"float\" default=\"1e-6\">Convergence threshold.</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("      <param name=\"node_type\" type=\"string|list\">Scope to a node label (subgraph). In-memory graphs only.</param>\n");
    xml.push_str("      <param name=\"where\" type=\"string\">Scope to nodes matching a predicate over the node variable `n` (e.g. 'n.is_test = false AND n.is_external = false'). In-memory graphs only. Applies to all centrality + community procedures (pagerank/degree/betweenness/closeness/louvain/leiden/label_propagation).</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL pagerank() YIELD node, score RETURN node.name, score ORDER BY score DESC LIMIT 10</ex>\n");
    xml.push_str("      <ex desc=\"filtered\">CALL pagerank({connection_types: 'CITES'}) YIELD node, score RETURN node.name, score ORDER BY score DESC</ex>\n");
    xml.push_str("      <ex desc=\"scoped to library functions\">CALL pagerank({node_type: 'Function', connection_types: 'CALLS', where: 'n.is_test = false'}) YIELD node, score RETURN node.name, score ORDER BY score DESC LIMIT 15</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </pagerank>\n");
}

pub(super) fn write_topic_betweenness(xml: &mut String) {
    xml.push_str("  <betweenness>\n");
    xml.push_str("    <desc>Compute betweenness centrality. High score = node lies on many shortest paths (bridge/broker).</desc>\n");
    xml.push_str("    <syntax>CALL betweenness({params}) YIELD node, score</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"normalized\" type=\"bool\" default=\"true\">Normalize scores to 0..1 range.</param>\n");
    xml.push_str("      <param name=\"sample_size\" type=\"int\" optional=\"true\">Approximate by sampling N source nodes (faster for large graphs).</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL betweenness() YIELD node, score RETURN node.name, score ORDER BY score DESC LIMIT 10</ex>\n");
    xml.push_str("      <ex desc=\"sampled\">CALL betweenness({sample_size: 100}) YIELD node, score RETURN node.name, round(score, 4) ORDER BY score DESC</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </betweenness>\n");
}

pub(super) fn write_topic_degree(xml: &mut String) {
    xml.push_str("  <degree>\n");
    xml.push_str("    <desc>Compute degree centrality (number of connections per node, optionally normalized).</desc>\n");
    xml.push_str("    <syntax>CALL degree({params}) YIELD node, score</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"normalized\" type=\"bool\" default=\"true\">Normalize by max possible degree.</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL degree() YIELD node, score RETURN node.name, score ORDER BY score DESC LIMIT 10</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </degree>\n");
}

pub(super) fn write_topic_closeness(xml: &mut String) {
    xml.push_str("  <closeness>\n");
    xml.push_str("    <desc>Compute closeness centrality (inverse of average shortest path distance). High = close to all others.</desc>\n");
    xml.push_str("    <syntax>CALL closeness({params}) YIELD node, score</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"normalized\" type=\"bool\" default=\"true\">Normalize scores.</param>\n");
    xml.push_str("      <param name=\"sample_size\" type=\"int\" optional=\"true\">Approximate by sampling N source nodes (faster for large graphs).</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL closeness() YIELD node, score RETURN node.name, score ORDER BY score DESC LIMIT 10</ex>\n");
    xml.push_str("      <ex desc=\"sampled\">CALL closeness({sample_size: 100}) YIELD node, score RETURN node.name, round(score, 4) ORDER BY score DESC</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </closeness>\n");
}

pub(super) fn write_topic_louvain(xml: &mut String) {
    xml.push_str("  <louvain>\n");
    xml.push_str("    <desc>Community detection using multilevel Louvain modularity optimisation (hierarchical). Assigns each node a community ID; YIELD optional 'level' for the community hierarchy (level 0 = finest).</desc>\n");
    xml.push_str("    <syntax>CALL louvain({params}) YIELD node, community [, level]</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"resolution\" type=\"float\" default=\"1.0\">Higher = more/smaller communities, lower = fewer/larger.</param>\n");
    xml.push_str("      <param name=\"weight_property\" type=\"string\" optional=\"true\">Edge property to use as weight.</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL louvain() YIELD node, community RETURN community, count(*) AS size, collect(node.name) AS members ORDER BY size DESC</ex>\n");
    xml.push_str("      <ex desc=\"hierarchy\">CALL louvain() YIELD node, community, level RETURN level, count(DISTINCT community) AS communities ORDER BY level</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </louvain>\n");
}

pub(super) fn write_topic_leiden(xml: &mut String) {
    xml.push_str("  <leiden>\n");
    xml.push_str("    <desc>Community detection using the Leiden algorithm (multilevel, hierarchical). Like Louvain but a refinement phase guarantees every community is well-connected (Louvain can return internally-disconnected communities). Deterministic. YIELD optional 'level' for the hierarchy (level 0 = finest).</desc>\n");
    xml.push_str("    <syntax>CALL leiden({params}) YIELD node, community [, level]</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"resolution\" type=\"float\" default=\"1.0\">Higher = more/smaller communities, lower = fewer/larger.</param>\n");
    xml.push_str("      <param name=\"weight_property\" type=\"string\" optional=\"true\">Edge property to use as weight.</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL leiden() YIELD node, community RETURN community, count(*) AS size, collect(node.name) AS members ORDER BY size DESC</ex>\n");
    xml.push_str("      <ex desc=\"hierarchy for GraphRAG\">CALL leiden() YIELD node, community, level RETURN level, community, collect(node.name) AS members ORDER BY level, community</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </leiden>\n");
}

pub(super) fn write_topic_label_propagation(xml: &mut String) {
    xml.push_str("  <label_propagation>\n");
    xml.push_str("    <desc>Community detection using label propagation. Fast, non-deterministic. Each node adopts its neighbors' majority label.</desc>\n");
    xml.push_str("    <syntax>CALL label_propagation({params}) YIELD node, community</syntax>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"max_iterations\" type=\"int\" default=\"100\">Iteration limit.</param>\n");
    xml.push_str("      <param name=\"connection_types\" type=\"string|list\">Filter to specific relationship types.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL label_propagation() YIELD node, community RETURN community, count(*) AS size ORDER BY size DESC</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </label_propagation>\n");
}

pub(super) fn write_topic_connected_components(xml: &mut String) {
    xml.push_str("  <connected_components>\n");
    xml.push_str("    <desc>Find weakly connected components. Nodes in the same component can reach each other ignoring edge direction. Optionally scope to a node-type universe and/or relationship type(s) via a parameter map — `{node_type, relationship}`, each a string or list — to analyse a single-relationship projection (e.g. components of the social graph) rather than the whole graph.</desc>\n");
    xml.push_str("    <syntax>CALL connected_components([{node_type, relationship}]) YIELD node, component</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic\">CALL connected_components() YIELD node, component RETURN component, count(*) AS size ORDER BY size DESC</ex>\n");
    xml.push_str("      <ex desc=\"scoped to a relationship\">CALL connected_components({node_type: 'Person', relationship: 'KNOWS'}) YIELD node, component RETURN component, count(*) AS size ORDER BY size DESC</ex>\n");
    xml.push_str("      <ex desc=\"find isolated\">CALL connected_components() YIELD node, component WITH component, count(*) AS size WHERE size = 1 RETURN count(*) AS isolated_nodes</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </connected_components>\n");
}

pub(super) fn write_topic_k_core(xml: &mut String) {
    xml.push_str("  <k_core>\n");
    xml.push_str("    <desc>k-core decomposition: the coreness of each node — the largest k for which the node belongs to a subgraph where every vertex has degree at least k. Optional `{node_type, relationship}` (string or list) scopes the analysis to a single-relationship projection. Filter `WHERE coreness >= k` to extract the k-core itself.</desc>\n");
    xml.push_str(
        "    <syntax>CALL k_core([{node_type, relationship}]) YIELD node, coreness</syntax>\n",
    );
    xml.push_str("    <aliases>coreness</aliases>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"dense core of a social graph\">CALL k_core({node_type: 'Person', relationship: 'KNOWS'}) YIELD node, coreness WHERE coreness >= 3 RETURN node.name, coreness ORDER BY coreness DESC</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </k_core>\n");
}

pub(super) fn write_topic_clustering_coefficient(xml: &mut String) {
    xml.push_str("  <clustering_coefficient>\n");
    xml.push_str("    <desc>Local clustering coefficient per node: how interconnected its neighbours are (fraction of possible links among neighbours that exist), in [0, 1]. Nodes with degree &lt; 2 are 0. Optional `{node_type, relationship}` (string or list) scopes to a single-relationship projection.</desc>\n");
    xml.push_str("    <syntax>CALL clustering_coefficient([{node_type, relationship}]) YIELD node, coefficient</syntax>\n");
    xml.push_str("    <aliases>local_clustering_coefficient</aliases>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"most-cliquey people\">CALL clustering_coefficient({node_type: 'Person', relationship: 'KNOWS'}) YIELD node, coefficient RETURN node.name, coefficient ORDER BY coefficient DESC LIMIT 10</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </clustering_coefficient>\n");
}

pub(super) fn write_topic_cluster(xml: &mut String) {
    xml.push_str("  <cluster>\n");
    xml.push_str("    <desc>Cluster nodes using DBSCAN or K-means. Reads nodes from preceding MATCH clause.</desc>\n");
    xml.push_str("    <syntax>MATCH (n:Type) CALL cluster({params}) YIELD node, cluster RETURN ...</syntax>\n");
    xml.push_str("    <modes>\n");
    xml.push_str("      <spatial>Omit 'properties' — auto-detects lat/lon from set_spatial() config. Uses haversine distance. eps is in meters. Geometry centroids used as fallback for WKT types.</spatial>\n");
    xml.push_str("      <property>Specify properties: ['col1','col2'] — euclidean distance on numeric values. Use normalize: true when feature scales differ.</property>\n");
    xml.push_str("    </modes>\n");
    xml.push_str("    <params>\n");
    xml.push_str("      <param name=\"method\" type=\"string\" default=\"dbscan\">'dbscan' or 'kmeans'.</param>\n");
    xml.push_str("      <param name=\"eps\" type=\"float\" default=\"0.5\">DBSCAN: max neighborhood distance. In meters for spatial mode.</param>\n");
    xml.push_str("      <param name=\"min_points\" type=\"int\" default=\"3\">DBSCAN: min neighbors to form a core point.</param>\n");
    xml.push_str(
        "      <param name=\"k\" type=\"int\" default=\"5\">K-means: number of clusters.</param>\n",
    );
    xml.push_str("      <param name=\"max_iterations\" type=\"int\" default=\"100\">K-means: iteration limit.</param>\n");
    xml.push_str("      <param name=\"normalize\" type=\"bool\" default=\"false\">Property mode: scale features to [0,1] before clustering.</param>\n");
    xml.push_str("      <param name=\"properties\" type=\"list\" optional=\"true\">Numeric property names for property mode. Omit for spatial mode.</param>\n");
    xml.push_str("    </params>\n");
    xml.push_str("    <yields>node (the matched node), cluster (int — cluster ID; -1 = noise for DBSCAN)</yields>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"spatial DBSCAN\">MATCH (f:Field) CALL cluster({method: 'dbscan', eps: 50000, min_points: 2}) YIELD node, cluster RETURN cluster, count(*) AS n, collect(node.name) AS fields ORDER BY n DESC</ex>\n");
    xml.push_str("      <ex desc=\"property K-means\">MATCH (w:Well) CALL cluster({properties: ['depth', 'temperature'], method: 'kmeans', k: 3, normalize: true}) YIELD node, cluster RETURN cluster, collect(node.name) AS wells</ex>\n");
    xml.push_str("      <ex desc=\"spatial K-means\">MATCH (s:Station) CALL cluster({method: 'kmeans', k: 4}) YIELD node, cluster RETURN cluster, count(*) AS n</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </cluster>\n");
}

pub(super) fn write_topic_explain(xml: &mut String) {
    xml.push_str("  <EXPLAIN>\n");
    xml.push_str("    <desc>Show query plan without executing. Returns a ResultView with columns [step, operation, estimated_rows].</desc>\n");
    xml.push_str("    <syntax>EXPLAIN &lt;any Cypher query&gt;</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic plan\">EXPLAIN MATCH (n:Person) WHERE n.age &gt; 30 RETURN n.name</ex>\n");
    xml.push_str("      <ex desc=\"inspect fused optimization\">EXPLAIN MATCH (n:Person) RETURN count(n)</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("    <notes>Cardinality estimates use type_indices counts. Fused optimizations shown as single steps.</notes>\n");
    xml.push_str("  </EXPLAIN>\n");
}

pub(super) fn write_topic_profile(xml: &mut String) {
    xml.push_str("  <PROFILE>\n");
    xml.push_str("    <desc>Execute query AND collect per-clause statistics. Returns normal results with a .profile property.</desc>\n");
    xml.push_str("    <syntax>PROFILE &lt;any Cypher query&gt;</syntax>\n");
    xml.push_str("    <profile_columns>clause (str), rows_in (int), rows_out (int), elapsed_us (int)</profile_columns>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"profile read query\">PROFILE MATCH (n:Person) WHERE n.age &gt; 30 RETURN n.name</ex>\n");
    xml.push_str("      <ex desc=\"profile mutation\">PROFILE CREATE (n:Temp {val: 1})</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("    <notes>Access stats via result.profile (list of dicts). None for non-profiled queries.</notes>\n");
    xml.push_str("  </PROFILE>\n");
}

// ── Tier 3: structural-validator rule procedures ─────────────────────────

pub(super) fn write_topic_orphan_node(xml: &mut String) {
    xml.push_str("  <orphan_node>\n");
    xml.push_str("    <desc>Yields nodes of {type} that have zero edges in any direction. Almost always ingest artifacts.</desc>\n");
    xml.push_str("    <syntax>CALL orphan_node({type: 'Wellbore'}) YIELD node</syntax>\n");
    xml.push_str("    <yield>node — bound to the orphaned NodeIndex (use node.id, node.title, etc.)</yield>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"count orphans\">CALL orphan_node({type: 'Discovery'}) YIELD node RETURN count(node) AS c</ex>\n");
    xml.push_str("      <ex desc=\"top-5 orphan ids\">CALL orphan_node({type: 'Wellbore'}) YIELD node RETURN node.id, node.title LIMIT 5</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </orphan_node>\n");
}

pub(super) fn write_topic_self_loop(xml: &mut String) {
    xml.push_str("  <self_loop>\n");
    xml.push_str("    <desc>Yields nodes of {type} that have an outgoing {edge} whose target is themselves. Always a data error in tree-shaped hierarchies; sometimes legitimate for self-referential domain edges.</desc>\n");
    xml.push_str(
        "    <syntax>CALL self_loop({type: 'Person', edge: 'KNOWS'}) YIELD node</syntax>\n",
    );
    xml.push_str("    <yield>node — bound to the self-looping NodeIndex</yield>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"find self-citations\">CALL self_loop({type: 'CourtDecision', edge: 'CITES'}) YIELD node RETURN node.id</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </self_loop>\n");
}

pub(super) fn write_topic_cycle_2step(xml: &mut String) {
    xml.push_str("  <cycle_2step>\n");
    xml.push_str("    <desc>Yields (node_a, node_b) pairs where a -[edge]-&gt; b -[edge]-&gt; a, both nodes of {type}, with id(a) &lt; id(b) (deduplicated).</desc>\n");
    xml.push_str("    <syntax>CALL cycle_2step({type: 'Person', edge: 'KNOWS'}) YIELD node_a, node_b</syntax>\n");
    xml.push_str("    <yield>node_a, node_b — two NodeIndex bindings (named to avoid CASE's reserved END keyword)</yield>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"find reciprocal pairs\">CALL cycle_2step({type: 'Person', edge: 'KNOWS'}) YIELD node_a, node_b RETURN node_a.name, node_b.name</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </cycle_2step>\n");
}

pub(super) fn write_topic_missing_required_edge(xml: &mut String) {
    xml.push_str("  <missing_required_edge>\n");
    xml.push_str("    <desc>Yields nodes of {type} that have NO outgoing {edge}. Direction-validated: refuses to execute when {type} is on the target side of {edge} in the graph's actual schema, suggesting missing_inbound_edge instead.</desc>\n");
    xml.push_str("    <syntax>CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node</syntax>\n");
    xml.push_str("    <yield>node — bound to the violating NodeIndex</yield>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"wellbores missing licence link\">CALL missing_required_edge({type: 'Wellbore', edge: 'IN_LICENCE'}) YIELD node RETURN count(node) AS missing</ex>\n");
    xml.push_str("      <ex desc=\"composed: PL057 wellbores missing DRILLED_BY\">MATCH (l:Licence {title: '057'})&lt;-[:IN_LICENCE]-(w:Wellbore) WITH collect(w.id) AS pl057 CALL missing_required_edge({type: 'Wellbore', edge: 'DRILLED_BY'}) YIELD node WHERE node.id IN pl057 RETURN count(*)</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </missing_required_edge>\n");
}

pub(super) fn write_topic_missing_inbound_edge(xml: &mut String) {
    xml.push_str("  <missing_inbound_edge>\n");
    xml.push_str("    <desc>Yields nodes of {type} that have NO incoming {edge}. Mirror of missing_required_edge with the same direction validation in reverse.</desc>\n");
    xml.push_str("    <syntax>CALL missing_inbound_edge({type: 'Discovery', edge: 'IN_DISCOVERY'}) YIELD node</syntax>\n");
    xml.push_str("    <yield>node — bound to the violating NodeIndex</yield>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"discoveries with no source wellbore\">CALL missing_inbound_edge({type: 'Discovery', edge: 'IN_DISCOVERY'}) YIELD node RETURN node.title</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </missing_inbound_edge>\n");
}

pub(super) fn write_topic_duplicate_title(xml: &mut String) {
    xml.push_str("  <duplicate_title>\n");
    xml.push_str("    <desc>Yields one row per node of {type} whose title is shared with at least one other node of the same type. Aggregate downstream to get per-group rollups.</desc>\n");
    xml.push_str("    <syntax>CALL duplicate_title({type: 'Prospect'}) YIELD node</syntax>\n");
    xml.push_str(
        "    <yield>node — bound to a NodeIndex whose title appears more than once</yield>\n",
    );
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"all duplicates\">CALL duplicate_title({type: 'Prospect'}) YIELD node RETURN count(node)</ex>\n");
    xml.push_str("      <ex desc=\"group + count\">CALL duplicate_title({type: 'Prospect'}) YIELD node WITH node.title AS title, collect(node) AS dups WITH title, size(dups) AS n WHERE n &gt; 1 RETURN title, n ORDER BY n DESC LIMIT 20</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </duplicate_title>\n");
}

pub(super) fn write_topic_null_property(xml: &mut String) {
    xml.push_str("  <null_property>\n");
    xml.push_str("    <desc>Yields nodes of {type} where {property} is missing, null, or empty string.</desc>\n");
    xml.push_str(
        "    <syntax>CALL null_property({type: 'Person', property: 'email'}) YIELD node</syntax>\n",
    );
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"count missing emails\">CALL null_property({type: 'Person', property: 'email'}) YIELD node RETURN count(node)</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </null_property>\n");
}

pub(super) fn write_topic_inverse_violation(xml: &mut String) {
    xml.push_str("  <inverse_violation>\n");
    xml.push_str("    <desc>Yields (a, b) pairs where (a)-[rel_a]-&gt;(b) exists but the inverse (b)-[rel_b]-&gt;(a) does not. Use when two relations are declared inverse (parent_of/child_of, manages/works_for, cites/cited_by).</desc>\n");
    xml.push_str("    <syntax>CALL inverse_violation({rel_a: 'parent_of', rel_b: 'child_of'}) YIELD a, b</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"unidirectional citations\">CALL inverse_violation({rel_a: 'CITES', rel_b: 'CITED_BY'}) YIELD a, b RETURN a.id, b.id LIMIT 50</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </inverse_violation>\n");
}

pub(super) fn write_topic_transitivity_violation(xml: &mut String) {
    xml.push_str("  <transitivity_violation>\n");
    xml.push_str("    <desc>Yields (a, b, c) triples where (a)-[rel]-&gt;(b)-[rel]-&gt;(c) exists but the direct (a)-[rel]-&gt;(c) edge does not. Use for taxonomy / call-graph / citation-chain hygiene.</desc>\n");
    xml.push_str(
        "    <syntax>CALL transitivity_violation({rel: 'subClassOf'}) YIELD a, b, c</syntax>\n",
    );
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"taxonomy fold\">CALL transitivity_violation({rel: 'subClassOf'}) YIELD a, b, c RETURN a.id, c.id, count(b) AS bridges</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </transitivity_violation>\n");
}

pub(super) fn write_topic_cardinality_violation(xml: &mut String) {
    xml.push_str("  <cardinality_violation>\n");
    xml.push_str("    <desc>Yields nodes of {type} whose outgoing-{edge} count is outside [min, max]. Setting max=1 catches functional-property violations; min=1 catches missing-required-edge.</desc>\n");
    xml.push_str("    <syntax>CALL cardinality_violation({type: 'Country', edge: 'HAS_CAPITAL', min: 1, max: 1}) YIELD node, count</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"countries with multiple capitals\">CALL cardinality_violation({type: 'Country', edge: 'HAS_CAPITAL', max: 1}) YIELD node, count RETURN node.title, count ORDER BY count DESC</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </cardinality_violation>\n");
}

pub(super) fn write_topic_type_domain_violation(xml: &mut String) {
    xml.push_str("  <type_domain_violation>\n");
    xml.push_str("    <desc>Yields edges of {edge} whose source node is not of {expected_source} type. Schema integrity check.</desc>\n");
    xml.push_str("    <syntax>CALL type_domain_violation({edge: 'CITES', expected_source: 'Case'}) YIELD source, target</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"non-case citations\">CALL type_domain_violation({edge: 'CITES', expected_source: 'Case'}) YIELD source, target RETURN labels(source), count(*)</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </type_domain_violation>\n");
}

pub(super) fn write_topic_type_range_violation(xml: &mut String) {
    xml.push_str("  <type_range_violation>\n");
    xml.push_str("    <desc>Yields edges of {edge} whose target node is not of {expected_target} type. Schema integrity check (mirror of type_domain_violation).</desc>\n");
    xml.push_str("    <syntax>CALL type_range_violation({edge: 'CITES', expected_target: 'Case'}) YIELD source, target</syntax>\n");
    xml.push_str("  </type_range_violation>\n");
}

pub(super) fn write_topic_parallel_edges(xml: &mut String) {
    xml.push_str("  <parallel_edges>\n");
    xml.push_str("    <desc>Yields (a, b) pairs connected by more than one edge of {edge}. Almost always a load-time bug (duplicate CSV rows, non-deduping upsert path).</desc>\n");
    xml.push_str("    <syntax>CALL parallel_edges({edge: 'CITES'}) YIELD a, b, count</syntax>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"top duplicates\">CALL parallel_edges({edge: 'CITES'}) YIELD a, b, count RETURN a.id, b.id, count ORDER BY count DESC LIMIT 20</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </parallel_edges>\n");
}

pub(super) fn write_topic_spatial(xml: &mut String) {
    xml.push_str("  <spatial>\n");
    xml.push_str("    <desc>Spatial functions for geographic queries. Requires set_spatial() config on the node type (location or geometry). All distance/area/perimeter results are in meters.</desc>\n");
    xml.push_str("    <setup>Python: g.set_spatial('Field', location=('lat', 'lon')) or g.set_spatial('Area', geometry='wkt')</setup>\n");
    xml.push_str("    <note>WKT uses (longitude latitude) order per OGC standard. point(lat, lon) uses latitude-first. These conventions differ — be careful when mixing them.</note>\n");
    xml.push_str("    <functions>\n");
    xml.push_str("      <fn name=\"distance(a, b)\">Geodesic distance in meters between two spatial nodes. Returns Null if either node has no location.</fn>\n");
    xml.push_str("      <fn name=\"contains(a, b)\">True if geometry a fully contains geometry b (or point b).</fn>\n");
    xml.push_str("      <fn name=\"intersects(a, b)\">True if geometries a and b overlap.</fn>\n");
    xml.push_str(
        "      <fn name=\"centroid(n)\">Returns {lat, lon} centroid of node's geometry.</fn>\n",
    );
    xml.push_str("      <fn name=\"area(n)\">Area of node's geometry in m².</fn>\n");
    xml.push_str("      <fn name=\"perimeter(n)\">Perimeter of node's geometry in meters.</fn>\n");
    xml.push_str("    </functions>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"distance between nodes\">MATCH (a:Field {name: 'Troll'}), (b:Field {name: 'Ekofisk'}) RETURN distance(a, b) / 1000.0 AS km</ex>\n");
    xml.push_str("      <ex desc=\"nearest neighbors\">MATCH (a:Field {name: 'Troll'}), (b:Field) WHERE a &lt;&gt; b RETURN b.name, round(distance(a, b) / 1000.0, 1) AS km ORDER BY km LIMIT 5</ex>\n");
    xml.push_str("      <ex desc=\"contains check\">MATCH (area:Block), (w:Well) WHERE contains(area, w) RETURN area.name, collect(w.name) AS wells</ex>\n");
    xml.push_str("      <ex desc=\"area calculation\">MATCH (b:Block) RETURN b.name, round(area(b) / 1e6, 1) AS km2</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </spatial>\n");
}

pub(super) fn write_topic_temporal(xml: &mut String) {
    xml.push_str("  <temporal>\n");
    xml.push_str("    <desc>Temporal filtering functions for date-range validity checks on nodes and relationships. Works with any date/datetime string or DateTime properties. NULL fields are treated as open-ended boundaries.</desc>\n");
    xml.push_str("    <functions>\n");
    xml.push_str("      <fn name=\"date(str) / datetime(str)\">Parse date string to DateTime value. Supports 'YYYY-MM-DD' format.</fn>\n");
    xml.push_str("      <fn name=\"date_diff(d1, d2)\">Days between two dates (d1 - d2). Same as date subtraction.</fn>\n");
    xml.push_str("      <fn name=\"date + N / date - N\">Add/subtract N days from a date.</fn>\n");
    xml.push_str("      <fn name=\"date - date\">Days between two dates (returns integer).</fn>\n");
    xml.push_str("      <fn name=\"d.year / d.month / d.day\">Extract year, month, or day from a DateTime value.</fn>\n");
    xml.push_str("      <fn name=\"valid_at(entity, date, 'from_field', 'to_field')\">True if entity.from_field &lt;= date &lt;= entity.to_field. NULL from_field = valid since beginning. NULL to_field = still valid.</fn>\n");
    xml.push_str("      <fn name=\"valid_during(entity, start, end, 'from_field', 'to_field')\">True if entity's validity period overlaps [start, end]. Overlap: entity.from_field &lt;= end AND entity.to_field &gt;= start. NULL = open-ended.</fn>\n");
    xml.push_str("    </functions>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"node valid at date\">MATCH (e:Estimate) WHERE valid_at(e, '2020-06-15', 'date_from', 'date_to') RETURN e.title, e.value</ex>\n");
    xml.push_str("      <ex desc=\"edge valid at date\">MATCH (a)-[r:EMPLOYED_AT]->(b) WHERE valid_at(r, '2023-01-01', 'start_date', 'end_date') RETURN a.name, b.name</ex>\n");
    xml.push_str("      <ex desc=\"range overlap\">MATCH (p:Prospect) WHERE valid_during(p, '2021-01-01', '2022-12-31', 'date_from', 'date_to') RETURN p.title</ex>\n");
    xml.push_str("      <ex desc=\"with date()\">MATCH (e:Estimate) WHERE valid_at(e, date('2020-06-15'), 'date_from', 'date_to') RETURN e.title</ex>\n");
    xml.push_str("      <ex desc=\"open-ended\">MATCH (c:Contract) WHERE valid_at(c, '2025-01-01', 'start_date', 'end_date') RETURN c.title -- NULL end_date = still valid</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("    <null_semantics>\n");
    xml.push_str("      <rule>NULL from_field = valid since the beginning (always passes the from check)</rule>\n");
    xml.push_str("      <rule>NULL to_field = still valid / open-ended (always passes the to check)</rule>\n");
    xml.push_str("      <rule>Both NULL = always valid (returns true)</rule>\n");
    xml.push_str("    </null_semantics>\n");
    xml.push_str("  </temporal>\n");
}

// ── Fluent API reference ──────────────────────────────────────────────────

const FLUENT_TOPIC_LIST: &str = "select, where, traverse, compare, spatial, temporal, \
    retrieval, statistics, algorithms, vectors, timeseries, mutation, \
    loading, export, indexes, set_ops, subgraph, schema, transactions";

/// Tier 2: compact fluent API reference grouped by functional area.
pub(super) fn write_fluent_overview(xml: &mut String) {
    xml.push_str("<fluent_api>\n");
    xml.push_str("  <note>Selection model: most methods return a new KnowledgeGraph with updated selection. Data is materialised only on retrieval (collect, to_df, etc.).</note>\n");

    // Selection & filtering
    xml.push_str("  <group name=\"selection\">\n");
    xml.push_str("    <method sig=\"select(type, sort=None, limit=None, include_secondary=False)\">Select all nodes of a type. include_secondary=True also selects nodes carrying type as a secondary label. Returns lazy selection.</method>\n");
    xml.push_str("    <method sig=\"where({prop: value})\">Filter by property: exact, comparison (&gt;,&lt;,&gt;=,&lt;=), string (contains, starts_with, ends_with, regex), in, is_null, is_not_null, negated variants.</method>\n");
    xml.push_str(
        "    <method sig=\"where_any([{...}, {...}])\">OR logic across condition sets.</method>\n",
    );
    xml.push_str("    <method sig=\"where_connected(conn_type, direction='any')\">Keep nodes that have a specific connection.</method>\n");
    xml.push_str("    <method sig=\"where_orphans(include_orphans=True)\">Filter by connectivity: orphans only or connected only.</method>\n");
    xml.push_str("    <method sig=\"sort(prop, ascending=True)\">Sort selection. Multi-col: sort([('a', True), ('b', False)]).</method>\n");
    xml.push_str("    <method sig=\"limit(n)\">Limit to first n results.</method>\n");
    xml.push_str("    <method sig=\"offset(n)\">Skip first n results (for pagination).</method>\n");
    xml.push_str("    <method sig=\"expand(hops=1)\">BFS expansion — include all nodes within n hops.</method>\n");
    xml.push_str("  </group>\n");

    // Traversal
    xml.push_str("  <group name=\"traversal\">\n");
    xml.push_str("    <method sig=\"traverse(conn_type, direction=None, target_type=None, where=None, where_connection=None, sort=None, limit=None)\">Follow graph edges. Returns target nodes as new selection level.</method>\n");
    xml.push_str("    <method sig=\"compare(target_type, method, filter=None, sort=None, limit=None)\">Spatial, semantic, or clustering comparison against a target type.</method>\n");
    xml.push_str("    <method sig=\"add_properties({Type: [props]})\">Enrich leaf nodes with properties from ancestor levels (copy, rename, aggregate, spatial).</method>\n");
    xml.push_str("    <method sig=\"create_connections(conn_type)\">Materialise direct edges from traversal chain.</method>\n");
    xml.push_str("  </group>\n");

    // Spatial
    xml.push_str("  <group name=\"spatial\">\n");
    xml.push_str("    <method sig=\"set_spatial(type, lat_field, lon_field, geometry_field=None)\">Declare spatial fields for a node type.</method>\n");
    xml.push_str("    <method sig=\"near_point(lat, lon, max_distance_deg)\">Filter by distance in degrees (fast, approximate).</method>\n");
    xml.push_str("    <method sig=\"near_point_m(lat, lon, max_distance_m)\">Filter by geodesic distance in meters (WGS84).</method>\n");
    xml.push_str("    <method sig=\"within_bounds(min_lat, min_lon, max_lat, max_lon)\">Bounding-box filter.</method>\n");
    xml.push_str("    <method sig=\"contains_point(lat, lon)\">Point-in-polygon test (requires WKT geometry).</method>\n");
    xml.push_str("    <method sig=\"intersects_geometry(wkt)\">Geometry overlap test.</method>\n");
    xml.push_str("    <method sig=\"bounds()\">Geographic bounding box of selection.</method>\n");
    xml.push_str("    <method sig=\"centroid()\">Average lat/lon of selection.</method>\n");
    xml.push_str("  </group>\n");

    // Temporal
    xml.push_str("  <group name=\"temporal\">\n");
    xml.push_str("    <method sig=\"valid_at(date, from_col='valid_from', to_col='valid_to')\">Point-in-time filter: keep nodes valid at a specific date.</method>\n");
    xml.push_str("    <method sig=\"valid_during(start, end, from_col='valid_from', to_col='valid_to')\">Range overlap filter: keep nodes valid during a period.</method>\n");
    xml.push_str("  </group>\n");

    // Retrieval
    xml.push_str("  <group name=\"retrieval\">\n");
    xml.push_str("    <method sig=\"collect(limit=None)\">Materialise selected nodes as a flat ResultView.</method>\n");
    xml.push_str("    <method sig=\"collect_grouped(group_by, parent_info=False)\">Materialise nodes grouped by parent type as dict.</method>\n");
    xml.push_str("    <method sig=\"to_df()\">Export selection as pandas DataFrame.</method>\n");
    xml.push_str(
        "    <method sig=\"to_gdf()\">Export as GeoDataFrame (requires WKT geometry).</method>\n",
    );
    xml.push_str(
        "    <method sig=\"ids()\">Lightweight retrieval: id + type + title only.</method>\n",
    );
    xml.push_str("    <method sig=\"node(type, id)\">O(1) lookup by type + id. Returns dict or None.</method>\n");
    xml.push_str("    <method sig=\"count(group_by=None)\">Count nodes, optionally grouped by property.</method>\n");
    xml.push_str("    <method sig=\"len()\">O(1) count of selected nodes.</method>\n");
    xml.push_str("    <method sig=\"sample(n)\">Random sample as ResultView.</method>\n");
    xml.push_str("    <method sig=\"titles()\">Title-only retrieval.</method>\n");
    xml.push_str("    <method sig=\"get_properties(props)\">Specific properties as list of tuples.</method>\n");
    xml.push_str("  </group>\n");

    // Statistics
    xml.push_str("  <group name=\"statistics\">\n");
    xml.push_str("    <method sig=\"statistics(properties=None, group_by=None)\">Descriptive stats: count, mean, std, min, max, sum.</method>\n");
    xml.push_str("    <method sig=\"calculate(expression, store_as=None)\">Math expressions on properties. store_as saves result as new property.</method>\n");
    xml.push_str("    <method sig=\"unique_values(property, store_as=None)\">Distinct values for a property.</method>\n");
    xml.push_str(
        "    <method sig=\"degrees(connection_type=None)\">Node degree counts.</method>\n",
    );
    xml.push_str("  </group>\n");

    // Graph algorithms
    xml.push_str("  <group name=\"algorithms\">\n");
    xml.push_str("    <method sig=\"shortest_path(source_type, source_id, target_type, target_id, connection_type=None, directed=True)\">Full path with node details.</method>\n");
    xml.push_str("    <method sig=\"shortest_path_length(...)\">Hop count only.</method>\n");
    xml.push_str("    <method sig=\"all_paths(source_type, source_id, target_type, target_id, max_hops=5)\">Enumerate all paths.</method>\n");
    xml.push_str("    <method sig=\"pagerank(damping_factor=0.85, connection_type=None)\">PageRank centrality.</method>\n");
    xml.push_str("    <method sig=\"betweenness_centrality(connection_type=None)\">Betweenness centrality.</method>\n");
    xml.push_str("    <method sig=\"louvain_communities(resolution=1.0, connection_type=None)\">Community detection (Louvain).</method>\n");
    xml.push_str("    <method sig=\"connected_components(mode='weak', connection_type=None)\">Connected component analysis.</method>\n");
    xml.push_str("  </group>\n");

    // Vector search
    xml.push_str("  <group name=\"vectors\">\n");
    xml.push_str("    <method sig=\"set_embedder(model_name_or_callable)\">Register embedding model for text search.</method>\n");
    xml.push_str("    <method sig=\"embed_texts(type, column)\">Compute and store embeddings for a text column.</method>\n");
    xml.push_str("    <method sig=\"search_text(query, type, column=None, top_k=10, min_score=None)\">Semantic text search (auto-embeds query).</method>\n");
    xml.push_str("    <method sig=\"vector(vector, type, column=None, top_k=10, min_score=None)\">Search with pre-computed query vector.</method>\n");
    xml.push_str("  </group>\n");

    // Timeseries
    xml.push_str("  <group name=\"timeseries\">\n");
    xml.push_str("    <method sig=\"set_timeseries(type, resolution, channels, units=None)\">Declare timeseries schema for a node type.</method>\n");
    xml.push_str("    <method sig=\"add_timeseries(df, type, fk_field, time_key, channels)\">Bulk load timeseries data from DataFrame.</method>\n");
    xml.push_str("    <method sig=\"timeseries(type, id, channel=None)\">Retrieve timeseries for a node (all channels or specific).</method>\n");
    xml.push_str("  </group>\n");

    // Mutation
    xml.push_str("  <group name=\"mutation\">\n");
    xml.push_str("    <method sig=\"update({prop: value}, conflict_handling='update')\">Batch property update on selected nodes.</method>\n");
    xml.push_str("  </group>\n");

    // Data loading
    xml.push_str("  <group name=\"loading\">\n");
    xml.push_str("    <method sig=\"add_nodes(df, type, id_field, title_field, columns=None, column_types=None, timeseries=None)\">Load nodes from DataFrame.</method>\n");
    xml.push_str("    <method sig=\"add_connections(df, conn_type, source_type, source_id, target_type, target_id)\">Load edges from DataFrame.</method>\n");
    xml.push_str("    <method sig=\"extend(other, conflict_handling='update')\">Merge another in-memory KnowledgeGraph into this one in place (node identity (type,id); unions secondary labels; dedups edges on (type,src,tgt)). Returns a report dict.</method>\n");
    xml.push_str("    <method sig=\"kglite.from_blueprint(path, verbose=False)\">Build graph from JSON blueprint + CSVs.</method>\n");
    xml.push_str("  </group>\n");

    // Export & persistence
    xml.push_str("  <group name=\"export\">\n");
    xml.push_str("    <method sig=\"export(path, format='graphml')\">Export as GraphML, GEXF, JSON (D3), or CSV.</method>\n");
    xml.push_str("    <method sig=\"export_csv(directory)\">CSV tree + blueprint.json (round-trips with from_blueprint).</method>\n");
    xml.push_str("    <method sig=\"save(path)\">Binary .kgl v3 file (auto-columnar, supports larger-than-RAM loading).</method>\n");
    xml.push_str("    <method sig=\"kglite.load(path)\">Restore from .kgl file.</method>\n");
    xml.push_str("  </group>\n");

    // Columnar storage
    xml.push_str("  <group name=\"columnar\">\n");
    xml.push_str("    <method sig=\"enable_columnar()\">Convert properties to per-type columnar stores (lower memory).</method>\n");
    xml.push_str("    <method sig=\"disable_columnar()\">Convert back to compact per-node storage.</method>\n");
    xml.push_str(
        "    <method sig=\"is_columnar\">Property: True if columnar storage is active.</method>\n",
    );
    xml.push_str("  </group>\n");

    // Set operations
    xml.push_str("  <group name=\"set_ops\">\n");
    xml.push_str("    <method sig=\"union(other)\">Nodes in either selection.</method>\n");
    xml.push_str("    <method sig=\"intersection(other)\">Nodes in both selections.</method>\n");
    xml.push_str("    <method sig=\"difference(other)\">Nodes in first but not second.</method>\n");
    xml.push_str("  </group>\n");

    // Indexes
    xml.push_str("  <group name=\"indexes\">\n");
    xml.push_str("    <method sig=\"create_index(type, property)\">Equality index for fast lookup.</method>\n");
    xml.push_str("    <method sig=\"create_range_index(type, property)\">B-tree for range queries.</method>\n");
    xml.push_str("    <method sig=\"create_composite_index(type, [prop1, prop2])\">Multi-column index.</method>\n");
    xml.push_str("  </group>\n");

    // Transactions
    xml.push_str("  <group name=\"transactions\">\n");
    xml.push_str(
        "    <method sig=\"begin()\">Read-write transaction (context manager).</method>\n",
    );
    xml.push_str("    <method sig=\"begin_read()\">Read-only transaction, O(1) cost (context manager).</method>\n");
    xml.push_str("  </group>\n");

    xml.push_str("  <hint>Use graph_overview(fluent=['traverse','where','spatial',...]) for detailed docs with examples.</hint>\n");
    xml.push_str("</fluent_api>\n");
}

/// Tier 3: detailed fluent API docs for specific topics with params and examples.
pub(super) fn write_fluent_topics(xml: &mut String, topics: &[String]) -> Result<(), String> {
    if topics.is_empty() {
        write_fluent_overview(xml);
        return Ok(());
    }

    xml.push_str("<fluent_api>\n");
    for topic in topics {
        let key = topic.to_lowercase();
        match key.as_str() {
            "select" | "selection" | "where" | "filtering" => write_fluent_topic_selection(xml),
            "traverse" | "traversal" => write_fluent_topic_traversal(xml),
            "compare" | "comparison" => write_fluent_topic_compare(xml),
            "spatial" => write_fluent_topic_spatial(xml),
            "temporal" => write_fluent_topic_temporal(xml),
            "retrieval" | "collect" => write_fluent_topic_retrieval(xml),
            "statistics" | "calculate" => write_fluent_topic_statistics(xml),
            "algorithms" | "graph_algorithms" => write_fluent_topic_algorithms(xml),
            "vectors" | "embeddings" | "search" => write_fluent_topic_vectors(xml),
            "timeseries" => write_fluent_topic_timeseries(xml),
            "mutation" | "update" => write_fluent_topic_mutation(xml),
            "loading" | "data_loading" => write_fluent_topic_loading(xml),
            "export" | "persistence" => write_fluent_topic_export(xml),
            "indexes" => write_fluent_topic_indexes(xml),
            "set_ops" => write_fluent_topic_set_operations(xml),
            "subgraph" => write_fluent_topic_subgraph(xml),
            "schema" => write_fluent_topic_schema(xml),
            "transactions" => write_fluent_topic_transactions(xml),
            _ => {
                return Err(format!(
                    "Unknown fluent API topic '{}'. Available: {}",
                    topic, FLUENT_TOPIC_LIST
                ));
            }
        }
    }
    xml.push_str("</fluent_api>\n");
    Ok(())
}

// ── Fluent tier 3: topic detail functions ──────────────────────────────────

pub(super) fn write_fluent_topic_selection(xml: &mut String) {
    xml.push_str("  <selection>\n");
    xml.push_str("    <desc>Select and filter nodes using method chaining. All filter methods return a new lazy selection.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"select(type, sort=None, limit=None, include_secondary=False)\">Start a selection on a node type. include_secondary=True also includes nodes carrying type as a secondary label.</m>\n");
    xml.push_str("      <m sig=\"where({prop: value})\">Exact match, comparison (&gt;, &lt;, &gt;=, &lt;=), string predicates (contains, starts_with, ends_with, regex), in-list, null checks, negated variants (not_in, not_contains).</m>\n");
    xml.push_str("      <m sig=\"where_any([{...}, {...}])\">OR logic: keep nodes matching any condition set.</m>\n");
    xml.push_str("      <m sig=\"where_connected(conn_type, direction='any')\">Keep only nodes that have a specific connection.</m>\n");
    xml.push_str(
        "      <m sig=\"where_orphans(include_orphans=True)\">Filter by connectivity.</m>\n",
    );
    xml.push_str("      <m sig=\"sort(prop, ascending=True)\">Sort by property. Multi-col: sort([('a', True), ('b', False)]).</m>\n");
    xml.push_str("      <m sig=\"limit(n) / offset(n)\">Pagination.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"exact match\">graph.select('Person').where({'city': 'Oslo'})</ex>\n",
    );
    xml.push_str("      <ex desc=\"comparison\">graph.select('Product').where({'price': {'&gt;=': 100, '&lt;=': 500}})</ex>\n");
    xml.push_str("      <ex desc=\"string search\">graph.select('Person').where({'name': {'contains': 'ali'}})</ex>\n");
    xml.push_str("      <ex desc=\"IN list\">graph.select('Person').where({'city': {'in': ['Oslo', 'Bergen']}})</ex>\n");
    xml.push_str("      <ex desc=\"null check\">graph.select('Person').where({'email': {'is_not_null': True}})</ex>\n");
    xml.push_str(
        "      <ex desc=\"regex\">graph.select('Person').where({'name': {'regex': '^A.*'}})</ex>\n",
    );
    xml.push_str("      <ex desc=\"OR logic\">graph.select('Person').where_any([{'city': 'Oslo'}, {'age': {'&gt;': 60}}])</ex>\n");
    xml.push_str("      <ex desc=\"pagination\">graph.select('Person').sort('name').offset(20).limit(10)</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </selection>\n");
}

pub(super) fn write_fluent_topic_traversal(xml: &mut String) {
    xml.push_str("  <traversal>\n");
    xml.push_str("    <desc>Follow graph edges to navigate the graph. traverse() adds target nodes as a new hierarchy level. For spatial/semantic/clustering operations, use compare() instead.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"traverse(conn_type, direction=None, target_type=None, where=None, where_connection=None, sort=None, limit=None)\">Follow edges. direction: 'outgoing', 'incoming', or None (both).</m>\n");
    xml.push_str("      <m sig=\"add_properties({Type: [props]})\">Enrich leaf nodes with properties from ancestor levels. Supports copy, rename, Agg helpers (count, sum, mean, min, max, std, collect), and Spatial helpers (distance, area, perimeter, centroid_lat, centroid_lon).</m>\n");
    xml.push_str("      <m sig=\"create_connections(conn_type)\">Materialise direct edges from a traversal chain.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic outgoing\">graph.select('Person').traverse('WORKS_AT').collect()</ex>\n");
    xml.push_str("      <ex desc=\"incoming with filter\">graph.select('Company').traverse('WORKS_AT', direction='incoming', where={'age': {'&gt;': 30}})</ex>\n");
    xml.push_str("      <ex desc=\"target type filter\">graph.select('Well').traverse('OF_FIELD', direction='incoming', target_type='ProductionProfile')</ex>\n");
    xml.push_str("      <ex desc=\"multi-hop chain\">graph.select('Person').traverse('WORKS_AT').traverse('LOCATED_IN').collect()</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </traversal>\n");
}

pub(super) fn write_fluent_topic_compare(xml: &mut String) {
    xml.push_str("  <compare>\n");
    xml.push_str("    <desc>Compare selected nodes against a target type using spatial, semantic, or clustering methods. Results are added as a new hierarchy level.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"compare(target_type, 'contains')\">Spatial: keep targets whose geometry contains the source point.</m>\n");
    xml.push_str("      <m sig=\"compare(target_type, 'intersects')\">Spatial: keep targets whose geometry intersects the source.</m>\n");
    xml.push_str("      <m sig=\"compare(target_type, {'type': 'distance', 'max_m': N})\">Spatial: keep targets within N meters.</m>\n");
    xml.push_str("      <m sig=\"compare(target_type, {'type': 'text_score', 'property': 'col', 'metric': 'cosine'|'poincare'})\">Semantic: rank by embedding similarity (default cosine; use 'poincare' for hierarchical data).</m>\n");
    xml.push_str("      <m sig=\"compare(target_type, {'type': 'cluster', 'k': N})\">Cluster targets by features (K-means or DBSCAN).</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"spatial containment\">graph.select('Structure').compare('Well', 'contains').collect()</ex>\n");
    xml.push_str("      <ex desc=\"distance\">graph.select('Well').compare('Well', {'type': 'distance', 'max_m': 5000})</ex>\n");
    xml.push_str("      <ex desc=\"semantic\">graph.select('Doc').compare('Doc', {'type': 'text_score', 'property': 'summary', 'threshold': 0.7})</ex>\n");
    xml.push_str("      <ex desc=\"clustering\">graph.select('Well').compare('Well', {'type': 'cluster', 'k': 5, 'features': ['lat', 'lon']})</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </compare>\n");
}

pub(super) fn write_fluent_topic_spatial(xml: &mut String) {
    xml.push_str("  <spatial>\n");
    xml.push_str("    <desc>Spatial filtering and aggregation. Requires set_spatial() or column_types during add_nodes().</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"set_spatial(type, lat_field, lon_field, geometry_field=None)\">Declare spatial fields for a node type.</m>\n");
    xml.push_str("      <m sig=\"near_point(lat, lon, max_distance_deg)\">Filter by distance in degrees (fast, approximate). ~111km per degree at equator.</m>\n");
    xml.push_str("      <m sig=\"near_point_m(lat, lon, max_distance_m)\">Geodesic distance filter in meters (WGS84, Vincenty).</m>\n");
    xml.push_str("      <m sig=\"within_bounds(min_lat, min_lon, max_lat, max_lon)\">Bounding-box filter.</m>\n");
    xml.push_str("      <m sig=\"contains_point(lat, lon)\">Point-in-polygon test (requires WKT geometry).</m>\n");
    xml.push_str("      <m sig=\"intersects_geometry(wkt)\">Geometry overlap test.</m>\n");
    xml.push_str("      <m sig=\"bounds()\">Bounding box of current selection: {min_lat, min_lon, max_lat, max_lon}.</m>\n");
    xml.push_str("      <m sig=\"centroid()\">Average lat/lon: {lat, lon}.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"setup\">graph.set_spatial('City', 'latitude', 'longitude')</ex>\n",
    );
    xml.push_str("      <ex desc=\"near point (degrees)\">graph.select('City').near_point(59.91, 10.75, 0.5)</ex>\n");
    xml.push_str("      <ex desc=\"near point (meters)\">graph.select('City').near_point_m(59.91, 10.75, 50000)</ex>\n");
    xml.push_str("      <ex desc=\"bounding box\">graph.select('Field').within_bounds(55.0, 0.0, 65.0, 15.0)</ex>\n");
    xml.push_str("      <ex desc=\"point in polygon\">graph.select('Block').contains_point(60.5, 4.2)</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </spatial>\n");
}

pub(super) fn write_fluent_topic_temporal(xml: &mut String) {
    xml.push_str("  <temporal>\n");
    xml.push_str("    <desc>Temporal validity filtering. Nodes must have valid_from / valid_to (or custom-named) date properties.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"valid_at(date, from_col='valid_from', to_col='valid_to')\">Keep nodes valid at a specific date. date can be 'YYYY-MM-DD' string or datetime.</m>\n");
    xml.push_str("      <m sig=\"valid_during(start, end, from_col='valid_from', to_col='valid_to')\">Keep nodes whose validity overlaps a date range.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"point in time\">graph.select('Licence').valid_at('2020-06-15')</ex>\n",
    );
    xml.push_str("      <ex desc=\"range overlap\">graph.select('Licence').valid_during('2020-01-01', '2020-12-31')</ex>\n");
    xml.push_str("      <ex desc=\"custom columns\">graph.select('Contract').valid_at('2023-01-01', from_col='start_date', to_col='end_date')</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </temporal>\n");
}

pub(super) fn write_fluent_topic_retrieval(xml: &mut String) {
    xml.push_str("  <retrieval>\n");
    xml.push_str("    <desc>Materialise selected nodes. Most selectors are lazy — these methods trigger data retrieval.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"collect(limit=None)\">Flat ResultView (iterable, indexable, .to_list(), .to_df()).</m>\n");
    xml.push_str("      <m sig=\"collect_grouped(group_by, parent_info=False)\">Nodes grouped by parent type as dict.</m>\n");
    xml.push_str("      <m sig=\"to_df()\">Pandas DataFrame with all properties as columns.</m>\n");
    xml.push_str("      <m sig=\"to_gdf()\">GeoDataFrame with geometry column (requires spatial config).</m>\n");
    xml.push_str("      <m sig=\"ids()\">Lightweight: id + type + title only.</m>\n");
    xml.push_str(
        "      <m sig=\"node(type, id)\">O(1) single-node lookup. Returns dict or None.</m>\n",
    );
    xml.push_str(
        "      <m sig=\"count(group_by=None)\">Count, optionally grouped by property.</m>\n",
    );
    xml.push_str("      <m sig=\"len()\">O(1) selection size.</m>\n");
    xml.push_str("      <m sig=\"sample(n)\">Random n nodes as ResultView.</m>\n");
    xml.push_str("      <m sig=\"titles()\">Title-only list.</m>\n");
    xml.push_str("      <m sig=\"get_properties(props)\">Specific properties as tuples.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"collect all\">results = graph.select('Person').where({'city': 'Oslo'}).collect()</ex>\n");
    xml.push_str("      <ex desc=\"to dataframe\">df = graph.select('Person').to_df()</ex>\n");
    xml.push_str("      <ex desc=\"single lookup\">node = graph.node('Person', 42)</ex>\n");
    xml.push_str("      <ex desc=\"existence check\">if graph.exists('Person', 42): ...</ex>\n");
    xml.push_str(
        "      <ex desc=\"count by group\">graph.select('Person').count(group_by='city')</ex>\n",
    );
    xml.push_str("      <ex desc=\"random sample\">graph.select('Person').sample(5)</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </retrieval>\n");
}

pub(super) fn write_fluent_topic_statistics(xml: &mut String) {
    xml.push_str("  <statistics>\n");
    xml.push_str("    <desc>Descriptive statistics, calculations, and aggregations on selected nodes.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"statistics(properties=None, group_by=None)\">Count, mean, std, min, max, sum for numeric properties.</m>\n");
    xml.push_str("      <m sig=\"calculate(expression, store_as=None)\">Math expression on properties. store_as persists result.</m>\n");
    xml.push_str("      <m sig=\"unique_values(property, store_as=None)\">Distinct values for a property.</m>\n");
    xml.push_str(
        "      <m sig=\"degrees(connection_type=None)\">In/out/total degree counts per node.</m>\n",
    );
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"basic stats\">graph.select('Product').statistics(['price', 'quantity'])</ex>\n");
    xml.push_str("      <ex desc=\"grouped stats\">graph.select('Product').statistics(['price'], group_by='category')</ex>\n");
    xml.push_str("      <ex desc=\"calculate\">graph.select('Product').calculate('price * quantity', store_as='revenue')</ex>\n");
    xml.push_str("      <ex desc=\"unique\">graph.select('Person').unique_values('city')</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </statistics>\n");
}

pub(super) fn write_fluent_topic_algorithms(xml: &mut String) {
    xml.push_str("  <algorithms>\n");
    xml.push_str("    <desc>Graph algorithms: paths, centrality, community detection.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"shortest_path(src_type, src_id, tgt_type, tgt_id, connection_type=None, directed=True)\">Full path with node details.</m>\n");
    xml.push_str("      <m sig=\"shortest_path_length(src_type, src_id, tgt_type, tgt_id, ...)\">Hop count only (integer).</m>\n");
    xml.push_str("      <m sig=\"shortest_path_ids(src_type, src_id, tgt_type, tgt_id, ...)\">Path as list of (type, id) tuples.</m>\n");
    xml.push_str("      <m sig=\"all_paths(src_type, src_id, tgt_type, tgt_id, max_hops=5)\">All paths up to max_hops.</m>\n");
    xml.push_str("      <m sig=\"pagerank(damping_factor=0.85, connection_type=None)\">PageRank centrality → ResultView.</m>\n");
    xml.push_str("      <m sig=\"betweenness_centrality(connection_type=None)\">Betweenness centrality → ResultView.</m>\n");
    xml.push_str("      <m sig=\"degree_centrality(connection_type=None, normalized=True)\">Degree centrality → dict.</m>\n");
    xml.push_str("      <m sig=\"closeness_centrality(connection_type=None)\">Closeness centrality → ResultView.</m>\n");
    xml.push_str("      <m sig=\"louvain_communities(resolution=1.0, connection_type=None)\">Community detection → ResultView.</m>\n");
    xml.push_str("      <m sig=\"label_propagation(max_iterations=100)\">Label propagation communities → ResultView.</m>\n");
    xml.push_str("      <m sig=\"connected_components(mode='weak', connection_type=None)\">Component analysis → ResultView.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"shortest path\">graph.shortest_path('Person', 1, 'Person', 42)</ex>\n",
    );
    xml.push_str("      <ex desc=\"path length\">graph.shortest_path_length('City', 'Oslo', 'City', 'Bergen', connection_type='ROAD')</ex>\n");
    xml.push_str("      <ex desc=\"pagerank\">graph.pagerank(connection_type='CITES')</ex>\n");
    xml.push_str("      <ex desc=\"communities\">graph.louvain_communities(resolution=1.5)</ex>\n");
    xml.push_str("      <ex desc=\"components\">graph.connected_components(mode='weak')</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </algorithms>\n");
}

pub(super) fn write_fluent_topic_vectors(xml: &mut String) {
    xml.push_str("  <vectors>\n");
    xml.push_str("    <desc>Embedding storage and semantic search. Requires set_embedder() or pre-computed vectors.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"set_embedder(model_name_or_callable)\">Register embedding model (sentence-transformers name or callable).</m>\n");
    xml.push_str("      <m sig=\"embed_texts(type, column)\">Compute and store embeddings for a text column.</m>\n");
    xml.push_str("      <m sig=\"set_embeddings(type, column, embeddings_dict)\">Provide pre-computed embeddings {id: vector}.</m>\n");
    xml.push_str("      <m sig=\"search_text(query, type, column=None, top_k=10, min_score=None)\">Semantic search — auto-embeds query string.</m>\n");
    xml.push_str("      <m sig=\"vector(vector, type, column=None, top_k=10, min_score=None)\">Search with explicit query vector.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"setup\">graph.set_embedder('all-MiniLM-L6-v2')</ex>\n");
    xml.push_str("      <ex desc=\"embed\">graph.embed_texts('Paper', 'abstract')</ex>\n");
    xml.push_str("      <ex desc=\"text search\">graph.search_text('machine learning for graphs', 'Paper', top_k=5)</ex>\n");
    xml.push_str(
        "      <ex desc=\"min score\">graph.search_text('NLP', 'Paper', min_score=0.7)</ex>\n",
    );
    xml.push_str("    </examples>\n");
    xml.push_str("  </vectors>\n");
}

pub(super) fn write_fluent_topic_timeseries(xml: &mut String) {
    xml.push_str("  <timeseries>\n");
    xml.push_str("    <desc>Time-indexed data per node. Declare schema, bulk-load from DataFrame, retrieve per node.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"set_timeseries(type, resolution, channels, units=None)\">Declare timeseries schema. resolution: 'day'|'month'|'year'.</m>\n");
    xml.push_str("      <m sig=\"add_timeseries(df, type, fk_field, time_key, channels)\">Bulk load from DataFrame with foreign key to nodes.</m>\n");
    xml.push_str("      <m sig=\"timeseries(type, id, channel=None)\">Retrieve all channels or a specific channel for one node.</m>\n");
    xml.push_str("      <m sig=\"timeseries_config(type)\">Query timeseries metadata (resolution, channels, units).</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"schema\">graph.set_timeseries('Field', resolution='month', channels=['oil', 'gas'], units={'oil': 'MSm3'})</ex>\n");
    xml.push_str("      <ex desc=\"bulk load\">graph.add_timeseries(prod_df, 'Field', fk_field='field_id', time_key='date', channels=['oil', 'gas'])</ex>\n");
    xml.push_str(
        "      <ex desc=\"retrieve\">ts = graph.timeseries('Field', 123, channel='oil')</ex>\n",
    );
    xml.push_str("      <ex desc=\"inline loading\">graph.add_nodes(df, 'Prod', 'id', 'name', timeseries={'time': 'date', 'channels': ['oil', 'gas']})</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </timeseries>\n");
}

pub(super) fn write_fluent_topic_mutation(xml: &mut String) {
    xml.push_str("  <mutation>\n");
    xml.push_str("    <desc>Update properties on selected nodes.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"update({prop: value}, conflict_handling='update')\">Batch property update. conflict_handling: 'update'|'preserve'|'replace'.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"set property\">graph.select('Person').where({'city': 'Oslo'}).update({'country': 'Norway'})</ex>\n");
    xml.push_str("      <ex desc=\"preserve existing\">graph.select('Person').update({'status': 'active'}, conflict_handling='preserve')</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </mutation>\n");
}

pub(super) fn write_fluent_topic_loading(xml: &mut String) {
    xml.push_str("  <loading>\n");
    xml.push_str(
        "    <desc>Load nodes and connections from DataFrames or blueprint files.</desc>\n",
    );
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"add_nodes(df, type, id_field, title_field, columns=None, column_types=None, conflict_handling='skip', timeseries=None)\">Load nodes. conflict_handling: 'update'|'replace'|'skip'|'preserve'|'sum'. column_types maps columns to spatial/temporal types.</m>\n");
    xml.push_str("      <m sig=\"add_connections(data, conn_type, source_type, source_id_field, target_type, target_id_field, columns=None, conflict_handling='update', query=None, extra_properties=None)\">Load edges from DataFrame (data=df) or Cypher query (data=None, query='MATCH...RETURN...'). conflict_handling: 'update'|'replace'|'skip'|'preserve'|'sum'. extra_properties stamps static props onto query-mode edges.</m>\n");
    xml.push_str("      <m sig=\"replace_connections(data, conn_type, source_type, source_id_field, target_type, target_id_field, ...)\">Atomic edge upsert: prune each source node's existing conn_type edges, then add the input's. Same args as add_connections; use to re-sync a derived edge set idempotently.</m>\n");
    xml.push_str("      <m sig=\"add_nodes_bulk(specs)\">Bulk load multiple node types: [{'node_type': ..., 'data': df, ...}].</m>\n");
    xml.push_str(
        "      <m sig=\"add_connections_bulk(specs)\">Bulk load multiple connection types.</m>\n",
    );
    xml.push_str("      <m sig=\"kglite.from_blueprint(path, verbose=False)\">Build graph from JSON blueprint + CSVs.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"basic nodes\">graph.add_nodes(df, 'Person', 'id', 'name')</ex>\n",
    );
    xml.push_str("      <ex desc=\"with spatial\">graph.add_nodes(df, 'City', 'id', 'name', column_types={'lat': 'location.lat', 'lon': 'location.lon'})</ex>\n");
    xml.push_str("      <ex desc=\"edges\">graph.add_connections(df, 'WORKS_AT', 'Person', 'person_id', 'Company', 'company_id')</ex>\n");
    xml.push_str("      <ex desc=\"edges from query\">graph.add_connections(None, 'ENCLOSES', 'Play', 'play_id', 'Area', 'area_id', query='MATCH (p:Play), (a:Area) WHERE contains(p, a) RETURN DISTINCT p.id AS play_id, a.id AS area_id')</ex>\n");
    xml.push_str("      <ex desc=\"blueprint\">graph = kglite.from_blueprint('blueprint.json', verbose=True)</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </loading>\n");
}

pub(super) fn write_fluent_topic_export(xml: &mut String) {
    xml.push_str("  <export>\n");
    xml.push_str("    <desc>Export graph data and persist to disk.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"export(path, format='graphml')\">Export as 'graphml', 'gexf', 'json' (D3), or 'csv'.</m>\n");
    xml.push_str(
        "      <m sig=\"export_string(format='graphml')\">Export to string (no file).</m>\n",
    );
    xml.push_str("      <m sig=\"export_csv(directory)\">CSV directory tree + blueprint.json (round-trips with from_blueprint).</m>\n");
    xml.push_str("      <m sig=\"save(path)\">Binary .kgl v3 file (auto-columnar, supports larger-than-RAM loading).</m>\n");
    xml.push_str("      <m sig=\"kglite.load(path)\">Restore from .kgl file.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str(
        "      <ex desc=\"graphml\">graph.export('graph.graphml', format='graphml')</ex>\n",
    );
    xml.push_str("      <ex desc=\"csv roundtrip\">graph.export_csv('output/'); g2 = kglite.from_blueprint('output/blueprint.json')</ex>\n");
    xml.push_str(
        "      <ex desc=\"binary\">graph.save('graph.kgl'); g2 = kglite.load('graph.kgl')</ex>\n",
    );
    xml.push_str("    </examples>\n");
    xml.push_str("  </export>\n");
}

pub(super) fn write_fluent_topic_indexes(xml: &mut String) {
    xml.push_str("  <indexes>\n");
    xml.push_str("    <desc>Create property indexes for faster lookups. Type indices are automatic.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"create_index(type, property)\">Equality index: fast exact-match lookup.</m>\n");
    xml.push_str("      <m sig=\"create_range_index(type, property)\">B-tree index: fast range queries (&gt;, &lt;, &gt;=, &lt;=).</m>\n");
    xml.push_str("      <m sig=\"create_composite_index(type, [prop1, prop2, ...])\">Multi-property index.</m>\n");
    xml.push_str("      <m sig=\"drop_index(type, property) / drop_range_index / drop_composite_index\">Remove indexes.</m>\n");
    xml.push_str("      <m sig=\"list_indexes() / list_composite_indexes()\">Enumerate existing indexes.</m>\n");
    xml.push_str(
        "      <m sig=\"index_stats(type, property)\">Index metadata and hit count.</m>\n",
    );
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"equality\">graph.create_index('Person', 'email')</ex>\n");
    xml.push_str("      <ex desc=\"range\">graph.create_range_index('Product', 'price')</ex>\n");
    xml.push_str("      <ex desc=\"composite\">graph.create_composite_index('Person', ['city', 'age'])</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </indexes>\n");
}

pub(super) fn write_fluent_topic_set_operations(xml: &mut String) {
    xml.push_str("  <set_ops>\n");
    xml.push_str("    <desc>Combine selections using set logic.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"union(other)\">Nodes in either selection.</m>\n");
    xml.push_str("      <m sig=\"intersection(other)\">Nodes in both selections.</m>\n");
    xml.push_str("      <m sig=\"difference(other)\">In first but not second.</m>\n");
    xml.push_str("      <m sig=\"symmetric_difference(other)\">In exactly one selection.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"union\">oslo_or_young = graph.select('Person').where({'city': 'Oslo'}).union(graph.select('Person').where({'age': {'&lt;': 25}}))</ex>\n");
    xml.push_str(
        "      <ex desc=\"intersection\">oslo_and_young = oslo.intersection(young)</ex>\n",
    );
    xml.push_str("    </examples>\n");
    xml.push_str("  </set_ops>\n");
}

pub(super) fn write_fluent_topic_subgraph(xml: &mut String) {
    xml.push_str("  <subgraph>\n");
    xml.push_str(
        "    <desc>Extract a subset of the graph into a new independent KnowledgeGraph.</desc>\n",
    );
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"to_subgraph()\">Extract selected nodes + inter-edges into a new graph.</m>\n");
    xml.push_str("      <m sig=\"subgraph_stats()\">Preview extraction: node/edge counts without materialising.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"extract\">sub = graph.select('Person').where({'city': 'Oslo'}).to_subgraph()</ex>\n");
    xml.push_str("      <ex desc=\"preview\">graph.select('Person').subgraph_stats()</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </subgraph>\n");
}

pub(super) fn write_fluent_topic_schema(xml: &mut String) {
    xml.push_str("  <schema>\n");
    xml.push_str("    <desc>Inspect and enforce graph schema.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"schema()\">Full schema dict: node types, connections, indexes, counts.</m>\n");
    xml.push_str("      <m sig=\"schema_text()\">Human-readable schema summary.</m>\n");
    xml.push_str("      <m sig=\"properties(type)\">Per-property statistics: type, non_null, unique, samples.</m>\n");
    xml.push_str(
        "      <m sig=\"connection_types()\">All connection types with counts and endpoints.</m>\n",
    );
    xml.push_str(
        "      <m sig=\"describe(types=['...'])\">AI-optimised XML for specific types.</m>\n",
    );
    xml.push_str("      <m sig=\"define_schema(schema_dict)\">Enforce schema constraints on future loads.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"full schema\">graph.schema()</ex>\n");
    xml.push_str("      <ex desc=\"text overview\">print(graph.schema_text())</ex>\n");
    xml.push_str("      <ex desc=\"property detail\">graph.properties('Person')</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </schema>\n");
}

pub(super) fn write_fluent_topic_transactions(xml: &mut String) {
    xml.push_str("  <transactions>\n");
    xml.push_str("    <desc>Transactional access with automatic rollback on error.</desc>\n");
    xml.push_str("    <methods>\n");
    xml.push_str("      <m sig=\"begin()\">Read-write transaction. Use as context manager.</m>\n");
    xml.push_str("      <m sig=\"begin_read()\">Read-only transaction (O(1) cost, no copy). Use as context manager.</m>\n");
    xml.push_str("    </methods>\n");
    xml.push_str("    <examples>\n");
    xml.push_str("      <ex desc=\"read-write\">with graph.begin() as tx: tx.select('Person').update({'verified': True})</ex>\n");
    xml.push_str("      <ex desc=\"read-only\">with graph.begin_read() as ro: count = ro.select('Person').len()</ex>\n");
    xml.push_str("    </examples>\n");
    xml.push_str("  </transactions>\n");
}

/// Tier 2: compact Cypher reference — all clauses, operators, functions, procedures.
/// No examples. Ends with hint to use tier 3.
pub(super) fn write_cypher_overview(xml: &mut String) {
    xml.push_str("<cypher>\n");

    // Clauses
    xml.push_str("  <clauses>\n");
    xml.push_str("    <clause name=\"MATCH\">Pattern-match nodes and relationships. OPTIONAL MATCH for left-join semantics.</clause>\n");
    xml.push_str("    <clause name=\"WHERE\">Filter by predicate (comparison, null check, regex, string predicates).</clause>\n");
    xml.push_str("    <clause name=\"RETURN\">Project columns. Supports DISTINCT, aliases (AS), aggregations.</clause>\n");
    xml.push_str("    <clause name=\"WITH\">Intermediate projection, aggregation, and variable scoping.</clause>\n");
    xml.push_str("    <clause name=\"ORDER BY\">Sort results. Append DESC for descending. Combine with SKIP n, LIMIT n.</clause>\n");
    xml.push_str("    <clause name=\"UNWIND\">Expand a list into individual rows: UNWIND expr AS var.</clause>\n");
    xml.push_str(
        "    <clause name=\"UNION\">Combine result sets. UNION ALL keeps duplicates.</clause>\n",
    );
    xml.push_str("    <clause name=\"CASE\">Conditional expression: CASE WHEN cond THEN val ... ELSE val END.</clause>\n");
    xml.push_str(
        "    <clause name=\"CREATE\">Create nodes and relationships with properties.</clause>\n",
    );
    xml.push_str("    <clause name=\"SET\">Set or update node/relationship properties.</clause>\n");
    xml.push_str("    <clause name=\"DELETE\">Delete nodes/relationships. REMOVE to drop individual properties.</clause>\n");
    xml.push_str(
        "    <clause name=\"MERGE\">Match existing or create new (upsert pattern).</clause>\n",
    );
    xml.push_str("    <clause name=\"CALL { }\">Read subquery — runs a nested MATCH/WITH/RETURN per outer row. Uncorrelated CALL { MATCH ... RETURN ... } runs once (cartesian-combined); correlated CALL { WITH p MATCH (p)-->... RETURN count(...) AS c } runs per outer row. Importing WITH lists bare variables only. Example: MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]-&gt;(f) RETURN count(f) AS c } RETURN p.name, c</clause>\n");
    xml.push_str("    <clause name=\"HAVING\">Post-aggregation filter on RETURN/WITH. Example: RETURN n.type, count(*) AS cnt HAVING cnt > 5</clause>\n");
    xml.push_str("    <clause name=\"EXPLAIN\">Prefix to show query plan as ResultView [step, operation, estimated_rows] without executing.</clause>\n");
    xml.push_str("    <clause name=\"PROFILE\">Prefix to execute and collect per-clause stats. Result has .profile with [clause, rows_in, rows_out, elapsed_us].</clause>\n");
    xml.push_str("  </clauses>\n");

    // Operators
    xml.push_str("  <operators>\n");
    xml.push_str("    <group name=\"math\">+ - * /</group>\n");
    xml.push_str("    <group name=\"string\">|| (concatenation)</group>\n");
    xml.push_str("    <group name=\"comparison\">= &lt;&gt; &lt; &gt; &lt;= &gt;= IN</group>\n");
    xml.push_str("    <group name=\"logical\">AND OR NOT XOR</group>\n");
    xml.push_str("    <group name=\"null\">IS NULL, IS NOT NULL</group>\n");
    xml.push_str("    <group name=\"regex\">=~ 'pattern'</group>\n");
    xml.push_str("    <group name=\"predicates\">CONTAINS, STARTS WITH, ENDS WITH</group>\n");
    xml.push_str("  </operators>\n");

    // Functions
    xml.push_str("  <functions>\n");
    xml.push_str("    <group name=\"math\">abs, ceil, floor, round(x [,decimals]), sqrt, sign, log, log10, exp, pow(x,y), pi, rand, randomUUID, toInteger, toFloat</group>\n");
    xml.push_str("    <group name=\"trig\">sin, cos, tan, asin, acos, atan, atan2(y,x), cot, haversin, degrees, radians (radians; NULL/non-numeric → NULL)</group>\n");
    xml.push_str("    <group name=\"string\">toString, toUpper, toLower, trim, lTrim, rTrim, replace, substring, left, right, split, reverse</group>\n");
    xml.push_str(
        "    <group name=\"aggregate\">count, sum, avg, min, max, collect, stDev</group>\n",
    );
    xml.push_str(
        "    <group name=\"graph\">size, length, id, labels, type, coalesce, range, keys</group>\n",
    );
    xml.push_str("    <group name=\"spatial\">distance(a,b)→m, contains(a,b), intersects(a,b), centroid(n), area(n)→m², perimeter(n)→m</group>\n");
    xml.push_str("    <group name=\"temporal\">date(str)/datetime(str), localdatetime()/localtime()/time() (ISO strings), date_diff(d1,d2), date ± N (days), date - date → int, d.year/d.month/d.day, valid_at(...), valid_during(...)</group>\n");
    xml.push_str("    <group name=\"window\">row_number() OVER (...), rank() OVER (...), dense_rank() OVER (...). OVER (PARTITION BY expr ORDER BY expr [DESC])</group>\n");
    xml.push_str("  </functions>\n");

    // Procedures
    xml.push_str("  <procedures>\n");
    xml.push_str("    <proc name=\"pagerank\" yields=\"node, score\">PageRank centrality for all nodes.</proc>\n");
    xml.push_str("    <proc name=\"betweenness\" yields=\"node, score\">Betweenness centrality for all nodes.</proc>\n");
    xml.push_str("    <proc name=\"degree\" yields=\"node, score\">Degree centrality for all nodes.</proc>\n");
    xml.push_str("    <proc name=\"closeness\" yields=\"node, score\">Closeness centrality for all nodes.</proc>\n");
    xml.push_str("    <proc name=\"louvain\" yields=\"node, community\">Community detection (Louvain algorithm).</proc>\n");
    xml.push_str("    <proc name=\"label_propagation\" yields=\"node, community\">Community detection (label propagation).</proc>\n");
    xml.push_str("    <proc name=\"connected_components\" yields=\"node, component\">Weakly connected components.</proc>\n");
    xml.push_str("    <proc name=\"cluster\" yields=\"node, cluster\">DBSCAN/K-means clustering on spatial or property data.</proc>\n");
    xml.push_str("  </procedures>\n");

    // Patterns
    xml.push_str("  <patterns>(n:Label), (n {prop: val}), (a)-[:TYPE]-&gt;(b), (a)-[:T*1..3]-&gt;(b), [x IN list WHERE pred | expr], n {.p1, .p2}</patterns>\n");

    xml.push_str("  <limitations>\n");
    xml.push_str("    <item feature=\"FOREACH\" workaround=\"UNWIND list AS x CREATE/SET ... (equivalent result)\"/>\n");
    xml.push_str("    <item feature=\"CALL {} subqueries\" workaround=\"Use WITH chaining or multiple cypher() calls\"/>\n");
    xml.push_str("    <item feature=\"LOAD CSV\" workaround=\"Use Python pandas/csv, then CREATE nodes from dicts\"/>\n");
    xml.push_str("    <item feature=\"CREATE INDEX\" note=\"Type indices are automatic; no manual index management needed\"/>\n");
    xml.push_str("    <item feature=\"Multi-label nodes\" note=\"Supported since 0.10.5. Each node has a primary type (set at creation, immutable) plus optional secondary labels via SET n:Label / CREATE (n:A:B) / g.add_label(...). MATCH (n:A:B) AND-intersects. Retype via SET n.type = 'NewType'.\"/>\n");
    xml.push_str("    <item feature=\"Variable-length weighted paths\" note=\"Unweighted variable-length paths (*1..3) are supported\"/>\n");
    xml.push_str("  </limitations>\n");
    xml.push_str("  <hint>Use graph_overview(cypher=['MATCH','cluster','spatial',...]) for detailed docs with examples.</hint>\n");
    xml.push_str("</cypher>\n");
}
