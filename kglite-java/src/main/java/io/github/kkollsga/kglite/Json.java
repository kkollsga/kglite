package io.github.kkollsga.kglite;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON reader/writer for the C ABI boundary.
 *
 * <p>The ABI speaks JSON in both directions — parameters go in as a JSON object
 * string, rows come back as a JSON array of objects with natural values — and
 * the JDK ships no JSON API. A ~200-line reader keeps the published artifact
 * dependency-free, which matters more here than anywhere else: a wrapper whose
 * whole selling point is "one JAR, one native library, no framework" should not
 * drag a serialization stack and its version conflicts into every consumer's
 * classpath. The grammar consumed on the read side is emitted by {@code
 * serde_json}, not by untrusted input.
 */
final class Json {

    private Json() {}

    // ---- writing ----------------------------------------------------------

    /**
     * Serialize a parameter map to a JSON object string.
     *
     * @param params the bindings; must have String keys
     * @return the JSON text
     * @throws KgliteException if a value has no JSON representation
     */
    static String writeObject(Map<String, ?> params) {
        StringBuilder out = new StringBuilder();
        writeValue(out, params);
        return out.toString();
    }

    private static void writeValue(StringBuilder out, Object value) {
        switch (value) {
            case null -> out.append("null");
            case String s -> writeString(out, s);
            case Boolean b -> out.append(b.booleanValue());
            case Number n -> writeNumber(out, n);
            case Map<?, ?> map -> writeMap(out, map);
            case Iterable<?> items -> writeArray(out, items);
            case Object[] items -> writeArray(out, java.util.Arrays.asList(items));
            default -> throw new KgliteException(
                    "cannot bind a " + value.getClass().getName() + " as a Cypher parameter;"
                            + " use null, String, Boolean, a Number, Map, Iterable or Object[]");
        }
    }

    private static void writeNumber(StringBuilder out, Number value) {
        switch (value) {
            case Byte b -> out.append(b.longValue());
            case Short s -> out.append(s.longValue());
            case Integer i -> out.append(i.longValue());
            case Long l -> out.append(l.longValue());
            case BigInteger b -> out.append(b);
            case BigDecimal b -> out.append(b.toPlainString());
            default -> {
                double d = value.doubleValue();
                if (Double.isNaN(d) || Double.isInfinite(d)) {
                    throw new KgliteException(
                            "cannot bind a non-finite number as a Cypher parameter: " + value);
                }
                out.append(d);
            }
        }
    }

    private static void writeMap(StringBuilder out, Map<?, ?> map) {
        out.append('{');
        boolean first = true;
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!(entry.getKey() instanceof String key)) {
                throw new KgliteException(
                        "Cypher parameter map keys must be Strings, got "
                                + (entry.getKey() == null ? "null" : entry.getKey().getClass().getName()));
            }
            if (!first) {
                out.append(',');
            }
            first = false;
            writeString(out, key);
            out.append(':');
            writeValue(out, entry.getValue());
        }
        out.append('}');
    }

    private static void writeArray(StringBuilder out, Iterable<?> items) {
        out.append('[');
        boolean first = true;
        for (Object item : items) {
            if (!first) {
                out.append(',');
            }
            first = false;
            writeValue(out, item);
        }
        out.append(']');
    }

    private static void writeString(StringBuilder out, String value) {
        out.append('"');
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        out.append('"');
    }

    // ---- reading ----------------------------------------------------------

    /**
     * Decode a Cypher result into rows keyed in column order.
     *
     * @param columnsJson the {@code ["a","b"]} array from the ABI, may be null
     * @param rowsJson    the {@code [{"a":1}]} array from the ABI
     * @return one insertion-ordered map per row
     * @throws KgliteException if either blob is missing or malformed
     */
    static List<Map<String, Object>> toRows(String columnsJson, String rowsJson) {
        if (rowsJson == null) {
            throw new KgliteException("the engine could not serialize the result rows");
        }
        List<String> columns = new ArrayList<>();
        if (columnsJson != null) {
            for (Object column : asList(parse(columnsJson), "result columns")) {
                columns.add(String.valueOf(column));
            }
        }
        List<Object> raw = asList(parse(rowsJson), "result rows");
        List<Map<String, Object>> rows = new ArrayList<>(raw.size());
        for (Object entry : raw) {
            if (!(entry instanceof Map<?, ?> cells)) {
                throw new KgliteException("expected a JSON object per result row, got " + entry);
            }
            Map<String, Object> row = new LinkedHashMap<>();
            for (String column : columns) {
                if (cells.containsKey(column)) {
                    row.put(column, cells.get(column));
                }
            }
            for (Map.Entry<?, ?> cell : cells.entrySet()) {
                row.putIfAbsent(String.valueOf(cell.getKey()), cell.getValue());
            }
            rows.add(Collections.unmodifiableMap(row));
        }
        return Collections.unmodifiableList(rows);
    }

    @SuppressWarnings("unchecked")
    private static List<Object> asList(Object value, String what) {
        if (!(value instanceof List<?>)) {
            throw new KgliteException("expected a JSON array for " + what + ", got " + value);
        }
        return (List<Object>) value;
    }

    /**
     * Parse a JSON document into {@code Map}/{@code List}/{@code String}/
     * {@code Long}/{@code Double}/{@code Boolean}/{@code null}.
     *
     * @param text the JSON text
     * @return the decoded value
     * @throws KgliteException on malformed input
     */
    static Object parse(String text) {
        Reader reader = new Reader(text);
        reader.skipWhitespace();
        Object value = reader.readValue();
        reader.skipWhitespace();
        if (!reader.atEnd()) {
            throw reader.fail("trailing content after the JSON value");
        }
        return value;
    }

    private static final class Reader {
        private final String src;
        private int pos;

        Reader(String src) {
            this.src = src;
        }

        boolean atEnd() {
            return pos >= src.length();
        }

        void skipWhitespace() {
            while (pos < src.length()) {
                char c = src.charAt(pos);
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    pos++;
                } else {
                    return;
                }
            }
        }

        KgliteException fail(String why) {
            return new KgliteException("malformed JSON at offset " + pos + ": " + why);
        }

        Object readValue() {
            if (atEnd()) {
                throw fail("unexpected end of input");
            }
            char c = src.charAt(pos);
            return switch (c) {
                case '{' -> readObject();
                case '[' -> readArray();
                case '"' -> readString();
                case 't' -> readLiteral("true", Boolean.TRUE);
                case 'f' -> readLiteral("false", Boolean.FALSE);
                case 'n' -> readLiteral("null", null);
                default -> readNumber();
            };
        }

        private Object readLiteral(String token, Object value) {
            if (!src.startsWith(token, pos)) {
                throw fail("expected " + token);
            }
            pos += token.length();
            return value;
        }

        private Map<String, Object> readObject() {
            pos++; // '{'
            Map<String, Object> map = new LinkedHashMap<>();
            skipWhitespace();
            if (!atEnd() && src.charAt(pos) == '}') {
                pos++;
                return map;
            }
            while (true) {
                skipWhitespace();
                if (atEnd() || src.charAt(pos) != '"') {
                    throw fail("expected an object key");
                }
                String key = readString();
                skipWhitespace();
                if (atEnd() || src.charAt(pos) != ':') {
                    throw fail("expected ':' after an object key");
                }
                pos++;
                skipWhitespace();
                map.put(key, readValue());
                skipWhitespace();
                if (atEnd()) {
                    throw fail("unterminated object");
                }
                char c = src.charAt(pos++);
                if (c == '}') {
                    return map;
                }
                if (c != ',') {
                    throw fail("expected ',' or '}' in an object");
                }
            }
        }

        private List<Object> readArray() {
            pos++; // '['
            List<Object> list = new ArrayList<>();
            skipWhitespace();
            if (!atEnd() && src.charAt(pos) == ']') {
                pos++;
                return list;
            }
            while (true) {
                skipWhitespace();
                list.add(readValue());
                skipWhitespace();
                if (atEnd()) {
                    throw fail("unterminated array");
                }
                char c = src.charAt(pos++);
                if (c == ']') {
                    return list;
                }
                if (c != ',') {
                    throw fail("expected ',' or ']' in an array");
                }
            }
        }

        private String readString() {
            pos++; // opening quote
            StringBuilder out = new StringBuilder();
            while (true) {
                if (atEnd()) {
                    throw fail("unterminated string");
                }
                char c = src.charAt(pos++);
                if (c == '"') {
                    return out.toString();
                }
                if (c != '\\') {
                    out.append(c);
                    continue;
                }
                if (atEnd()) {
                    throw fail("unterminated escape");
                }
                char esc = src.charAt(pos++);
                switch (esc) {
                    case '"' -> out.append('"');
                    case '\\' -> out.append('\\');
                    case '/' -> out.append('/');
                    case 'b' -> out.append('\b');
                    case 'f' -> out.append('\f');
                    case 'n' -> out.append('\n');
                    case 'r' -> out.append('\r');
                    case 't' -> out.append('\t');
                    case 'u' -> {
                        if (pos + 4 > src.length()) {
                            throw fail("truncated \\u escape");
                        }
                        out.append((char) Integer.parseInt(src.substring(pos, pos + 4), 16));
                        pos += 4;
                    }
                    default -> throw fail("unknown escape \\" + esc);
                }
            }
        }

        private Object readNumber() {
            int start = pos;
            boolean floating = false;
            while (pos < src.length()) {
                char c = src.charAt(pos);
                if (c == '-' || c == '+' || (c >= '0' && c <= '9')) {
                    pos++;
                } else if (c == '.' || c == 'e' || c == 'E') {
                    floating = true;
                    pos++;
                } else {
                    break;
                }
            }
            if (start == pos) {
                throw fail("expected a value");
            }
            String token = src.substring(start, pos);
            try {
                if (floating) {
                    return Double.valueOf(token);
                }
                return Long.valueOf(token);
            } catch (NumberFormatException e) {
                try {
                    return Double.valueOf(token);
                } catch (NumberFormatException nested) {
                    throw fail("not a number: " + token);
                }
            }
        }
    }
}
