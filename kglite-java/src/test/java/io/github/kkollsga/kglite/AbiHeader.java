package io.github.kkollsga.kglite;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Reads {@code crates/kglite-c/include/kglite.h} and reduces it to the two
 * things the wrapper depends on: the exported function declarations, and the
 * numeric status codes. Used only by {@link AbiContractTest}.
 */
final class AbiHeader {

    private AbiHeader() {}

    private static final Pattern BLOCK_COMMENT = Pattern.compile("/\\*.*?\\*/", Pattern.DOTALL);
    private static final Pattern LINE_COMMENT = Pattern.compile("//[^\n]*");
    private static final Pattern DIRECTIVE = Pattern.compile("(?m)^\\s*#[^\n]*");
    private static final Pattern FUNCTION = Pattern.compile("\\b(kglite_\\w+)\\s*\\(");
    private static final Pattern STATUS = Pattern.compile("KGLITE_STATUS_CODE_(\\w+)\\s*=\\s*(\\d+)");

    /** The header this run checks against, from {@code -Dkglite.header.path}. */
    static Path headerPath() {
        String configured = System.getProperty("kglite.header.path");
        if (configured == null || configured.isBlank()) {
            throw new IllegalStateException("-Dkglite.header.path is not set");
        }
        return Path.of(configured);
    }

    static String read(Path path) {
        try {
            return Files.readString(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + path, e);
        }
    }

    /**
     * Every {@code kglite_*} function the header declares, in header order,
     * mapped from symbol name to its whitespace-normalized declaration.
     */
    static Map<String, String> functions(String header) {
        String stripped = DIRECTIVE.matcher(
                        LINE_COMMENT.matcher(
                                        BLOCK_COMMENT.matcher(header).replaceAll(" "))
                                .replaceAll(" "))
                .replaceAll(" ");
        Map<String, String> declarations = new LinkedHashMap<>();
        for (String chunk : stripped.split(";")) {
            String normalized = chunk.replaceAll("\\s+", " ").trim();
            Matcher matcher = FUNCTION.matcher(normalized);
            if (!matcher.find()) {
                continue;
            }
            // The declaration starts at the return type, which is whatever
            // precedes the last balanced-open paren group; the normalized chunk
            // already begins there once the preceding ';' split it off.
            declarations.put(matcher.group(1), normalized);
        }
        return declarations;
    }

    /** The {@code KgliteStatusCode} discriminants the header declares. */
    static Map<String, Integer> statusCodes(String header) {
        Map<String, Integer> codes = new LinkedHashMap<>();
        Matcher matcher = STATUS.matcher(BLOCK_COMMENT.matcher(header).replaceAll(" "));
        while (matcher.find()) {
            codes.put(matcher.group(1), Integer.parseInt(matcher.group(2)));
        }
        return codes;
    }
}
