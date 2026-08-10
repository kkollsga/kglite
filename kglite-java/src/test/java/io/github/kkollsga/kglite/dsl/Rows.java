package io.github.kkollsga.kglite.dsl;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Row comparison shared by the corpus gates. */
final class Rows {

    private Rows() {}

    /**
     * Compares result rows, respecting order only when the statement asked for one.
     *
     * @param expected the expected rows
     * @param actual the rows the engine returned
     * @param orderSensitive whether the statement carries an ORDER BY
     * @param message the assertion message prefix
     */
    static void assertRows(
            List<Map<String, Object>> expected,
            List<Map<String, Object>> actual,
            boolean orderSensitive,
            String message) {
        if (orderSensitive) {
            assertEquals(expected, actual, message);
            return;
        }
        assertEquals(expected.size(), actual.size(), message + ": row count");
        List<Map<String, Object>> remaining = new ArrayList<>(actual);
        for (Map<String, Object> row : expected) {
            if (!remaining.remove(row)) {
                throw new AssertionError(
                        message + ": expected row " + row + " is absent from " + actual);
            }
        }
    }
}
