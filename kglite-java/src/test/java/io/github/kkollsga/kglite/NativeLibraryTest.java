package io.github.kkollsga.kglite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The three resolution tiers, exercised without initializing {@link Abi}.
 *
 * <p>{@code Abi} links in its static initializer, so every resolution outcome
 * reachable there is either "the whole class works" or "a
 * ExceptionInInitializerError". These tests drive {@link NativeLibrary}
 * directly instead, which is the only way the <em>failure</em> shapes — an
 * unbundled platform, an empty staging directory, a bad override — can be
 * asserted at all.
 *
 * <p>The extraction tests use the real library staged by the build's
 * {@code stageNativesForTests} task, and finish by asking the JVM's linker to
 * open the extracted file. Copying bytes into a cache directory proves nothing
 * on its own; the property that matters to a consumer is that the file the
 * cache hands back is loadable, and only {@code libraryLookup} decides that.
 */
final class NativeLibraryTest {

    private static final String OS = System.getProperty("os.name", "");
    private static final String ARCH = System.getProperty("os.arch", "");

    /** The platform directory the bundled resource for this JVM lives under. */
    private static String hostPlatform() {
        return NativeLibrary.platform(OS, ARCH);
    }

    private static String hostFileName() {
        return NativeLibrary.libraryFileName(OS);
    }

    private static String hostResource() {
        return NativeLibrary.resourcePath(hostPlatform(), hostFileName());
    }

    /** The staged native for this platform, or {@code null} if it was not staged. */
    private static byte[] stagedNative() throws IOException {
        try (InputStream stream =
                NativeLibraryTest.class.getClassLoader().getResourceAsStream(hostResource())) {
            return stream == null ? null : stream.readAllBytes();
        }
    }

    // ---- platform identity -------------------------------------------------

    @Test
    void platformIdentifiersCoverEveryBundledTarget() {
        assertEquals("darwin-aarch64", NativeLibrary.platform("Mac OS X", "aarch64"));
        assertEquals("darwin-aarch64", NativeLibrary.platform("Mac OS X", "arm64"));
        assertEquals("linux-x86_64", NativeLibrary.platform("Linux", "amd64"));
        assertEquals("linux-x86_64", NativeLibrary.platform("Linux", "x86_64"));
        assertEquals("linux-aarch64", NativeLibrary.platform("Linux", "aarch64"));
        assertEquals("windows-x86_64", NativeLibrary.platform("Windows 11", "amd64"));

        // Every identifier this mapping can produce for a shipped target must
        // be one the packaging side actually stages, or the JAR carries
        // natives under names nothing looks for.
        for (String supported : List.of(
                NativeLibrary.platform("Mac OS X", "aarch64"),
                NativeLibrary.platform("Linux", "amd64"),
                NativeLibrary.platform("Linux", "aarch64"),
                NativeLibrary.platform("Windows 11", "amd64"))) {
            assertTrue(
                    NativeLibrary.BUNDLED_PLATFORMS.contains(supported),
                    supported + " is produced by the mapping but is not in BUNDLED_PLATFORMS");
        }

        // Intel macOS is a named gap, not an oversight: it must still produce a
        // clean identifier so the error message can name it.
        assertEquals("darwin-x86_64", NativeLibrary.platform("Mac OS X", "x86_64"));
        assertTrue(!NativeLibrary.BUNDLED_PLATFORMS.contains("darwin-x86_64"));
    }

    @Test
    void libraryFileNameFollowsThePlatformConvention() {
        assertEquals("libkglite_c.dylib", NativeLibrary.libraryFileName("Mac OS X"));
        assertEquals("libkglite_c.so", NativeLibrary.libraryFileName("Linux"));
        assertEquals("kglite_c.dll", NativeLibrary.libraryFileName("Windows Server 2022"));
        assertEquals(
                "natives/linux-aarch64/libkglite_c.so",
                NativeLibrary.resourcePath("linux-aarch64", "libkglite_c.so"));
    }

    // ---- tier 3: the bundled resource --------------------------------------

    @Test
    void extractedResourceIsByteIdenticalAndLoadable(@TempDir Path cache) throws IOException {
        byte[] expected = stagedNative();
        assumeTrue(expected != null, "no native staged at /" + hostResource());

        Path extracted = NativeLibrary.fromClasspath(
                getClass().getClassLoader(), hostResource(), hostPlatform(), hostFileName(), cache);

        assertTrue(extracted != null && Files.isRegularFile(extracted));
        assertEquals(hostFileName(), extracted.getFileName().toString());
        assertTrue(extracted.startsWith(cache.resolve(hostPlatform())));
        org.junit.jupiter.api.Assertions.assertArrayEquals(expected, Files.readAllBytes(extracted));

        // The whole point of the tier: the linker can open what we extracted.
        try (Arena arena = Arena.ofConfined()) {
            SymbolLookup lookup = openLibrary(extracted, arena);
            assertTrue(
                    lookup.find("kglite_abi_version").isPresent(),
                    "extracted library does not export kglite_abi_version");
        }
    }

    @SuppressWarnings("restricted") // proving the extracted file is loadable is the assertion
    private static SymbolLookup openLibrary(Path library, Arena arena) {
        return SymbolLookup.libraryLookup(library, arena);
    }

    @Test
    void extractionIsContentAddressedAndHappensOnce(@TempDir Path cache) throws IOException {
        assumeTrue(stagedNative() != null, "no native staged at /" + hostResource());

        Path first = NativeLibrary.fromClasspath(
                getClass().getClassLoader(), hostResource(), hostPlatform(), hostFileName(), cache);
        // Backdated far enough that a second write could not produce the same
        // stamp by coincidence on a coarse-grained filesystem.
        FileTime stamp = FileTime.fromMillis(System.currentTimeMillis() - 600_000);
        Files.setLastModifiedTime(first, stamp);

        Path second = NativeLibrary.fromClasspath(
                getClass().getClassLoader(), hostResource(), hostPlatform(), hostFileName(), cache);

        assertEquals(first, second);
        assertEquals(
                stamp.toMillis(),
                Files.getLastModifiedTime(second).toMillis(),
                "the second resolve rewrote the library instead of reusing the cached copy");

        // Extract-once also means no debris: a leftover .tmp would be loaded by
        // nothing but would grow without bound.
        try (Stream<Path> entries = Files.walk(cache)) {
            List<Path> temps = entries.filter(p -> p.getFileName().toString().endsWith(".tmp")).toList();
            assertEquals(List.of(), temps, "extraction left temporary files behind");
        }
    }

    @Test
    void differentContentCachesUnderADifferentDigest(@TempDir Path cache) {
        ClassLoader oldBuild = loaderFor("natives/test-platform/lib.so", new byte[] {1, 2, 3, 4});
        ClassLoader newBuild = loaderFor("natives/test-platform/lib.so", new byte[] {1, 2, 3, 5});

        Path a = NativeLibrary.fromClasspath(
                oldBuild, "natives/test-platform/lib.so", "test-platform", "lib.so", cache);
        Path b = NativeLibrary.fromClasspath(
                newBuild, "natives/test-platform/lib.so", "test-platform", "lib.so", cache);

        // Same file name, different digest directory: an upgrade in a
        // long-lived cache must not be able to overwrite the copy a running JVM
        // already mapped.
        assertNotEquals(a, b);
        assertEquals(a.getFileName(), b.getFileName());
        assertTrue(Files.isRegularFile(a) && Files.isRegularFile(b));
    }

    @Test
    void anAbsentResourceFallsThroughButAnEmptyOneIsAPackagingFailure(@TempDir Path cache) {
        assertNull(
                NativeLibrary.fromClasspath(
                        loaderFor("natives/other/lib.so", new byte[] {1}),
                        "natives/test-platform/lib.so",
                        "test-platform",
                        "lib.so",
                        cache),
                "an unbundled platform must fall through to the error message, not throw here");

        KgliteException empty = assertThrows(
                KgliteException.class,
                () -> NativeLibrary.fromClasspath(
                        loaderFor("natives/test-platform/lib.so", new byte[0]),
                        "natives/test-platform/lib.so",
                        "test-platform",
                        "lib.so",
                        cache));
        assertTrue(
                empty.getMessage().contains("unpopulated natives/ staging directory"),
                empty.getMessage());
    }

    // ---- the tier chain ----------------------------------------------------

    @Test
    void explicitPathWinsAndNeverFallsThrough(@TempDir Path root) throws IOException {
        Path libDir = Files.createDirectories(root.resolve("elsewhere"));
        Path library = Files.write(libDir.resolve("libkglite_c.so"), new byte[] {9});

        // Directory form and file form both resolve to the same file.
        for (String configured : List.of(libDir.toString(), library.toString())) {
            assertEquals(
                    library.toAbsolutePath(),
                    NativeLibrary.locate(
                            configured, root, emptyLoader(), root.resolve("cache"), "Linux", "amd64"));
        }

        // A set-but-wrong override is a configuration error, not a cue to go
        // load some other library: silently resolving elsewhere is how you
        // spend an afternoon debugging the wrong binary.
        KgliteException wrong = assertThrows(
                KgliteException.class,
                () -> NativeLibrary.locate(
                        root.resolve("nope").toString(),
                        root,
                        emptyLoader(),
                        root.resolve("cache"),
                        "Linux",
                        "amd64"));
        assertTrue(wrong.getMessage().contains(NativeLibrary.PATH_PROPERTY), wrong.getMessage());
    }

    @Test
    void workspaceTierPicksTheNewestProfileAndOutranksTheBundledResource(@TempDir Path root)
            throws IOException {
        Path release = Files.createDirectories(root.resolve("target/release")).resolve("libkglite_c.so");
        Path debug = Files.createDirectories(root.resolve("target/debug")).resolve("libkglite_c.so");
        Files.write(release, new byte[] {1});
        Files.write(debug, new byte[] {2});
        Files.setLastModifiedTime(release, FileTime.fromMillis(1_000_000));
        Files.setLastModifiedTime(debug, FileTime.fromMillis(2_000_000));

        // Started from a nested directory: the walk has to climb to find it.
        Path nested = Files.createDirectories(root.resolve("a/b/c"));
        ClassLoader bundled = loaderFor("natives/linux-x86_64/libkglite_c.so", new byte[] {3});

        assertEquals(
                debug,
                NativeLibrary.locate(null, nested, bundled, root.resolve("cache"), "Linux", "amd64"),
                "a fresher debug build must outrank a stale release build");

        Files.setLastModifiedTime(release, FileTime.fromMillis(3_000_000));
        assertEquals(
                release,
                NativeLibrary.locate(null, nested, bundled, root.resolve("cache"), "Linux", "amd64"));
    }

    @Test
    void bundledResourceIsUsedWhenNoWorkspaceIsAround(@TempDir Path root) throws IOException {
        Path start = Files.createDirectories(root.resolve("no-workspace-here"));
        Path cache = root.resolve("cache");
        ClassLoader bundled = loaderFor("natives/linux-aarch64/libkglite_c.so", new byte[] {7, 7, 7});

        Path resolved = NativeLibrary.locate(null, start, bundled, cache, "Linux", "aarch64");

        assertTrue(resolved.startsWith(cache.resolve("linux-aarch64")), resolved.toString());
        org.junit.jupiter.api.Assertions.assertArrayEquals(
                new byte[] {7, 7, 7}, Files.readAllBytes(resolved));
    }

    @Test
    void aMissingNativeNamesEveryLocationItLookedIn(@TempDir Path root) throws IOException {
        Path start = Files.createDirectories(root.resolve("app"));
        Path cache = root.resolve("cache");

        KgliteException failure = assertThrows(
                KgliteException.class,
                () -> NativeLibrary.locate(null, start, emptyLoader(), cache, "Mac OS X", "x86_64"));
        String message = failure.getMessage();

        // The unbundled platform, all three tiers, and the way out. A message
        // that says only "library not found" sends the reader to the wrong
        // place — usually to LD_LIBRARY_PATH, which this loader never consults.
        assertTrue(message.contains("darwin-x86_64"), message);
        assertTrue(message.contains(NativeLibrary.PATH_PROPERTY), message);
        assertTrue(message.contains("target/{release,debug}"), message);
        assertTrue(message.contains(start.toString()), message);
        assertTrue(message.contains("natives/darwin-x86_64/libkglite_c.dylib"), message);
        assertTrue(message.contains("cargo build -p kglite-c --release"), message);
        for (String platform : NativeLibrary.BUNDLED_PLATFORMS) {
            assertTrue(message.contains(platform), platform + " missing from:\n" + message);
        }
    }

    @Test
    void theCacheRootIsUnderTheUsersOwnDirectory() {
        Path root = NativeLibrary.defaultCacheRoot();
        assertEquals("natives", root.getFileName().toString());
        assertEquals("kglite", root.getParent().getFileName().toString());
        // Never the shared temp directory: it is world-writable on a multi-user
        // host and swept by reaper daemons that would delete a library out from
        // under a running JVM.
        assertTrue(
                !root.startsWith(Path.of(System.getProperty("java.io.tmpdir"))),
                "extraction cache must not live in java.io.tmpdir: " + root);
    }

    // ---- helpers -----------------------------------------------------------

    /** A classloader exposing exactly one resource, so tier 3 can be driven in isolation. */
    private static ClassLoader loaderFor(String resource, byte[] content) {
        return new ClassLoader(null) {
            @Override
            public InputStream getResourceAsStream(String name) {
                return name.equals(resource) ? new ByteArrayInputStream(content) : null;
            }
        };
    }

    private static ClassLoader emptyLoader() {
        return loaderFor("nothing-at-all", new byte[0]);
    }
}
