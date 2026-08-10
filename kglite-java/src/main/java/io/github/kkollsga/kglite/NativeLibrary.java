package io.github.kkollsga.kglite;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Where {@code libkglite_c} comes from, in three tiers.
 *
 * <p>Separate from {@link Abi} because it is the one part of the binding that
 * has to be exercised <em>without</em> loading anything: {@code Abi} resolves
 * and links in its static initializer, so a resolution bug there is only
 * observable as a class-initialization failure. Everything here takes its
 * inputs as parameters and returns a {@link Path}, which is what lets the tests
 * drive each tier — including the two failure shapes — directly.
 *
 * <h2>The tiers, in order</h2>
 *
 * <ol>
 *   <li><b>Explicit override</b> — {@code -Dkglite.native.path=<file-or-dir>}.
 *       The escape hatch for a platform we do not bundle and for anyone testing
 *       a locally built engine against a released JAR. Never falls through: a
 *       path that was set and does not resolve is a configuration error, and
 *       silently loading some other library instead is how you debug the wrong
 *       binary for an afternoon.</li>
 *   <li><b>Workspace build</b> — the newest of {@code target/{release,debug}}
 *       walking up from the working directory. This is the repo's own dev loop;
 *       it stays ahead of the bundled resource on purpose, so an engine change
 *       is picked up by {@code gradle test} without re-staging anything.</li>
 *   <li><b>Bundled resource</b> — {@code /natives/<platform>/<file>} on the
 *       classpath, extracted to a per-user cache and loaded from there (the
 *       sqlite-jdbc pattern). This is the only tier a consumer of the published
 *       JAR ever reaches, and the only one that has to work with no environment
 *       set up at all.</li>
 * </ol>
 *
 * <p>Extraction is content-addressed: the cache path contains a digest of the
 * resource bytes, so two kglite versions on one machine cannot collide, an
 * interrupted extraction cannot be mistaken for a complete one, and the common
 * case after the first run is a single {@code isRegularFile} check.
 */
final class NativeLibrary {

    private NativeLibrary() {}

    /** System property naming an explicit library file or directory. */
    static final String PATH_PROPERTY = "kglite.native.path";

    /** System property overriding the extraction cache root. */
    static final String CACHE_PROPERTY = "kglite.native.cache";

    /** Classpath directory holding the bundled per-platform natives. */
    static final String RESOURCE_ROOT = "natives";

    /**
     * The platforms the published JAR carries a native for.
     *
     * <p>{@code darwin-x86_64} is deliberately absent — Intel macOS is the one
     * platform with no runner in {@code publish_java.yml}, and shipping an
     * unexercised cross-build is worse than a clear error naming the override.
     * Intel-Mac users build the engine once
     * ({@code cargo build -p kglite-c --release}) and pass
     * {@code -Dkglite.native.path}. Same posture as the wheel matrix's
     * documented gaps: an absent platform is a named gap, never a broken load.
     */
    static final Set<String> BUNDLED_PLATFORMS =
            Set.of("darwin-aarch64", "linux-aarch64", "linux-x86_64", "windows-x86_64");

    /** The two cargo profiles tier 2 chooses between, newest wins. */
    private static final String[] PROFILES = {"release", "debug"};

    // ---- platform identity -------------------------------------------------

    /**
     * The {@code <os>-<arch>} directory name for the running JVM.
     *
     * <p>Returns the identifier even when it is not in {@link
     * #BUNDLED_PLATFORMS}: an unbundled platform has to reach the error message
     * that names it, not a different, earlier error about being unrecognised.
     *
     * @param osName   the {@code os.name} property
     * @param osArch   the {@code os.arch} property
     * @return the platform identifier, e.g. {@code "linux-aarch64"}
     */
    static String platform(String osName, String osArch) {
        return operatingSystem(osName) + "-" + architecture(osArch);
    }

    private static String operatingSystem(String osName) {
        String os = osName == null ? "" : osName.toLowerCase(Locale.ROOT);
        if (os.contains("win")) {
            return "windows";
        }
        if (os.contains("mac") || os.contains("darwin")) {
            return "darwin";
        }
        if (os.contains("linux")) {
            return "linux";
        }
        // Kept verbatim rather than mapped to a guess: the resource lookup then
        // misses and the error names exactly what the JVM reported, which is
        // the only useful thing to put in a bug report from a platform we have
        // never seen.
        return os.replaceAll("[^a-z0-9]+", "");
    }

    private static String architecture(String osArch) {
        String arch = osArch == null ? "" : osArch.toLowerCase(Locale.ROOT);
        if (arch.equals("aarch64") || arch.equals("arm64")) {
            return "aarch64";
        }
        if (arch.equals("x86_64") || arch.equals("amd64")) {
            return "x86_64";
        }
        return arch.replaceAll("[^a-z0-9_]+", "");
    }

    /**
     * The platform's file name for the {@code kglite_c} shared library.
     *
     * <p>Derived from {@code os.name} rather than {@link System#mapLibraryName}
     * so the bundled-resource path for a platform can be computed on any host —
     * the packaging side has to name all four, and only one of them is ever the
     * running one.
     *
     * @param osName the {@code os.name} property
     * @return {@code kglite_c.dll}, {@code libkglite_c.dylib} or {@code libkglite_c.so}
     */
    static String libraryFileName(String osName) {
        return switch (operatingSystem(osName)) {
            case "windows" -> "kglite_c.dll";
            case "darwin" -> "libkglite_c.dylib";
            default -> "libkglite_c.so";
        };
    }

    /** The classpath resource path for one platform's native. */
    static String resourcePath(String platform, String fileName) {
        return RESOURCE_ROOT + "/" + platform + "/" + fileName;
    }

    // ---- resolution --------------------------------------------------------

    /** Resolve the native library for the running JVM. */
    static Path locate() {
        return locate(
                System.getProperty(PATH_PROPERTY),
                Path.of(System.getProperty("user.dir", ".")).toAbsolutePath(),
                NativeLibrary.class.getClassLoader(),
                defaultCacheRoot(),
                System.getProperty("os.name", ""),
                System.getProperty("os.arch", ""));
    }

    /**
     * The three-tier resolution, with every environment input passed in.
     *
     * @param configured   value of {@link #PATH_PROPERTY}, or {@code null}
     * @param searchStart  directory the workspace walk starts from
     * @param loader       classloader the bundled resource is read through
     * @param cacheRoot    directory extracted natives are cached under
     * @param osName       the {@code os.name} property
     * @param osArch       the {@code os.arch} property
     * @return the resolved library file
     * @throws KgliteException if no tier produced one, naming every location tried
     */
    static Path locate(
            String configured,
            Path searchStart,
            ClassLoader loader,
            Path cacheRoot,
            String osName,
            String osArch) {
        String fileName = libraryFileName(osName);
        String platform = platform(osName, osArch);
        List<String> tried = new ArrayList<>();

        if (configured != null && !configured.isBlank()) {
            // Terminal on purpose — see the class doc.
            return fromExplicitPath(configured, fileName);
        }
        tried.add("-D" + PATH_PROPERTY + " (not set)");

        Path workspace = fromWorkspace(searchStart, fileName);
        if (workspace != null) {
            return workspace;
        }
        tried.add("target/{release,debug}/" + fileName + " walking up from " + searchStart);

        String resource = resourcePath(platform, fileName);
        Path extracted = fromClasspath(loader, resource, platform, fileName, cacheRoot);
        if (extracted != null) {
            return extracted;
        }
        tried.add("classpath resource /" + resource + " (extracted to " + cacheRoot + ")");

        throw new KgliteException(missingLibraryMessage(platform, tried));
    }

    private static String missingLibraryMessage(String platform, List<String> tried) {
        StringBuilder message = new StringBuilder("no kglite native library found for platform ")
                .append(platform)
                .append(". Looked in:");
        for (String location : tried) {
            message.append("\n  - ").append(location);
        }
        message.append("\nThe published JAR bundles ")
                .append(String.join(", ", BUNDLED_PLATFORMS.stream().sorted().toList()))
                .append(". On any other platform, build the engine once with"
                        + " `cargo build -p kglite-c --release` and start the JVM with"
                        + " -D" + PATH_PROPERTY + "=<dir-or-file>.");
        return message.toString();
    }

    /** Tier 1 — an operator-supplied file or directory. */
    private static Path fromExplicitPath(String configured, String fileName) {
        Path candidate = Path.of(configured);
        if (Files.isDirectory(candidate)) {
            candidate = candidate.resolve(fileName);
        }
        if (!Files.isRegularFile(candidate)) {
            throw new KgliteException(
                    PATH_PROPERTY + " does not resolve to " + fileName + ": " + candidate);
        }
        return candidate.toAbsolutePath();
    }

    /**
     * Tier 2 — the newest of {@code target/{release,debug}} in an enclosing
     * checkout, or {@code null}.
     *
     * <p>Newest, never a fixed profile preference: a stale release library left
     * over from a benchmark otherwise shadows the debug library the current
     * source just produced, and the ABI contract test then reports drift
     * against code that does exist. Same rule as
     * {@code tests/conftest.py::workspace_binary}.
     */
    private static Path fromWorkspace(Path searchStart, String fileName) {
        for (Path dir = searchStart; dir != null; dir = dir.getParent()) {
            Path newest = null;
            for (String profile : PROFILES) {
                Path candidate = dir.resolve("target").resolve(profile).resolve(fileName);
                if (Files.isRegularFile(candidate) && (newest == null || newer(candidate, newest))) {
                    newest = candidate;
                }
            }
            if (newest != null) {
                return newest;
            }
        }
        return null;
    }

    /** Whether {@code candidate} was modified strictly later than {@code other}. */
    private static boolean newer(Path candidate, Path other) {
        try {
            return Files.getLastModifiedTime(candidate).compareTo(Files.getLastModifiedTime(other))
                    > 0;
        } catch (IOException e) {
            return false;
        }
    }

    // ---- tier 3: extract from the classpath --------------------------------

    /**
     * Tier 3 — extract {@code resource} into {@code cacheRoot} and return the
     * extracted file, or {@code null} when the classpath carries no such
     * resource.
     *
     * @param loader   classloader to read the resource through
     * @param resource resource path, without a leading slash
     * @param platform platform identifier, used as a cache subdirectory
     * @param fileName file name the extracted library must keep
     * @param cacheRoot directory to extract under
     * @return the extracted file, or {@code null} if the resource is absent
     */
    static Path fromClasspath(
            ClassLoader loader, String resource, String platform, String fileName, Path cacheRoot) {
        byte[] bytes;
        try (InputStream stream = openResource(loader, resource)) {
            if (stream == null) {
                return null;
            }
            bytes = stream.readAllBytes();
        } catch (IOException e) {
            throw new KgliteException("could not read bundled native " + resource, e);
        }
        if (bytes.length == 0) {
            // An empty resource is a packaging failure, not an absent platform:
            // falling through would report "no native found" and send the
            // reader looking for a missing file that is right there.
            throw new KgliteException("bundled native " + resource + " is empty — the JAR was "
                    + "assembled from an unpopulated natives/ staging directory");
        }
        return cache(bytes, cacheRoot.resolve(platform).resolve(digest(bytes)), fileName, resource);
    }

    private static InputStream openResource(ClassLoader loader, String resource) {
        return loader == null
                ? ClassLoader.getSystemResourceAsStream(resource)
                : loader.getResourceAsStream(resource);
    }

    /**
     * Write {@code bytes} to {@code directory/fileName} unless an extraction of
     * the same content is already there.
     *
     * <p>{@code directory} is content-addressed by the caller, so "already
     * there with the right length" is proof of identical bytes and the fast
     * path is a stat. The write goes to a sibling temp file and is moved into
     * place, so a JVM killed mid-extraction leaves a {@code .tmp} behind rather
     * than a truncated library that every later run would happily load.
     */
    private static Path cache(byte[] bytes, Path directory, String fileName, String resource) {
        Path target = directory.resolve(fileName);
        try {
            if (Files.isRegularFile(target) && Files.size(target) == bytes.length) {
                return target;
            }
            Files.createDirectories(directory);
            Path staged = Files.createTempFile(directory, fileName, ".tmp");
            try {
                Files.write(staged, bytes);
                makeExecutable(staged);
                try {
                    Files.move(staged, target, StandardCopyOption.ATOMIC_MOVE);
                } catch (FileAlreadyExistsException e) {
                    // Another JVM won the race. Its file has the same digest,
                    // so it is byte-identical; replacing it could pull the
                    // library out from under a load in progress on Windows.
                    Files.deleteIfExists(staged);
                }
            } catch (IOException e) {
                Files.deleteIfExists(staged);
                throw e;
            }
            return target;
        } catch (IOException e) {
            throw new KgliteException(
                    "could not extract bundled native " + resource + " to " + target, e);
        }
    }

    private static void makeExecutable(Path file) throws IOException {
        // POSIX only; on Windows the loader does not consult a mode bit and
        // setPosixFilePermissions would throw UnsupportedOperationException.
        if (file.getFileSystem().supportedFileAttributeViews().contains("posix")) {
            Files.setPosixFilePermissions(
                    file, java.nio.file.attribute.PosixFilePermissions.fromString("rwxr-xr-x"));
        }
    }

    /** First 16 hex characters of the content's SHA-256. */
    private static String digest(byte[] bytes) {
        MessageDigest sha256;
        try {
            sha256 = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new KgliteException("SHA-256 is unavailable in this JVM", e);
        }
        byte[] hash = sha256.digest(bytes);
        StringBuilder hex = new StringBuilder(16);
        for (int i = 0; i < 8; i++) {
            hex.append(Character.forDigit((hash[i] >> 4) & 0xF, 16));
            hex.append(Character.forDigit(hash[i] & 0xF, 16));
        }
        return hex.toString();
    }

    // ---- cache location ----------------------------------------------------

    /**
     * Where extracted natives live: the platform's per-user cache directory,
     * {@code kglite/natives} beneath it.
     *
     * <p>Not {@code java.io.tmpdir}: on a shared host that is a world-writable
     * directory another user can pre-populate, and on every host it is subject
     * to reaper daemons that would delete a library out from under a running
     * JVM. {@code -Dkglite.native.cache} overrides it for images with a
     * read-only home.
     *
     * @return the cache root, which may not exist yet
     */
    static Path defaultCacheRoot() {
        String override = System.getProperty(CACHE_PROPERTY);
        if (override != null && !override.isBlank()) {
            return Path.of(override);
        }
        String os = operatingSystem(System.getProperty("os.name", ""));
        String home = System.getProperty("user.home", ".");
        Path base =
                switch (os) {
                    case "windows" -> environmentDirectory("LOCALAPPDATA", Path.of(home, "AppData", "Local"));
                    case "darwin" -> Path.of(home, "Library", "Caches");
                    default -> environmentDirectory("XDG_CACHE_HOME", Path.of(home, ".cache"));
                };
        return base.resolve("kglite").resolve("natives");
    }

    private static Path environmentDirectory(String variable, Path fallback) {
        String value = System.getenv(variable);
        return value == null || value.isBlank() ? fallback : Path.of(value);
    }
}
