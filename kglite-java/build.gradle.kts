plugins {
    `java-library`
}

group = "io.github.kkollsga"

// ---------------------------------------------------------------------------
// Version: read from the Cargo workspace, never written down here.
//
// `[workspace.package] version` in the repo-root Cargo.toml is the single
// source of truth for every published kglite artifact (`make bump-version`
// rewrites it and `make gate` fails on drift). The Java artifact ships in the
// same lockstep, so it reads that value rather than keeping a second copy that
// could disagree. `-PkgliteVersion=X.Y.Z` overrides it at publish time (e.g. a
// release-candidate suffix) without editing anything.
// ---------------------------------------------------------------------------
val workspaceCargoToml = layout.projectDirectory.file("../Cargo.toml").asFile

fun readWorkspaceVersion(): String {
    val text = workspaceCargoToml.readText()
    // Anchored to the line start: the surrounding comments in Cargo.toml
    // mention `[workspace.package]` in prose, and an unanchored match finds
    // those first.
    val table = Regex("^\\[workspace\\.package\\]$([\\s\\S]*?)(?=^\\[|\\z)", RegexOption.MULTILINE)
        .find(text)
        ?: error("no [workspace.package] table in ${workspaceCargoToml.path}")
    return Regex("^\\s*version\\s*=\\s*\"([^\"]+)\"", RegexOption.MULTILINE)
        .find(table.groupValues[1])
        ?.groupValues?.get(1)
        ?: error("no version key under [workspace.package] in ${workspaceCargoToml.path}")
}

version = (findProperty("kgliteVersion") as String?) ?: readWorkspaceVersion()

// ---------------------------------------------------------------------------
// Java floor: 22.
//
// 22 is the release that finalized the Foreign Function & Memory API (JEP 454),
// which is the whole binding mechanism — there is no kglite-java below it. We
// take nothing from 23-26, so raising the floor to 25 (the current LTS) would
// only shut out 22/23/24 deployments while gaining zero capability; 25 LTS is
// already inside the 22+ range. Compiled with the JDK 26 toolchain against
// `--release 22` so the bytecode and the API surface are both 22-clean.
// ---------------------------------------------------------------------------
val javaFloor = 22

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(26)
    }
    withSourcesJar()
    withJavadocJar()
}

tasks.withType<JavaCompile>().configureEach {
    options.release = javaFloor
    options.encoding = "UTF-8"
    options.compilerArgs.addAll(listOf("-Xlint:all", "-Werror"))
}

repositories {
    mavenCentral()
}

dependencies {
    // No runtime dependencies. The wrapper wraps the C ABI chokepoint and
    // parses the JSON the ABI emits with a ~200-line internal reader, so the
    // published artifact has an empty transitive surface.
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}

// ---------------------------------------------------------------------------
// Native library for tests: the repo's own build, newest profile wins.
//
//   cargo build -p kglite-c             ->  target/debug/libkglite_c.dylib
//   cargo build -p kglite-c --release   ->  target/release/libkglite_c.dylib
//
// Deliberately NOT pinned to one profile. Pinning `target/release` meant a
// stale release library left over from a benchmark shadowed the debug library
// the current source had just produced: every symbol the working tree added
// was missing, and the ABI contract test reported drift against code that did
// exist. Abi.resolveLibrary picks the most recently built of the two — the
// same rule as tests/conftest.py::workspace_binary — and
// `-Dkglite.native.path=<dir-or-file>` still overrides it outright.
//
// The packaging phase bundles per-platform natives as JAR resources instead.
// ---------------------------------------------------------------------------
val workspaceRoot = layout.projectDirectory.dir("..").asFile

/** The header `AbiContractTest` validates, and the two profiles `Abi.resolveLibrary` picks between. */
val abiHeaderFile = File(workspaceRoot, "crates/kglite-c/include/kglite.h")
val nativeLibraryCandidates = listOf("release", "debug").map {
    File(workspaceRoot, "target/$it/${System.mapLibraryName("kglite_c")}")
}

tasks.test {
    useJUnitPlatform()
    // ---------------------------------------------------------------------
    // Both real inputs of this suite are produced by cargo, outside the Gradle
    // project tree, and are reached at *runtime* — the header through a system
    // property, the native library through `Abi.resolveLibrary`. Gradle's
    // up-to-date check hashes a system property's *value*, so passing the
    // header as an absolute path string tracked the path and never the
    // content: editing `kglite.h` (or rebuilding `libkglite_c`) left this task
    // `UP-TO-DATE` and the build `SUCCESSFUL` without running the drift
    // detector at all. Verified 2026-08-10 by renaming a declared function in
    // the header — `gradle test` reported UP-TO-DATE and passed.
    //
    // Declaring them as content inputs is what makes the green able to go red.
    // `optional` because a fresh checkout has neither profile built yet, and a
    // missing library must fail in `Abi`'s initializer with its build hint
    // rather than in Gradle's input snapshotter.
    inputs.file(abiHeaderFile)
        .withPropertyName("kgliteAbiHeader")
        .withPathSensitivity(PathSensitivity.NONE)
    inputs.files(nativeLibraryCandidates)
        .withPropertyName("kgliteNativeLibrary")
        .withPathSensitivity(PathSensitivity.NONE)
        .optional()
    // JDK 24+ (JEP 472) warns on restricted native access without this; a
    // future release makes it an error. Consumers pass the same flag.
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    // Gradle does not forward -D to the forked test JVM; relay an explicit
    // override only, and otherwise let the resolver find the newest build.
    providers.systemProperty("kglite.native.path").orNull?.let {
        systemProperty("kglite.native.path", it)
    }
    systemProperty(
        "kglite.header.path",
        providers.systemProperty("kglite.header.path")
            .getOrElse(File(workspaceRoot, "crates/kglite-c/include/kglite.h").absolutePath),
    )
    systemProperty(
        "kglite.contract.path",
        layout.projectDirectory.file("src/test/resources/abi-contract.txt").asFile.absolutePath,
    )
    // Gradle does not forward -D to the forked test JVM; the regeneration
    // switch has to be relayed explicitly.
    systemProperty(
        "kglite.contract.update",
        providers.systemProperty("kglite.contract.update").getOrElse("false"),
    )
    testLogging {
        events("failed", "skipped")
        showStandardStreams = false
    }
}

tasks.javadoc {
    // Full doclint, including the "missing" group — every documented member
    // must carry its @param/@return.
    (options as CoreJavadocOptions).addStringOption("Xdoclint:all", "-quiet")
}
