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
// Native library for tests: the repo's own release build.
//
//   cargo build -p kglite-c --release   ->   target/release/libkglite_c.dylib
//
// The packaging phase bundles per-platform natives as JAR resources; until
// then tests point at the workspace build directory explicitly.
// ---------------------------------------------------------------------------
val workspaceRoot = layout.projectDirectory.dir("..").asFile

tasks.test {
    useJUnitPlatform()
    // JDK 24+ (JEP 472) warns on restricted native access without this; a
    // future release makes it an error. Consumers pass the same flag.
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty("kglite.native.path", File(workspaceRoot, "target/release").absolutePath)
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
