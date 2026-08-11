plugins {
    `java-library`
    // Sonatype's Central Portal is the only route for a new namespace (OSSRH
    // stopped accepting them), and this plugin is the one that speaks it end to
    // end: bundle assembly, the Portal upload API, in-memory PGP signing, and
    // the POM completeness Central rejects a deployment for. Sonatype's own
    // `org.sonatype.central` plugin covers the upload but leaves signing and
    // the POM to be wired by hand — two more places for a release to fail after
    // the artifacts are already built.
    id("com.vanniktech.maven.publish") version "0.34.0"
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
    // No withSourcesJar()/withJavadocJar() here. Central requires both, and the
    // publishing plugin builds both (`sourcesJar`, `plainJavadocJar`) — but the
    // JDK's javadoc jar and the plugin's are two tasks writing the same
    // `kglite-<version>-javadoc.jar`, which Gradle 9 rejects outright as an
    // undeclared dependency between them. The plugin's pair is the one that
    // ends up in the POM, so it is the one that exists; `assemble` is wired to
    // build them below so a plain `build` still proves they work.
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
// exist. `NativeLibrary` picks the most recently built of the two — the
// same rule as tests/conftest.py::workspace_binary — and
// `-Dkglite.native.path=<dir-or-file>` still overrides it outright.
//
// The published JAR bundles per-platform natives as resources instead; see the
// packaging section below.
// ---------------------------------------------------------------------------
val workspaceRoot = layout.projectDirectory.dir("..").asFile

/** The header `AbiContractTest` validates, and the two profiles `NativeLibrary` picks between. */
val abiHeaderFile = File(workspaceRoot, "crates/kglite-c/include/kglite.h")
val nativeLibraryCandidates = listOf("release", "debug").map {
    File(workspaceRoot, "target/$it/${System.mapLibraryName("kglite_c")}")
}

// ---------------------------------------------------------------------------
// Packaging: natives as JAR resources (the sqlite-jdbc pattern).
//
// `NativeLibrary` reads `/natives/<platform>/<libfile>` off the classpath,
// extracts it to a content-addressed per-user cache and loads it from there. So
// the packaging job is exactly: get one file per platform under those names.
//
// The staging directory `kglite-java/natives/` is the single input. CI's
// `publish_java.yml` populates all four platform subdirectories from the
// matrix build artifacts; it is gitignored, and on a developer machine it is
// normally empty. Rather than making a local `gradle build` produce a JAR that
// cannot load on the machine that built it, the host's own cargo-built library
// is staged as a fallback when the staging directory does not already carry
// one for this platform.
//
// A partial set WARNS. Failing would make the ordinary local build red for a
// reason that has nothing to do with the change being tested, and every
// developer would learn to ignore it. The place where an absent platform is
// genuinely fatal is the release, so the publish job passes
// `-PrequireAllNatives=true` and gets a hard failure there instead — the
// difference between a warning nobody reads and a gate that can go red.
// ---------------------------------------------------------------------------

/** Platform identifiers, matching `NativeLibrary.BUNDLED_PLATFORMS` exactly. */
val bundledPlatforms = listOf("darwin-aarch64", "linux-aarch64", "linux-x86_64", "windows-x86_64")

/** `<os>-<arch>` for the machine running this build, matching `NativeLibrary.platform`. */
val hostPlatform: String = run {
    val os = System.getProperty("os.name", "").lowercase()
    val arch = System.getProperty("os.arch", "").lowercase()
    val osPart = when {
        os.contains("win") -> "windows"
        os.contains("mac") || os.contains("darwin") -> "darwin"
        os.contains("linux") -> "linux"
        else -> os.replace(Regex("[^a-z0-9]+"), "")
    }
    val archPart = when (arch) {
        "aarch64", "arm64" -> "aarch64"
        "x86_64", "amd64" -> "x86_64"
        else -> arch.replace(Regex("[^a-z0-9_]+"), "")
    }
    "$osPart-$archPart"
}

val nativesStagingDir = layout.projectDirectory.dir("natives")
val hostStagedNative = nativesStagingDir.file("$hostPlatform/${System.mapLibraryName("kglite_c")}").asFile
val workspaceHostNative = nativeLibraryCandidates.filter { it.isFile }.maxByOrNull { it.lastModified() }

/**
 * Assembles the resource tree the JAR and the test classpath both consume:
 * `<build>/natives-resources/natives/<platform>/<libfile>`.
 *
 * Producing the `natives/` prefix here rather than at each consumer is what
 * lets the tests see the same resource path a published JAR exposes — the
 * extraction tier is then exercised by the real layout, not a rehearsal of it.
 */
val stageNatives = tasks.register<Sync>("stageNatives") {
    group = "build"
    description = "Stages per-platform kglite natives as JAR resources."
    into(layout.buildDirectory.dir("natives-resources"))
    into("natives") {
        from(nativesStagingDir)
        if (workspaceHostNative != null && !hostStagedNative.isFile) {
            from(workspaceHostNative) { into(hostPlatform) }
        }
    }
    // Without this the coverage check below is decorative: the task goes
    // UP-TO-DATE from the previous run and its `doLast` never executes, so
    // `-PrequireAllNatives=true` silently passes on an incomplete staging
    // directory — the exact "green that cannot go red" shape CLAUDE.md names.
    inputs.property("requireAllNatives", providers.gradleProperty("requireAllNatives").orElse("false"))
    doLast {
        val staged = destinationDir.resolve("natives")
        val present = bundledPlatforms.filter { staged.resolve(it).listFiles()?.isNotEmpty() == true }
        val missing = bundledPlatforms - present.toSet()
        if (missing.isEmpty()) {
            logger.lifecycle("natives: all ${bundledPlatforms.size} platforms staged")
        } else if (providers.gradleProperty("requireAllNatives").orNull == "true") {
            error(
                "natives staging is incomplete: missing ${missing.joinToString(", ")}. " +
                    "Present: ${present.joinToString(", ").ifEmpty { "none" }}. " +
                    "A JAR published without a platform's native fails at load time for those " +
                    "users with no signal anywhere in the release."
            )
        } else {
            logger.warn(
                "natives: staged ${present.joinToString(", ").ifEmpty { "none" }}; " +
                    "MISSING ${missing.joinToString(", ")}. This JAR only loads on the staged " +
                    "platforms. CI stages all ${bundledPlatforms.size} from the matrix build."
            )
        }
    }
}

tasks.jar {
    from(stageNatives)
    manifest {
        // An automatic module name, not a module-info.java. Without it the JPMS
        // name is derived from the *file name*, so `kglite-0.15.9.jar` becomes
        // the module `kglite` and a consumer's `requires kglite` silently binds
        // to whatever jar happens to be named that — and the derived name
        // changes if the artifact is ever renamed. Declaring it pins the name
        // to the package namespace and costs one manifest line.
        //
        // Deliberately not a real module descriptor: `module-info.java` would
        // raise the compile floor's tooling requirements for no gain here (the
        // wrapper has no dependencies to encapsulate), and it would have to
        // declare the native-access requirements that JEP 472 already handles
        // through the command line.
        attributes("Automatic-Module-Name" to "io.github.kkollsga.kglite")
    }
    // The repo's licence travels inside the artifact, not only in the POM: the
    // JAR bundles compiled Rust, and a redistributor unpacking it has to find
    // the terms without going back to a URL.
    from(File(workspaceRoot, "LICENSE")) { into("META-INF") }
}

// The test classpath carries the same tree, so `NativeLibraryTest` extracts
// from a real `/natives/<platform>/…` resource. Wired as a plain srcDir plus an
// explicit dependency rather than `srcDir(taskProvider)`, which does not carry
// the task dependency for a SourceDirectorySet.
sourceSets.test {
    resources.srcDir(layout.buildDirectory.dir("natives-resources"))
}
tasks.processTestResources {
    dependsOn(stageNatives)
    // The staged library and a same-named file from src/test/resources would be
    // a packaging bug, not something to silently pick a winner for.
    duplicatesStrategy = DuplicatesStrategy.FAIL
}

tasks.test {
    useJUnitPlatform()
    // The forked test worker loads the native engine through FFM; its default
    // heap is sized from the runner's cgroup, which GitHub reports too low, so
    // the first CI run OOM'd here. Bound it explicitly — adequate everywhere,
    // and 2 GB is comfortable on the 7 GB Linux runners alongside the daemon.
    maxHeapSize = "2g"
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
    // ...and -Werror, because doclint alone only *prints*. Verified 2026-08-10
    // by deleting an `@param`: javadoc reported "warning: no @param for value"
    // and the build still went SUCCESSFUL, so the whole Tier-2 documentation
    // requirement was a message in a log nobody reads. Warnings here are
    // exactly the defect the check exists to catch — an undocumented parameter
    // reaching a published artifact — so they end the build, the same posture
    // as `-Werror` on JavaCompile above.
    (options as CoreJavadocOptions).addBooleanOption("Werror", true)
}

// ---------------------------------------------------------------------------
// Publication — Maven Central via the Sonatype Central Portal.
//
// Explicit task only. `publishToMavenCentral` is never reachable from `build`
// or `check`: an accidental deployment is not undoable (Central never deletes a
// published version, the same rule this repo already carries for PyPI and
// crates.io), so the only way to reach it is to name it.
//
// Credentials and key both arrive as environment variables, never files —
// `ORG_GRADLE_PROJECT_<name>` is Gradle's own env→property mapping, so nothing
// here reads a keyring, a `gradle.properties` in `$HOME`, or a decrypted file
// on disk:
//
//   ORG_GRADLE_PROJECT_mavenCentralUsername   <- CENTRAL_TOKEN_USERNAME
//   ORG_GRADLE_PROJECT_mavenCentralPassword   <- CENTRAL_TOKEN_PASSWORD
//   ORG_GRADLE_PROJECT_signingInMemoryKey     <- GPG_SIGNING_KEY (ASCII-armoured)
//   ORG_GRADLE_PROJECT_signingInMemoryKeyPassword <- GPG_PASSPHRASE
// ---------------------------------------------------------------------------

/**
 * Signing is conditional on a key being present, and deliberately not on a CI
 * flag: a local `publishToMavenLocal` must not fail for want of a key, and a
 * release must not be able to reach Central unsigned because a flag was
 * mis-set. `verifyPublishable` below is what makes the release case loud.
 */
val hasSigningKey = providers.gradleProperty("signingInMemoryKey").isPresent

mavenPublishing {
    publishToMavenCentral()
    if (hasSigningKey) {
        signAllPublications()
    }
    coordinates(group.toString(), "kglite", version.toString())
    configure(
        com.vanniktech.maven.publish.JavaLibrary(
            javadocJar = com.vanniktech.maven.publish.JavadocJar.Javadoc(),
            sourcesJar = true,
        ),
    )

    pom {
        name = "kglite"
        description =
            "Embedded graph database for the JVM: Cypher, vector search and a single-file " +
                "graph, in-process over the kglite C ABI. No server, no daemon, no JNI."
        inceptionYear = "2026"
        url = "https://github.com/kkollsga/kglite"
        licenses {
            license {
                // Matches the repo's own LICENSE (MIT). Central rejects a POM
                // with no licence block, and a licence here that disagreed with
                // the file in the JAR would be worse than either alone.
                name = "MIT License"
                url = "https://github.com/kkollsga/kglite/blob/main/LICENSE"
                distribution = "repo"
            }
        }
        developers {
            developer {
                id = "kkollsga"
                name = "Kristian de Figueiredo Kollsgård"
                url = "https://github.com/kkollsga"
            }
        }
        scm {
            url = "https://github.com/kkollsga/kglite"
            connection = "scm:git:https://github.com/kkollsga/kglite.git"
            developerConnection = "scm:git:ssh://git@github.com/kkollsga/kglite.git"
        }
        issueManagement {
            system = "GitHub Issues"
            url = "https://github.com/kkollsga/kglite/issues"
        }
    }
}

/**
 * The preconditions a Central deployment has that `build` does not.
 *
 * Each one fails *before* the upload rather than after: a Portal deployment
 * that is rejected for an unsigned artifact or an incomplete native set still
 * consumed the version's one shot at that coordinate, and the repair is manual.
 * Run it as `gradle -p kglite-java verifyPublishable -PrequireAllNatives=true`.
 */
val verifyPublishable = tasks.register("verifyPublishable") {
    group = "verification"
    description = "Asserts this build is signable and complete enough to deploy to Central."
    dependsOn(stageNatives, tasks.jar, tasks.named("sourcesJar"), tasks.named("plainJavadocJar"))
    val versionForCheck = version.toString()
    val signing = hasSigningKey
    val credentials = providers.gradleProperty("mavenCentralUsername").isPresent &&
        providers.gradleProperty("mavenCentralPassword").isPresent
    doLast {
        check(signing) {
            "no signing key: set ORG_GRADLE_PROJECT_signingInMemoryKey (and " +
                "…KeyPassword). Central rejects unsigned deployments, and it rejects them " +
                "after the artifacts have been uploaded."
        }
        check(credentials) {
            "no Central credentials: set ORG_GRADLE_PROJECT_mavenCentralUsername and " +
                "ORG_GRADLE_PROJECT_mavenCentralPassword from the portal token."
        }
        check(!versionForCheck.endsWith("SNAPSHOT")) {
            "refusing to deploy a SNAPSHOT version ($versionForCheck) to Central releases"
        }
        logger.lifecycle("publishable: kglite $versionForCheck, signed, credentials present")
    }
}

// A plain `build` builds the two jars Central requires, so a javadoc error or a
// broken sources jar surfaces on the branch rather than mid-deployment — where
// the artifacts are already uploaded and the version's one shot is spent.
tasks.assemble {
    dependsOn(tasks.named("sourcesJar"), tasks.named("plainJavadocJar"))
}
