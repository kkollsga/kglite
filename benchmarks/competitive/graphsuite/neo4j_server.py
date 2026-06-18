"""Ephemeral Neo4j provisioners for the benchmark — two flavors.

The `ad_neo4j` adapter talks to a *running* server over Bolt; it reads the
connection from ``GRAPHSUITE_NEO4J_URI`` / ``GRAPHSUITE_NEO4J_USER`` /
``GRAPHSUITE_NEO4J_PASSWORD``. These provisioners *start* such a server,
export those three env vars, hand the URI back, and tear the server down on
exit — so the benchmark can stand up a clean Neo4j per run without anyone
hand-managing a database.

Two flavors, both reusing the one Bolt adapter:

- ``"docker"`` — run ``neo4j:<tag>`` in a container. Uses
  ``testcontainers[neo4j]`` when installed (the canonical prebuilt
  harness), else a thin ``docker run`` fallback. Portable and isolated;
  on macOS it pays the Docker-VM tax, so it's the *baseline* Neo4j number.
- ``"local"`` — launch the natively-installed Neo4j (``neo4j`` on PATH or
  ``$NEO4J_HOME``) against a throwaway ``NEO4J_CONF`` pointing data/logs at
  a tempdir. No container, no VM — the *higher-performance* Neo4j path, and
  the fair "Neo4j at its best" number.

Each flavor degrades to a clean, explained *unavailable* (Docker daemon
down / no Java / no neo4j CLI) so a no-prereq run stays green and just
skips that column.

There is no JVM/Java *harness* to write here: Neo4j's only embeddable API
is JVM-only, which we deliberately avoid. Both flavors are pure-Python
process management talking to the server over Bolt.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import tempfile
import time

# Credentials the provisioners set up and the adapter then reads. Neo4j 5+/
# 2026.x requires an initial password of at least 8 characters.
_USER = "neo4j"
_PASSWORD = "benchmarkpw"
_DEFAULT_IMAGE = "neo4j:5-community"
_READY_TIMEOUT_S = 120.0


class ProvisionError(RuntimeError):
    """Raised when a server was expected to come up but didn't."""


def provisioner_for(flavor: str):
    if flavor == "docker":
        return DockerNeo4jServer()
    if flavor == "local":
        return LocalNeo4jServer()
    raise ValueError(f"unknown Neo4j flavor: {flavor!r}")


# ── shared helpers ──────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_bolt(uri: str, deadline: float, watch: subprocess.Popen | None = None) -> None:
    """Poll until the driver can verify connectivity, or raise on timeout /
    early child exit."""
    import neo4j

    last_err: Exception | None = None
    while time.perf_counter() < deadline:
        if watch is not None and watch.poll() is not None:
            tail = ""
            if watch.stdout is not None:
                with contextlib.suppress(Exception):
                    tail = watch.stdout.read() or ""
            raise ProvisionError(f"server process exited early (code {watch.returncode})\n{tail[-2000:]}")
        try:
            drv = neo4j.GraphDatabase.driver(uri, auth=(_USER, _PASSWORD))
            drv.verify_connectivity()
            drv.close()
            return
        except Exception as e:  # not up yet
            last_err = e
            time.sleep(1.0)
    raise ProvisionError(f"Neo4j did not become reachable at {uri} within {_READY_TIMEOUT_S:.0f}s: {last_err}")


class _EnvScope:
    """Set GRAPHSUITE_NEO4J_* for the adapter, restoring prior values on exit."""

    _KEYS = ("GRAPHSUITE_NEO4J_URI", "GRAPHSUITE_NEO4J_USER", "GRAPHSUITE_NEO4J_PASSWORD")

    def __init__(self) -> None:
        self._saved: dict[str, str | None] = {}

    def apply(self, uri: str) -> None:
        for k in self._KEYS:
            self._saved[k] = os.environ.get(k)
        os.environ["GRAPHSUITE_NEO4J_URI"] = uri
        os.environ["GRAPHSUITE_NEO4J_USER"] = _USER
        os.environ["GRAPHSUITE_NEO4J_PASSWORD"] = _PASSWORD

    def restore(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._saved.clear()


# ── native local server (the higher-performance path) ───────────────────


class LocalNeo4jServer:
    flavor = "local"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._tmp: str | None = None
        self._env = _EnvScope()

    @staticmethod
    def _launcher() -> str | None:
        home = os.environ.get("NEO4J_HOME")
        if home:
            cand = os.path.join(home, "bin", "neo4j")
            if os.path.exists(cand):
                return cand
        return shutil.which("neo4j")

    @staticmethod
    def _admin(launcher: str) -> str | None:
        cand = os.path.join(os.path.dirname(launcher), "neo4j-admin")
        return cand if os.path.exists(cand) else shutil.which("neo4j-admin")

    def available(self) -> tuple[bool, str]:
        if self._launcher() is None:
            return False, "no `neo4j` CLI on PATH or $NEO4J_HOME (install Neo4j to benchmark the native server)"
        try:
            import neo4j  # noqa: F401
        except Exception as e:
            return False, f"neo4j driver missing: {e}"
        return True, ""

    def start(self) -> str:
        launcher = self._launcher()
        if launcher is None:
            raise ProvisionError("no neo4j launcher found")
        bolt_port = _free_port()
        self._tmp = tempfile.mkdtemp(prefix="graphsuite_neo4j_")
        conf_dir = os.path.join(self._tmp, "conf")
        os.makedirs(conf_dir, exist_ok=True)
        # A throwaway neo4j.conf: bind a free Bolt port, disable HTTP/HTTPS to
        # avoid clashing with any system server, and point all writable dirs
        # at the tempdir so we never touch the install. The bare `neo4j`
        # launcher reads conf from $NEO4J_CONF (the NEO4J_<setting> env trick
        # is Docker-entrypoint-only, so we can't use it here).
        conf = "\n".join(
            [
                f"server.bolt.listen_address=:{bolt_port}",
                "server.http.enabled=false",
                "server.https.enabled=false",
                f"server.directories.data={self._tmp}/data",
                f"server.directories.logs={self._tmp}/logs",
                f"server.directories.run={self._tmp}/run",
                f"server.directories.import={self._tmp}/import",
                "dbms.security.auth_enabled=true",
                "",
            ]
        )
        with open(os.path.join(conf_dir, "neo4j.conf"), "w") as fh:
            fh.write(conf)

        child_env = dict(os.environ, NEO4J_CONF=conf_dir)
        # Seed the initial password into the (fresh) temp data dir.
        admin = self._admin(launcher)
        if admin is not None:
            pw_set = subprocess.run(
                [admin, "dbms", "set-initial-password", _PASSWORD],
                env=child_env,
                capture_output=True,
                text=True,
            )
            if pw_set.returncode != 0:
                # older single-verb form
                subprocess.run(
                    [admin, "set-initial-password", _PASSWORD],
                    env=child_env,
                    capture_output=True,
                    text=True,
                )

        self._proc = subprocess.Popen(
            [launcher, "console"],
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        uri = f"bolt://127.0.0.1:{bolt_port}"
        _wait_for_bolt(uri, time.perf_counter() + _READY_TIMEOUT_S, watch=self._proc)
        self._env.apply(uri)
        return uri

    def stop(self) -> None:
        self._env.restore()
        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(Exception):
                # kill the whole process group (the launcher forks a JVM)
                os.killpg(os.getpgid(self._proc.pid), 15)
            with contextlib.suppress(Exception):
                self._proc.wait(timeout=30)
            if self._proc.poll() is None:
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgid(self._proc.pid), 9)
        self._proc = None
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None


# ── Docker container (the baseline path) ─────────────────────────────────


class DockerNeo4jServer:
    flavor = "docker"

    def __init__(self) -> None:
        self._image = os.environ.get("GRAPHSUITE_NEO4J_IMAGE", _DEFAULT_IMAGE)
        self._container_id: str | None = None  # raw docker run
        self._tc = None  # testcontainers instance
        self._env = _EnvScope()

    @staticmethod
    def _docker_up() -> bool:
        if shutil.which("docker") is None:
            return False
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0

    def available(self) -> tuple[bool, str]:
        if shutil.which("docker") is None:
            return False, "no `docker` CLI (install Docker to benchmark Neo4j in a container)"
        if not self._docker_up():
            return False, "docker daemon is not running"
        try:
            import neo4j  # noqa: F401
        except Exception as e:
            return False, f"neo4j driver missing: {e}"
        return True, ""

    def start(self) -> str:
        # Prefer testcontainers (the prebuilt harness) when present; it does
        # image pull + readiness + teardown for us.
        try:
            from testcontainers.neo4j import Neo4jContainer  # type: ignore
        except Exception:
            Neo4jContainer = None  # noqa: N806

        if Neo4jContainer is not None:
            self._tc = Neo4jContainer(self._image, password=_PASSWORD)
            self._tc.start()
            host = self._tc.get_container_host_ip()
            port = self._tc.get_exposed_port(7687)
            uri = f"bolt://{host}:{port}"
            _wait_for_bolt(uri, time.perf_counter() + _READY_TIMEOUT_S)
            self._env.apply(uri)
            return uri

        # Fallback: raw `docker run` with a host-mapped Bolt port.
        port = _free_port()
        run = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "-p",
                f"{port}:7687",
                "-e",
                f"NEO4J_AUTH={_USER}/{_PASSWORD}",
                self._image,
            ],
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            raise ProvisionError(f"`docker run {self._image}` failed: {run.stderr.strip()}")
        self._container_id = run.stdout.strip()
        uri = f"bolt://127.0.0.1:{port}"
        try:
            _wait_for_bolt(uri, time.perf_counter() + _READY_TIMEOUT_S)
        except Exception:
            self.stop()
            raise
        self._env.apply(uri)
        return uri

    def stop(self) -> None:
        self._env.restore()
        if self._tc is not None:
            with contextlib.suppress(Exception):
                self._tc.stop()
            self._tc = None
        if self._container_id is not None:
            with contextlib.suppress(Exception):
                subprocess.run(["docker", "rm", "-f", self._container_id], capture_output=True)
            self._container_id = None
