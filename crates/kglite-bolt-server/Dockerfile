# kglite-bolt-server — serve a `.kgl` knowledge graph over the Bolt wire
# protocol, so any Neo4j-compatible driver can query a kglite graph.
#
# Multi-stage: compile the binary against the workspace in a Rust builder,
# then ship just the binary on a slim Debian runtime. Build from the repo
# root so the whole workspace (the bolt server's path-dep on `kglite` core)
# is in the build context:
#
#   docker build -f crates/kglite-bolt-server/Dockerfile -t kglite-bolt-server:local .
#   docker run --rm -p 7687:7687 \
#       -v "$PWD/graph.kgl:/data/graph.kgl:ro" \
#       kglite-bolt-server:local
#
# Then point any Bolt driver at bolt://localhost:7687 (default auth scheme
# is `none`, so any credentials are accepted). Override the served graph /
# flags by appending args after the image name (they replace CMD).

FROM rust:bookworm AS build
WORKDIR /src
COPY . .
# Only the bolt server (+ its `kglite` core dependency) is compiled; the
# PyO3 wheel crate is not built, so no Python toolchain is needed here.
RUN cargo build --release -p kglite-bolt-server

FROM debian:bookworm-slim
# Shared libs the binary may dlopen at runtime (TLS for bolt+s, the .kgl
# (de)compression codecs). ca-certificates for outbound TLS trust.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -r -u 10001 -m kglite && mkdir -p /data && chown kglite:kglite /data
COPY --from=build /src/target/release/kglite-bolt-server /usr/local/bin/kglite-bolt-server
USER kglite
EXPOSE 7687
ENTRYPOINT ["kglite-bolt-server"]
# Default: serve the graph at /data/graph.kgl on all interfaces. Provide the
# graph by bind-mounting onto /data/graph.kgl, or `docker cp` it into the
# container before `docker start` (mount-backend-independent — what the
# benchmark adapter does).
CMD ["--graph", "/data/graph.kgl", "--bind", "0.0.0.0", "--port", "7687"]
