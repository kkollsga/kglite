# AGENTS.md deviations from the doctrine baseline

Local improvements in `AGENTS.md` and `.agents/skills/` that differ in
behaviour from `../doctrine/reference/AGENTS.baseline.md` and the doctrine
reference skills, each with the reason it adds value here. Adaptations
(KGLite's own commands, paths and thresholds) are not listed. A behavioural
difference without an entry is staleness. Each doctrine sync re-justifies every
entry (doctrine R14; format in the baseline).

Last checked against doctrine 0.2.0 on 2026-11-01. One deviation since.

## Merging a green PR to main needs no approval

- **Where:** `AGENTS.md` → "Commits & releases"
- **Baseline says:** a push to `main` needs the user's explicit go-ahead in the same turn, except inside `/release` and the CI fix-and-push loop.
- **Local rule:** fast-forwarding a finished, CI-green PR to `main` is standard procedure and needs no approval; only the release (version bump, publish) needs the user's explicit request.
- **Why it adds value here:** the owner's decision of 2026-11-14. A merge without a version bump publishes nothing here (the release workflow skips an already-published version), and waiting for a second approval left a green PR unmerged while `main` was red on its next run.
- **Upstream:** candidate
- **Last reviewed:** 2026-11-14
