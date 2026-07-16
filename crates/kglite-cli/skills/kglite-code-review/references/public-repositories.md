# Reviewing a public repository

Use a predictable cache outside the target repository. Clone data only; do not
run package managers, build scripts, generators, hooks, or repository binaries.

```bash
cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/kglite/repos"
repo_dir="$cache_root/<owner>/<repo>"
mkdir -p "$(dirname "$repo_dir")"
git clone --filter=blob:none --no-checkout \
  'https://github.com/<owner>/<repo>.git' "$repo_dir"
git -C "$repo_dir" fetch origin '<revision>'
codingest build "$repo_dir" --rev '<revision>' \
  --output "$repo_dir/.kglite/code-review.kgl" --format json
```

For an existing cache, fetch only the explicit refs needed. The revision build
uses `git archive` and never checks out or runs the repository. Do not silently
switch a user's working checkout. Verify GitHub claims against the resolved
commit and cite stable source links when reporting findings.
