# Setup runbook

Two repositories, deliberately decoupled.

```
hex9            engine, training, web bundle
                CI builds WASM, deploys to <you>.github.io/hex9/

personal-site   Hugo + Blowfish
                project page iframes the deployed demo
```

The demo stays independently linkable — `<you>.github.io/hex9/` is what goes in
an application form, where nobody wants to scroll past a blog header first. The
site never holds a copy of the bundle, so rebuilding one never rebuilds the
other.

---

## Repo 1 — hex9

```
hex9/
├── CLAUDE.md
├── CMakeLists.txt
├── README.md
├── .gitignore
├── .github/workflows/ci.yml
├── include/hex/board.hpp
├── tests/test_board.cpp
└── bench/bench_board.cpp
```

```bash
mkdir -p hex9/include/hex hex9/tests hex9/bench hex9/.github/workflows
cd hex9
# place the files, then:
cmake -B build && cmake --build build -j
./build/test_board && ./build/bench_board

git init && git add -A
git commit -m "Phase 0: game core with property-based verification"
gh repo create hex9 --public --source=. --push
```

CI runs on the first push: release build plus tests, and an ASan/UBSan job. The
TSAN and Pages-deploy jobs are commented out in the workflow — uncomment them at
phases 3 and 4 respectively.

Before the deploy job can work: **Settings → Pages → Source: GitHub Actions.**

### Dependencies by phase

| Phase | Needs |
|---|---|
| 0–1 | nothing beyond a C++20 compiler and CMake |
| 2 | Python, PyTorch, numpy |
| 3 | ONNX Runtime C++ with the CUDA execution provider |
| 4 | emsdk, ONNX Runtime Web (from jsDelivr, nothing to vendor) |

Phase 3 uses ONNX Runtime rather than LibTorch on purpose: it is the same model
format the browser already needs, so there is one export path for the whole
project and the self-play binary stays a self-contained artifact.

---

## Repo 2 — personal-site

```
personal-site/
├── .gitignore
├── config/_default/
│   ├── hugo.toml
│   ├── module.toml
│   └── menus.en.toml
└── content/projects/hex9/index.md
```

```bash
mkdir personal-site && cd personal-site
hugo mod init github.com/<you>/personal-site
hugo mod get github.com/nunocoracao/blowfish/v2
# place the config files and content, then:
hugo server -D
```

`-D` renders drafts. The project page ships with `draft: true` so it stays
invisible until the demo actually exists.

Blowfish has far more configuration than the three files here. Copy
`params.toml` from the annotated example in the Blowfish repo rather than
writing it by hand, and check the current docs — the option names move between
major versions.

Updating the theme later: `hugo mod get -u ./...`

---

## Hosting

Cloudflare Pages, connected to the `personal-site` repo:

- Build command: `hugo --gc --minify`
- Output directory: `public`
- Environment variable: `HUGO_VERSION` set to a specific version, not `latest`

Cloudflare knows Hugo natively, so there is no build workflow to maintain and no
submodule handling to debug. GitHub Pages, Netlify, and Vercel all work equally
well if you would rather keep everything in one place — Hugo just emits static
HTML, so the host is not a commitment.

---

## Gotchas, in the order you will hit them

**`unsafe = true` under `markup.goldmark.renderer`.** Without it Hugo strips raw
HTML from markdown. The iframe vanishes and leaves blank space, with no error
and no warning. Already set in `hugo.toml`.

**Use Hugo Modules, not a git submodule, for Blowfish.** The submodule route
needs `submodules: recursive` in every CI checkout, and forgetting it produces a
site that builds locally and deploys completely unstyled. Modules avoid the
whole category.

**Pin `HUGO_VERSION`.** Blowfish tracks recent Hugo features; a host silently
upgrading Hugo underneath you produces build failures that look like your fault.

**Git LFS does not work on GitHub Pages.** It serves the pointer file, not the
blob. Commit ONNX weights as ordinary files — a quantised 6×64 net is well under
a megabyte, so this costs nothing.

**Do not enable cross-origin isolation.** Setting COOP/COEP to unlock
`SharedArrayBuffer` applies to the whole page, and then every embedded resource
needs CORP headers to load — fonts, CDN scripts, comment widgets all break. The
browser build stays single-threaded. Concurrency lives in phase 3 training,
where it belongs.

**Threaded self-play is not reproducible.** Keep a `--threads=1` path that is
bit-for-bit deterministic given a seed, and seed each game from
`(run_seed, game_index)` rather than a shared generator. You will need it
constantly for debugging, and reproducibility on demand is worth a line in the
writeup.
