# Cutting the v1.0 release

The release is deliberately boring: every claim is already measured and every
number in the README has a command behind it (`docs/REPRODUCING.md`). What is
left is packaging and the parts a human has to approve.

## 0. Before anything else

```bash
python tools/release_gate.py      # runs every check below and prints a table
```

Or one at a time:

```bash
git status                 # must be clean
cargo tree                 # must be one line
cargo test --release       # 75 tests
cargo clippy --all-targets --features cuda -- -D warnings
cargo clippy --all-targets --no-default-features -- -D warnings
cargo fmt --all -- --check
python tools/check_no_turkish.py
python tools/check_docs.py
```

All of them green. If any is red, the release waits — that is the whole point of
having them.

**Then check what CI sees, not what this machine sees.** Twice during this release
a check passed locally and failed in CI, because `data/` and `runs/` are gitignored:
they exist here and not in a fresh clone. Reproduce the CI environment exactly —
`git archive` contains precisely the tracked files:

```bash
tmp=$(mktemp -d)
git archive --format=tar HEAD | tar -x -C "$tmp"
cd "$tmp"
python tools/check_docs.py
python tools/check_numbers.py      # gitignored artifacts report as "skipped"
python tools/check_no_turkish.py
CARGO_TARGET_DIR="$tmp/target" cargo test --release --no-default-features
```

That last line is the CI cpu-only job: it must end in `55 passed; 0 failed`.

## 1. The measurement that must exist in the README

These numbers cannot be copied from an older run:

| Number | Command | Lives in |
|---|---|---|
| val loss, perplexity, BPC | `en-eval 8 data/en/tiny stonefish_en.final.state` | §6.2, §7 |
| out-of-distribution loss | `en-eval 8 data/en/probe stonefish_en.final.state` | §7 |
| generation samples | `en-gen "Once upon a time" 150 0` and `... 0.8` | §6.2 |
| throughput at 222M | training log, `tok/s` column | §7.4 |
| PyTorch speed and parity | `tools/torch_bench.py`, `tools/torch_parity.py` | §7.5 |
| charts | `tools/make_charts.py`, `tools/make_report.py` | `docs/*.png` |

## 2. Tag

```bash
git tag -a v1.0.0 -m "v1.0.0 - 222M from-scratch language model, zero dependencies"
git push origin main --tags
```

Only after the repository is switched from private to public (or the tag will
point at a commit nobody can see).

## 3. GitHub release

Title: **v1.0.0 — a language model from scratch, zero dependencies**

Body: start from `CHANGELOG.md`, then put the four numbers a reader can check in
the first screen:

```
222,331,904 parameters · 2.05B tokens · 75 tests · 0 dependencies
val loss X.XXXX · BPC 0.XXXX · 10,7xx tokens/s on one RTX 5070
```

Attach: the loss curve and the comparison chart. **Do not attach the 848 MB
checkpoint** to a GitHub release — use Hugging Face (below) and link it.

## 4. Weights

The checkpoint is `SFEN001`, 848 MB, and cannot be rebuilt without 56 hours of
GPU time, so it belongs somewhere durable:

- Hugging Face model repo (`Edipcm-hash/stonefish-222m`), file
  `stonefish_en.final.state`, with `docs/huggingface-README.md` renamed to
  `README.md` as the model card - it carries the YAML frontmatter the Hub needs,
  which the in-repo model card does not.
- `data/en/tiny.bpe` (126 KB) must ship with the weights: the checkpoint alone
  cannot decode, and a downloader who has the state file but not the vocabulary
  gets an error, not a model. Put it in the repo as `tiny.bpe`.

Do not put the weights in the main git repo. `.gitignore` already blocks
`*.state`.

### Prerequisites, checked on this machine

| Requirement | State |
|---|---|
| `git lfs` | **3.8.1 installed** (needed: the file is over the 10 MB limit) |
| Hugging Face account + write token | not present in the environment; needed |
| `huggingface_hub` / `hf` CLI | not installed; the git + LFS route below avoids it |

### Upload over git + LFS

The whole thing is one script - it checks for git-lfs, clones the Hub repo, tracks
*.state, copies the weights, the vocabulary and the card, commits, and lets git
push prompt for the token (so the token is never written to a file):

`ash
./tools/publish_hf.ps1                 # defaults to Edipcm-hash/stonefish-222m
./tools/publish_hf.ps1 -Repo other/name
`

Doing it by hand also works:

```bash
# 1. create the empty repo on huggingface.co (web UI), then:
git clone https://huggingface.co/Edipcm-hash/stonefish-222m /tmp/sf-hf
cd /tmp/sf-hf

# 2. large-file tracking, before adding the checkpoint
git lfs install
git lfs track "*.state"
cp <repo>/stonefish_en.final.state .
cp <repo>/data/en/tiny.bpe .
cp <repo>/docs/huggingface-README.md README.md

# 3. push (username + write token as the password)
git add .gitattributes README.md tiny.bpe
git add stonefish_en.final.state
git commit -m "StonefishAI v2 - 222M, 2.05B tokens"
git push
```

Verify afterwards that the model card renders and that the file page shows an
LFS pointer with the real size, not a 130-byte text file.

## 5. Repository settings

- **About**: description from `Cargo.toml`, topics `rust llm transformer autograd
  cuda from-scratch zero-dependencies gpt bpe machine-learning`.
- **Pin** the repository on the profile, otherwise a fresh account shows nothing.
- **Enable Issues** — a from-scratch project will get "how do I…" questions, and
  that is a good problem.
- **Social preview**: `docs/architecture.png` works at that aspect ratio.

## 6. Announcement

One paragraph, no superlatives, the numbers that can be verified:

> I wrote a language model from scratch in Rust — autograd, BPE, transformer,
> CUDA kernels — with zero dependencies. 222M parameters trained on 2.05B tokens
> on one RTX 5070. 75 tests, 32 of them numerical gradient checks. It is 1.8×
> faster than PyTorch on the same GPU at the same architecture, and it is **not**
> a chat assistant: it continues text.

Then the honest part, because it is what makes the post credible: what it cannot
do, and what the next step (bf16) would buy.

## 7. After the release

- [ ] README's status line updated from "training" to the measured numbers
- [ ] `Cargo.toml` version already 1.0.0
- [ ] Tag matches the commit that was tested
- [ ] Hugging Face weights downloadable and the model card renders
- [ ] The verify-it-yourself table still passes on a fresh clone
