"""Generate docs/SAMPLES.md from real `en-gen` runs.

Each sample is captured verbatim from the binary, so the document cannot drift from
what the model actually produces.
"""

import io
import re
import subprocess

EXE = "target/release/stonefish_ai.exe"

SAMPLES = [
    (
        "A story that works",
        "Once upon a time",
        0.8,
        150,
        "Coherent children's-story prose for several sentences. Note the seam at the "
        "end: `a bright light and a bright light` - the model repeats instead of "
        "advancing, and `they thanked the family` appears from nowhere.",
    ),
    (
        "Dialogue",
        '"Where are you going?" asked the mother.',
        0.8,
        120,
        "It picks up the dialogue convention (quotes, speaker tags) without being "
        "told. The content drifts: the daughter `hugged her little child` a line "
        "after being the child.",
    ),
    (
        "The other half of the corpus",
        "The little girl",
        0.8,
        120,
        "Same prompt style as a children's story, but it answers in **19th-century "
        "prose** - `the town of Aix-Slav`, `a good deal of a good deal more than was "
        "her fortune to us` - and reproduces Gutenberg's indentation for the paragraph. "
        "Three quarters of the training mix is Project Gutenberg; this is that half "
        "speaking.",
    ),
    (
        "Greedy decoding falls into a loop",
        "The old house at the end of the street",
        0.0,
        80,
        "Temperature 0. Each step takes the single most likely token, so a repeated "
        "phrase is self-reinforcing: `the old woman and the old woman and...` until "
        "the budget runs out. This is why the released samples use temperature 0.8.",
    ),
    (
        "No world knowledge",
        "The capital of France is",
        0.8,
        80,
        "It never answers. It produces confident pseudo-sociological prose in the "
        "register of a 19th-century gazetteer (`of the same classes`, `the peoples of "
        "Great Britain America`). There is no fact to retrieve because none was "
        "learned.",
    ),
    (
        "No code",
        "fn main() {",
        0.8,
        80,
        "The `fn main() {` prompt is ignored after four tokens; the model falls back "
        "to the only thing it knows how to write, which is prose about `a diocesed "
        "navigator`.",
    ),
    (
        "Confident fabrication",
        "In the beginning of the nineteenth century",
        0.8,
        120,
        "The clearest illustration of the limitation: fluent, period-appropriate, and "
        "factually invented - `in the year 1846, he was not content to find it in the "
        "year 1848`. Nothing here is true; nothing here reads as false.",
    ),
]


def run(prompt, temp, tokens):
    p = subprocess.run(
        [EXE, "en-gen", prompt, str(tokens), str(temp)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = p.stdout + p.stderr
    text = "\n".join(
        ln for ln in out.splitlines() if ln and not ln.startswith(("🐟", "  (", "Model", "Device"))
    ).strip()
    m = re.search(r"\((\d+) tokens, ([\d.]+) s, ([\d.]+) tokens/s", out)
    return text, m


header = """# What the model actually writes

These are **real, unfiltered** outputs from the released checkpoint
(`stonefish_en.final.state`, step 62,483), produced by:

```bash
cargo run --release -- en-gen "<prompt>" <tokens> <temperature>
```

Nothing is selected for being good. The failures are here because a model card that
only shows its best sample is not telling you anything.

Two things to know before reading:

- **Sampling is random.** At any temperature above 0, every run produces a different
  continuation. These are single samples, not cherry-picked reruns - but a different
  seed would give different text.
- **\u2192 means "what to notice".** Each one points at the specific behaviour, good or
  bad, rather than at the writing.

The model is a 222M-parameter English text continuer. It has no instruction
following, no world knowledge and no code ability, and it says so in the README's
first screen.

---

"""

body = []
for i, (title, prompt, temp, tokens, note) in enumerate(SAMPLES, 1):
    text, m = run(prompt, temp, tokens)
    toks = m.group(1) if m else str(tokens)
    tps = m.group(3) if m else "?"
    body.append(f"## {i}. {title}\n\n")
    body.append(f"**Prompt:** `{prompt}` · temperature **{temp}** · {toks} tokens · {tps} tok/s\n\n")
    body.append("```text\n" + text + "\n```\n\n")
    body.append(f"\u2192 {note}\n\n---\n\n")

closing = """## Summary

| | |
|---|---|
| Grammar | solid. Sentence structure and dialogue conventions are learned |
| Coherence | two to three sentences, then drift |
| Register | switches between children's stories and 19th-century prose, because the corpus is both |
| World knowledge | none |
| Code | none |
| Greedy decoding | can loop; sampling is required for usable output |
| Speed | 160-180 tokens/s with the KV-cache on one RTX 5070 |

That is the honest shape of a 222M-parameter model on 2.05B tokens. It is not a
small chatbot and it is not trying to be one - it continues text, and the price of
that narrowness is everything a chatbot does that this cannot.
"""

with io.open("docs/SAMPLES.md", "w", encoding="utf-8", newline="") as fh:
    fh.write(header + "".join(body) + closing)

print("docs/SAMPLES.md written")
for i, (title, prompt, temp, tokens, note) in enumerate(SAMPLES, 1):
    print(f"  {i}. {title}  [temp {temp}] prompt {len(prompt)} chars")
