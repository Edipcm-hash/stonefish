# What the model actually writes

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
- **→ means "what to notice".** Each one points at the specific behaviour, good or
  bad, rather than at the writing.

The model is a 222M-parameter English text continuer. It has no instruction
following, no world knowledge and no code ability, and it says so in the README's
first screen.

---

## 1. A story that works

**Prompt:** `Once upon a time` · temperature **0.8** · 150 tokens · 174.7 tok/s

```text
Once upon a time? The family was very sad. They needed help to find a way to get home and stay cool. They looked around and finally found a hidden door. The family was very excited to see what was behind the door. They opened the door and saw a bright light and a bright light.
The family was so happy to be home and they thanked the family for being so kind. They continued to walk home and had a wonderful time together.
<|endoftext|>
One day, a little boy named Tim went to the museum with his mom. They saw a big box with a picture of a big, scary dog. Tim saw a man with a funny hat and a funny hat. He asked his mom, "Can we buy the dog?"
His mom said, "Yes, let's buy the dog." So, they bought the dog and brought it home. Tim
```

→ Coherent children's-story prose for several sentences. Note the seam at the end: `a bright light and a bright light` - the model repeats instead of advancing, and `they thanked the family` appears from nowhere.

---

## 2. Dialogue

**Prompt:** `"Where are you going?" asked the mother.` · temperature **0.8** · 120 tokens · 185.3 tok/s

```text
"Where are you going?" asked the mother.—"I was just going to make the village a special place for the garden." 
The mother's face lit up with excitement. She said, "Don't worry, I will make the village a better place for you." 
The mother smiled and said, "That's right! The village can be very organized!" 
The daughter thanked her mother and hugged her little child. She said, "You're welcome, my good mother." 
The mother hugged her and said, "I'm very proud of you, but not today. I have a plan to keep you safe."
<|endoftext|>
One day, a little girl named Mia went to the park with her mom. Mia saw a big, red ball. She wanted
```

→ It picks up the dialogue convention (quotes, speaker tags) without being told. The content drifts: the daughter `hugged her little child` a line after being the child.

---

## 3. The other half of the corpus

**Prompt:** `The little girl` · temperature **0.8** · 120 tokens · 186.4 tok/s

```text
The little girl,
      which had been the first to tell us of her arrival in the town of Aix-Slav
      (a good deal of a good deal more than was her fortune to us). In the
      course of a week the house was occupied in the house of the Slav de Fleming
      Tophilla, which was on the other side of the coast, and a small part of
      the town
```

→ Same prompt style as a children's story, but it answers in **19th-century prose** - `the town of Aix-Slav`, `a good deal of a good deal more than was her fortune to us` - and reproduces Gutenberg's indentation for the paragraph. Three quarters of the training mix is Project Gutenberg; this is that half speaking.

---

## 4. Greedy decoding falls into a loop

**Prompt:** `The old house at the end of the street` · temperature **0.0** · 80 tokens · 178.9 tok/s

```text
The old house at the end of the street,
      and the old man was in the habit of going to the house to see the old
      woman and the old woman and the old woman and the old woman and the
      old woman and the old woman and the old woman and the old woman and the
      old woman and
```

→ Temperature 0. Each step takes the single most likely token, so a repeated phrase is self-reinforcing: `the old woman and the old woman and...` until the budget runs out. This is why the released samples use temperature 0.8.

---

## 5. No world knowledge

**Prompt:** `The capital of France is` · temperature **0.8** · 80 tokens · 168.5 tok/s

```text
The capital of France islands, and of the same classes,
which are now in the habit of the people whom the country affords. In
the new districts of France it is probable that the population of this
country is not the more remarkable among the peoples of Great Britain
America than of England; but the people who have been educated have been
taught to be the most intelligent and the
```

→ It never answers. It produces confident pseudo-sociological prose in the register of a 19th-century gazetteer (`of the same classes`, `the peoples of Great Britain America`). There is no fact to retrieve because none was learned.

---

## 6. No code

**Prompt:** `fn main() {` · temperature **0.8** · 80 tokens · 182.4 tok/s

```text
fn main() { in the case of a diocesed navigator (which has
been regarded as a section of the same estate) in the case of the
elements of the former section of the same Institute. It would be
admitted that a section of the provisions of the elements of the
estatesmanship of the elements was not to
```

→ The `fn main() {` prompt is ignored after four tokens; the model falls back to the only thing it knows how to write, which is prose about `a diocesed navigator`.

---

## 7. Confident fabrication

**Prompt:** `In the beginning of the nineteenth century` · temperature **0.8** · 120 tokens · 176.2 tok/s

```text
In the beginning of the nineteenth century—
      in the year 1846, he was not content to find it in the year 1848, but to
      give it his own account of the great business of life, and then to give
      up the first great and necessary means of which he was to begin the
      work of which he had been educated and made up the scale of all his
      possessions, and to keep up the system of production by making a great
```

→ The clearest illustration of the limitation: fluent, period-appropriate, and factually invented - `in the year 1846, he was not content to find it in the year 1848`. Nothing here is true; nothing here reads as false.

---

## Summary

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
