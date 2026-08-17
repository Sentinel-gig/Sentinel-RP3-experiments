# Experiment A: Re-run Protocol

Sentinel Working Paper No. 3. Everything needed to regenerate Experiment A
from scratch, with a clean separation of who produces what.

---

## Why the division of labour matters

Three artifacts feed the scoring, and they must come from three different places
or the experiment proves nothing.

| Artifact | Produced by | Why |
|---|---|---|
| Gold annotations | **You**, reading the source articles | You are the author and take responsibility for the answer key. An AI-written key scored against an AI extractor measures agreement between two language models, not extraction accuracy. |
| LLM extraction outputs | **A language model you choose**, not Claude | The manuscript was drafted with Claude's assistance. Evaluating Claude would make the evaluation dependent on the tool that wrote the paper. Any capable model is fine; record which one. |
| Rule baseline outputs | **Code in this repository** | Deterministic. Reproduces on any machine. |

State the model and date in the outputs file. That is what makes the scoring
checkable even though the generation is not reproducible.

---

## Step 1: Label the corpus yourself

Open each URL in `corpus_manifest.json` and record six fields per article.
Label from the article as published; do not infer beyond what the text supports.

| Field | Values | Guidance |
|---|---|---|
| `is_incident` | true / false | Does the text report a discrete safety-relevant incident or hazard that actually occurred? A report that a complaint was fabricated is **false**. |
| `type` | road_accident, robbery, theft, fraud, fire, assault, other, none | Choose the dominant classification. Impersonation to obtain money is `fraud`, not `robbery`. |
| `city` | free text | The city or district where the incident **occurred**, not the publishing desk. Items 12 and 13 are published under Kanpur but the incidents are elsewhere. |
| `location_text` | free text | The most specific location phrase in the article: landmark, road, area, police station. Empty if none. |
| `severity` | fatal / injury / property_only / none | Fatal if anyone died. Injury if injured but no death. |
| `gig_relevance` | true / false | Does the incident involve or specifically endanger a mobile worker: delivery agent, courier, collection agent, distribution agent? |

**Item 10 needs a decision from you.** That article reports two separate fatal
crashes. Either label the article as one record for the dominant incident, or
split it into `10a` and `10b`. Whichever you choose, apply the same convention
in the outputs file, and tell me which you used so the manuscript describes it
correctly.

**Optional but valuable.** Have Ashutosh or Shreya independently label six or
more of the same articles without seeing your labels. Send both sets and I will
report inter-annotator agreement, which answers the reviewer's standing point
about single-annotator gold labels.

Save as `gold_labels.json`:

```json
{
  "annotator": "Arav Misra",
  "annotation_date": "2026-08-16",
  "labels": {
    "1": {"is_incident": true, "type": "robbery", "city": "Delhi",
          "location_text": "Pragati Maidan tunnel", "severity": "property_only",
          "gig_relevance": true}
  }
}
```

---

## Step 2: Run the extraction on a model of your choice

For each article, paste the article text and the prompt below into the model.
Use a **fresh conversation per article** so that earlier items cannot influence
later ones. Do not give the model examples, and do not correct it between items.

### The prompt, verbatim

```
You are extracting structured safety-incident records from Indian news text for a
worker-safety risk model. Read the text and return ONLY a JSON object with fields:
is_incident (boolean, does the text report a discrete safety-relevant incident or
hazard?), type (one of: road_accident, robbery, theft, fraud, fire, assault, other,
none), city (the city or district where the incident occurred, disambiguated from
context, noting the state if the name is ambiguous), location_text (the most specific
location phrase in the text: landmark, road, area, or empty if none), severity
(fatal / injury / property_only / none), gig_relevance (boolean, does the incident
involve or specifically endanger a mobile worker such as a delivery agent, courier,
or collection agent?). Extract only what the text supports; do not invent locations
or victims; if the text is not an incident report, set is_incident to false and type
to none.
```

Paste the article text below that prompt. For Hindi articles, paste the Hindi
text unchanged. Do not translate it first: cross-lingual handling without a
translation stage is one of the things the experiment measures.

Save as `llm_outputs.json`:

```json
{
  "model": "exact model name and version string",
  "provider": "e.g. OpenAI / Google / Meta",
  "run_date": "2026-08-16",
  "conversation_policy": "fresh conversation per article, no examples",
  "extractions": {
    "1": {"is_incident": true, "type": "robbery", "city": "Delhi",
          "location_text": "Pragati Maidan tunnel", "severity": "property_only",
          "gig_relevance": true}
  }
}
```

Record the model string exactly as the provider reports it. "GPT" or "Gemini"
is not sufficient; the version matters and a reviewer will want it.

---

## Step 3: Check the files parse

```bash
python check_materials.py
```

This verifies that both files are valid JSON, that every manifest item has a
gold label and an extraction, that all field values are in the permitted sets,
and that the required metadata is present. Fix anything it reports before
sending.

---

## Step 4: Send back

- `gold_labels.json`
- `llm_outputs.json`
- the second annotator's labels, if you obtained them
- which convention you used for item 10

I will run the rule baseline, score all three conditions with Wilson intervals,
and rewrite Section 7.1 and the abstract to whatever the numbers show.

---

## What to expect

The corpus is 13 items, smaller than the 22 reported in the previous version,
so the intervals will be **wider**, not narrower. At n = 13 a perfect score
carries a 95 percent interval of roughly [0.77, 1.00]. It is entirely possible
that no field comparison separates, in which case the manuscript will say so and
Experiment A will be reported as a viability demonstration rather than a
comparison. That is an acceptable outcome and a more defensible one than the
previous version's.

The three hard cases are where the interesting results will be: the fabricated
robbery complaint (item 3), the two-incidents-in-one-article case (item 10), and
the two geographic misattribution traps (items 12 and 13).
