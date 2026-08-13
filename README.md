# Crime Weighted Geospatial Risk Modeling: Study Materials

Reproducibility package for:

> Misra, A. (2026). *Crime Weighted Geospatial Risk Modeling for Last Mile
> Mobility Corridors in Tier 2 Indian Cities: Large Language Models as Open
> World Data Engines for Worker Safety Intelligence.* Sentinel Working Paper
> No. 3, Secured Systems Technologies Pvt. Ltd.

---

## Read this first: what is and is not reproducible

This package is more honest than it is complete, and the distinction matters.
The three experiments in the manuscript have different reproducibility status,
and the manuscript states this in its limitations section.

| Component | Status |
|---|---|
| Experiment A, rule baseline | **Deterministic.** Reproduces exactly from `experiment_a.py`. |
| Experiment A, corpus | **Not redistributed.** Copyrighted news text. Source URLs, dates, outlets and gold labels are published so any reader can verify every annotation against the original. |
| Experiment A, LLM condition | **Not reproducible.** Model outputs depend on model version and sampling. Stored outputs make the *scoring* reproducible; they do not make the *generation* reproducible. |
| Experiment B, scenario sets | **Fully published.** Author written text, included in `experiment_b.py`. |
| Experiment B, rule baseline | **Deterministic.** Reproduces exactly. |
| Experiment B, LLM condition | **Not reproducible**, same reason as above. |
| Experiment C | **Deterministic given a seed.** Reproduces exactly on the same NumPy and SciPy versions. |

Anyone claiming that an LLM evaluation is exactly reproducible is either pinning
a model version that will eventually be retired, or is not being careful. We
publish what makes the scoring checkable and state plainly what cannot be fixed.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

python experiment_b.py                      # deterministic, runs immediately
python experiment_c.py --replicates 20      # simulation, a few minutes
python experiment_a.py                      # requires the data files below
```

---

## Experiment A: files you must supply

Three files, with templates provided:

**`corpus_manifest.json`** — one record per item: outlet, publication date,
source URL, language, and a short identifying quotation. No full text. See
`corpus_manifest.template.json`.

**`gold_labels.json`** — the annotations, keyed by item id, with the six fields
`is_incident`, `type`, `city`, `location_text`, `severity`, `gig_relevance`.

**`llm_outputs.json`** — the model outputs that were scored, in the same shape as
the gold labels, plus three required metadata fields:

```json
{
  "model": "exact-model-identifier",
  "run_date": "YYYY-MM-DD",
  "prompt_sha256": "hash of the verbatim prompt in Appendix A",
  "extractions": { "1": { "is_incident": true, "...": "..." } }
}
```

The `model` and `run_date` fields are not optional. Without them a reader cannot
know what was evaluated, and the evaluation is uninterpretable.

---

## Experiment A: known weaknesses

**Sample size.** The corpus is 22 items. At that size a reported accuracy of
100 percent carries a 95 percent Wilson interval of roughly [85, 100], and most
field level comparisons between the LLM and the rule baseline do **not** reach
statistical separation. `experiment_a.py` reports intervals for exactly this
reason. The manuscript restricts its claims accordingly: location extraction,
where the gap is large, and cross lingual transfer, where the LLM required no
Hindi rules. The other fields are reported as directionally favourable and
underpowered.

**Single annotator.** Gold labels were produced by the author, who also ran the
systems being compared. Inter annotator agreement is unavailable. Independent
double annotation of a subset would materially strengthen this experiment and is
the most valuable single improvement available.

---

## Experiment B: scope of the conclusion

Both scenario sets are author constructed, and the novel set was written with
knowledge of the rule inventory it is tested against. The experiment therefore
demonstrates a property of scripted extractors that follows from how they are
built: an enumerated inventory produces no output outside its enumeration. It
does **not** establish how often such scenarios occur in real reporting, and it
does **not** establish a general rate at which any LLM handles real out of
inventory material.

An independently sourced test set, drawn from real reporting and annotated by
someone other than the system author, would be a substantially stronger design.

---

## Experiment C: what the simulation shows

A synthetic risk field is degraded by a realistic observation process (spatially
biased under reporting, geocoding noise) and then recovered from the degraded
observations. Recovery quality is measured by the predictive accuracy index
against a withheld month, and the recovered field is then used as a prior for
sensor based detection.

Run with `--replicates 20` or more. Single replication figures vary
substantially: across replications the predictive accuracy index of the
recovered field spans a wide range, and reporting a single draw would overstate
precision. The script reports means and intervals across replications.

Setting `--seed 7 --replicates 1` reproduces the first replication reported in
the original version of the manuscript.

---

## Files

| File | Purpose |
|---|---|
| `experiment_a.py` | rule baseline, scorer with Wilson intervals, corpus loader |
| `experiment_b.py` | scenario sets, rule inventory, Good and Turing estimator |
| `experiment_c.py` | risk field simulation and risk conditioned detection |
| `corpus_manifest.template.json` | required schema for the corpus manifest |

## License

MIT for the code. The corpus manifest references third party copyrighted
material which is not licensed under these terms and is not redistributed here.
