"""
Experiment A: extraction from real multilingual incident reports.

Sentinel Working Paper No. 3.

WHY THIS FILE DOES NOT CONTAIN THE CORPUS
-----------------------------------------
The evaluation corpus consists of excerpts from published news reporting by
Amar Ujala, Zee News Hindi, Hindustan Times, The Tribune, Deccan Herald and PTI.
That text is copyrighted and cannot be redistributed here.

What is published instead, which is the standard practice for news derived
datasets, is everything needed to reconstruct and verify the evaluation:

  corpus_manifest.json   one record per item: source outlet, publication date,
                         URL, language, and a short identifying quotation
  gold_labels.json       the annotations, which are the author's own work
  llm_outputs.json       the model outputs that were scored, with the model
                         identifier and run date
  this file              the deterministic rule baseline and the scorer

A reader can follow each URL, read the original, and check the gold label
against it. The scoring step is fully reproducible from the stored outputs.

TWO LIMITATIONS THE READER SHOULD WEIGH
---------------------------------------
1. Single annotator. The gold labels were produced by the author, who also ran
   the systems being compared. Inter annotator agreement is not available.
   Independent double annotation of a subset is identified as future work.

2. The LLM condition is not deterministically reproducible. Model outputs depend
   on model version and sampling. The stored outputs make the SCORING
   reproducible; they do not make the GENERATION reproducible. Re running the
   extraction against a different model, or the same model later, may give
   different results. The model identifier and date are recorded so that a
   reader knows exactly what was evaluated.
"""
import json
import os
import re

FIELDS = ['is_incident', 'type', 'city', 'location_text', 'severity', 'gig_relevance']

# ---------------------------------------------------------------- rule baseline
# A tuned keyword and regex extractor of the kind a diligent engineering team
# would script. It was built WITH knowledge of the corpus languages and cities,
# which makes it a deliberately strong version of the scripted alternative.
TYPE_RULES = [
    ('robbery',       r'loot|robbed|robbery|snatch|gunpoint|armed|waylaid|\u0932\u0942\u091f'),
    ('theft',         r'theft|thieves|thief|stole|burgl|\u091a\u094b\u0930\u0940|\u091a\u094b\u0930\u094b\u0902'),
    ('fraud',         r'fraud|fake|posing|\u092b\u0930\u094d\u091c\u0940'),
    ('fire',          r'\bfire\b|blaze|\u0906\u0917'),
    ('assault',       r'assault|beaten|attacked'),
    ('road_accident', r'accident|collision|collided|overturn|hit (his|her|the)|ran (him|her) over|'
                      r'\u091f\u0915\u094d\u0915\u0930|\u0939\u093e\u0926\u0938\u093e'),
]
CITY_GAZETTEER = ['Kanpur', 'Delhi', 'Bengaluru', 'Zirakpur', 'Amritsar',
                  'Budaun', 'Hassan', 'Ramanagaram', '\u0915\u093e\u0928\u092a\u0941\u0930']


def rule_extract(text):
    """Deterministic scripted extractor. Reproducible on any machine."""
    t = 'none'
    for name, pattern in TYPE_RULES:
        if re.search(pattern, text, re.I):
            t = name
            break
    city = ''
    for c in CITY_GAZETTEER:
        if c in text:
            city = 'Kanpur' if c == '\u0915\u093e\u0928\u092a\u0941\u0930' else c
            break
    if re.search(r'killed|died|dead|death|\u092e\u094c\u0924', text, re.I):
        sev = 'fatal'
    elif re.search(r'injur|hurt|hospital|\u0918\u093e\u092f\u0932|\u091c\u0916\u094d\u092e\u0940', text, re.I):
        sev = 'injury'
    elif t == 'none':
        sev = 'none'
    else:
        sev = 'property_only'
    m = re.search(r'near ([A-Z][\w\- ]{2,30}?)(?:[,.;]| on| when| by)', text)
    loc = m.group(1).strip() if m else ''
    gig = bool(re.search(r'delivery (agent|boy)|courier', text, re.I))
    return {'is_incident': t != 'none', 'type': t, 'city': city,
            'location_text': loc, 'severity': sev, 'gig_relevance': gig}


# ---------------------------------------------------------------- scoring
def tokens(s):
    return set(re.findall(r'\w+', (s or '').lower()))


def location_match(pred, gold, threshold=0.5):
    """Token overlap, since location phrasing varies legitimately."""
    if not pred and not gold:
        return True
    if not pred or not gold:
        return False
    a, b = tokens(pred), tokens(gold)
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= threshold


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def score(predictions, gold):
    """Field accuracy with Wilson intervals. n is small, so the intervals are
    wide and the manuscript reports them rather than bare point estimates."""
    n = len(gold)
    out = {}
    for field in FIELDS:
        if field == 'location_text':
            k = sum(1 for key in gold
                    if location_match(predictions[key].get(field), gold[key].get(field)))
        else:
            k = sum(1 for key in gold
                    if predictions[key].get(field) == gold[key].get(field))
        lo, hi = wilson(k, n)
        out[field] = {'correct': k, 'n': n, 'accuracy': k / n, 'ci95': [lo, hi]}
    tp = sum(1 for key in gold if predictions[key].get('is_incident') and gold[key].get('is_incident'))
    fp = sum(1 for key in gold if predictions[key].get('is_incident') and not gold[key].get('is_incident'))
    fn = sum(1 for key in gold if not predictions[key].get('is_incident') and gold[key].get('is_incident'))
    out['relevance_precision'] = tp / max(1, tp + fp)
    out['relevance_recall'] = tp / max(1, tp + fn)
    return out


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main():
    manifest = load('corpus_manifest.json')
    gold = load('gold_labels.json')
    llm = load('llm_outputs.json')

    if manifest is None or gold is None:
        print('corpus_manifest.json or gold_labels.json not found.')
        print('See README for the required schema. The corpus text itself is not')
        print('redistributed; the manifest records the source URL for each item so')
        print('that a reader can verify every annotation against the original.')
        return

    print(f'corpus: {len(manifest["items"])} items, '
          f'{sum(1 for i in manifest["items"] if i.get("language") == "hi")} in Hindi')

    texts = {i['id']: i.get('text') for i in manifest['items']}
    if any(v is None for v in texts.values()):
        print('\nManifest contains no text field (expected: copyright).')
        print('To score the rule baseline, first run reconstruct_corpus.py, which')
        print('fetches each source URL locally into corpus_reconstructed/ and is')
        print('gitignored. Only the scoring of stored LLM outputs can proceed here.')
    else:
        rule_pred = {k: rule_extract(v) for k, v in texts.items()}
        print('\n=== rule baseline (deterministic) ===')
        for field, r in score(rule_pred, gold['labels']).items():
            if isinstance(r, dict):
                print(f"  {field:16s} {r['accuracy']:.3f} "
                      f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  ({r['correct']}/{r['n']})")

    if llm:
        print(f"\n=== stored LLM outputs ({llm.get('model','UNSPECIFIED')}, "
              f"{llm.get('run_date','UNSPECIFIED')}) ===")
        for field, r in score(llm['extractions'], gold['labels']).items():
            if isinstance(r, dict):
                print(f"  {field:16s} {r['accuracy']:.3f} "
                      f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  ({r['correct']}/{r['n']})")
    else:
        print('\nNo llm_outputs.json present. See README for the required schema.')


if __name__ == '__main__':
    main()
