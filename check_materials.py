"""
Validate the Experiment A materials before scoring.

Run this after producing gold_labels.json and llm_outputs.json. It checks that
both files parse, cover every item in the manifest, use permitted field values,
and carry the metadata the manuscript needs to describe what was evaluated.

    python check_materials.py
"""
import json
import os
import sys

FIELDS = ['is_incident', 'type', 'city', 'location_text', 'severity', 'gig_relevance']
TYPES = {'road_accident', 'robbery', 'theft', 'fraud', 'fire', 'assault', 'other', 'none'}
SEVERITIES = {'fatal', 'injury', 'property_only', 'none'}

problems = []
notes = []


def load(path, required=True):
    if not os.path.exists(path):
        if required:
            problems.append(f'MISSING FILE: {path}')
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        problems.append(f'INVALID JSON in {path}: {e}')
        return None


def check_record(tag, key, rec):
    for f in FIELDS:
        if f not in rec:
            problems.append(f'{tag} item {key}: missing field "{f}"')
    if 'is_incident' in rec and not isinstance(rec['is_incident'], bool):
        problems.append(f'{tag} item {key}: is_incident must be true or false, got {rec["is_incident"]!r}')
    if 'gig_relevance' in rec and not isinstance(rec['gig_relevance'], bool):
        problems.append(f'{tag} item {key}: gig_relevance must be true or false, got {rec["gig_relevance"]!r}')
    if rec.get('type') not in TYPES and 'type' in rec:
        problems.append(f'{tag} item {key}: type "{rec["type"]}" not in {sorted(TYPES)}')
    if rec.get('severity') not in SEVERITIES and 'severity' in rec:
        problems.append(f'{tag} item {key}: severity "{rec["severity"]}" not in {sorted(SEVERITIES)}')
    if rec.get('is_incident') is False and rec.get('type') not in (None, 'none'):
        notes.append(f'{tag} item {key}: is_incident is false but type is "{rec.get("type")}"')


def main():
    manifest = load('corpus_manifest.json')
    gold = load('gold_labels.json')
    llm = load('llm_outputs.json')
    if manifest is None:
        print('corpus_manifest.json not found. Nothing to check against.')
        sys.exit(1)

    ids = {i['id'] for i in manifest['items']}
    print(f'manifest: {len(ids)} items '
          f'({sum(1 for i in manifest["items"] if i["language"] == "hi")} Hindi)')

    if gold is not None:
        if 'annotator' not in gold:
            problems.append('gold_labels.json: missing "annotator"')
        if 'annotation_date' not in gold:
            problems.append('gold_labels.json: missing "annotation_date"')
        labels = gold.get('labels', {})
        extra = set(labels) - ids
        missing = ids - set(labels)
        # allow split records such as 10a / 10b
        missing = {m for m in missing if not any(k.startswith(m) for k in labels)}
        if missing:
            problems.append(f'gold_labels.json: no label for items {sorted(missing)}')
        if extra:
            notes.append(f'gold_labels.json: extra keys not in manifest {sorted(extra)} '
                         f'(fine if these are split records such as 10a and 10b)')
        for k, rec in labels.items():
            check_record('gold', k, rec)
        print(f'gold labels: {len(labels)} records by {gold.get("annotator", "UNKNOWN")}')

    if llm is not None:
        for meta in ['model', 'run_date']:
            if not llm.get(meta):
                problems.append(f'llm_outputs.json: missing or empty "{meta}" '
                                f'(the manuscript cannot describe the evaluation without it)')
        ex = llm.get('extractions', {})
        if gold is not None:
            gk = set(gold.get('labels', {}))
            if set(ex) != gk:
                only_gold = sorted(gk - set(ex))
                only_llm = sorted(set(ex) - gk)
                if only_gold:
                    problems.append(f'llm_outputs.json: no extraction for {only_gold}')
                if only_llm:
                    problems.append(f'llm_outputs.json: extraction for unlabelled items {only_llm}')
        for k, rec in ex.items():
            check_record('llm', k, rec)
        print(f'llm outputs: {len(ex)} records from {llm.get("model", "UNKNOWN")} '
              f'on {llm.get("run_date", "UNKNOWN")}')

    print()
    if notes:
        print('NOTES (not blocking):')
        for n in notes:
            print(f'  - {n}')
        print()
    if problems:
        print(f'{len(problems)} PROBLEM(S) TO FIX:')
        for p in problems:
            print(f'  - {p}')
        sys.exit(1)
    print('All checks passed. Materials are ready for scoring.')


if __name__ == '__main__':
    main()
