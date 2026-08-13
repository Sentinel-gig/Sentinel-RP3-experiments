"""
Experiment B: coverage of scripted versus novel scenarios.

Sentinel Working Paper No. 3.

WHAT THIS EXPERIMENT IS AND IS NOT
----------------------------------
This is a proof of concept evaluation on author constructed material. Both
scenario sets were written by the author for this paper. They are not sampled
from any corpus, they are not independently validated, and the novel set was
written with knowledge of the rule inventory it is tested against.

That construction is deliberate and it bounds the conclusion. The experiment
demonstrates that a scripted extractor built from an enumerated inventory
produces no output on scenario types outside that inventory, which is a
property of scripted extractors that follows from their construction. It does
NOT establish the rate at which such scenarios occur in real reporting, and it
does NOT establish that any particular LLM handles real out of inventory
scenarios at the rate observed here.

An independently sourced out of inventory test set, drawn from real reporting
and annotated by someone other than the system author, would be a substantially
stronger design. It is identified as future work in the manuscript.

WHAT IS DETERMINISTIC HERE
--------------------------
The rule baseline is deterministic: given the scenario text it produces the same
output on every run, on any machine. That half of the experiment is fully
reproducible from this file.

The LLM condition is NOT deterministic and cannot be made so. Model outputs
depend on the model version and on sampling. The published model outputs are
stored in llm_outputs.json with the model identifier and date of the run that
produced them, so that the SCORING is reproducible even though the GENERATION
is not. Re running the extraction with a different model, or the same model at
a later date, may produce different outputs.
"""
import json
import os
import re

# ---------------------------------------------------------------- rule inventory
# The scripted baseline: a finite ontology of incident types with keyword
# patterns. This is the artefact whose coverage the experiment measures.
TYPE_RULES = [
    ('robbery',       r'loot|robbed|robbery|snatch|gunpoint|armed|waylaid'),
    ('theft',         r'theft|thieves|thief|stole|stolen|burgl'),
    ('fraud',         r'fraud|fake|posing as|impersonat|defraud'),
    ('fire',          r'\bfire\b|blaze|caught fire'),
    ('assault',       r'assault|beaten|thrashed|attacked him|attacked her'),
    ('road_accident', r'accident|collision|collided|overturn|hit (his|her|the)|ran (him|her) over|rammed'),
]

RULE_TYPES = [name for name, _ in TYPE_RULES]


def rule_classify(text):
    """The scripted extractor. Returns a type from the inventory, or UNHANDLED."""
    for name, pattern in TYPE_RULES:
        if re.search(pattern, text, re.I):
            return name
    return 'UNHANDLED'


# ---------------------------------------------------------------- scenario sets
# Scripted set: plain phrasing instances of the six types in the inventory.
SCRIPTED = [
    {'id': 'S1',  'true_type': 'robbery',
     'text': 'Armed men on a motorcycle looted cash from a bank collection agent at gunpoint near the market crossing.'},
    {'id': 'S2',  'true_type': 'theft',
     'text': 'Thieves broke into a shop overnight and stole electronics worth several lakhs.'},
    {'id': 'S3',  'true_type': 'road_accident',
     'text': 'A speeding truck hit the motorcycle at the bypass junction and the rider died on the spot.'},
    {'id': 'S4',  'true_type': 'road_accident',
     'text': 'Two bikes collided head on near the flyover and both riders were injured.'},
    {'id': 'S5',  'true_type': 'fire',
     'text': 'A fire broke out in a godown near the industrial area and fire tenders rushed to the spot.'},
    {'id': 'S6',  'true_type': 'fraud',
     'text': 'A man posing as a bank officer defrauded a shopkeeper of fifty thousand rupees.'},
    {'id': 'S7',  'true_type': 'robbery',
     'text': 'Miscreants snatched a chain from a woman while she waited at the bus stop.'},
    {'id': 'S8',  'true_type': 'assault',
     'text': 'A group assaulted a vendor over a payment dispute at the mandi and he was hospitalised.'},
    {'id': 'S9',  'true_type': 'road_accident',
     'text': 'A car overturned on the highway after the driver lost control in the rain and two were injured.'},
    {'id': 'S10', 'true_type': 'theft',
     'text': 'Burglars stole jewellery from a locked house while the family was away.'},
    {'id': 'S11', 'true_type': 'fire',
     'text': 'A scooter caught fire at the petrol pump and no injuries were reported.'},
    {'id': 'S12', 'true_type': 'robbery',
     'text': 'Robbers waylaid a courier and snatched the parcel he was carrying.'},
]

# Novel set: hazard scenarios written to fall outside the inventory above.
# Several are modelled on documented Indian road hazards. All text is original.
NOVEL = [
    {'id': 'N1',
     'text': 'Razor coated kite string stretched across the flyover exit slashed a scooterist across the neck and he is bleeding heavily.',
     'note': 'manja injury; documented seasonal hazard in several Indian cities'},
    {'id': 'N2',
     'text': 'A crowd has blocked both lanes at the crossing over a procession dispute and stones are being hurled at passing two wheelers.'},
    {'id': 'N3',
     'text': 'Two men on a scooter threw a corrosive liquid at a woman near the toll plaza and rode away.'},
    {'id': 'N4',
     'text': 'A stray cattle herd wandered onto the unlit bypass after dark and a five vehicle pileup followed.'},
    {'id': 'N5',
     'text': 'An unmarked open manhole on the service road swallowed a scooter front wheel and threw the rider.'},
    {'id': 'N6',
     'text': 'An electric pole is leaning after the storm and a live wire is hanging at handlebar height across the lane.'},
    {'id': 'N7',
     'text': 'Men in khaki at an unofficial checkpoint are stopping delivery riders and demanding cash to let them pass.'},
    {'id': 'N8',
     'text': 'A drone has been following a courier along her route for the last twenty minutes and hovers whenever she stops.'},
    {'id': 'N9',
     'text': 'The underpass is waterlogged and a snapped cable is submerged in it; a commuter received a shock while wading through.'},
    {'id': 'N10',
     'text': 'A fuel tanker is leaking on the incline, there is a strong petrol smell, and vehicles are crawling through the spill.'},
    {'id': 'N11',
     'text': 'A crowd has surrounded a rider after a minor scrape at the junction and the situation is turning violent.'},
    {'id': 'N12',
     'text': 'A sinkhole opened mid carriageway on the ring road and two wheelers are being diverted onto the shoulder into oncoming traffic.'},
]


def run_rule_baseline():
    """Deterministic. Reproducible on any machine."""
    scripted = []
    for s in SCRIPTED:
        pred = rule_classify(s['text'])
        scripted.append({'id': s['id'], 'true_type': s['true_type'],
                         'predicted': pred, 'correct': pred == s['true_type']})
    novel = []
    for s in NOVEL:
        pred = rule_classify(s['text'])
        novel.append({'id': s['id'], 'predicted': pred,
                      'produced_output': pred != 'UNHANDLED'})
    return {
        'scripted': scripted,
        'novel': novel,
        'scripted_correct': sum(x['correct'] for x in scripted),
        'scripted_n': len(scripted),
        'novel_with_output': sum(x['produced_output'] for x in novel),
        'novel_n': len(novel),
    }


def score_llm_outputs(path='llm_outputs.json'):
    """Score stored LLM outputs. Returns None if no outputs file is present.

    The outputs file must record the model identifier and run date. This
    function does not call any model: it scores what was stored, so that the
    scoring step is reproducible even though generation is not.
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    novel = data.get('novel_assessments', {})
    handled = sum(1 for v in novel.values() if v.get('handled'))
    return {
        'model': data.get('model', 'UNSPECIFIED'),
        'run_date': data.get('run_date', 'UNSPECIFIED'),
        'prompt_sha256': data.get('prompt_sha256'),
        'novel_with_output': handled,
        'novel_n': len(novel),
        'assessments': novel,
    }


def good_turing_missing_mass(type_counts):
    """Good and Turing estimate of the probability mass of unseen types.

    E[M0] is approximated by N1 / n, where N1 is the number of types observed
    exactly once and n is the total number of observations.

    NOTE ON SCOPE: this estimates the missing mass of the OBSERVED SAMPLE. It
    bounds the coverage of an inventory constructed from that sample. It is not
    a statement about any particular alternative architecture.
    """
    n = sum(type_counts.values())
    n1 = sum(1 for c in type_counts.values() if c == 1)
    return {'n': n, 'singletons': n1, 'missing_mass_estimate': n1 / n if n else None}


if __name__ == '__main__':
    r = run_rule_baseline()
    print('=== Experiment B: rule baseline (deterministic) ===')
    print(f"scripted: {r['scripted_correct']}/{r['scripted_n']} correctly typed")
    print(f"novel:    {r['novel_with_output']}/{r['novel_n']} produced any classification")
    print()
    for x in r['novel']:
        print(f"  {x['id']}: {x['predicted']}")
    llm = score_llm_outputs()
    print()
    if llm:
        print(f"=== stored LLM outputs ({llm['model']}, {llm['run_date']}) ===")
        print(f"novel: {llm['novel_with_output']}/{llm['novel_n']} actionable assessments")
    else:
        print('No llm_outputs.json present. The LLM condition requires a stored')
        print('outputs file produced by running the Appendix A prompt against a')
        print('pinned model. See README for the required schema.')
