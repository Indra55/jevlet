"""Record real CUDA predictions for the explanation guide."""
import json
from pathlib import Path
import torch
from jevlet.config import write_json
from jevlet.model import Jevlet

ROOT=Path(__file__).resolve().parents[1]
assert torch.cuda.is_available(), 'Examples must use CUDA'
r=json.loads((ROOT/'results/jevlet_refinement.json').read_text())
assert r.get('completed')
model=Jevlet.load(r['selected_checkpoint'],'cuda')
examples=[{'text':'I lost my debit card and need to order a replacement.',
 'questions':[{'id':'intent','type':'choice','text':"What is the customer's intent?",
               'options':['lost or stolen card','cash withdrawal','balance inquiry']},
              {'id':'lost','type':'yesno','text':'Is this about lost or stolen card?'},
              {'id':'cash','type':'yesno','text':'Is this about cash withdrawal?'}]},
 {'text':'The performances are wonderful, and the story is moving and beautifully told.',
  'questions':[{'id':'sentiment','type':'score','text':'How positive is this review?'}]},
 {'text':'This movie is dull, badly acted, and a complete waste of time.',
  'questions':[{'id':'sentiment','type':'score','text':'How positive is this review?'}]}]
for e in examples: e['prediction']=model.predict(e['text'],e['questions'])
write_json(ROOT/'results/jevlet_explainer_examples.json',examples)
print(json.dumps(examples,indent=2))
