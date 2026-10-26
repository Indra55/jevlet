"use strict";
const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = value => `${(value * 100).toFixed(1)}%`;
const icons = {
  arrow:'<path d="M5 12h14m-5-5 5 5-5 5"/>',
  check:'<path d="m5 12 4 4L19 6"/>',
  layers:'<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
  lock:'<rect x="5" y="10" width="14" height="11" rx="3"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2"/>',
  spark:'<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z"/>',
  person:'<circle cx="12" cy="7" r="4"/><path d="M4 21v-3a8 8 0 0 1 16 0v3"/>',
  route:'<rect x="3" y="3" width="6" height="6" rx="2"/><rect x="15" y="15" width="6" height="6" rx="2"/><path d="M9 6h7a3 3 0 0 1 0 6H8a3 3 0 0 0 0 6h7"/>',
};
const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.spark}</svg>`;
const CASES = {
  banking_multi: {number:"05", title:"Multiple banking intents", type:"multi-label", subtitle:"A banking message can combine an action, an object, and a problem.", description:"Select all applicable intent components using a model trained on real NLU++ banking annotations.", inputLabel:"Customer message", button:"Find banking intents"},
  emotions: {number:"04", title:"Multiple emotions", type:"multi-label", subtitle:"One text can express several emotions. Select every label that applies.", description:"Explore independent emotion probabilities learned from human-annotated GoEmotions text.", inputLabel:"Text to understand", button:"Find emotions"},
  banking: {number:"01", title:"Banking support", type:"choice + yes/no", subtitle:"One customer message. An intent and two checks, answered together.", description:"Find the intent behind a customer message and check two topics in the same model pass.", inputLabel:"Customer message", button:"Analyze message"},
  reviews: {number:"02", title:"Review sentiment", type:"score · 1–5", subtitle:"Turn a short movie review into a five-level sentiment decision.", description:"Explore an ordinal score and its probability distribution, from very negative to very positive.", inputLabel:"Movie review", button:"Score review"},
  routing: {number:"03", title:"Confidence routing", type:"choice + policy", subtitle:"Let the model predict. Let a simple confidence rule decide the next step.", description:"Preview a support route or human review, using a confidence threshold you can adjust.", inputLabel:"Customer message", button:"Preview routing"},
};
const BANK_EXAMPLES = [
  {name:"Duplicate charge", text:"I was charged twice for my card payment.", options:["card arrival","transaction charged twice","card payment fee charged"], checks:["transaction charged twice","card arrival"]},
  {name:"Missing card", text:"I ordered my card two weeks ago and it still hasn't arrived.", options:["card arrival","card delivery estimate","lost or stolen card"], checks:["card arrival","lost or stolen card"]},
  {name:"Lost card", text:"I lost my bank card yesterday. What should I do?", options:["lost or stolen card","card not working","card arrival"], checks:["lost or stolen card","card not working"]},
  {name:"Ambiguous message", text:"There's an extra charge on my statement and I don't understand why.", options:["transaction charged twice","extra charge on statement","card payment fee charged"], checks:["transaction charged twice","card payment fee charged"]},
];
const REVIEW_EXAMPLES = [
  {name:"Positive", text:"A warm, funny film with excellent performances."},
  {name:"Negative", text:"A dull, painfully slow movie with wooden acting and a predictable ending."},
  {name:"Mixed", text:"The performances are strong, but the story never quite comes together."},
  {name:"Neutral", text:"The movie follows two people who meet on a train and spend a day together."},
];
const EMOTION_EXAMPLES = [
  {name:"Grateful and happy", text:"Thank you so much! I am really happy you helped me.", options:["gratitude","joy","admiration","sadness","anger","neutral"], checks:[]},
  {name:"Love and excitement", text:"I love this! I cannot wait to see you again!", options:["love","excitement","joy","fear","disappointment","neutral"], checks:[]},
  {name:"Sad and disappointed", text:"I am so disappointed. I really hoped we would get to stay together.", options:["sadness","disappointment","love","optimism","anger","neutral"], checks:[]},
  {name:"Curious", text:"How did you make that? I would love to learn more.", options:["curiosity","admiration","love","confusion","disgust","neutral"], checks:[]},
];
const BANK_MULTI_EXAMPLES = [
  {name:"Transfer problem", text:"I am trying to make a transfer but it is not working.", options:["make open apply setup get activate","transfer payment deposit","wrong notworking notshowing","card","balance","lost stolen"], checks:[]},
  {name:"Lost card", text:"I lost my card and want to freeze it.", options:["lost stolen","card","cancel close leave freeze","new","pin","withdrawal"], checks:[]},
  {name:"Change PIN", text:"How can I change my card PIN?", options:["how","change","pin","card","fees interests","loan"], checks:[]},
  {name:"Account balance", text:"How much money is in my savings account?", options:["how much","balance","savings","account","loan","card"], checks:[]},
];
const isMultiCase = key => key === "emotions" || key === "banking_multi";
function multiConfig(key = currentCase) {
  const banking = key === "banking_multi";
  return {banking, head:banking ? "banking_multilabel" : "multilabel",
    labels:banking ? info?.banking_multilabel_labels : info?.multilabel_labels,
    threshold:banking ? info?.banking_multilabel_threshold : info?.multilabel_threshold,
    noun:banking ? "intent components" : "emotions", catalogue:banking ? "NLU++ banking" : "GoEmotions",
    total:banking ? 48 : 28};
}
const BANKING_LABEL_TITLES = {
  "make open apply setup get activate":"Make / open / activate",
  "transfer payment deposit":"Transfer / payment / deposit",
  "wrong notworking notshowing":"Incorrect / not working",
  "cancel close leave freeze":"Cancel / close / freeze",
  "lost stolen":"Lost / stolen", "fees interests":"Fees / interest",
  "less lower before":"Less / lower / before", "more higher after":"More / higher / after"
};
const displayLabel = label => currentCase === "banking_multi" ? BANKING_LABEL_TITLES[label] || label : label;
const examplesFor = key => key === "banking_multi" ? BANK_MULTI_EXAMPLES : key === "emotions" ? EMOTION_EXAMPLES : key === "reviews" ? REVIEW_EXAMPLES : BANK_EXAMPLES;
const SENTIMENT = ["Very negative","Negative","Neutral","Positive","Very positive"];
let info = null, currentCase = null, controller = null, sequence = 0;
const states = Object.fromEntries(Object.keys(CASES).map(key => [key, {
  ...structuredClone(examplesFor(key)[0]),
  threshold:.9, result:null, sent:null, tab:"use", busy:false, error:""
}]));
function illustration(name) {
  const start = '<svg viewBox="0 0 240 135" fill="none" aria-hidden="true">';
  if (name === "emotions") return start + `<rect x="24" y="16" width="192" height="105" rx="9" fill="white" stroke="#ddd2ee"/><text x="39" y="35" fill="#6944ac" font-family="monospace" font-size="8">SEVERAL LABELS CAN APPLY</text><text x="39" y="57" fill="#6944ac" font-family="sans-serif" font-size="10">gratitude</text><rect x="108" y="48" width="88" height="10" rx="4" fill="#6944ac"/><text x="39" y="81" fill="#6944ac" font-family="sans-serif" font-size="10">joy</text><rect x="108" y="72" width="69" height="10" rx="4" fill="#9473c1"/><text x="39" y="105" fill="#a99cb9" font-family="sans-serif" font-size="10">anger</text><rect x="108" y="96" width="9" height="10" rx="4" fill="#e8e0f1"/></svg>`;
  if (name === "banking" || name === "banking_multi") return start + `<rect x="22" y="17" width="166" height="87" rx="9" fill="white" stroke="#d5dcf9"/><rect x="38" y="34" width="26" height="26" rx="7" fill="#edf0ff"/><path d="M45 43h12m-12 6h8" stroke="#5770da" stroke-width="2" stroke-linecap="round"/><path d="M76 39h75m-75 11h50M38 76h110m-110 10h70" stroke="#d3d9ed" stroke-width="4" stroke-linecap="round"/><rect x="114" y="80" width="109" height="31" rx="8" fill="#2c4ee5"/><path d="m125 96 3 3 6-7" stroke="white" stroke-width="1.5"/><text x="144" y="100" fill="white" font-family="sans-serif" font-size="9">Intent selected</text></svg>`;
  if (name === "reviews") return start + `<rect x="27" y="16" width="187" height="103" rx="9" fill="white" stroke="#eddbc9"/><text x="42" y="36" fill="#af967f" font-family="monospace" font-size="8">SENTIMENT DISTRIBUTION</text><path d="M43 94h155" stroke="#e7ded4"/><rect x="46" y="82" width="20" height="11" rx="3" fill="#eadaca"/><rect x="77" y="71" width="20" height="22" rx="3" fill="#eadaca"/><rect x="108" y="59" width="20" height="34" rx="3" fill="#dfc2a9"/><rect x="139" y="48" width="20" height="45" rx="3" fill="#a75822"/><rect x="170" y="74" width="20" height="19" rx="3" fill="#eadaca"/><text x="51" y="108" fill="#ab9681" font-family="monospace" font-size="8">1</text><text x="82" y="108" fill="#ab9681" font-family="monospace" font-size="8">2</text><text x="113" y="108" fill="#ab9681" font-family="monospace" font-size="8">3</text><text x="144" y="108" fill="#a75822" font-family="monospace" font-size="8">4</text><text x="175" y="108" fill="#ab9681" font-family="monospace" font-size="8">5</text></svg>`;
  return start + `<path d="M65 68h32m42 0h19V37h22m-41 31h19v31h22" stroke="#8fb49d" stroke-width="1.5"/><rect x="15" y="52" width="52" height="32" rx="7" fill="white" stroke="#c7dece"/><text x="26" y="72" fill="#698975" font-family="monospace" font-size="9">intent</text><rect x="99" y="48" width="39" height="39" rx="8" transform="rotate(45 118.5 67.5)" fill="white" stroke="#c7dece"/><text x="112" y="71" fill="#266a52" font-family="sans-serif" font-size="11">≥ t</text><rect x="178" y="21" width="54" height="32" rx="7" fill="#266a52"/><text x="187" y="41" fill="white" font-family="sans-serif" font-size="9">Route</text><rect x="178" y="83" width="54" height="32" rx="7" fill="white" stroke="#c7dece"/><text x="186" y="103" fill="#698975" font-family="sans-serif" font-size="9">Review</text></svg>`;
}
function home() {
  document.title = "Jevlet Lab · Small model, clear decisions";
  $("#main").innerHTML = `<div class="fade-in">${!info ? '<div class="error offline-banner">The model server is unavailable. Start it with <code>make demo</code>, then reload this page.</div>' : ""}
    <section class="hero"><div class="hero-copy"><div class="eyebrow"><span class="live-dot"></span> A local decision lab <span class="pill">DistilBERT inside</span></div><h1>Small model.<br><em>Clear decisions.</em></h1><p>Give it text. Ask typed questions. Get probabilities in one forward pass. Explore what your trained Jevlet can do.</p><div class="hero-notes"><span>${icon("check")} Real model outputs</span><span>${icon("lock")} Runs on your machine</span></div></div>
    <div class="architecture" aria-label="Text enters a single encoder pass, producing choice, score, yes/no, and multi-label outputs"><span class="arch-label">The idea, in one diagram</span><div class="text-node">text + typed questions</div><div class="arch-line"></div><div class="encoder"><div><b>One small encoder</b><span>one forward pass · multiple answers</span></div><div class="encoder-icon">${'<i></i>'.repeat(9)}</div></div><div class="arch-branch"><i></i><i></i><i></i><i></i></div><div class="typed-nodes"><span><b>↳</b> choice</span><span><b>↳</b> score</span><span><b>↳</b> yes/no</span><span><b>↳</b> labels</span></div></div></section>
    <section aria-labelledby="explore-title"><div class="section-heading"><h2 id="explore-title">Pick a case study</h2><p>Five ways to explore. Each comes with a live demo and a guide.</p></div><div class="case-grid">${Object.entries(CASES).map(([key,c]) => `<a class="case-card ${key}" href="/cases/${key}" data-nav><div class="card-top"><span>Case study ${c.number}</span><span>${c.type}</span></div><div class="card-art">${illustration(key)}</div><h3>${c.title}</h3><p>${c.description}</p><div class="card-link"><span>Open case study</span><span aria-hidden="true">↗</span></div></a>`).join("")}</div></section>
    <div class="under-cards"><span>${icon("layers")} Banking77 choices, NLU++ banking components, SST-5 reviews, and GoEmotions labels.</span><span>${icon("spark")} A focused research prototype. Explore its limits, too.</span></div></div>`;
}
function requestFor(key) {
  const s = states[key];
  return {case:key, text:s.text, ...(key === "reviews" ? {} : isMultiCase(key) ? {options:s.options} : {options:s.options, checks:s.checks.filter(Boolean)})};
}
function renderCase(key) {
  if (isMultiCase(key)) {renderMultiCase(); return;}
  const c = CASES[key], s = states[key];
  document.title = `${c.title} · Jevlet Lab`;
  $("#main").innerHTML = `<div class="case-shell ${key} fade-in"><a class="back" href="/" data-nav><span aria-hidden="true">←</span> All case studies</a><div class="case-title"><div><div class="eyebrow">Case study ${c.number} <span class="pill">${c.type}</span></div><h1>${c.title}</h1><p>${c.subtitle}</p></div><div class="case-number">${c.number} / 05</div></div>
    <div class="workspace"><section class="panel" aria-labelledby="input-title"><div class="panel-header"><span class="step">01</span><span id="input-title">Try it yourself</span><span class="tag">Editable input</span></div><div class="panel-body"><form id="decision-form"><div class="label-row"><label class="field-label" for="text">${c.inputLabel}</label><span>English text</span></div><textarea id="text" name="text" maxlength="12000" required>${esc(s.text)}</textarea><div class="example-label">Try an example</div><div class="examples">${(key === "reviews" ? REVIEW_EXAMPLES : BANK_EXAMPLES).map((e,i) => `<button type="button" class="example ${s.text === e.text ? "active" : ""}" data-example="${i}">${e.name}</button>`).join("")}</div>
    ${key === "reviews" ? `<label class="field-label">Fixed sentiment scale</label><div class="rubric">${SENTIMENT.map((name,i) => `<div><b>${i+1}</b><span>${name}</span></div>`).join("")}</div><p class="field-help">Use a short movie review. The model learned this rubric from SST-5, not from general product ratings.</p>` : '<div id="option-controls"></div>'}
    ${key === "routing" ? `<div class="policy"><label class="threshold-label" for="threshold">Demo confidence threshold <output id="threshold-output" for="threshold">${pct(s.threshold)}</output></label><input type="range" id="threshold" min="0.5" max="1" step="0.01" value="${s.threshold}"><div class="range-labels"><span>50% · More routes</span><span>100% · More review</span></div><p class="field-help">After a prediction, adjust this rule instantly. The model does not run again.</p></div>` : ""}
    <div id="input-error" role="alert">${s.error ? `<div class="error">${esc(s.error)}</div>` : ""}</div><button class="button-primary" type="submit" id="run-button" ${!info || s.busy ? "disabled" : ""}><span>${s.busy ? "Running local model…" : c.button}</span>${s.busy ? '<span class="spinner" aria-hidden="true"></span>' : icon("arrow")}</button><p class="input-foot">${key === "reviews" ? "1 typed question" : "Up to 3 typed questions"} · 1 encoder pass · Calibrated probabilities</p></form></div></section>
    <section class="panel" aria-labelledby="result-title"><div class="panel-header"><span class="step">02</span><span id="result-title">${key === "routing" ? "Routing preview" : "Model output"}</span><span class="tag">${info ? esc(info.checkpoint) : "Server offline"}</span></div><div id="result-body" class="panel-body" aria-live="polite" aria-busy="${s.busy}"></div></section></div>
    <section class="guide" aria-label="Case study guide"><div class="guide-tabs" role="tablist" aria-label="Case study guide">${[["use","How to use"],["works","How it works"],["integrate","Integrate"],["evidence","Evaluation notes"]].map(([id,title]) => `<button type="button" id="tab-${id}" role="tab" aria-controls="guide-body" aria-selected="${s.tab === id}" tabindex="${s.tab === id ? 0 : -1}" data-tab="${id}">${title}</button>`).join("")}</div><div id="guide-body" role="tabpanel" tabindex="0"></div></section></div>`;
  if (key !== "reviews") updateOptions();
  renderResult(); renderGuide();
}
function updateOptions() {
  if (isMultiCase(currentCase)) {updateMultiOptions(); return;}
  const s = states[currentCase], catalogue = info?.labels || [], selected = s.options;
  $("#option-controls").innerHTML = `<div class="label-row"><label class="field-label" for="add-option">Candidate intents</label><span>${selected.length} / 10</span></div><div class="option-chips">${selected.map((label,i) => `<span class="option-chip">${esc(label)}<button type="button" data-remove="${i}" aria-label="Remove ${esc(label)}">×</button></span>`).join("")}</div><select id="add-option" class="add-select" ${selected.length >= 10 || !info ? "disabled" : ""}><option value="">+ Add an intent from Banking77</option>${catalogue.filter(l => !selected.includes(l)).map(l => `<option value="${esc(l)}">${esc(l)}</option>`).join("")}</select><p class="field-help">Choose 2–10 plausible intents. A choice is relative to these options; include the correct intent when testing.</p><div class="checks"><label class="field-label" for="check-0">Two yes/no checks <span style="font-weight:400;color:var(--muted)">(optional)</span></label><div class="check-select">${[0,1].map(i => `<select id="check-${i}" aria-label="Yes/no topic check ${i+1}"><option value="">No check</option>${catalogue.map(l => `<option value="${esc(l)}" ${s.checks[i] === l ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>`).join("")}</div><p class="field-help">Each asks “Is this about [selected intent]?”</p></div>`;
}
function stale() {
  const s = states[currentCase];
  return s.result && JSON.stringify(s.sent) !== JSON.stringify(requestFor(currentCase));
}
function renderResult() {
  if (!currentCase) return;
  const s = states[currentCase], r = s.result, el = $("#result-body");
  el.setAttribute("aria-busy", String(s.busy));
  if (s.busy) {el.innerHTML = '<div class="result-placeholder"><span class="spinner" aria-hidden="true"></span><h3>Reading your text</h3><p>Your local encoder is answering the packed questions. No external model is called.</p></div>'; return;}
  if (!r) {el.innerHTML = `<div class="result-placeholder"><div class="empty-icon">${icon(currentCase === "routing" ? "route" : "layers")}</div><h3>A decision starts with your text</h3><p>Edit the example, then run the model to see its answer and confidence.</p><div class="empty-types">${(currentCase === "reviews" ? ["score","distribution"] : currentCase === "routing" ? ["intent","confidence","policy"] : ["choice","yes / no"]).map(t => `<span>${t}</span>`).join("")}</div></div>`; return;}
  if (isMultiCase(currentCase)) {renderMultiResult(); return;}
  const answer = r.answers[0], isReview = currentCase === "reviews";
  el.innerHTML = `<div id="stale-notice">${stale() ? '<div class="stale-note">Input changed. These results are from your previous run.</div>' : ""}</div><div class="result-meta"><span><i class="live-dot"></i> Real prediction</span><span><strong>${r.inference_ms.toFixed(1)} ms</strong> inference</span><span>${r.answers.length} question${r.answers.length === 1 ? "" : "s"} · ${r.encoder_passes} pass</span></div><div class="answer-label">${isReview ? "Predicted sentiment" : "Selected intent"}</div><div class="answer-main"><h2 class="answer-title">${isReview ? `${answer.answer} · ${SENTIMENT[answer.answer-1]}` : esc(answer.answer)}</h2><div class="confidence"><b>${pct(answer.confidence)}</b><span>confidence</span></div></div>
    ${isReview ? `<div class="score-summary"><span>Expected sentiment score</span><b>${answer.expected_score.toFixed(2)} <small>/ 5</small></b></div>` : ""}
    <div class="distribution" aria-label="${isReview ? "Sentiment" : "Candidate intent"} probabilities">${answer.probabilities.map((p,i) => `<div class="prob-row ${p === answer.confidence ? "winner" : ""}"><div class="prob-label"><span>${esc(isReview ? SENTIMENT[i] : answer.options[i])}</span></div><div class="prob-value">${pct(p)}</div><div class="bar"><i style="width:${p*100}%"></i></div></div>`).join("")}</div>
    ${r.answers.length > 1 ? `<div class="yesno-results">${r.answers.slice(1).map(a => `<div class="yesno-item"><span class="yesno-question">${esc(r.questions.find(q => q.id === a.id).text)}</span><span class="yesno-answer"><b>${a.answer ? "Yes" : "No"}</b><span>${pct(a.confidence)} conf.</span></span></div>`).join("")}</div>` : ""}
    ${currentCase === "routing" ? '<div id="routing-decision"></div>' : ""}
    <p class="result-note">${isReview ? "The expected score is the probability-weighted average of levels 1–5. This is a sentiment estimate, not a generated star rating." : "Choice confidence is relative to the candidate set. Each yes/no confidence is the probability of its displayed answer; the heads can disagree."}</p>
    ${r.context_tokens_dropped ? `<div class="warning">Long input: ${r.context_tokens_dropped} context tokens were dropped to fit the encoder. All questions were preserved.</div>` : ""}
    <details class="json-details"><summary>View the actual response JSON</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details>`;
  if (currentCase === "routing") renderRouting();
}
function renderRouting() {
  const s = states.routing, r = s.result;
  if (!r || !$("#routing-decision")) return;
  const answer = r.answers[0], route = answer.confidence >= s.threshold;
  $("#routing-decision").innerHTML = `<div class="routing-decision"><div class="route-track"><span class="route-node">Intent</span><span aria-hidden="true">→</span><span class="route-node">${pct(answer.confidence)} ${route ? "≥" : "<"} ${pct(s.threshold)}</span><span aria-hidden="true">→</span><span class="route-node">${route ? "Route" : "Review"}</span></div><div class="route-outcome ${route ? "routed" : "review"}">${icon(route ? "check" : "person")}<div><h3>${route ? "Would route to a support queue" : "Would defer to human review"}</h3><p>${route ? `Proposed queue: <strong>${esc(answer.answer)}</strong>. Choice confidence meets your threshold.` : "Choice confidence is below your threshold. A person would review the message and candidate set."}</p><small>Demo policy preview · No ticket is sent or financial action taken.</small></div></div></div>`;
}
function integrationSnippet() {
  const payload = JSON.stringify(requestFor(currentCase),null,2);
  let code = `import json\nfrom urllib.request import Request, urlopen\n\npayload = ${payload}\nrequest = Request(\n    "${location.origin}/api/predict",\n    data=json.dumps(payload).encode(),\n    headers={"Content-Type": "application/json"},\n)\nwith urlopen(request) as response:\n    result = json.load(response)\n\nprint(result["answers"])`;
  if (currentCase === "routing") code += `\n\n# Application policy; this does not rerun the model.\nintent = result["answers"][0]\nthreshold = ${states.routing.threshold.toFixed(2)}\nif intent["confidence"] >= threshold:\n    print("Proposed queue:", intent["answer"])\nelse:\n    print("Needs human review")`;
  return code;
}
function renderGuide() {
  if (isMultiCase(currentCase)) {renderMultiGuide(); return;}
  const key = currentCase, s = states[key], target = $("#guide-body");
  target.setAttribute("aria-labelledby",`tab-${s.tab}`);
  if (s.tab === "use") {
    const instructions = key === "reviews" ? [["Start with a movie review","Pick a positive, negative, or mixed example. Replace it with your own short English movie review."],["Read the whole distribution","Run the model. The largest probability selects a level; the expected score averages across all five levels."],["Explore uncertain reviews","Compare strongly worded and mixed reviews. Read the evaluation notes before applying this to another domain."]] : key === "routing" ? [["Choose plausible routes","Enter a support message. Select 2–10 known intents and optional yes/no checks, then run the preview."],["Move the decision threshold","Slide the threshold after a prediction. Below the threshold means review; at or above it means a proposed queue."],["Treat the rule as an experiment","Try the ambiguous example. The 90% default is a demo setting; choose a real threshold using held-out cases and review cost."]] : [["Choose a customer message","Pick an example or enter a short English banking support message. Keep it within the Banking77 topics."],["Set the candidate intents","Choose 2–10 known intents and up to two topic checks. The model can only choose an intent you include."],["Inspect all three answers","Run once to see choice probabilities and yes/no checks. Try similar candidates and ambiguous messages to explore uncertainty."]];
    target.innerHTML = `<div class="guide-grid">${instructions.map(([title,text],i) => `<div class="guide-item"><span class="guide-step">0${i+1}</span><div><h3>${title}</h3><p>${text}</p></div></div>`).join("")}</div>`;
  } else if (s.tab === "works") {
    target.innerHTML = `<div class="guide-prose"><h3>${key === "routing" ? "A model decision, followed by an ordinary rule" : "Typed outputs from one shared encoder"}</h3><p>${key === "reviews" ? "The server packs the fixed sentiment question and your review into a sequence. DistilBERT reads it once. A CORAL-style ordinal head at the question marker predicts four cumulative logits, which are converted to five class probabilities." : "The server packs the intent question, candidate option markers, up to two topic questions, and your message into one sequence. DistilBERT reads it once. A pointer head scores the option markers; a binary head answers each topic question."}</p><p>Held-out temperature scaling calibrates each head. Confidence is the largest class probability, not a promise of correctness. Long context is truncated after the questions, and the response reports how many context tokens were dropped.</p>${key === "routing" ? '<p>Routing is an application policy. The page compares the existing choice confidence with your threshold using ordinary code. Changing the slider reuses the last prediction; it makes no model request.</p>' : ""}<p>Text is sent to the server on this machine. This demo does not save submitted messages, call a hosted model, or require API keys.</p></div>`;
  } else if (s.tab === "integrate") {
    target.innerHTML = `<p class="code-note">Run this Python snippet while the local server is running. It reflects your current input${key === "routing" ? " and threshold" : ""}. The interactive schema is at <a href="/docs" target="_blank" rel="noopener" style="text-decoration:underline">/docs</a>.</p><div class="code-panel"><div class="code-top"><span>Python · standard library</span><button type="button" class="copy-button" id="copy-code">Copy code</button></div><pre id="integration-code">${esc(integrationSnippet())}</pre></div>`;
  } else {
    const head = key === "reviews" ? "score" : "choice", m = info?.benchmark?.[head];
    target.innerHTML = `<div class="guide-prose"><h3>${key === "reviews" ? "Sentiment is the harder task" : "A benchmark for candidate-based banking decisions"}</h3>${m ? `<div class="benchmark-row"><div><b>${pct(m.accuracy)}</b><span>${key === "reviews" ? "Exact five-level accuracy" : "Candidate choice accuracy"}</span></div><div><b>${m.ece.toFixed(3)}</b><span>Expected calibration error · 15 bins</span></div>${key === "reviews" ? `<div><b>${m.expected_score_mae.toFixed(3)}</b><span>Expected-score MAE</span></div>` : ""}<div><b>${m.samples.toLocaleString()}</b><span>Held-out test examples</span></div></div><p>Measured for <strong>${esc(info.checkpoint)}</strong>, after temperature scaling. These are saved test-set results, not performance measured from your demo inputs.</p>` : '<p>No matching full evaluation file was found for this checkpoint. Run the evaluation command to generate benchmark evidence; the live predictions still use the loaded model.</p>'}<p>${key === "reviews" ? "This head was trained and evaluated on SST-5 movie reviews. Exact five-class accuracy is modest and calibration has a visible gap. Mixed wording and adjacent sentiment levels can be difficult. Product reviews, sarcasm, and other languages are not established by this benchmark." : "Banking77 evaluation samples 2–10 candidate intents and always includes the correct intent. This is not accuracy for choosing among all 77 labels at once. Topic yes/no pairs come from the same banking labels, so they do not establish general question answering."}</p>${key === "routing" ? '<p>The routing demo has no separate routing benchmark. A high confidence value does not establish that a ticket should be auto-routed. Validate a threshold on representative messages, including missing true intents and unfamiliar topics, before using a real queue.</p>' : ""}<p>Inference timing covers packing and the model call on this server; it excludes startup, network time, and waiting for another request. Calibration is fitted on a separate calibration split; thresholds use validation data.</p></div>`;
  }
}
function updateStale() {
  if ($("#stale-notice")) $("#stale-notice").innerHTML = stale() ? '<div class="stale-note">Input changed. These results are from your previous run.</div>' : "";
  if (currentCase && states[currentCase].tab === "integrate") renderGuide();
}
function showError(message) {
  states[currentCase].error = message;
  $("#input-error").innerHTML = message ? `<div class="error">${esc(message)}</div>` : "";
}
async function run() {
  const key = currentCase, s = states[key], token = ++sequence;
  if (!info || s.busy) return;
  if (key !== "reviews" && (s.options.length < 2 || s.options.length > 10)) {showError("Choose between 2 and 10 candidate intents."); return;}
  if (key !== "reviews" && new Set(s.checks.filter(Boolean)).size !== s.checks.filter(Boolean).length) {showError("Choose different topics for the two checks, or leave one blank."); return;}
  const payload = structuredClone(requestFor(key));
  s.busy = true; showError(""); renderCase(key);
  $("#decision-form").querySelectorAll("input:not([type=range]),textarea,select,button").forEach(el => {el.disabled = true;});
  controller = new AbortController();
  try {
    const response = await fetch("/api/predict", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload),signal:controller.signal});
    const result = await response.json();
    if (!response.ok) throw new Error(Array.isArray(result.detail) ? result.detail.map(e => e.msg).join(" ") : result.detail || "Prediction failed.");
    s.result = result; s.sent = payload;
  } catch (error) {
    if (error.name !== "AbortError") s.error = error.message === "Failed to fetch" ? "Cannot reach the local model. Check the server terminal and try again." : error.message;
  } finally {
    s.busy = false;
    if (token === sequence && currentCase === key) renderCase(key);
  }
}
function navigate() {
  if (controller) controller.abort();
  sequence++;
  Object.values(states).forEach(s => {s.busy = false;});
  const match = location.pathname.match(/^\/cases\/(banking|reviews|routing|emotions|banking_multi)\/?$/);
  currentCase = match ? match[1] : null;
  if (currentCase) renderCase(currentCase); else home();
}
document.addEventListener("click", async event => {
  const link = event.target.closest("a[data-nav]");
  if (link && !event.ctrlKey && !event.metaKey && !event.shiftKey && event.button === 0) {event.preventDefault();history.pushState({},"",link.href);navigate();window.scrollTo(0,0);$("#main").focus({preventScroll:true});return;}
  if (!currentCase) return;
  const s = states[currentCase];
  const example = event.target.closest("[data-example]");
  if (example && !s.busy) {Object.assign(s,structuredClone(examplesFor(currentCase)[Number(example.dataset.example)]));s.error = "";renderCase(currentCase);return;}
  const remove = event.target.closest("[data-remove]");
  if (remove && !s.busy) {s.options.splice(Number(remove.dataset.remove),1);updateOptions();updateStale();return;}
  const tab = event.target.closest("[data-tab]");
  if (tab) {s.tab = tab.dataset.tab;$(".guide-tabs").querySelectorAll("button").forEach(b => {b.setAttribute("aria-selected",String(b === tab));b.tabIndex = b === tab ? 0 : -1;});renderGuide();return;}
  if (event.target.closest("#copy-code")) {
    const btn = $("#copy-code");
    try {await navigator.clipboard.writeText(integrationSnippet());btn.textContent="Copied";} catch {btn.textContent="Select code to copy";}
    setTimeout(() => {if (btn.isConnected) btn.textContent = "Copy code";},2000);
  }
});
document.addEventListener("submit", event => {if (event.target.id === "decision-form") {event.preventDefault();run();}});
document.addEventListener("input", event => {
  if (!currentCase) return;
  const s = states[currentCase];
  if (event.target.id === "text") {s.text=event.target.value;$(".examples").querySelectorAll("button").forEach(b => {b.classList.remove("active");});updateStale();}
  if (event.target.id === "threshold") {s.threshold=Number(event.target.value);$("#threshold-output").value=pct(s.threshold);if (isMultiCase(currentCase)) renderResult(); else renderRouting();if (s.tab === "integrate") renderGuide();}
});
document.addEventListener("change", event => {
  if (!currentCase) return;
  const s = states[currentCase];
  if (event.target.id === "add-option" && event.target.value) {s.options.push(event.target.value);updateOptions();updateStale();}
  if (event.target.id.startsWith("check-")) {s.checks[Number(event.target.id.slice(-1))]=event.target.value;updateStale();}
});
document.addEventListener("keydown", event => {
  if (!event.target.matches("[role=tab]") || !["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
  const tabs = Array.from($(".guide-tabs").querySelectorAll("button")), index = tabs.indexOf(event.target);
  const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length-1 : (index+(event.key === "ArrowRight" ? 1 : -1)+tabs.length)%tabs.length;
  event.preventDefault();tabs[next].focus();tabs[next].click();
});
window.addEventListener("popstate",navigate);
async function start() {
  try {
    const response = await fetch("/api/info");
    if (!response.ok) throw new Error("Unavailable");
    info = await response.json();
    states.emotions.threshold = info.multilabel_threshold ?? 0.5;
    states.banking_multi.threshold = info.banking_multilabel_threshold ?? 0.5;
    $("#connection").className="connection ready";
    $("#connection").innerHTML=`<i></i> Local ${esc(info.device.toUpperCase())} · Ready`;
  } catch {
    $("#connection").className="connection failed";
    $("#connection").innerHTML="<i></i> Server unavailable";
  }
  navigate();
}
start();

function renderMultiCase() {
  const key = currentCase, s = states[key], c = CASES[key], m = multiConfig(), supported = Boolean(m.labels?.length);
  document.title = `${c.title} · Jevlet Lab`;
  $("#main").innerHTML = `<div class="case-shell ${key} fade-in"><a class="back" href="/" data-nav>← All case studies</a><div class="case-title"><div><div class="eyebrow">Case study ${c.number} <span class="pill">multi-label</span></div><h1>${c.title}</h1><p>${c.subtitle}</p></div><div class="case-number">${c.number} / 05</div></div>
    ${!supported ? `<div class="warning">The loaded checkpoint does not support ${m.noun} selection. Start the server with a checkpoint trained for this task.</div>` : ''}
    <div class="workspace"><section class="panel"><div class="panel-header"><span class="step">01</span><span>Try it yourself</span><span class="tag">Editable input</span></div><div class="panel-body"><form id="decision-form"><label class="field-label" for="text">${c.inputLabel}</label><textarea id="text" maxlength="12000" required>${esc(s.text)}</textarea><div class="example-label">Try an example</div><div class="examples">${examplesFor(key).map((e,i) => `<button type="button" class="example ${s.text === e.text ? 'active' : ''}" data-example="${i}">${e.name}</button>`).join('')}</div><div id="option-controls"></div>
    <div class="policy"><label class="threshold-label" for="threshold">Label selection threshold <output id="threshold-output" for="threshold">${pct(s.threshold)}</output></label><input type="range" id="threshold" min="0.05" max="0.95" step="0.01" value="${s.threshold}"><div class="range-labels"><span>More labels</span><span>Fewer labels</span></div><p class="field-help">Adjust which labels are selected from the same probabilities. This does not rerun the model. Saved validation threshold: ${pct(m.threshold ?? .5)}.</p></div>
    <div id="input-error" role="alert">${s.error ? `<div class="error">${esc(s.error)}</div>` : ''}</div><button class="button-primary" id="run-button" type="submit" ${!supported || s.busy ? 'disabled' : ''}><span>${s.busy ? 'Running local model…' : c.button}</span>${icon('arrow')}</button><p class="input-foot">1 question · 1 encoder pass · Independent label probabilities</p></form></div></section>
    <section class="panel"><div class="panel-header"><span class="step">02</span><span>Selected ${m.noun}</span><span class="tag">${esc(info?.checkpoint || 'Server offline')}</span></div><div class="panel-body" id="result-body" aria-live="polite"></div></section></div>
    <section class="guide"><div class="guide-tabs" role="tablist" aria-label="Case study guide">${[['use','How to use'],['works','How it works'],['integrate','Integrate'],['evidence','Evaluation notes']].map(([id,title]) => `<button type="button" id="tab-${id}" role="tab" aria-controls="guide-body" aria-selected="${s.tab === id}" tabindex="${s.tab === id ? 0 : -1}" data-tab="${id}">${title}</button>`).join('')}</div><div id="guide-body" role="tabpanel" tabindex="0"></div></section></div>`;
  updateMultiOptions(); renderResult(); renderGuide();
}
function updateMultiOptions() {
  const s = states[currentCase], m = multiConfig(), labels = m.labels || [];
  $("#option-controls").innerHTML = `<div class="label-row"><label class="field-label" for="add-option">Candidate ${m.noun}</label><span>${s.options.length} / 10</span></div><div class="option-chips">${s.options.map((label,i) => `<span class="option-chip" title="${esc(m.banking ? info?.banking_multilabel_descriptions?.[label] || label : label)}">${esc(displayLabel(label))}<button type="button" data-remove="${i}" aria-label="Remove ${esc(label)}">×</button></span>`).join('')}</div><select id="add-option" class="add-select" ${s.options.length >= 10 || !labels.length ? 'disabled' : ''}><option value="">+ Add ${m.banking ? 'an intent component' : 'an emotion'}</option>${labels.filter(l => !s.options.includes(l)).map(l => `<option value="${esc(l)}">${esc(displayLabel(l))}</option>`).join('')}</select><p class="field-help">Choose 2–10 labels from the ${m.total}-label ${m.catalogue} catalogue. Several, one, or none can be selected. Omitted labels cannot appear in the answer.${m.banking ? ' These are modular actions, objects, and states; they differ from the Banking77 choice labels.' : ''}</p>`;
}
function renderMultiResult() {
  const s = states[currentCase], m = multiConfig(), r = s.result, a = r.answers[0], selected = a.options.filter((_,i) => a.probabilities[i] >= s.threshold);
  $("#result-body").innerHTML = `<div id="stale-notice">${stale() ? '<div class="stale-note">Input changed. These results are from your previous run.</div>' : ''}</div><div class="result-meta"><span><i class="live-dot"></i> Real prediction</span><span>${r.inference_ms.toFixed(1)} ms inference</span><span>1 question · ${r.encoder_passes} pass</span></div><div class="answer-label">${selected.length} label${selected.length === 1 ? '' : 's'} selected at ${pct(s.threshold)}</div><div class="answer-main"><h2 class="answer-title">${selected.length ? selected.map(label => esc(displayLabel(label))).join(' · ') : 'No candidate passed the threshold'}</h2></div><div class="distribution" aria-label="Independent ${m.noun} probabilities">${a.probabilities.map((p,i) => `<div class="prob-row ${p >= s.threshold ? 'winner' : ''}"><div class="prob-label"><span>${p >= s.threshold ? '✓ ' : ''}${esc(displayLabel(a.options[i]))}</span></div><div class="prob-value">${pct(p)}</div><div class="bar"><i style="width:${p*100}%"></i></div></div>`).join('')}</div><p class="result-note">Each probability describes one label. They do not sum to 100%. No selection means none passed this threshold, not that the text is out of domain.${m.banking ? ' The model identifies intent components; it does not perform transactions or change an account.' : ' These predictions are not a psychological assessment.'}</p>${r.context_tokens_dropped ? `<div class="warning">Long input: ${r.context_tokens_dropped} context tokens were dropped.</div>` : ''}<details class="json-details"><summary>View the actual response JSON</summary><pre>${esc(JSON.stringify(r,null,2))}</pre><p class="field-help">JSON records the server threshold. The slider changes the displayed selection locally.</p></details>`;
}
function renderMultiGuide() {
  const s = states[currentCase], c = multiConfig(), target = $("#guide-body"), m = info?.benchmark?.[c.head];
  target.setAttribute('aria-labelledby',`tab-${s.tab}`);
  if (s.tab === 'integrate') {
    target.innerHTML = `<p class="code-note">The server uses this task's saved validation threshold. Add a threshold field to request a different selection policy.</p><div class="code-panel"><div class="code-top"><span>Python · standard library</span><button type="button" class="copy-button" id="copy-code">Copy code</button></div><pre id="integration-code">${esc(integrationSnippet())}</pre></div>`;
  } else if (s.tab === 'evidence') {
    target.innerHTML = `<div class="guide-prose"><h3>Real multi-label annotations</h3><p>${c.banking ? 'Trained on NLU++ banking messages annotated with modular intent components. Separate source folds are used for training, validation, calibration, and test, with text duplicates removed. This is one fixed split, not the original paper’s cross-validation result.' : 'The emotion head uses human-annotated GoEmotions comments. Its original test partition remains separate, and development examples are divided between validation and calibration after duplicate removal. The combined checkpoint also rehearses emotion examples while learning banking.'}</p>${m ? `<div class="benchmark-row"><div><b>${pct(m.micro_f1)}</b><span>Candidate micro-F1</span></div><div><b>${pct(m.macro_f1)}</b><span>Candidate macro-F1</span></div><div><b>${pct(m.exact_set_accuracy)}</b><span>Exact candidate-set accuracy</span></div><div><b>${m.samples}</b><span>Held-out texts</span></div></div>` : '<p>No matching evaluation results are available for this checkpoint.</p>'}<p>Evaluation uses 2–10 supplied candidates, including cases with missing true labels. It is not full-catalogue classification across all ${c.total} labels.${c.banking ? ' NLU++ components do not map directly to Banking77’s single-intent classes. Slot extraction is not implemented.' : ' Neutral is an annotated label, distinct from no selected candidates.'}</p></div>`;
  } else if (s.tab === 'works') {
    target.innerHTML = `<div class="guide-prose"><h3>Several labels from one encoder pass</h3><p>The shared DistilBERT encoder reads the text and supplied labels. A dedicated ${c.banking ? 'banking' : 'emotion'} head scores each option marker. Sigmoid converts each score to an independent probability. Training uses binary cross-entropy with observed-label masking.</p><p>Each multi-label task has its own temperature and selection threshold. Calibration uses the calibration split; threshold selection uses validation micro-F1. No maximum-probability confidence is assigned to the whole label set.</p></div>`;
  } else {
    target.innerHTML = `<div class="guide-prose"><h3>Read all applicable ${c.noun}</h3><p>Enter a short English ${c.banking ? 'banking message' : 'text'}, choose plausible candidate labels, and run once. Several labels may be selected. Adjust the threshold to explore the same probabilities without another model request.</p><p>${c.banking ? 'A transfer problem can combine an action, transfer/payment, and a not-working state. A lost-card request can combine lost/stolen, card, and freeze/cancel. Hover over a selected component for its dataset definition.' : 'Include the emotions you want to test. Labels outside the supplied candidate set cannot appear in the answer.'} The editable examples are prompts, not guaranteed predictions.</p></div>`;
  }
}
