"""Build and render the model explanation from the actual selected checkpoint outputs."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/jevlet_model_explained.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
result = json.loads((ROOT / "results/jevlet_refinement.json").read_text())
assert result["completed"]
examples = json.loads((ROOT / "results/jevlet_explainer_examples.json").read_text())
metrics = result["test"]["calibrated"]
navy = colors.HexColor("#17324D")
teal = colors.HexColor("#007D87")
light = colors.HexColor("#EDF6F7")
gray = colors.HexColor("#536475")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleJ", fontName="Helvetica-Bold", fontSize=29, leading=34, textColor=navy, spaceAfter=17))
styles.add(ParagraphStyle(name="DeckJ", fontName="Helvetica", fontSize=13, leading=19, textColor=gray, spaceAfter=19))
styles.add(ParagraphStyle(name="H1J", fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=navy, spaceAfter=15))
styles.add(ParagraphStyle(name="H2J", fontName="Helvetica-Bold", fontSize=12, leading=17, textColor=teal, spaceBefore=12, spaceAfter=7))
styles.add(ParagraphStyle(name="BodyJ", fontName="Helvetica", fontSize=10.5, leading=16, spaceAfter=10))
styles.add(ParagraphStyle(name="SmallJ", fontName="Helvetica", fontSize=8.5, leading=12, textColor=gray, spaceAfter=7))
styles.add(ParagraphStyle(name="MonoJ", fontName="Courier", fontSize=9, leading=14, backColor=light, borderPadding=10, spaceAfter=14))

story = []
def p(text, style="BodyJ"):
    story.append(Paragraph(text, styles[style]))
def heading(k, title):
    p(f"{k:02d} / JEVLET EXPLAINED", "SmallJ")
    p(title, "H1J")
def page():
    story.append(PageBreak())
def table(rows, widths):
    cells = [[Paragraph(str(c), styles["SmallJ"]) for c in row] for row in rows]
    t = Table(cells, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0), light), ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LINEBELOW",(0,0),(-1,0),1,teal), ("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#CCD8E0")),
        ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story.append(t); story.append(Spacer(1,10))
def pipeline():
    d = Drawing(500,235)
    def box(x,y,w,h,label,fill=light):
        d.add(Rect(x,y,w,h,rx=7,ry=7,fillColor=fill,strokeColor=teal,strokeWidth=.8))
        lines=label.split("\n")
        for i,line in enumerate(lines):
            d.add(String(x+w/2,y+h/2+(len(lines)-1)*7-i*14,line,fontName="Helvetica-Bold",fontSize=10,textAnchor="middle",fillColor=navy))
    def arrow(x,y,x2,y2):
        d.add(Line(x,y,x2,y2,strokeColor=gray,strokeWidth=1.2))
        d.add(Polygon([x2,y2,x2-4,y2+7,x2+4,y2+7],fillColor=gray,strokeColor=gray))
    box(65,191,370,35,"Context + one to three typed questions")
    arrow(250,191,250,173)
    box(65,137,370,36,"Tokenizer + [Q] / [OPT] markers")
    arrow(250,137,250,118)
    box(65,82,370,36,"Shared DistilBERT encoder: one packed pass")
    for x,label in [(5,"Choice\noption scores"),(177,"Sentiment\nfive logits"),(349,"Yes / no\none logit")]:
        arrow(250,82,x+73,57)
        box(x,10,146,47,label)
    return d
def bars(labels, values):
    d=Drawing(500,90)
    width=480/len(values)
    for i,(label,value) in enumerate(zip(labels,values)):
        x=i*width+8
        d.add(Rect(x,24,width-16,max(1,value*49),fillColor=teal,strokeColor=None))
        d.add(String(x+(width-16)/2,10,str(label),fontSize=8,textAnchor="middle",fillColor=gray))
        d.add(String(x+(width-16)/2,27+value*49,f"{value*100:.1f}%",fontSize=8,textAnchor="middle",fillColor=navy))
    return d

p("Jevlet\n", "TitleJ")
p("How one shared encoder turns text and questions into typed decisions", "DeckJ")
p("A practical guide to the implementation, training process, probability outputs, and real predictions from the selected checkpoint.")
story.append(pipeline())
p("Three useful answers, one interface", "H2J")
p("<b>Choice:</b> choose a banking intent from the supplied options.<br/><b>Score:</b> assign an English review a sentiment level from 1 to 5.<br/><b>Yes/no:</b> decide whether a banking message concerns a queried intent.")
p("The context is written once in a packed sequence. Each question has its own marker, so its answer can be read from the corresponding encoder representation. The heads return probabilities and a typed answer rather than generated prose.")
p("Authors: Hitanshu Gala, Jay Dalvi, Satishkumar Varma, and Richa Sharma<br/>Dwarkadas J. Sanghvi College of Engineering, Mumbai, India", "SmallJ")
page()

heading(2,"From a request to a packed sequence")
p("1. Validate the request", "H2J")
p("The implementation accepts one to three questions. Every question needs a unique ID, a type, and nonempty text. Choice questions need two to ten distinct options. Literal special-token strings in user input are removed so they cannot create extra markers.")
p("2. Add markers and tokenize", "H2J")
p("[CLS] [Q] Which intent? [OPT] lost card<br/>[OPT] cash withdrawal [Q] Is this about lost card?<br/>[SEP] I lost my debit card. [SEP]", "MonoJ")
p("This schematic shows two questions, not the full model input for every request. The tokenizer turns words into subword token IDs and records the positions of each [Q] and [OPT] marker.")
p("3. Preserve the question block", "H2J")
p("The input budget is 256 tokens. Questions and options are retained first; only the trailing context is truncated if necessary. The response includes <font face='Courier'>context_tokens_dropped</font> so the application can see when evidence was shortened. A question block that leaves no room for context is rejected.")
p("4. Encode the sequence once", "H2J")
p("DistilBERT-base has six transformer layers, hidden width 768, and 12 attention heads. Self-attention allows question and option markers to interact with the shared text. The implementation gathers marker vectors from the final hidden states. It does not independently encode every option.")
table([["Vector read", "Head receiving it"],["Each [OPT] representation", "Shared linear option scorer"],
       ["[Q] representation for sentiment", "Five-class sentiment head"],
       ["[Q] representation for a predicate", "Binary head"]],[225,275])
p("Packed requests can save repeated encoder work. They also share the token budget: adding long questions can leave less space for context. Sharing a context is not a guarantee of lower latency than every conventional classifier.","SmallJ")
page()

heading(3,"How each head predicts")
p("Choice: compare only the supplied options", "H2J")
p("One linear layer maps each option-marker vector to a scalar logit. A mask excludes unused option slots. Softmax converts the candidate logits into probabilities that sum to one; the highest-probability option becomes the answer.")
p("p(option j) = exp(logit j / T) / sum exp(logit k / T)","MonoJ")
p("The result is relative to this shortlist. The training/evaluation shortlist contains the correct label; accuracy does not establish retrieval of the correct intent from all 77 possible labels.")
p("Sentiment: five levels and an expected score", "H2J")
p("The selected categorical head maps a question-marker vector to five logits. Softmax produces a probability for each level. The winning index plus one gives the discrete answer. The expected score is the probability-weighted average of levels 1 to 5.")
table([["Level", "Meaning"],["1", "Very negative"],["2", "Negative"],["3", "Neutral"],["4", "Positive"],["5", "Very positive"]],[100,400])
p("An earlier variant uses a scalar and four ordered thresholds to form cumulative probabilities. The five-class version directly optimizes class discrimination; it does not impose an ordering constraint on its logits.")
p("Yes/no: a probability of the predicate being true", "H2J")
p("A linear layer produces one logit. Sigmoid gives P(yes); P(no) is one minus that value. The higher probability determines the boolean answer. A confidence of 0.9 means the model assigns that answer 90% probability; it is not proof that the answer is correct.")
p("Temperature T is positive and fitted separately for each head on calibration data. It changes confidence. Positive temperature scaling preserves categorical and binary argmax decisions; it can change the ordinal head's winning level.","SmallJ")
page()

heading(4,"Training, validation, and calibration")
table([["Partition", "Banking texts", "Review texts", "Purpose"],["Training","7,993","8,531","Learn encoder and head weights"],
       ["Validation","999","550","Select checkpoint/settings"],["Calibration","1,000","550","Fit confidence temperatures"],
       ["Test","3,079","2,210","Report final quality"]],[87,95,94,224])
p("Learning from labels", "H2J")
p("Training derives intent choices, binary intent predicates, and review sentiment questions from public dataset labels. Cross-entropy supervises choice and categorical sentiment; binary cross-entropy supervises yes/no. A Brier penalty of weight 0.5 encourages useful probability distributions. Losses are averaged within each present head and then across those heads.")
p("Shared learning", "H2J")
p("Backpropagation updates the task heads and shared encoder. A change that helps sentiment can alter banking predictions because both tasks use the same backbone. GPU training uses mixed precision, gradient accumulation, AdamW, and gradient clipping; evaluation uses float32.")
p("What the expanded validation changes", "H2J")
p("Five fixed question views per validation text produce 7,745 requests and 17,735 decisions. Banking views vary wording, candidates, and queried predicates. Review views vary the wording of the same five-level sentiment task. Training also varies question wording. There are still only 1,549 unique validation contexts: repeated views are correlated, not new independent labels.")
p("Three refinement settings", "H2J")
p("The search continues from the previous five-class checkpoint using three seed/learning-rate settings, for at most three epochs each. Expanded-validation sentiment accuracy chooses the winner, subject to a one-percentage-point banking regression limit. The original checkpoint remains an eligible fallback. Candidate test scores are not used for ranking.")
p("Calibration is separate", "H2J")
p("After training, one temperature per head minimizes calibration-set negative log-likelihood. The test split is evaluated only after selection. Earlier test results have already been inspected, so this study remains exploratory and needs fresh-data confirmation.","SmallJ")
page()

heading(5,"Real predictions from the checkpoint")
p("These hand-written examples are qualitative demonstrations, not extra benchmark samples. Every probability below was obtained by running the selected Jevlet checkpoint on CUDA.","SmallJ")
bank=examples[0]
p("Banking: several questions share one message", "H2J")
p(f"<i>{bank['text']}</i>")
table([["Question", "Answer", "Confidence"]]+[[q["text"],str(a["answer"]),f"{100*a['confidence']:.1f}%"]
      for q,a in zip(bank["questions"],bank["prediction"]["answers"])],[295,120,85])
for ex in examples[1:]:
    a=ex['prediction']['answers'][0]
    p("Review: five-level sentiment", "H2J")
    p(f"<i>{ex['text']}</i>")
    p(f"Predicted level: <b>{a['answer']}</b> | Expected score: <b>{a['expected_score']:.2f}</b> | Confidence: <b>{100*a['confidence']:.1f}%</b>","SmallJ")
    story.append(bars(["1","2","3","4","5"],a['probabilities']))
p("A review can receive a confident wrong answer. Discrete accuracy, macro-F1, ordinal error, and calibration measure different aspects of prediction quality. Inspecting a few examples cannot replace held-out evaluation.","SmallJ")
page()

heading(6,"What the results establish")
table([["Selected checkpoint: calibrated test", "Measurement"],
       ["Banking shortlist accuracy",f"{100*metrics['choice']['accuracy']:.2f}%"],
       ["Sentiment accuracy",f"{100*metrics['score']['accuracy']:.2f}%"],
       ["Sentiment macro-F1",f"{metrics['score']['macro_f1']:.4f}"],
       ["Sentiment mean absolute error",f"{metrics['score']['mae']:.4f}"],
       ["Binary intent accuracy",f"{100*metrics['yesno']['accuracy']:.2f}%"]],[350,150])
p("What to look at in a response", "H2J")
p("<b>answer:</b> the chosen option, level, or boolean.<br/><b>probabilities:</b> uncertainty across the answer space.<br/><b>confidence:</b> the probability of the winning answer.<br/><b>expected_score:</b> a continuous sentiment estimate for score questions.<br/><b>context_tokens_dropped:</b> evidence removed by truncation.")
p("Where Jevlet is useful", "H2J")
p("A single compact checkpoint can answer several bounded questions over the same text, with a consistent probability interface. It supports application logic that needs typed decisions. The evidence covers known English banking intents and five-level review sentiment, rather than arbitrary natural-language reasoning.")
p("Where caution is needed", "H2J")
p("Unseen options, question paraphrases outside the tested templates, different languages, missing gold labels, and new domains are not established capabilities. A score question about banking sentiment was not supervised by the review labels. Refinements share an initial checkpoint, and repeated validation views do not establish cross-validation or independent training-seed robustness.")
p("Implementation map", "H2J")
p("<font face='Courier'>data.py</font>: sampling, markers, token packing.<br/><font face='Courier'>model.py</font>: shared encoder, typed heads, prediction.<br/><font face='Courier'>train.py</font>: learning and checkpoint selection.<br/><font face='Courier'>metrics.py</font>: calibration and quality measures.<br/><font face='Courier'>refine.py</font>: expanded-validation refinement search.","SmallJ")
p("Sources: the local Jevlet implementation and measured run artifacts; DistilBERT (arxiv.org/abs/1910.01108); BANKING77 (aclanthology.org/2020.nlp4convai-1.5/); SST (aclanthology.org/D13-1170/); temperature scaling (proceedings.mlr.press/v70/guo17a.html).", "SmallJ")

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#CCD8E0")); canvas.line(18*mm,17*mm,A4[0]-18*mm,17*mm)
    canvas.setFont("Helvetica",8); canvas.setFillColor(gray)
    canvas.drawString(18*mm,12*mm,"Jevlet | Model and prediction guide")
    canvas.drawRightString(A4[0]-18*mm,12*mm,str(doc.page))
SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,
                  topMargin=18*mm,bottomMargin=23*mm,title="Jevlet: How the model works and predicts",
                  author="Hitanshu Gala and Jay Dalvi").build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(OUT)
assert len(reader.pages)==6, f"Unexpected pagination: {len(reader.pages)}"
render=ROOT / "tmp/pdfs/jevlet_explainer"
render.mkdir(parents=True,exist_ok=True)
pdf=pdfium.PdfDocument(str(OUT))
for i in range(len(pdf)):
    pdf[i].render(scale=1.4).to_pil().save(render/f"page-{i+1}.png")
print(f"Created {OUT}; {len(reader.pages)} pages rendered for review")
