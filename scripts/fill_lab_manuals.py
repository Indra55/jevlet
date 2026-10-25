"""Fill four retained manuals with progressively completed, valid LaTeX sources."""
import copy
import hashlib
import json
from pathlib import Path
import re
import zipfile
from lxml import etree as E

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/docx';OUT.mkdir(parents=True,exist_ok=True)
LATEX=ROOT/'output/latex';LATEX.mkdir(parents=True,exist_ok=True)
paper=(ROOT/'paper/paper.tex').read_text()
ref=json.loads((ROOT/'results/jevlet_refinement.json').read_text())
comparison=json.loads((ROOT/'results/gpu_comparison_summary.json').read_text())
assert ref['completed']
start=paper[:paper.index(r'\section{Introduction}')]
refs=paper[paper.index(r'\begin{thebibliography}'):paper.index(r'\end{thebibliography}')+len(r'\end{thebibliography}')]
sections={}
for m in re.finditer(r'\\section\{([^}]+)\}(.*?)(?=\\section\{|\\begin\{thebibliography\})',paper,re.S):
    sections[m.group(1)]=r'\section{'+m.group(1)+'}'+m.group(2)
architecture=sections['Methodology and System Architecture']
learning=sections['Learning and Expanded Validation']
results=sections['Results and Discussion']
intro=sections['Introduction']
literature=sections['Literature Review']
dataset=learning[learning.index(r'\subsection{Dataset'):learning.index(r'\subsection{Mathematical')]
initial_training=learning[learning.index(r'\subsection{Mathematical'):learning.index(r'\subsection{Model Tuning')]
metrics=learning[learning.index(r'\subsection{Calibration, Metrics'):]
chosen_metrics=ref['test']['calibrated']
five=comparison['models']['jevlet_softmax']['calibrated']
def quality(t):
    return r'''\section{Performance Evaluation and Result Analysis}
The canonical test contains 3,079 choice, 2,210 sentiment, and 6,158 binary decisions.
\begin{table}[ht]\centering
\begin{tabular}{lrr}\toprule
Head & Accuracy (percent) & Secondary metric\\\midrule
'''+f"Choice & {100*t['choice']['accuracy']:.2f} & macro-F1 {t['choice']['macro_f1']:.4f} "+r'\\'+'\n'+f"Sentiment & {100*t['score']['accuracy']:.2f} & macro-F1 {t['score']['macro_f1']:.4f} "+r'\\'+'\n'+f"Binary & {100*t['yesno']['accuracy']:.2f} & AUROC {t['yesno']['auroc']:.4f} "+r'\\'+r'''
\bottomrule\end{tabular}
\caption{Calibrated Jevlet test quality.}
\end{table}
Confidence is evaluated separately from correctness. The shortlist contains the gold label, and approximately balanced binary questions are derived from known banking intents. Sentiment labels come from English reviews.
'''
conclusions={
7:'The dataset, preprocessing pipeline, and shared DistilBERT architecture were selected for Jevlet. Typed heads provide a practical design for intent, sentiment, and binary decisions.',
8:'The five-class Jevlet model was trained and evaluated on held-out data. Sentiment reached 52.22% accuracy, while calibration and task-specific metrics exposed remaining weaknesses.',
9:'Question augmentation improved binary robustness across five validation forms. Canonical banking accuracy improved, but sentiment accuracy changed from 52.22% to 51.72%, showing a multi-task tradeoff.',
10:'The Elsevier-format paper presents Jevlet with measured tables, graphs, and references. Its strengths are task-specific, while fresh data and independent training runs are needed for stronger conclusions.'}
def finish(body,conclusion):
    # Keep bibliography items only when cited by the stage source.
    used=set()
    for group in re.findall(r'\\cite(?:p|t)?\{([^}]+)\}',body): used.update(group.split(','))
    items=re.findall(r'(\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}.*?)(?=\\bibitem|\\end\{thebibliography\})',refs,re.S)
    bibliography='\\begin{thebibliography}{10}\n'+'\n'.join(text.strip() for text,key in items if key in used)+'\n\\end{thebibliography}\n'
    return body+'\n\\section{Conclusion}\n'+conclusion.replace('%',r'\%')+'\n'+bibliography+r'\end{document}'+'\n'
source7=finish(start+intro+literature+architecture+r'\section{Dataset Selection and Preprocessing}'+'\n'+dataset,conclusions[7])
source8=finish(start+intro+literature+architecture+r'\section{Dataset Training and Testing}'+'\n'+dataset+initial_training+metrics+quality(five),conclusions[8])
without_external=results[:results.index(r'\subsection{External Comparisons}')]+results[results.index(r'\subsection{Calibration and Confidence}'):]
source9=finish(start+intro+literature+architecture+learning+without_external,conclusions[9])
stages={7:source7,8:source8,9:source9,10:paper}
# Stage abstracts describe the matching stage rather than later measurements.
abstracts={7:'Jevlet is designed as a shared DistilBERT encoder with typed choice, five-level sentiment, and binary heads. BANKING77 and SST-5 supply supervised labels. This stage documents the literature, problem statement, source data, preprocessing, mathematical model, and marker-based system architecture.',
8:'Jevlet was trained and evaluated on separate training, validation, calibration, and test partitions. The five-class model reaches 52.22 percent sentiment accuracy, 95.45 percent banking shortlist accuracy, and 96.83 percent binary intent accuracy. Probability calibration and task-specific evaluation identify strengths and remaining limitations.',
9:'This stage studies Jevlet regularization, question-template augmentation, parameter tuning, calibration, and prediction. Five validation views per context support checkpoint selection. The selected refinement improves banking test metrics and known-template binary robustness, while canonical sentiment accuracy changes from 52.22 to 51.72 percent.'}
steps={
7:['Literature survey and problem identification: bounded decisions require typed probability outputs rather than free-form text.',
   'Data gathering and cleaning: BANKING77 and SST-5 are deduplicated and split before question generation.',
   'Model selection: a shared DistilBERT encoder feeds choice, five-class sentiment, and binary heads.'],
8:['Literature survey and problem identification lead to typed banking and review decisions; BANKING77 and SST-5 supply labels.',
   'Preprocessing removes normalized duplicate contexts and partitions texts before deriving questions.',
   'Model selection uses a shared DistilBERT encoder with choice, five-class sentiment, and binary heads.',
   'Training and testing use disjoint source partitions and GPU model execution with held-out checkpoint selection.',
   'Performance evaluation reports accuracy, macro-F1, MAE, AUROC, ECE, NLL, and Brier error.'],
9:['Literature review, problem identification, and data collection are completed for typed banking and review decisions.',
   'Normalized duplicates are removed and source texts are split before question generation.',
   'A shared DistilBERT model and five-class sentiment head are selected.',
   'Initial training uses AdamW, warm-up, gradient accumulation, and GPU mixed precision.',
   'Held-out accuracy, macro-F1, MAE, AUROC, and calibration metrics evaluate performance.',
   'Three predefined seed and learning-rate settings refine the existing checkpoint using five validation views per text.',
   'Encoder dropout, weight decay, early stopping, gradient clipping, and the Brier penalty regularize training; predictions expose typed answers and probabilities.',
   'Result analysis documents improved banking/template robustness and the canonical sentiment tradeoff.'],
10:['Literature review identifies the typed decision problem, and BANKING77/SST-5 provide supervised source data.',
    'Preprocessing normalizes duplicate checks, splits source texts, and packs question and option markers.',
    'Jevlet uses a shared DistilBERT encoder and typed choice, five-class sentiment, and binary heads.',
    'Initial training fine-tunes the shared encoder and heads on GPU with AdamW and validation-based checkpointing.',
    'Task-specific accuracy, macro-F1, MAE, AUROC, and confidence metrics evaluate performance.',
    'Refinement compares three predefined seed/learning-rate settings on expanded validation, with banking regression guardrails.',
    'Regularization uses dropout, weight decay, early stopping, gradient clipping, and a Brier penalty; prediction returns typed answers and confidence.',
    'Analysis reports canonical test results, question-template robustness, calibration, and limitations.',
    'Measured results are presented in data, validation, quality, calibration, and timing tables, plus architecture and comparison graphs.',
    'The final LaTeX source follows the supplied Elsevier author-year format and includes all required sections, four authors, conclusions, and references.']}

NS='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def q(n): return '{'+NS+'}'+n
def text_of(p): return ''.join(p.itertext()) if p.tag!=q('p') else ''.join(p.xpath('.//w:t/text()',namespaces={'w':NS}))
def paragraph(text,code=False,bold=False):
    p=E.Element(q('p')); pr=E.SubElement(p,q('pPr'))
    E.SubElement(pr,q('pStyle')).set(q('val'),'Normal')
    E.SubElement(pr,q('jc')).set(q('val'),'left')
    spacing=E.SubElement(pr,q('spacing'));spacing.set(q('after'),'0' if code else '120');spacing.set(q('line'),'240');spacing.set(q('lineRule'),'auto')
    if code:
        E.SubElement(pr,q('keepNext')).set(q('val'),'0')
        E.SubElement(pr,q('keepLines')).set(q('val'),'0')
        E.SubElement(pr,q('widowControl')).set(q('val'),'0')
    r=E.SubElement(p,q('r'));rp=E.SubElement(r,q('rPr'))
    fonts=E.SubElement(rp,q('rFonts'))
    for attr in ('ascii','hAnsi','cs','eastAsia'): fonts.set(q(attr),'Consolas' if code else 'Times New Roman')
    E.SubElement(rp,q('sz')).set(q('val'),'18' if code else '24')
    E.SubElement(rp,q('color')).set(q('val'),'000000')
    if bold:E.SubElement(rp,q('b'))
    t=E.SubElement(r,q('t'));t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=text
    return p

files={7:'DL EXP 7 Manual 1.docx',8:'DL EXP 8 Manual (1).docx',9:'DL EXP 9 Manual (1).docx',10:'DL EXP 10 Manual.docx'}
for n,name in files.items():
    src=Path('C:/Users/Jay/Downloads')/name
    source=stages[n]
    if n<10: source=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstracts[n]+'\n'+r'\end{abstract}',source,flags=re.S)
    latex_path=LATEX/f'exp{n}_jevlet.tex';latex_path.write_text(source,encoding='utf-8')
    with zipfile.ZipFile(src) as z:
        original={p:z.read(p) for p in z.namelist()}
        root=E.fromstring(original['word/document.xml'])
        body=root.find(q('body'));children=list(body)
        a=next(i for i,p in enumerate(children) if text_of(p).strip()=='EXERCISE')
        b=next(i for i,p in enumerate(children) if text_of(p).strip()=='CONCLUSIONS:')
        for p in children[a+1:b]:body.remove(p)
        insert_at=list(body).index(children[a])+1
        content=[paragraph('Completed project work',bold=True)]
        content += [paragraph(f'Step {i+1}  '+step) for i,step in enumerate(steps[n])]
        content += [paragraph('LaTeX implementation',bold=True)]
        if n == 10:
            E.SubElement(content[-1].find(q('pPr')),q('pageBreakBefore')).set(q('val'),'1')
        content += [paragraph(line,code=True) for line in source.splitlines()]
        content += [paragraph('')]
        for i,p in enumerate(content):body.insert(insert_at+i,p)
        index=list(body).index(children[b])+1
        while index<len(body) and body[index].tag==q('p') and not text_of(body[index]).strip():body.remove(body[index])
        body.insert(index,paragraph(conclusions[n]))
        # Repair source encoding artifacts without changing instructions/metadata.
        for t in root.iter(q('t')):
            if t.text and '\ufffd' in t.text:t.text=t.text.replace('SDN\ufffdIoT','SDN-IoT').replace('\ufffd','"')
        # The source top margin cannot contain its repeated 0.76-inch logo.
        # Give it room on later pages while retaining the header/image parts.
        body.find(q('sectPr')).find(q('pgMar')).set(q('top'),'2160')
        for p in body:
            if p.tag==q('p') and text_of(p).strip()=='LaTeX Template Link':
                pp=p.find(q('pPr'))
                if pp is None:pp=E.SubElement(p,q('pPr'))
                E.SubElement(pp,q('pageBreakBefore')).set(q('val'),'1')
        while len(body)>1 and body[-2].tag==q('p') and not text_of(body[-2]).strip():
            if body[-2].xpath('.//w:drawing',namespaces={'w':NS}):break
            body.remove(body[-2])
        rewritten=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
        out=OUT/f'DL_Exp_{n}_Jevlet_Filled.docx'
        with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as dest:
            for item in z.infolist():dest.writestr(item,rewritten if item.filename=='word/document.xml' else original[item.filename])
    with zipfile.ZipFile(out) as z:
        assert set(z.namelist())==set(original)
        assert all(z.read(p)==data for p,data in original.items() if p!='word/document.xml')
    print(n,out,'LaTeX lines:',len(source.splitlines()),'Preserve-only package parts unchanged')
