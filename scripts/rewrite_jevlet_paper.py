"""Render the current evidence-backed manuscript; --legacy restores the prior CUDA-only writer."""
import runpy
import sys
from pathlib import Path

if "--legacy" not in sys.argv:
    runpy.run_path(str(Path(__file__).with_name("update_multilabel_paper.py")), run_name="__main__")
    raise SystemExit
sys.argv.remove("--legacy")

"""Write the existing Jevlet-focused manuscript from completed measurements."""
import json
import argparse
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
ref=json.loads((ROOT/'results/jevlet_refinement.json').read_text())
assert ref.get('completed'), 'Refinement must complete before reporting its results'
comparison=json.loads((ROOT/'results/gpu_comparison_summary.json').read_text())
models=comparison['models']
test=ref['test']
c=test['calibrated']
path=ROOT/'paper/paper.tex'
old=path.read_text(encoding='utf-8-sig')
author=old[old.index(r'\author{'):old.index(r'\begin{document}')] if r'\IEEEauthorblockN' in old else ''
bib=old[old.index(r'\begin{thebibliography}'):old.index(r'\end{thebibliography}')+len(r'\end{thebibliography}')]
entries=dict(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})',bib,re.S))
order=['distilbert','bert','laya','coral','guo','brier','banking','sst','svm','roberta']
bib='\\begin{thebibliography}{10}\n'+'\n'.join('\\bibitem{'+k+'}'+entries[k].strip() for k in order)+'\n\\end{thebibliography}\n'

def pct(x): return f'{100*x:.2f}'
def f(x): return f'{x:.4f}'
labels={'jevlet_ordinal':'Jevlet (ordinal)','jevlet_softmax':'Jevlet (five-class)',
 'tfidf_logistic':'TF-IDF + logistic','tfidf_svm':'TF-IDF + linear SVM',
 'distilbert_classifier':'DistilBERT classifier','bert_classifier':'BERT-base classifier',
 'roberta_classifier':'RoBERTa-base classifier'}
quality=[]
for n,m in models.items():
    t=m['calibrated']
    quality.append(f"{labels[n]} & {pct(t['choice']['accuracy'])} & {pct(t['score']['accuracy'])} & {f(t['score']['macro_f1'])} & {f(t['score']['mae'])} & {pct(t['yesno']['accuracy'])} "+r'\\')
if ref['selected']!='initial':
    quality.append(f"Jevlet (refined, selected) & {pct(c['choice']['accuracy'])} & {pct(c['score']['accuracy'])} & {f(c['score']['macro_f1'])} & {f(c['score']['mae'])} & {pct(c['yesno']['accuracy'])} "+r'\\')
validation=[]
for n,item in [('Initial five-class',{'validation':ref['initial_validation']}),*[(n,v) for n,v in ref['candidates'].items()]]:
    v=item['validation']
    display=n.replace('_',r'\_')
    validation.append(f"{display} & {pct(v['choice']['accuracy'])} & {pct(v['score']['accuracy'])} & {pct(v['yesno']['accuracy'])} "+r'\\')
calrows=[]
for h,label in [('choice','Choice'),('score','Sentiment'),('yesno','Binary')]:
    raw=test['raw'][h]; after=c[h]
    calrows.append(f"{label} & {f(raw['ece'])} & {f(after['ece'])} & {f(raw['nll'])} & {f(after['nll'])} & {f(raw['brier'])} & {f(after['brier'])} "+r'\\')
lat=test['latency']['measurements']
def time(mode,size): return next(r for r in lat if r['mode']==mode and r['batch_size']==size)
latrows=[]
for mode,label in [('packed','Packed'),('sequential','Sequential'),('batched_questions','Batched questions')]:
    for size in (1,32):
        row=time(mode,size)
        latrows.append(f"{label} & {size} & {row['p50_forward_ms_per_request']:.2f} & {row['p50_pack_and_forward_ms_per_request']:.2f} "+r'\\')
chosen=ref['selected']
winner='the initial five-class checkpoint' if chosen=='initial' else chosen.replace('_',r'\_')
selected_validation=ref['initial_validation'] if chosen=='initial' else ref['candidates'][chosen]['validation']
oldsoft=models['jevlet_softmax']['calibrated']
delta=100*(c['score']['accuracy']-oldsoft['score']['accuracy'])
max_external=max((n for n in models if not n.startswith('jevlet')),key=lambda n:models[n]['calibrated']['score']['accuracy'])
external=models[max_external]['calibrated']
f1best=c['score']['macro_f1']>=max(m['calibrated']['score']['macro_f1'] for n,m in models.items() if not n.startswith('jevlet'))
f1claim=('The selected Jevlet has the highest sentiment macro-F1 among the evaluated external configurations.'
         if f1best else 'The external configurations and Jevlet have different strengths across the reported measures.')
report_change=(f"The selected checkpoint changes canonical test sentiment accuracy from {pct(oldsoft['score']['accuracy'])}\\% to {pct(c['score']['accuracy'])}\\%, a difference of {delta:+.2f} percentage points."
               if chosen!='initial' else 'No refinement candidate passed selection with higher validation sentiment accuracy, so the previous five-class checkpoint is retained.')

text=r'''\documentclass[conference]{IEEEtran}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb,booktabs,cite,url}
\usepackage[hidelinks]{hyperref}
\title{Jevlet: A Shared Encoder for Typed Text Decisions with Expanded Question Validation}
'''+author+r'''
\begin{document}
\maketitle
\begin{abstract}
\bfseries
Many text-processing applications need a bounded decision: a customer intent, a sentiment rating, or a boolean. Jevlet provides these outputs through one shared encoder, typed prediction heads, and explicit probabilities. Question and option markers allow up to three questions to share the same text context in one packed sequence. We describe the architecture, supervised objectives, calibration procedure, and an expanded validation protocol that evaluates five question views per source text. The validation set contains 1,549 unique contexts, yielding 7,745 requests and 17,735 decisions; repeated views test question wording and candidate variation without being treated as independent examples. A five-class sentiment head addresses poor intermediate-class coverage observed with the original ordinal head. Three further refinement settings are selected using validation sentiment accuracy with banking regression guardrails. The selected checkpoint obtains @CHOICE@\% banking shortlist accuracy, @SENT@\% five-level sentiment accuracy, sentiment macro-F1 of @F1@, and @BINARY@\% binary intent accuracy. We report test quality, probability calibration, and synchronized inference cost. Jevlet offers a compact typed interface for this bounded workload, while its performance outside the evaluated domains remains an open question.
\end{abstract}
\begin{IEEEkeywords}
\bfseries
Jevlet, shared encoder, typed text decisions, multi-task learning, sentiment analysis, probability calibration
\end{IEEEkeywords}

\section{Introduction}
A banking support system may need to classify a message, decide whether it concerns a particular issue, and attach a confidence value to each answer. A review-analysis system may instead require a sentiment level from one to five. In both cases, the downstream application expects a small, structured answer space. Returning a typed value and its probability distribution makes that expectation explicit.

Jevlet uses a shared DistilBERT encoder and three task-specific heads to answer these bounded questions. A request contains a context and one to three questions. Each question specifies whether its answer is an option, a five-level score, or a boolean. The question block and context form one token sequence, so several questions can reuse an encoder pass. The implementation exposes confidence, an expected sentiment score, and the number of context tokens removed by truncation.

The purpose of this work is to explain and evaluate Jevlet as a practical model for typed decisions. Our contributions are the marker-conditioned multi-question architecture, an empirical sentiment-head revision, and a validation protocol that varies wording and banking candidate sets. We also measure calibration and inference cost. These contributions concern a supervised English banking-and-review workload; they do not establish unrestricted question answering or domain-independent sentiment understanding.

\section{Background and Design Motivation}
DistilBERT provides a compact pretrained transformer representation through knowledge distillation~\cite{distilbert}, building on bidirectional language representations studied in BERT~\cite{bert}. Jevlet fine-tunes this representation jointly with its typed heads rather than treating the encoder as a frozen feature extractor. Its engineering design draws on Laya's typed interface and marker scoring~\cite{laya}, but Jevlet implements its own heads and packed multi-question sequences.

Ordered sentiment levels motivate cumulative ordinal regression~\cite{coral}. However, a mathematically ordered distribution can still assign too little probability to intermediate classes under an argmax decoder. We therefore examine a categorical five-class head as a direct alternative. Temperature scaling~\cite{guo} and Brier error~\cite{brier} are used to evaluate probability quality separately from answer correctness.

\section{Jevlet Architecture}
\subsection{Request Interface and Token Packing}
The input is a context $x$ with questions $Q=\{q_1,\ldots,q_m\}$, where $1\leq m\leq3$. Each question has a unique identifier, a type, and nonempty text. Choice questions additionally supply two to ten distinct options. The packed representation is
\begin{equation}
[\mathrm{CLS}]\;B(Q)\;[\mathrm{SEP}]\;x\;[\mathrm{SEP}],
\end{equation}
where $B(Q)$ contains a \texttt{[Q]} marker before each question and a \texttt{[OPT]} marker before every choice option. Both markers are added to the tokenizer vocabulary and learned during fine-tuning.

The sequence has a 256-token budget. The question block is retained first, and the context is truncated only if necessary. An oversized question block that leaves no context room is rejected. Literal reserved-token strings in input text are removed before tokenization to prevent accidental extra markers. Padding masks hide unused sequence positions. Marker positions are recorded during packing so the output heads can gather the correct vectors.

\subsection{Shared Encoder and Marker Representations}
DistilBERT-base supplies six transformer layers, hidden dimension 768, and twelve attention heads. Self-attention contextualizes the question and option markers jointly with the text. If $H$ is the encoder output, Jevlet reads $h_{q_i}$ from the marker for question $i$ and $h_{o_{ij}}$ from the marker for its option $j$. The heads are linear projections of these vectors.

Several questions share the context tokens and the encoder computation. This is a concrete implementation property rather than evidence of general reasoning: the model learns from the task labels and question forms present during supervision. Packing also creates a tradeoff, because longer questions and option lists reduce the remaining context budget.

\subsection{Choice and Binary Heads}
The choice scorer maps every option representation to a scalar,
\begin{equation}
z_{ij}=w_c^\top h_{o_{ij}}+b_c,\qquad
p_{ij}=\frac{\exp(z_{ij}/T_c)}{\sum_{k\in O_i}\exp(z_{ik}/T_c)}.
\end{equation}
Only supplied options participate in the softmax. Unused option slots are masked, and the maximum-probability option is returned as the typed answer. Its probability is relative to the current shortlist.

The binary head reads a question marker and produces
\begin{equation}
p_i(\mathrm{yes})=\sigma((w_b^\top h_{q_i}+b_b)/T_b).
\end{equation}
The no probability is its complement. The predicted answer is a boolean, and confidence is the probability of that predicted answer. Training predicates refer to known banking intents; the interface alone does not establish understanding of arbitrary predicates.

\subsection{Five-Level Sentiment Head}
The categorical head maps the sentiment question marker to five logits:
\begin{equation}
\boldsymbol{p}_i=\operatorname{softmax}((W_s h_{q_i}+b_s)/T_s).
\end{equation}
The levels correspond to very negative, negative, neutral, positive, and very positive review sentiment. The discrete answer is $1+\arg\max_j p_{ij}$, and the expected score is $\sum_{j=1}^{5}j p_{ij}$. The latter retains uncertainty that a single discrete level hides.

The original head instead learns a scalar $s$ and four ordered thresholds $\beta_1<\cdots<\beta_4$, using softplus increments to preserve order. With $u_k=\sigma((s-\beta_k)/T_s)$, the probabilities are $(1-u_1,u_1-u_2,u_2-u_3,u_3-u_4,u_4)$. Narrow threshold spacing can leave intermediate levels weakly represented under argmax. The categorical revision changes both the parameterization and the supervised sentiment loss, so their effects cannot be separated by this ablation alone.

\section{Learning and Expanded Validation}
\subsection{Data and Supervision}
BANKING77 supplies 77 banking intent labels~\cite{banking}. SST-5 supplies English review sentiment labels~\cite{sst}. Contexts are partitioned before questions are derived. Case-folded, whitespace-normalized duplicate contexts are removed with test taking precedence; the manifest records 15 within-split and 11 cross-split removals. Retained counts appear in Table~\ref{tab:data}.

\begin{table}[t]
\caption{Unique source texts; unchanged across the Jevlet experiments}
\label{tab:data}\centering
\begin{tabular}{lrrr}\toprule
Partition & Banking & Reviews & Total\\\midrule
Training & 7,993 & 8,531 & 16,524\\
Validation & 999 & 550 & 1,549\\
Calibration & 1,000 & 550 & 1,550\\
Test & 3,079 & 2,210 & 5,289\\\bottomrule
\end{tabular}\end{table}

Banking choice questions include the true intent and one to nine sampled alternatives. Yes/no labels are sampled positive with probability 0.5. In packed requests, queried labels occur among the supplied options, removing a simple option-presence shortcut. Training samples choice-only, binary-only, paired, and three-question recipes, with deterministic epoch-wise variation in candidates and question order. The fixed canonical evaluation request has one choice and two binary questions per banking text, or one sentiment question per review.

The choice task evaluates a shortlist containing the correct label, not retrieval over all 77 labels. Binary prevalence is approximately balanced by construction. Banking texts do not supply sentiment labels, and review texts do not supply banking intent labels.

\subsection{Objectives and Initial Training}
Choice and categorical sentiment use cross-entropy; binary prediction uses binary cross-entropy. The ordinal sentiment variant uses binary cross-entropy over four cumulative targets. Every question also contributes a Brier penalty $0.5\sum_j(p_j-y_j)^2$. Losses are averaged within each represented head, then across those heads. All heads update the same encoder, so sentiment changes can affect banking performance.

The initial runs use seed 42, AdamW, learning rate $3\times10^{-5}$, weight decay 0.01, 10\% warm-up, gradient clipping at one, batch size eight, and two-step accumulation. They train for at most six epochs, with patience two on validation loss. The categorical variant is chosen using uncalibrated validation sentiment accuracy. Both variants begin from the same pinned pretrained encoder revision.

\subsection{Five Question Views and Refinement Search}
The original validation partition was already a collection of 1,549 texts, not a single example. Its review question used one canonical template. We expand its question coverage to five fixed views per source text. Banking views vary question phrasing, candidate sets, and queried predicates; review views use five phrasings of the same sentiment task. This yields 4,995 banking requests and 2,750 review requests: 7,745 requests and 17,735 decisions overall.

Views share their source text and target labels. They are correlated measurements and are not counted as new independent contexts. This protocol is neither five-fold cross-validation nor a larger source dataset. Its purpose is to reduce dependence on one prompt template and one banking shortlist sample.

Further training starts from the earlier five-class checkpoint and samples the same five question templates during training. Three predefined seed/learning-rate settings are tried: $(42,5\times10^{-6})$, $(43,10^{-5})$, and $(44,2\times10^{-5})$. Each has a three-epoch maximum, patience two, and otherwise retains the initial optimizer settings. Within a candidate, the checkpoint maximizes expanded-validation sentiment accuracy. Across candidates, selection uses that accuracy subject to choice and binary accuracy each remaining within one percentage point of the initial checkpoint on expanded validation. The initial checkpoint remains eligible as a fallback.

These are three refinement settings sharing an initialization, not three independent from-scratch estimates of model quality. Fixed validation and canonical test sampling use request seed 42. Test scores are not used to rank the refinement candidates. The selected checkpoint is evaluated after validation selection.

\subsection{Calibration, Metrics, and Execution}
One positive temperature per head minimizes negative log-likelihood (NLL) on the separate calibration partition. Fitting uses L-BFGS with a bounded log-temperature for numerical stability. Positive temperature scaling preserves categorical and binary argmax predictions, while the ordinal decoder can change its winning class.

We report accuracy for all heads, intent macro-F1, sentiment macro-F1 and mean absolute error (MAE), and binary AUROC. Probability quality uses 15-bin top-label expected calibration error (ECE), NLL, and summed multiclass Brier error. A model's winning probability is its confidence; it is not a correctness guarantee.

Learned-model training, validation/test forward passes, and temperature optimization use CUDA. CPU routines handle tokenization, summaries, and serialization. The device is an NVIDIA GeForce RTX 4060 Laptop GPU with 8~GiB VRAM. Device memory, utilization, temperature, and power are sampled every ten seconds. GPU form factor affects throughput, power, and memory constraints; it does not intrinsically change the learning objective or guarantee a different accuracy.

\section{Results and Discussion}
\subsection{Jevlet Quality and Validation Selection}
The original ordinal Jevlet attains 35.20\% calibrated sentiment accuracy, while the earlier five-class Jevlet reaches 52.22\%. Its sentiment macro-F1 also rises from 0.2770 to 0.5141. This revision improves intermediate-level coverage, but banking shortlist accuracy changes from 97.04\% to 95.45\%, and binary accuracy changes from 97.99\% to 96.83\%. Shared learning therefore makes the head revision a task-specific tradeoff rather than a universal improvement.

Table~\ref{tab:validation} reports the expanded-validation measurements. The selected checkpoint is @WINNER@. Binary accuracy across the five validation views changes from @OLDVBINARY@\% to @NEWVBINARY@\%. This exposes sensitivity to question wording that a canonical-template evaluation can hide. Because the same five forms are sampled in refinement training, the result shows adaptation to known forms rather than unrestricted paraphrase generalization. @CHANGE@ Its calibrated canonical test results are @CHOICE@\% shortlist accuracy, @SENT@\% sentiment accuracy, sentiment macro-F1 @F1@, sentiment MAE @MAE@, and @BINARY@\% binary accuracy. The test comprises 3,079 choice, 2,210 sentiment, and 6,158 binary decisions.

\begin{table}[t]
\caption{Expanded-validation accuracy (\%); five correlated views per text. Seed and learning rate identify refinement settings.}
\label{tab:validation}\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrr}\toprule
Checkpoint & Choice & Sentiment & Binary\\\midrule
@VALIDATION@
\bottomrule\end{tabular}\end{table}

\subsection{External Comparisons}
To place Jevlet's measured quality in context, Table~\ref{tab:comparison} includes GPU-trained TF-IDF logistic and squared-hinge linear classifiers~\cite{svm}, as well as conventional DistilBERT, BERT, and RoBERTa classifiers~\cite{roberta}. They use the same source partitions and canonical test requests. Context classifiers read text alone and produce 77 intent logits, five sentiment logits, and 77 binary predicate logits. Supplied label IDs select shortlist and binary outputs; they do not interpret arbitrary question text.

TF-IDF uses training-fitted unigrams/bigrams, sublinear term frequency, and at most 20,000 features. Its linear classifiers are optimized with CUDA PyTorch operations, not a dedicated library SVM solver. The logistic objective includes the same Brier penalty; the SVM uses squared one-vs-rest hinge loss without that term. Linear models have a 40-epoch maximum, batch size 256, learning rate 0.01, and patience five. Transformer classifiers have the initial six-epoch budget. Checkpoints use validation loss, and temperatures use separate calibration data.

\begin{table*}[t]
\caption{Calibrated canonical test results. Accuracies are percentages. Jevlet refinement has additional tuning exposure; the table is not an equal-compute comparison.}
\label{tab:comparison}\centering\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrr}\toprule
Model & Choice \% & Sentiment \% & Sentiment F1 & Sentiment MAE & Binary \%\\\midrule
@QUALITY@
\bottomrule\end{tabular}\end{table*}

@F1CLAIM@ Its typed interface combines these decisions in one checkpoint and supports packed questions. The highest external sentiment accuracy is @EXTACC@\%, obtained by @EXTNAME@. A larger exact-level accuracy and a larger macro-F1 can favor different models because class coverage matters. The table therefore supports specific strengths rather than a claim that Jevlet wins every task.

Source contexts and label decisions are matched, but inputs and optimization differ. Jevlet shares 256 tokens between questions and context, whereas the context classifiers allocate that budget to text alone and use fixed label identities. Tokenization, truncation, task objectives, and tuning budgets differ. Jevlet's additional refinement cannot be treated as a controlled encoder-only ablation. The original comparator runs use one seed; their scores do not establish optimally tuned baselines.

\subsection{Calibration and Confidence}
Table~\ref{tab:calibration} reports the selected Jevlet before and after temperature fitting. Discrete categorical predictions remain unchanged by temperature scaling, so any categorical accuracy change is attributable to the trained checkpoint rather than to this post-processing step. ECE, NLL, and Brier error measure different aspects of confidence, and a temperature fitted to calibration-set NLL need not minimize every test metric.

\begin{table*}[t]
\caption{Selected Jevlet probability quality before (B) and after (A) calibration; lower values are better.}
\label{tab:calibration}\centering\small
\begin{tabular}{lrrrrrr}\toprule
Head & ECE B & ECE A & NLL B & NLL A & Brier B & Brier A\\\midrule
@CALROWS@
\bottomrule\end{tabular}\end{table*}

\subsection{Packed Prediction Cost}
Table~\ref{tab:timing} measures the selected checkpoint on fixed three-question banking requests. Packed mode uses one sequence per context. Sequential mode uses three passes; batched-question mode sends separate question sequences together. All three modes retain the same context tokens. Timing includes probability construction, five warm-up iterations, twenty measured repetitions, and CUDA synchronization. Batch-32 times are amortized milliseconds per request, not individual response latency.

\begin{table}[t]
\caption{Selected Jevlet CUDA float32 median inference time (ms/request). Complete time includes packing and transfer.}
\label{tab:timing}\centering\footnotesize
\begin{tabular}{lrrr}\toprule
Mode & Batch & Forward & Complete\\\midrule
@LATROWS@
\bottomrule\end{tabular}\end{table}

For batch one, packed forward time is @PACK@~ms and sequential forward time is @SEQ@~ms. This matched workload demonstrates the effect of avoiding repeated encoding within Jevlet. It does not isolate encoder arithmetic from probability selection, establish service throughput under concurrency, or prove a speedup over every context-only classifier. Preprocessing, sequence length, padding, and thermal conditions influence application-level time.

\section{Limitations and Future Work}
Expanded validation increases coverage of known question templates and banking candidates, not the number of independent contexts. Five paraphrases of one label task do not establish broad prompt understanding. The refinement settings share the same earlier checkpoint, and their seed/learning-rate pairs vary together, so they cannot separate either factor or quantify independent training-seed uncertainty.

Earlier test results were inspected during sentiment diagnosis. Although the refinement search ranks candidates only on validation data, the study remains exploratory. Fresh held-out data, repeated from-scratch runs, and a separately tuned comparator protocol are needed for stronger conclusions. No accuracy gain is guaranteed by simply adding validation requests.

Banking choices contain the true label, and binary queries are approximately balanced. Production shortlists may omit the intent or have a different query prevalence. Sentiment supervision comes only from English reviews; bank-domain sentiment, unseen labels, other languages, and out-of-domain confidence are not established. Multiple questions and repeated validation views share contexts and are correlated. Future work should test these shifts, quantify class-specific errors, and examine task balancing that improves sentiment without sacrificing banking quality.

\section{Conclusion}
Jevlet provides typed choice, sentiment, and binary predictions through one shared encoder and explicit probability outputs. Marker-based packing lets several questions reuse the same text representation. The five-class head addresses a practical weakness of the earlier ordinal sentiment decoder, and expanded validation checks several question forms without confusing repeated views with new labeled examples. The selected checkpoint reaches @SENT@\% sentiment accuracy and macro-F1 @F1@ alongside @CHOICE@\% banking shortlist and @BINARY@\% binary accuracy. @F1CLAIM@ Its value is supported by those measured task strengths and a unified interface; universal superiority is not established. Fresh-data evaluation and stronger task balancing remain necessary to assess how well these benefits generalize.

'''+bib+r'\end{document}'
values={'CHOICE':pct(c['choice']['accuracy']),'SENT':pct(c['score']['accuracy']),
 'F1':f(c['score']['macro_f1']),'BINARY':pct(c['yesno']['accuracy']),'MAE':f(c['score']['mae']),
 'WINNER':winner,'CHANGE':report_change,'VALIDATION':'\n'.join(validation),
 'OLDVBINARY':pct(ref['initial_validation']['yesno']['accuracy']),
 'NEWVBINARY':pct(selected_validation['yesno']['accuracy']),
 'QUALITY':'\n'.join(quality),'F1CLAIM':f1claim,'EXTACC':pct(external['score']['accuracy']),
 'EXTNAME':labels[max_external],'CALROWS':'\n'.join(calrows),'LATROWS':'\n'.join(latrows),
 'PACK':f"{time('packed',1)['p50_forward_ms_per_request']:.2f}",
 'SEQ':f"{time('sequential',1)['p50_forward_ms_per_request']:.2f}"}
for key,value in values.items(): text=text.replace('@'+key+'@',value)
# The Experiment 10 manual specifies elsarticle-template-harv.tex.
body=text[text.index(r'\begin{abstract}'):]
body=body.replace('\\bfseries\n','')
body=body.replace(r'\begin{IEEEkeywords}',r'\begin{keyword}').replace(r'\end{IEEEkeywords}',r'\end{keyword}')
body=body.replace('Jevlet, shared encoder, typed text decisions, multi-task learning, sentiment analysis, probability calibration',
                  r'Jevlet \sep shared encoder \sep typed text decisions \sep multi-task learning \sep sentiment analysis \sep probability calibration')
body=body.replace(r'\end{keyword}',r'\end{keyword}'+'\n'+r'\end{frontmatter}',1)
body=body.replace(r'\cite{',r'\citep{')
body=body.replace(r'\centering\small\setlength{\tabcolsep}{5pt}',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}')
body=body.replace(r'Model & Choice \% & Sentiment \% & Sentiment F1 & Sentiment MAE & Binary \%',
                  r'Model & Choice \% & Sent. \% & Sent. F1 & MAE & Binary \%')
author_year={'distilbert':'Sanh et al.(2019)','bert':'Devlin et al.(2019)','laya':'Mukkunnoth and Convai Innovations(n.d.)',
 'coral':'Cao et al.(2020)','guo':'Guo et al.(2017)','brier':'Brier(1950)',
 'banking':'Casanueva et al.(2020)','sst':'Socher et al.(2013)','svm':'Cortes and Vapnik(1995)','roberta':'Liu et al.(2019)'}
for key,display in author_year.items():
    body=body.replace(r'\bibitem{'+key+'}',r'\bibitem['+display+']{'+key+'}')
body=body.replace(r'\section{Background and Design Motivation}',r'\section{Literature Review}')
body=body.replace(r'\section{Jevlet Architecture}',r'\section{Methodology and System Architecture}')
body=body.replace(r'\subsection{Request Interface and Token Packing}',r'\subsection{Preprocessing and Request Packing}')
body=body.replace(r'\subsection{Data and Supervision}',r'\subsection{Dataset Description and Supervision}')
body=body.replace(r'\subsection{Objectives and Initial Training}',r'\subsection{Mathematical Objectives and Initial Training}')
body=body.replace(r'\subsection{Five Question Views and Refinement Search}',r'\subsection{Model Tuning Optimization and Regularization}')
diagram=r'''
\begin{figure}[t]
\centering
\begin{tikzpicture}[node distance=1.25cm, every node/.style={draw,rounded corners,align=center,font=\small,minimum height=0.8cm}]
\node (input) {Context and typed questions};
\node (tokens) [below of=input] {Tokenization and marker packing};
\node (encoder) [below of=tokens] {Shared DistilBERT encoder};
\node (choice) [below of=encoder,xshift=-3.3cm] {Choice head\\option probabilities};
\node (score) [below of=encoder] {Sentiment head\\five probabilities};
\node (binary) [below of=encoder,xshift=3.3cm] {Binary head\\yes/no probabilities};
\draw[->] (input)--(tokens); \draw[->] (tokens)--(encoder);
\draw[->] (encoder)--(choice); \draw[->] (encoder)--(score); \draw[->] (encoder)--(binary);
\end{tikzpicture}
\caption{Jevlet's system architecture. Marker representations feed typed heads after one packed encoder pass.}
\label{fig:architecture}
\end{figure}
'''
body=body.replace(r'\subsection{Choice and Binary Heads}',diagram+'\n'+r'\subsection{Choice and Binary Heads}')
prediction=r'''
\subsection{Making Predictions}
At inference time, the tokenizer records question and option marker positions, the encoder contextualizes the packed sequence, and the heads produce logits. Calibrated probabilities determine the typed answer. The response returns the question identifier, answer, probability vector, winning confidence, and context truncation count. Sentiment also returns its expected score. These fields allow an application to inspect uncertainty or defer low-confidence decisions; a confidence threshold is not validated as a production safety rule in this study.
'''
body=body.replace(r'\subsection{Calibration, Metrics, and Execution}',prediction+'\n'+r'\subsection{Calibration, Metrics, and Execution}')
chart_names=['Ordinal Jevlet','Five-class Jevlet','Refined Jevlet','TF-IDF logistic','TF-IDF SVM','DistilBERT','BERT-base','RoBERTa-base']
chart_metrics=[models['jevlet_ordinal']['calibrated'],models['jevlet_softmax']['calibrated'],c,
               *[models[n]['calibrated'] for n in ('tfidf_logistic','tfidf_svm','distilbert_classifier','bert_classifier','roberta_classifier')]]
charts=[]
for key,title,maxval in [('accuracy','Sentiment accuracy (percent)',60),('macro_f1','Sentiment macro-F1',.6)]:
    coordinates=' '.join(f"({100*m['score'][key] if key=='accuracy' else m['score'][key]:.4f},{i})" for i,m in enumerate(chart_metrics))
    charts.append(r'''\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[xbar,width=0.82\textwidth,height=7cm,xmin=0,xmax='''+str(maxval)+r''',
ytick={0,1,2,3,4,5,6,7},yticklabels={'''+','.join(chart_names)+r'''},
y dir=reverse,xlabel={'''+title+r'''},bar width=9pt,enlarge y limits=0.1,
nodes near coords,nodes near coords align={horizontal},
every node near coord/.append style={font=\scriptsize,/pgf/number format/fixed,
/pgf/number format/precision='''+str(2 if key=='accuracy' else 4)+r''',/pgf/number format/fixed zerofill},tick label style={font=\small},
xmajorgrids=true,grid style={gray!20}]
\addplot[fill=teal!65,draw=teal!80!black] coordinates {'''+coordinates+r'''};
\end{axis}\end{tikzpicture}
\caption{'''+title+r''' on the shared canonical test set. Additional refinement tuning and differing model inputs prevent an equal-compute architectural claim.}
\end{figure}''')
body=body.replace(r'\subsection{Calibration and Confidence}','\n'.join(charts)+'\n'+r'\subsection{Calibration and Confidence}')
text=r'''\documentclass[preprint,12pt,authoryear]{elsarticle}
% Based on the elsarticle-template-harv.tex supplied with Experiment 10.
\usepackage{amsmath,amssymb,booktabs,url}
\usepackage{tikz,pgfplots}
\pgfplotsset{compat=1.18}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\journal{Deep Learning Laboratory Mini Project}
\begin{document}
\begin{frontmatter}
\title{Jevlet: A Shared Encoder for Typed Text Decisions with Expanded Question Validation}
\author[org]{Hitanshu Gala}
\author[org]{Jay Dalvi}
\author[org]{Satishkumar Varma}
\author[org]{Richa Sharma}
\address[org]{Department of Information Technology, Dwarkadas Jivanlal Sanghvi College of Engineering, Mumbai, India}
'''+body
parser=argparse.ArgumentParser()
parser.add_argument('--format',choices=['ieee','elsevier'],default='ieee')
if parser.parse_args().format=='ieee':
    ieee_body=body.replace(r'\begin{keyword}',r'\begin{IEEEkeywords}').replace(r'\end{keyword}',r'\end{IEEEkeywords}')
    ieee_body=ieee_body.replace(r'\end{frontmatter}','').replace(r'\sep',',').replace(r'\citep{',r'\cite{')
    ieee_body=re.sub(r'\\bibitem\[[^\]]*\]',lambda _:r'\bibitem',ieee_body)
    ieee_body=ieee_body.replace(r'\begin{abstract}',r'\begin{abstract}'+'\n'+r'\bfseries',1)
    ieee_body=ieee_body.replace(r'\begin{IEEEkeywords}',r'\begin{IEEEkeywords}'+'\n'+r'\bfseries',1)
    # Give the architecture full-width space; keep result plots in one column.
    architecture_start=ieee_body.index(r'\begin{figure}[t]')
    architecture_end=ieee_body.index(r'\end{figure}',architecture_start)+len(r'\end{figure}')
    architecture=ieee_body[architecture_start:architecture_end].replace(r'\begin{figure}',r'\begin{figure*}').replace(r'\end{figure}',r'\end{figure*}')
    ieee_body=ieee_body[:architecture_start]+architecture+ieee_body[architecture_end:]
    ieee_body=ieee_body.replace(r'width=0.82\textwidth,height=7cm',r'width=0.67\columnwidth,height=5.2cm')
    ieee_body=ieee_body.replace('xmin=0,xmax=60,','xmin=0,xmax=70,').replace('xmin=0,xmax=0.6,','xmin=0,xmax=0.7,')
    ieee_body=ieee_body.replace(r'tick label style={font=\small}',r'tick label style={font=\footnotesize}')
    text=r'''\documentclass[conference]{IEEEtran}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb,booktabs,cite,url}
\usepackage{tikz,pgfplots}
\pgfplotsset{compat=1.18}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\title{Jevlet: A Shared Encoder for Typed Text Decisions with Expanded Question Validation}
\makeatletter
\newcommand{\linebreakand}{\end{@IEEEauthorhalign}\hfill\mbox{}\par\vspace{1em}\mbox{}\hfill\begin{@IEEEauthorhalign}}
\makeatother
\author{
\IEEEauthorblockN{\normalfont Hitanshu Gala}
\IEEEauthorblockA{\textit{Department of Information Technology}\\
\textit{Dwarkadas Jivanlal Sanghvi}\\
\textit{College of Engineering}\\
Mumbai, India}
\and
\IEEEauthorblockN{\normalfont Jay Dalvi}
\IEEEauthorblockA{\textit{Department of Information Technology}\\
\textit{Dwarkadas Jivanlal Sanghvi}\\
\textit{College of Engineering}\\
Mumbai, India}
\linebreakand
\IEEEauthorblockN{\normalfont Satishkumar Varma}
\IEEEauthorblockA{\textit{Department of Information Technology}\\
\textit{Dwarkadas Jivanlal Sanghvi}\\
\textit{College of Engineering}\\
Mumbai, India}
\and
\IEEEauthorblockN{\normalfont Richa Sharma}
\IEEEauthorblockA{\textit{Department of Information Technology}\\
\textit{Dwarkadas Jivanlal Sanghvi}\\
\textit{College of Engineering}\\
Mumbai, India}
}
\begin{document}
\maketitle
'''+ieee_body
assert not re.search('@[A-Z]+@',text)
assert text.count('NVIDIA GeForce RTX 4060 Laptop GPU')==1
path.write_text(text+'\n',encoding='utf-8')
print('Updated existing Jevlet manuscript:',path)
