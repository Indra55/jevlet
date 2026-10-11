"""Rewrite the existing standalone manuscript from completed, audited run artifacts."""
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "results/gpu_comparison_summary.json").read_text())
if not summary.get("completed"):
    raise RuntimeError("All seven comparisons must finish before writing the final paper")
models = summary["models"]
selected = summary["selected_jevlet"]
labels = {"jevlet_ordinal": "Jevlet (ordinal)", "jevlet_softmax": "Jevlet (softmax)",
          "tfidf_logistic": "TF-IDF + logistic", "tfidf_svm": "TF-IDF + linear SVM",
          "distilbert_classifier": "DistilBERT classifier", "bert_classifier": "BERT-base classifier",
          "roberta_classifier": "RoBERTa-base classifier"}
names = list(labels)
target = ROOT / "paper/paper.tex"
current = target.read_text(encoding="utf-8-sig")
author = current[current.index("\\author{"):current.index("\\begin{document}")]
bibliography = current[current.index("\\begin{thebibliography}"):current.index("\\end{thebibliography}")]
bibliography = bibliography.replace("{8}", "{10}", 1)
bibliography += r'''
\bibitem{roberta} Y. Liu et al., ``RoBERTa: A robustly optimized BERT pretraining approach,'' \emph{arXiv preprint arXiv:1907.11692}, 2019. [Online]. Available: \url{https://arxiv.org/abs/1907.11692}
\bibitem{svm} C. Cortes and V. Vapnik, ``Support-vector networks,'' \emph{Machine Learning}, vol. 20, pp. 273--297, 1995, doi: 10.1007/BF00994018.
\end{thebibliography}
'''
bib_entries = dict(re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})",
                             bibliography, flags=re.S))
bib_order = ["bert", "distilbert", "roberta", "svm", "laya", "coral", "guo", "brier", "banking", "sst"]
bibliography = "\\begin{thebibliography}{10}\n" + "\n".join(
    "\\bibitem{" + key + "}" + bib_entries[key].strip() for key in bib_order) + "\n\\end{thebibliography}\n"

def fmt(value):
    return f"{value:.4f}"

def acc(value):
    return f"{100*value:.2f}"

def timing(name, size):
    rows = models[name]["latency"]["measurements"]
    return next(r for r in rows if r["batch_size"] == size and
                (not name.startswith("jevlet") or r["mode"] == "packed"))

quality_rows, resource_rows = [], []
for name in names:
    m = models[name]
    c = m["calibrated"]
    quality_rows.append(f"{labels[name]} & {acc(c['choice']['accuracy'])} & {fmt(c['choice']['macro_f1'])} & "
                        f"{acc(c['score']['accuracy'])} & {fmt(c['score']['macro_f1'])} & {fmt(c['score']['mae'])} & "
                        f"{acc(c['yesno']['accuracy'])} & {fmt(c['yesno']['auroc'])} " + r"\\")
    resource_rows.append(f"{labels[name]} & {m['parameters']/1e6:.2f} & {m['epochs_completed']} & "
                         f"{m['training_seconds']/60:.2f} & {m['peak_gpu_bytes']/2**30:.2f} & "
                         f"{timing(name,1)['p50_forward_ms_per_request']:.2f} & "
                         f"{timing(name,32)['p50_forward_ms_per_request']:.2f} " + r"\\")

chosen = models[selected]
c = chosen["calibrated"]
ordinal = models["jevlet_ordinal"]["calibrated"]["score"]
softmax = models["jevlet_softmax"]["calibrated"]["score"]
diff = 100 * (softmax["accuracy"] - ordinal["accuracy"])
best_sentiment = max(names, key=lambda n: models[n]["calibrated"]["score"]["accuracy"])
best_choice = max(names, key=lambda n: models[n]["calibrated"]["choice"]["accuracy"])
calibration_rows = []
for head, label in [("choice", "Choice"), ("score", "Sentiment"), ("yesno", "Binary")]:
    raw, after = chosen["raw"][head], c[head]
    calibration_rows.append(f"{label} & {fmt(raw['ece'])} & {fmt(after['ece'])} & {fmt(raw['nll'])} & "
                             f"{fmt(after['nll'])} & {fmt(raw['brier'])} & {fmt(after['brier'])} " + r"\\")

# Paired context bootstrap: uncertainty from held-out examples, not training seeds.
predictions = {}
for name in names:
    p = json.loads((ROOT / f"results/predictions/gpu_compare_{name}.json").read_text())
    rows = sorted([r for r in p["calibrated"] if r["type"] == "score"], key=lambda r: r["source_id"])
    predictions[name] = rows
reference = predictions[selected]
ids = [r["source_id"] for r in reference]
truth = np.array([r["target"] for r in reference])
assert all([r["source_id"] for r in predictions[n]] == ids for n in names)
assert all(np.array_equal([r["target"] for r in predictions[n]], truth) for n in names)
correct = {n: np.array([r["predicted"] for r in predictions[n]]) == truth for n in names}
rng = np.random.default_rng(42)
bootstrap = {n: [] for n in names}
for _ in range(2000):
    ix = rng.integers(len(truth), size=len(truth))
    for n in names:
        bootstrap[n].append(float((correct[selected][ix].astype(float) - correct[n][ix]).mean()))
intervals = {n: np.percentile(bootstrap[n], [2.5,97.5]).tolist() for n in names if n != selected}
(ROOT / "results/gpu_comparison_bootstrap.json").write_text(json.dumps({"reference":selected,
    "samples":len(truth), "resamples":2000, "paired_sentiment_accuracy_difference_95_percentile_intervals":intervals},indent=2))
diagnostic = {n: np.bincount([r["predicted"] for r in predictions[n]], minlength=5).tolist()
              for n in ("jevlet_ordinal", "jevlet_softmax")}
interval_rows = [f"{labels[n]} & {100*(correct[selected].mean()-correct[n].mean()):+.2f} & "
                 f"[{100*intervals[n][0]:+.2f}, {100*intervals[n][1]:+.2f}] " + r"\\"
                 for n in names if n not in ("jevlet_ordinal", "jevlet_softmax")]

text = r'''\documentclass[conference]{IEEEtran}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb,booktabs,cite,url}
\usepackage[hidelinks]{hyperref}
\title{Jevlet: Comparing Typed Shared-Encoder Decisions with Classical and Transformer Text Classifiers}
''' + author + r'''
\begin{document}
\maketitle
\begin{abstract}
\bfseries
Text-processing applications often need an intent, a sentiment level, or a boolean rather than a generated response. Jevlet answers these bounded questions using a shared encoder and explicit probability outputs. This study retrains its original ordinal sentiment variant and a five-class softmax variant, and compares them with GPU-trained TF-IDF linear classifiers and conventional DistilBERT, BERT, and RoBERTa classifiers. All systems use the same BANKING77 and SST-5 source partitions and the same derived evaluation questions. Variant selection uses validation sentiment accuracy, while temperatures are fitted on a separate calibration partition. The selected Jevlet variant attains @CHOICE@\% banking shortlist accuracy, @SENTIMENT@\% five-level sentiment accuracy, and @BINARY@\% binary intent accuracy. Replacing the ordinal sentiment head changes test accuracy by @DIFF@ percentage points. The highest sentiment accuracy in this run is @BESTSENT@\%, obtained by @BESTMODEL@. We also report calibration, synchronized GPU inference time, training cost, and peak allocated memory. These single-seed measurements describe a practical quality--cost tradeoff rather than establishing a universal ranking of model families.
\end{abstract}
\begin{IEEEkeywords}
\bfseries
text classification, multi-task learning, sentiment analysis, DistilBERT, probability calibration, GPU inference
\end{IEEEkeywords}

\section{Introduction}
A support application may need to choose a banking intent and answer two follow-up predicates about the same message. A review-analysis application may need a sentiment level from one to five. Each decision has a small answer space, so a classifier can expose the decision directly without requiring an application to parse generated prose. Probability outputs also make it possible to evaluate confidence separately from answer correctness.

Jevlet implements this interface with a shared DistilBERT encoder, question and option markers, and three typed heads. Several questions can occupy one sequence with a common text context. However, a useful interface is not itself evidence of better classification. A conventional text classifier can also reuse a context representation and answer known label queries efficiently. The relevant question is therefore how Jevlet compares with simpler linear classifiers and other pretrained encoders on the same tasks.

An earlier exploratory run exposed a weakness in the ordinal sentiment head: its discrete predictions omitted intermediate levels. This motivated a five-class alternative. We retrain both variants from the pretrained encoder, select between them on validation data, and compare their quality and computational cost with five external classifier configurations. The contribution is a reproducible empirical comparison of typed marker scoring and closed-label context classification, together with a sentiment-head ablation. It is not a claim of a new general-purpose language model or unrestricted question answering.

\section{Related Work}
BERT uses bidirectional transformer representations that can be fine-tuned for downstream language tasks~\cite{bert}. DistilBERT provides a smaller distilled encoder~\cite{distilbert}. RoBERTa studies changes to BERT pretraining and provides another pretrained representation family~\cite{roberta}. These architectures motivate our transformer comparators, but published benchmark scores are not substituted for measurements on our data.

TF-IDF provides a sparse lexical representation that is useful for a lightweight baseline. We pair it with regularized logistic classifiers and linear squared-hinge classifiers inspired by the support-vector-machine framework~\cite{svm}. To satisfy the GPU execution requirement, the learned linear models are optimized in PyTorch. These are explicitly implemented classifier configurations, not claims to reproduce a particular library solver's optimum.

Laya's typed interface and marker scoring informed Jevlet's engineering design~\cite{laya}. Ordered cumulative prediction is motivated by rank-consistent ordinal regression~\cite{coral}. Temperature scaling~\cite{guo} and Brier error~\cite{brier} provide complementary tools for studying probability quality. Their inclusion does not establish calibration outside the evaluated distribution.

\section{Models and Objectives}
\subsection{Jevlet and the Sentiment Ablation}
Jevlet encodes a question block followed by the shared context:
\begin{equation}
[\mathrm{CLS}]\;B(Q)\;[\mathrm{SEP}]\;x\;[\mathrm{SEP}].
\end{equation}
Each question begins with a learned \texttt{[Q]} marker, and choice options begin with \texttt{[OPT]}. One encoder pass produces marker representations. A linear scorer reads each option marker for a choice question; a binary scorer reads a question marker for a yes/no question. Choice probabilities are a softmax over the supplied options, while a sigmoid produces the binary probability. Requests contain one to three questions, and choice questions contain two to ten distinct options.

The original sentiment head learns a scalar score $s$ and four ordered thresholds $\beta_1<\cdots<\beta_4$. Softplus increments enforce ordering. With a positive temperature $T_s$,
\begin{equation}
u_k=P(Y>k)=\sigma((s-\beta_k)/T_s),
\quad k=1,\ldots,4.
\end{equation}
The class probabilities are $(1-u_1,u_1-u_2,u_2-u_3,u_3-u_4,u_4)$. The most probable class supplies the discrete sentiment level. Because class probabilities are differences of cumulative probabilities, narrow learned threshold spacing can yield poorly represented intermediate classes under this decoder. The ordered model remains a valid distribution, but validity does not guarantee useful class coverage.

The alternative replaces this head with five independent logits,
\begin{equation}
\boldsymbol{p}_s=\operatorname{softmax}((W_s h_q+b_s)/T_s).
\end{equation}
This directly optimizes five-class discrimination but removes the explicit ordering constraint. Both heads return an expected score $\sum_{j=1}^{5}j p_j$. The choice and binary heads, marker packing, and encoder are otherwise unchanged.

\subsection{Conventional Context Classifiers}
The DistilBERT, BERT-base, and RoBERTa-base comparators encode the text alone and use its first-token representation with dropout and linear task heads. There are 77 intent logits, five sentiment logits, and 77 binary predicate logits. A supplied shortlist selects the corresponding intent logits before softmax, while a queried intent selects one binary logit. Thus, these comparators answer known label IDs; they do not interpret arbitrary question or option text. They share a context representation across the questions just as a conventional application could.

The TF-IDF comparators use word unigrams and bigrams, sublinear term frequency, L2-normalized features, and at most 20,000 features. The vocabulary and document-frequency statistics are fitted only on training contexts. They use the same three output-head dimensions and shortlist/query selection as the transformer classifiers. Only the current minibatch is densified for CUDA matrix operations. Feature extraction is performed on the CPU; fitting and learned-model inference run on CUDA.

These input designs differ meaningfully: Jevlet reads natural-language questions and candidate labels, whereas the context classifiers use fixed label identities. Transformer comparators have a 256-token context budget; Jevlet shares that budget with its question block. Model-specific tokenization and truncation therefore prevent a claim of identical token-level evidence. The experiment matches source contexts and label decisions, not complete internal computations.

\subsection{Loss and Calibration}
For Jevlet and the neural/logistic comparators, choice uses cross-entropy, binary prediction uses binary cross-entropy, and categorical sentiment uses cross-entropy. Ordinal sentiment uses binary cross-entropy over four cumulative targets. Each question also contributes $0.5\sum_j(p_j-y_j)^2$, where $y_j$ is a one-hot class target. Losses are averaged within each head represented in a batch, then across those heads.

The SVM configuration uses regularized squared one-vs-rest hinge losses on candidate classes and signed binary margins, without the Brier term. It is trained by minibatch AdamW rather than a dedicated dual solver. Its decision margins are converted to softmax or sigmoid distributions for evaluation. This conversion, followed by calibration, is a modeling choice; raw SVM margins are not intrinsically probabilities.

After checkpoint selection, one positive temperature per head is fitted on the calibration split by minimizing negative log-likelihood (NLL), using L-BFGS on CUDA. Temperatures rescale logits before probability construction. Confidence is the largest class probability. For softmax and binary heads, positive temperature scaling preserves discrete decisions. It can change an ordinal head's most probable level. We report both probability quality and discrete quality rather than assuming that calibration improves every measure.

\section{Experimental Protocol}
\subsection{Data and Matched Decisions}
BANKING77 supplies banking intent labels~\cite{banking}; SST-5 supplies five-level review sentiment~\cite{sst}. The retained source partitions are shown in Table~\ref{tab:data}. Contexts are split before questions are derived. Case-folded, whitespace-normalized duplicate contexts are removed with test taking precedence; the manifest records 15 within-split and 11 cross-split removals. Banking training data is divided approximately 80/10/10 into training, validation, and calibration. SST development data is divided equally between validation and calibration.

\begin{table}[t]
\caption{Retained source contexts shared by every configuration}
\label{tab:data}
\centering
\begin{tabular}{lrrr}\toprule
Partition & Banking & SST-5 & Total\\\midrule
Training & 7,993 & 8,531 & 16,524\\
Validation & 999 & 550 & 1,549\\
Calibration & 1,000 & 550 & 1,550\\
Test & 3,079 & 2,210 & 5,289\\\bottomrule
\end{tabular}
\end{table}

Banking choices contain the correct intent and 1--9 sampled alternatives. Both queried binary labels occur among the supplied options in packed requests, removing an option-presence shortcut. Binary positives are sampled with probability 0.5; the two predicates are sampled independently and may have equal truth values. Training varies the choice/binary packing recipe, candidates, and question order deterministically by epoch. The common request-generation seed gives identical per-epoch supervision; early stopping changes total exposure. Fixed evaluation requests contain one choice and two binary questions per banking context, and one sentiment question per review. Test counts are 3,079 choice, 6,158 binary, and 2,210 sentiment decisions.

The shortlist always includes the true label. Reported choice accuracy is therefore not all-77 intent detection or retrieval quality. Binary decision prevalence is approximately balanced by construction rather than representative of all production queries. SST sentiment labels do not provide banking sentiment supervision.

\subsection{Training, Selection, and Resources}
The study uses seed 42 and new checkpoints for every configuration. Both Jevlet variants start from the same pinned pretrained DistilBERT revision. Transformers train for at most six epochs using AdamW, learning rate $3\times10^{-5}$, weight decay 0.01, 10\% linear warm-up, gradient clipping at one, batch size eight, and two-step accumulation. CUDA mixed precision is enabled during training. Validation and timing use float32. Checkpoints minimize validation loss with patience two; the Jevlet variant is chosen using uncalibrated validation sentiment accuracy rather than test accuracy.

The linear configurations have a separate optimization budget: at most 40 epochs, batch size 256, learning rate 0.01, weight decay $10^{-4}$, and patience five. This accommodates their different scale and convergence behavior; it is not an equal-compute comparison or a comprehensive hyperparameter search. Their logistic objective includes the Brier term, so ``logistic'' denotes the classifier link rather than an unmodified maximum-likelihood solver.

All learned-model forward/backward passes, validation/test inference, and temperature fitting use CUDA, with no CPU fallback. Tokenization, TF-IDF extraction, serialization, and metric summaries use CPU routines. The experimental device is an NVIDIA GeForce RTX 4060 Laptop GPU with 8~GiB VRAM. A monitor samples device memory, utilization, temperature, and power every ten seconds. Peak allocated memory is separately recorded by PyTorch. Models run sequentially and release cached allocations between configurations.

Training times exclude initial feature/model preparation, downloads, calibration, and test evaluation. They include the training loop, validation, and best-checkpoint saving. Hardware power and thermal conditions may influence elapsed time; these measurements do not imply that the device form factor changes the learning objective or intrinsically lowers model quality.

\subsection{Metrics and Inference Timing}
Choice reports accuracy and macro-F1 over the 77 original intent IDs. Sentiment reports accuracy, macro-F1, discrete MAE, and expected-score MAE. Binary prediction reports accuracy and AUROC. Probability metrics are 15-bin top-label expected calibration error (ECE), NLL, and summed multiclass Brier error.

Timing uses the same fixed banking requests, five warm-up iterations, and twenty measured repetitions, with CUDA synchronization around each interval. Batch one uses one request; batch 32 uses 32 requests and reports amortized milliseconds per request. Jevlet uses packed execution, while context classifiers encode the context once and score the known labels. Forward timing includes probability construction; a separate scope includes fresh preprocessing and transfer. Initial loading and JSON serialization are excluded. Python output selection differs between implementations, and input lengths/tokenizers differ, so these are implementation-level workload measurements rather than isolated architectural speedups.

\section{Results and Discussion}
\subsection{Sentiment Head Selection}
The ordinal variant obtains @VALORD@\% validation sentiment accuracy, compared with @VALSOFT@\% for the categorical variant. The validation-selected configuration is @SELECTED@. Their calibrated test sentiment accuracies are @ORDACC@\% and @SOFTACC@\%, respectively, with MAEs of @ORDMAE@ and @SOFTMAE@. The softmax-minus-ordinal difference is @DIFF@ percentage points. This is one trained pair, so the difference should not be interpreted as an average over random initializations.

The ordinal model's predicted counts for levels one through five are @ORDCOUNTS@. The softmax model's corresponding counts are @SOFTCOUNTS@. Class coverage helps explain the aggregate sentiment results, but it is not a substitute for per-class accuracy. Replacing the head changes both the parameterization and the sentiment objective, so this ablation does not separate those two factors.

The improvement also has a multi-task cost. Choice accuracy changes from @ORDCHOICE@\% in ordinal Jevlet to @SOFTCHOICE@\% in softmax Jevlet, while binary accuracy changes from @ORDBINARY@\% to @SOFTBINARY@\%. Both variants share their encoder across tasks, so changing sentiment gradients can affect the banking heads. The validation selection rule prioritizes sentiment accuracy; it should not be interpreted as a universal preference for every application.

\begin{table*}[t]
\caption{Calibrated test comparison. Accuracies are percentages. Choice F1 is over original intent IDs; sentiment F1 is over five levels. Every row is newly trained with seed 42.}
\label{tab:quality}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}\toprule
Model & Choice \% & Choice F1 & Sent. \% & Sent. F1 & MAE & Binary \% & AUROC\\\midrule
@QUALITYROWS@
\bottomrule\end{tabular}
\end{table*}

\subsection{Comparison with External Classifiers}
Table~\ref{tab:quality} shows the complete test comparison. The highest sentiment accuracy is @BESTSENT@\%, achieved by @BESTMODEL@; the highest shortlist accuracy is @BESTCHOICE@\%, achieved by @BESTCHOICEMODEL@. The selected Jevlet configuration has sentiment macro-F1 @SENTF1@, MAE @SENTMAE@, and expected-score MAE @EXPECTEDMAE@. These measures distinguish exact-level correctness, coverage across classes, and ordinal distance.

The TF-IDF logistic and squared-hinge configurations achieve @LRCHOICE@\% and @SVMCHOICE@\% shortlist accuracy, respectively, with sentiment accuracies of @LRSENT@\% and @SVMSENT@\%. Their banking results show why lightweight lexical models belong in this comparison. Their sentiment results also illustrate that a model family can have different strengths across the two datasets. Choosing a system therefore requires the task-specific results and resource measurements together.

The context classifiers are genuine alternatives for these fixed datasets: they can reuse one encoded context and select known output labels without marker text. Jevlet's typed interface is therefore an engineering benefit rather than automatic evidence of superior quality or lower cost. A model can perform well on banking and poorly on sentiment, and a stronger sentiment encoder may require more parameters or training memory. Table~\ref{tab:resources} makes those costs explicit instead of collapsing the study into a single leaderboard score.

Table~\ref{tab:intervals} gives paired bootstrap intervals for the selected Jevlet's sentiment accuracy difference against external comparators. We resample the same 2,210 review contexts 2,000 times with seed 42. Positive differences favor Jevlet. These intervals reflect variation in sampled evaluation contexts conditional on the fitted models; they do not measure training-seed uncertainty or correct for the exploratory nature of the study.

\begin{table}[t]
\caption{Selected Jevlet minus comparator sentiment accuracy, in percentage points; paired 95\% percentile intervals}
\label{tab:intervals}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrr}\toprule
Comparator & Difference & Interval\\\midrule
@INTERVALROWS@
\bottomrule\end{tabular}
\end{table}

\subsection{Probability Quality}
Table~\ref{tab:calibration} reports the selected Jevlet's metrics before and after calibration. Calibration is evaluated on test data after temperature fitting on a disjoint calibration partition. The softmax variant's discrete predictions are invariant to positive temperature scaling, so any gain from its new sentiment head should not be attributed to calibration alone. NLL, Brier error, and ECE can change independently; a temperature chosen to minimize calibration-set NLL is not guaranteed to minimize test ECE.

\begin{table*}[t]
\caption{Selected Jevlet probability quality: before (B) and after (A) temperature fitting. Lower values are better.}
\label{tab:calibration}
\centering\small
\begin{tabular}{lrrrrrr}\toprule
Head & ECE B & ECE A & NLL B & NLL A & Brier B & Brier A\\\midrule
@CALIBRATIONROWS@
\bottomrule\end{tabular}
\end{table*}

\subsection{Training and Inference Cost}
\begin{table*}[t]
\caption{Measured resources and CUDA float32 median forward time. Parameter counts are millions; memory is peak allocated GiB. Batch-32 time is amortized ms/request. Times include the evaluated probability-selection implementation.}
\label{tab:resources}
\centering\small
\begin{tabular}{lrrrrrr}\toprule
Model & Parameters & Epochs & Train min & Peak GiB & Batch-1 ms & Batch-32 ms\\\midrule
@RESOURCEROWS@
\bottomrule\end{tabular}
\end{table*}

The selected Jevlet's median forward times are @LAT1@~ms for batch one and @LAT32@~ms per request for batch 32. Including preprocessing and transfer gives @FULL1@~ms and @FULL32@~ms, respectively. Its training interval is @TRAINMIN@ minutes, with peak allocated memory of @PEAK@~GiB. The table reports the complete configurations rather than assuming that encoder size alone determines runtime. Sparse lexical preprocessing, padding, question length, output selection, and hardware conditions all contribute to application-level cost.

\section{Limitations}
The study uses one random seed and one predefined optimization schedule per model family. It does not estimate run-to-run variability or establish best possible accuracy for BERT, RoBERTa, linear SVMs, or Jevlet. The original test set was inspected during earlier diagnosis, which motivated the sentiment ablation. Although checkpoint and variant selection in these runs use validation data, the final comparison is exploratory and should be confirmed on fresh held-out data or another dataset.

Source splits and evaluation decisions are shared, but tokenization, truncation, input representations, objectives, batch sizes, and optimization budgets differ as documented. These differences prevent a claim that every observed gap is caused solely by encoder architecture. The context comparators operate on known label IDs; Jevlet accepts option and question text, but unseen-label transfer and free-form question understanding are not evaluated.

Candidate sets contain the gold label, and binary queries are approximately balanced. Confidence may shift when a production shortlist omits the true intent, when query prevalence changes, or when text comes from another domain. Sentiment is evaluated only on English reviews with a fixed five-level rubric. Multiple banking decisions share a context and are correlated. Timing uses a small fixed workload rather than a concurrent service-load test. Downloaded weights and training data should be redistributed only under their applicable terms.

\section{Conclusion}
This study compares retrained typed Jevlet variants with GPU-trained lexical and transformer classifiers using a shared source-data and evaluation protocol. The categorical sentiment head provides a direct alternative to ordered cumulative scoring, and validation-based selection avoids choosing a variant from its test score. The measured results show where each configuration stands in exact classification, ordinal error, probability quality, and computational cost. Jevlet's practical role should be judged against those alternatives for the intended task, rather than inferred from its typed interface alone. Repeated seeds, stronger tuning, and fresh evaluation data are the next steps toward more general conclusions.

''' + bibliography + r'\end{document}'
values = {"CHOICE":acc(c["choice"]["accuracy"]), "SENTIMENT":acc(c["score"]["accuracy"]),
          "BINARY":acc(c["yesno"]["accuracy"]), "DIFF":f"{diff:+.2f}",
          "BESTSENT":acc(models[best_sentiment]["calibrated"]["score"]["accuracy"]), "BESTMODEL":labels[best_sentiment],
          "BESTCHOICE":acc(models[best_choice]["calibrated"]["choice"]["accuracy"]), "BESTCHOICEMODEL":labels[best_choice],
          "VALORD":acc(models["jevlet_ordinal"]["validation"]["score"]["accuracy"]),
          "VALSOFT":acc(models["jevlet_softmax"]["validation"]["score"]["accuracy"]), "SELECTED":labels[selected],
          "ORDACC":acc(ordinal["accuracy"]), "SOFTACC":acc(softmax["accuracy"]),
          "ORDMAE":fmt(ordinal["mae"]), "SOFTMAE":fmt(softmax["mae"]),
          "ORDCHOICE":acc(models["jevlet_ordinal"]["calibrated"]["choice"]["accuracy"]),
          "SOFTCHOICE":acc(models["jevlet_softmax"]["calibrated"]["choice"]["accuracy"]),
          "ORDBINARY":acc(models["jevlet_ordinal"]["calibrated"]["yesno"]["accuracy"]),
          "SOFTBINARY":acc(models["jevlet_softmax"]["calibrated"]["yesno"]["accuracy"]),
          "LRCHOICE":acc(models["tfidf_logistic"]["calibrated"]["choice"]["accuracy"]),
          "SVMCHOICE":acc(models["tfidf_svm"]["calibrated"]["choice"]["accuracy"]),
          "LRSENT":acc(models["tfidf_logistic"]["calibrated"]["score"]["accuracy"]),
          "SVMSENT":acc(models["tfidf_svm"]["calibrated"]["score"]["accuracy"]),
          "ORDCOUNTS":", ".join(map(str,diagnostic["jevlet_ordinal"])),
          "SOFTCOUNTS":", ".join(map(str,diagnostic["jevlet_softmax"])),
          "QUALITYROWS":"\n".join(quality_rows), "RESOURCEROWS":"\n".join(resource_rows),
          "CALIBRATIONROWS":"\n".join(calibration_rows), "INTERVALROWS":"\n".join(interval_rows),
          "SENTF1":fmt(c["score"]["macro_f1"]), "SENTMAE":fmt(c["score"]["mae"]),
          "EXPECTEDMAE":fmt(c["score"]["expected_score_mae"]),
          "LAT1":f"{timing(selected,1)['p50_forward_ms_per_request']:.2f}",
          "LAT32":f"{timing(selected,32)['p50_forward_ms_per_request']:.2f}",
          "FULL1":f"{timing(selected,1)['p50_pack_and_forward_ms_per_request']:.2f}",
          "FULL32":f"{timing(selected,32)['p50_pack_and_forward_ms_per_request']:.2f}",
          "TRAINMIN":f"{chosen['training_seconds']/60:.2f}", "PEAK":f"{chosen['peak_gpu_bytes']/2**30:.2f}"}
for key,value in values.items():
    text=text.replace(f"@{key}@",value)
assert not re.search(r"@[A-Z0-9]+@",text)
assert text.count("NVIDIA GeForce RTX 4060 Laptop GPU") == 1
target.write_text(text + "\n", encoding="utf-8")
print(f"Updated {target}; validation-selected variant: {selected}")
