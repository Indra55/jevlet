"""Generate README and paper tables using measured, completed runs only."""

import json

from . import ROOT


def number(value):
    return "—" if value is None else f"{value:.4f}"


def tex_escape(text):
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def generate_report():
    results = [json.loads(path.read_text()) for path in sorted((ROOT / "results").glob("*.json"))]
    results = [r for r in results if "calibrated" in r]
    headings = "| Run | Model | Head | Accuracy | F1 / MAE / AUROC | ECE | NLL | Brier |"
    markdown = [headings, "|---|---|---|---:|---:|---:|---:|---:|"]
    paper_rows = []
    for result in results:
        for head, scores in result["calibrated"].items():
            secondary = scores.get({"choice": "macro_f1", "score": "mae", "yesno": "auroc"}[head])
            values = [scores["accuracy"], secondary, scores["ece"], scores["nll"], scores["brier"]]
            markdown.append(f"| {result['run']} | {result['model']} | {head} | " +
                            " | ".join(number(v) for v in values) + " |")
            if not result["smoke"]:
                paper_rows.append(" & ".join([tex_escape(result["run"] + "/" + result["model"]), head] +
                                  ["--" if v is None else f"{v:.4f}" for v in values]) + " \\\\")
    if not results:
        markdown.append("| not run | — | — | — | — | — | — | — |")
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text()
        start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if start in text and end in text:
            before, remaining = text.split(start, 1)
            _, after = remaining.split(end, 1)
            readme.write_text(before + start + "\n\n" + "\n".join(markdown) + "\n\n" + end + after)
    generated = ROOT / "paper/generated"
    generated.mkdir(parents=True, exist_ok=True)
    table = ["\\begin{table*}[t]", "\\centering", "\\scriptsize",
             "\\caption{Measured test results after calibration. Secondary metrics: macro-F1 (choice), MAE (score), AUROC (yes/no).}",
             "\\begin{tabular}{llrrrrr}", "\\hline",
             "Run/model & Head & Accuracy & Secondary & ECE & NLL & Brier \\\\", "\\hline"]
    table.extend(paper_rows or ["\\multicolumn{7}{c}{Full experiments have not been run.} \\\\"])
    table += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    (generated / "results.tex").write_text("\n".join(table) + "\n")
    latency_rows = []
    for result in results:
        if result["smoke"]:
            continue
        for row in (result.get("latency") or {}).get("measurements", []):
            latency_rows.append(" & ".join([
                tex_escape(result["run"] + "/" + result["model"]),
                tex_escape(row.get("mode", "task baseline")), tex_escape(row["device"]),
                str(row["batch_size"]), f"{row['p50_pack_and_forward_ms_per_request']:.2f}",
            ]) + " \\\\")
    latency_table = ["\\begin{table*}[t]", "\\centering", "\\scriptsize",
        "\\caption{Measured median preprocessing and scoring time per request in milliseconds. Batch 32 is amortized throughput.}",
        "\\begin{tabular}{lllrr}", "\\hline",
        "Run/model & Layout & Device & Batch & ms/request \\\\", "\\hline"]
    latency_table.extend(latency_rows or ["\\multicolumn{5}{c}{Full latency experiments have not been run.} \\\\"])
    latency_table += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    (generated / "latency.tex").write_text("\n".join(latency_table) + "\n")
    # Aggregate only identical experiment settings, varying training seed alone.
    from collections import defaultdict
    import numpy as np
    groups = defaultdict(list)
    for result in results:
        if not result["smoke"]:
            settings = dict(result["config"])
            settings.pop("seed", None)
            settings.pop("device", None)
            groups[(result["model"], json.dumps([settings, result["data"]], sort_keys=True))].append(result)
    summary = []
    for (name, _), group in groups.items():
        if len({r["seed"] for r in group}) < 3:
            continue
        # Duplicate invocations of the same seed do not count as repetitions.
        group = list({r["seed"]: r for r in group}.values())
        for head in group[0]["calibrated"]:
            values = [r["calibrated"][head]["accuracy"] for r in group]
            summary.append(f"{tex_escape(name)} & {head} & {len(group)} & " +
                           f"{np.mean(values):.4f} $\\pm$ {np.std(values, ddof=1):.4f} \\\\")
    if summary:
        (generated / "summary.tex").write_text("\n".join([
            "\\begin{table}[t]", "\\centering", "\\scriptsize",
            "\\caption{Across-seed accuracy: mean and sample standard deviation.}",
            "\\begin{tabular}{llrl}", "\\hline", "Model & Head & Seeds & Accuracy \\\\",
            "\\hline", *summary, "\\hline", "\\end{tabular}", "\\end{table}"]) + "\n")
    else:
        (generated / "summary.tex").write_text("% Fewer than three matched seeds; no aggregate claims.\n")
    full_main = [r for r in results if r["model"] == "jevlet" and not r["smoke"]]
    figures = []
    if full_main:
        chosen = full_main[-1]
        stem = f"{chosen['run']}_{chosen['model']}"
        for head in chosen["calibrated"]:
            for kind in ("reliability", "risk_coverage"):
                filename = ROOT / "results/figs" / stem / f"{head}_{kind}.pdf"
                if filename.exists():
                    figures += ["\\begin{figure}[t]", "\\centering",
                        "\\includegraphics[width=0.95\\columnwidth]{../results/" + filename.relative_to(ROOT / "results").as_posix() + "}",
                        "\\caption{" + tex_escape(head + " " + kind.replace("_", " ")) + "}", "\\end{figure}"]
    (generated / "figures.tex").write_text("\n".join(figures) + "\n")
    print(f"Generated results tables from {len(results)} completed evaluations; smoke excluded from paper.")
