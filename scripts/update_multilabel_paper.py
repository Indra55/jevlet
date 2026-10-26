"""Render the current paper from a checked-in snapshot of measured run artifacts."""
import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
TITLE = 'Jevlet: A Typed Shared Encoder for Single-Label and Multi-Label Text Decisions'
AUTHORS = ('Hitanshu Gala', 'Jay Dalvi', 'Satishkumar Varma', 'Richa Sharma')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def capture(root):
    from jevlet.metrics import score_records
    snapshot = {'schema_version': 1, 'scope': 'measured supplied-candidate decisions; no full-catalogue claim', 'runs': {}}
    for run in ('multilabel-emotions', 'multilabel-banking'):
        directory = root / 'checkpoints' / run
        training = read(directory / 'training.json')
        manifest = read(directory / 'data.json')
        status = read(directory / 'run_status.json')
        if status['status'] != 'complete':
            raise ValueError(f'{run} is incomplete')
        head = 'banking_multilabel' if run.endswith('banking') else 'multilabel'
        result_path = root / 'results' / f'{run}_jevlet.json'
        result = read(result_path)
        comparison_path = directory / f'{head}_results.json'
        comparison = read(comparison_path)
        baseline_path = directory / f'{head}_baseline.json'
        files = [directory / n for n in ('jevlet.json', 'training.json', 'data.json', 'calibration.json', 'run_status.json')]
        files += [result_path, comparison_path, baseline_path]
        cfg = training['config']
        settings = {k: cfg[k] for k in ('seed', 'device', 'cpu_threads', 'max_length', 'train', 'eval')}
        settings['initial_checkpoint'] = 'checkpoints/' + Path(cfg['initial_checkpoint']).name
        counts = {s: dict(Counter(r['domain'] for r in rows)) for s, rows in manifest['splits'].items()}
        main_rows = [r for r in manifest['splits']['test'] if r['domain'] == head]
        snapshot['runs'][run] = {
            'head': head, 'settings': settings, 'versions': training['versions'],
            'specification': read(directory / 'jevlet.json'),
            'calibration': read(directory / 'calibration.json'), 'partition_counts': counts,
            'sources': manifest['sources'], 'history': training['history'],
            'training_seconds': training['training_seconds'], 'status': {k: v for k, v in status.items() if k != 'data'},
            'metrics': result['calibrated'], 'raw_metrics': result['raw'],
            'baseline': {k: v for k, v in read(baseline_path).items() if k != 'predictions'},
            'test_cardinality_counts': dict(Counter(len(r['labels']) for r in main_rows)),
            'weights_sha256': digest(directory / 'model.safetensors'),
            'artifact_sha256': {str(p.relative_to(root)): digest(p) for p in files},
        }
    old = read(root / 'results/predictions/multilabel-emotions_jevlet.json')['calibrated']
    new = read(root / 'results/predictions/multilabel-banking_jevlet.json')['calibrated']
    old_by_key = {(r['source_id'], r['id']): r for r in old}
    matched = []
    for row in new:
        if row['type'] != 'multilabel':
            continue
        previous = old_by_key[(row['source_id'], row['id'])]
        if previous['target'] != row['target'] or previous['option_labels'] != row['option_labels']:
            raise ValueError('Matched emotion candidates or targets differ')
        matched.append(previous)
    old_temperatures = snapshot['runs']['multilabel-emotions']['calibration']['temperatures']
    before, _ = score_records(matched, old_temperatures)
    snapshot['matched_emotion_test'] = {
        'samples': len(matched), 'before': before['multilabel'],
        'after': snapshot['runs']['multilabel-banking']['metrics']['multilabel'],
        'comparison_scope': 'same source IDs, targets and candidate lists; each checkpoint keeps its own calibration and threshold',
    }
    for head in ('choice', 'score', 'yesno'):
        a = {(r['source_id'], r['id']): r for r in old if r['type'] == head}
        b = {(r['source_id'], r['id']): r for r in new if r['type'] == head}
        if set(a) != set(b) or any(a[k]['target'] != b[k]['target'] or a[k].get('option_labels') != b[k].get('option_labels') for k in a):
            raise ValueError(f'Legacy {head} checks are not matched')
    snapshot['legacy_retention_matched'] = True
    historical = subprocess.check_output(['git', 'show', '38ed101:paper/paper.tex'], cwd=root, text=True)
    table = historical.split(r'\label{tab:quality}', 1)[1].split(r'\end{table*}', 1)[0]
    rows = []
    for line in table.splitlines():
        if ' & ' not in line or not re.match(r'^(Jevlet|TF-IDF|DistilBERT|BERT-base|RoBERTa-base)', line):
            continue
        parts = line.rstrip().removesuffix(r'\\').split(' & ')
        rows.append({'model': parts[0], 'choice_accuracy_percent': float(parts[1]),
                     'sentiment_accuracy_percent': float(parts[3]), 'sentiment_macro_f1': float(parts[4]),
                     'binary_accuracy_percent': float(parts[6])})
    if len(rows) != 7:
        raise ValueError('Expected seven archived comparison rows')
    snapshot['historical_gpu_comparison'] = {
        'source': 'paper/paper.tex at Git commit 38ed101',
        'source_sha256': hashlib.sha256(historical.encode()).hexdigest(), 'rows': rows,
        'status': 'archived manuscript values; not rerun or independently re-audited in this extension',
    }
    bibliography = historical.split(r'\begin{thebibliography}{10}', 1)[1].split(r'\end{thebibliography}', 1)[0].strip()
    snapshot['historical_bibliography_tex'] = bibliography
    snapshot['authors'] = list(AUTHORS)
    return snapshot


def tex(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('#', r'\#')


def author_block():
    blocks = []
    for index, author in enumerate(AUTHORS):
        if index:
            blocks.append(r'\linebreakand' if index == 2 else r'\and')
        blocks.append(r'\IEEEauthorblockN{\normalfont ' + author + '}\n' +
                      r'\IEEEauthorblockA{\textit{Department of Information Technology}\\' + '\n' +
                      r'\textit{Dwarkadas Jivanlal Sanghvi}\\' + '\n' +
                      r'\textit{College of Engineering}\\' + '\nMumbai, India}')
    return '\n'.join(blocks)


def render(root, evidence, style):
    runs = evidence['runs']
    e, b = runs['multilabel-emotions'], runs['multilabel-banking']
    em, bm = e['metrics']['multilabel'], b['metrics']['banking_multilabel']
    eb, bb = e['baseline']['metrics'], b['baseline']['metrics']
    if em['samples'] != 768 or bm['samples'] != 204 or not evidence['legacy_retention_matched']:
        raise ValueError('The manuscript protocol does not match this evidence snapshot')
    for run in (e, b):
        if run['specification']['score_mode'] != 'ordinal':
            raise ValueError('Update the architecture text for a categorical deployed checkpoint')
        if run['settings']['seed'] != 42 or run['settings']['device'] != 'cpu' or len(run['history']) != 3:
            raise ValueError('Update the protocol text before changing run settings')
    f = lambda x: f'{x:.4f}'
    pct = lambda x: f'{100*x:.2f}'
    replacements = {'TITLE': TITLE, 'AUTHORS': author_block(),
        'EMOTION_MICRO': pct(em['micro_f1']), 'EMOTION_BASELINE_MICRO': pct(eb['micro_f1']),
        'BANK_MICRO': pct(bm['micro_f1']), 'BANK_BASELINE_MICRO': pct(bb['micro_f1']),
        'EMOTION_DELTA': f"{100*(em['micro_f1']-eb['micro_f1']):.2f}",
        'BANK_DELTA': f"{100*(bm['micro_f1']-bb['micro_f1']):.2f}",
        'EMOTION_THRESHOLD': f"{e['specification']['multilabel_threshold']:.2f}",
        'BANK_THRESHOLD': f"{b['specification']['banking_multilabel_threshold']:.2f}",
        'FINAL_EMOTION_THRESHOLD': f"{b['specification']['multilabel_threshold']:.2f}",
        'HISTORICAL_BIBLIOGRAPHY': evidence['historical_bibliography_tex']}
    metric_rows = []
    for task, name, m in (('Emotion', 'Jevlet (stage E)', em), ('Emotion', 'TF-IDF logistic', eb),
                           ('Banking', 'Jevlet (stage B)', bm), ('Banking', 'TF-IDF logistic', bb)):
        metric_rows.append(' & '.join([task, name, str(m['samples'])] + [f(m[k]) for k in
            ('micro_f1','macro_f1','precision','recall','exact_set_accuracy','hamming_loss')]) + r'\\')
    replacements['MULTILABEL_ROWS'] = '\n'.join(metric_rows)
    replacements['PROBABILITY_ROWS'] = '\n'.join(' & '.join([task, name] + [f(m[k]) for k in
        ('brier','nll','ece','candidate_recall')]) + r'\\' for task, name, m in
        (('Emotion','Jevlet (E)',em),('Emotion','TF-IDF',eb),('Banking','Jevlet (B)',bm),('Banking','TF-IDF',bb)))
    replacements['EMOTION_RETENTION_ROWS'] = '\n'.join(' & '.join([name,str(m['samples'])] + [f(m[k]) for k in
        ('micro_f1','macro_f1','exact_set_accuracy')]) + r'\\' for name,m in
        (('Stage E',evidence['matched_emotion_test']['before']),('Stage B',evidence['matched_emotion_test']['after'])))
    replacements['LEGACY_RETENTION_ROWS'] = '\n'.join(' & '.join([name,
        pct(run['metrics']['choice']['accuracy']),pct(run['metrics']['score']['accuracy']),
        pct(run['metrics']['yesno']['accuracy'])]) + r'\\' for name,run in (('Stage E',e),('Stage B',b)))
    replacements['HISTORICAL_ROWS'] = '\n'.join(' & '.join([tex(r['model']),
        f"{r['choice_accuracy_percent']:.2f}", f"{r['sentiment_accuracy_percent']:.2f}",
        f(r['sentiment_macro_f1']),f"{r['binary_accuracy_percent']:.2f}"]) + r'\\' for r in evidence['historical_gpu_comparison']['rows'])
    replacements['TRAINING_ROWS'] = '\n'.join(' & '.join([name,
        f"{run['training_seconds']/60:.2f}",str(len(run['history'])),
        f(run['history'][-1]['train_loss']),f(run['history'][-1]['validation_loss'])]) + r'\\'
        for name,run in (('Emotion E',e),('Banking B',b)))
    replacements['TEMPERATURE_ROWS'] = '\n'.join(' & '.join([label,
        f(e['calibration']['temperatures'][head]) if head in e['calibration']['temperatures'] else '--',
        f(b['calibration']['temperatures'][head])]) + r'\\' for head,label in
        (('choice','Choice'),('score','Ordinal sentiment'),('yesno','Binary'),('multilabel','Emotion'),('banking_multilabel','Banking components')))
    replacements['EMOTION_SOURCE_REV'] = e['sources']['multilabel']['revision']
    replacements['BANK_SOURCE_REV'] = b['sources']['banking_multilabel']['revision']
    template = (root/'paper/multilabel-template.tex').read_text()
    for name,value in replacements.items():
        template = template.replace('@'+name+'@',value)
    if re.search(r'@[A-Z_]+@',template):
        raise ValueError('Unresolved manuscript placeholder')
    bibliography_start = template.index(r'\begin{thebibliography}')
    entries = dict(re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})',
                              template[bibliography_start:], re.S))
    citation_order = list(dict.fromkeys(key.strip() for group in
        re.findall(r'\\cite\{([^}]+)\}', template[:bibliography_start]) for key in group.split(',')))
    if set(citation_order) != set(entries):
        raise ValueError('Manuscript citations and bibliography entries differ')
    template = (template[:bibliography_start] + r'\begin{thebibliography}{' + str(len(entries)) + '}\n' +
                '\n'.join(r'\bibitem{' + key + '}' + entries[key].strip() for key in citation_order) +
                '\n' + r'\end{thebibliography}' + '\n' + r'\end{document}' + '\n')
    if style == 'elsevier':
        start = template.index(r'\begin{document}')
        body = template[start+len(r'\begin{document}'):].replace(r'\maketitle','',1)
        abstract_end = body.index(r'\end{IEEEkeywords}') + len(r'\end{IEEEkeywords}')
        front,rest = body[:abstract_end],body[abstract_end:]
        front = front.replace(r'\begin{IEEEkeywords}',r'\begin{keyword}').replace(r'\end{IEEEkeywords}',r'\end{keyword}')
        template = (r'\documentclass[preprint,12pt,numbers]{elsarticle}'+'\n'+
                    r'\usepackage{mathptmx,amsmath,amssymb,booktabs,url,tikz,graphicx}'+'\n'+
                    r'\usetikzlibrary{positioning,arrows.meta}'+'\n'+r'\usepackage[hidelinks]{hyperref}'+'\n'+
                    r'\begin{document}\begin{frontmatter}'+'\n'+r'\title{'+TITLE+'}\n'+
                    '\n'.join(r'\author[org]{'+a+'}' for a in AUTHORS)+'\n'+
                    r'\address[org]{Department of Information Technology, Dwarkadas Jivanlal Sanghvi College of Engineering, Mumbai, India}'+'\n'+
                    front+r'\end{frontmatter}'+rest)
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--format', choices=('ieee','elsevier'), default='ieee')
    parser.add_argument('--refresh-evidence', action='store_true', help='Capture the completed local runs and archived comparison')
    parser.add_argument('--check', action='store_true', help='Check that the current source matches the saved evidence/template')
    args = parser.parse_args()
    root = args.root.resolve()
    evidence_path = root/'paper/evidence.json'
    if args.refresh_evidence:
        evidence = capture(root)
        evidence_path.write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    evidence = read(evidence_path)
    text = render(root,evidence,args.format)
    target = root/'paper/paper.tex'
    if args.check:
        if target.read_text() != text:
            raise SystemExit('Paper source differs from the saved evidence/template')
        print('Paper source matches saved measured evidence')
    else:
        target.write_text(text)
        print(f'Updated {target} from {evidence_path}')


if __name__ == '__main__':
    main()
