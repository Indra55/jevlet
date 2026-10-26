"""Compatibility of upstream question refinement with the local multi-label tasks."""
import json
from unittest.mock import patch

import pytest

from test_multilabel import model
from jevlet import ROOT
from jevlet.config import load_config
from jevlet.data import requests_for
from jevlet.model import Jevlet


@pytest.fixture
def mixed_bundle():
    names = ['card', 'cash', 'pin']
    splits = {}
    for split in ('train', 'validation', 'calibration', 'test'):
        splits[split] = [
            {'id': split+'/bank', 'domain': 'banking', 'text': 'card', 'label': 0},
            {'id': split+'/score', 'domain': 'sst', 'text': 'text', 'label': 2},
            {'id': split+'/emotion', 'domain': 'multilabel', 'text': 'joy love', 'labels': [0, 1], 'data_kind': 'human_annotated'},
            {'id': split+'/bankmulti', 'domain': 'banking_multilabel', 'text': 'card pin', 'labels': [0, 2], 'data_kind': 'human_annotated'}]
    return {'label_names': names, 'multilabel_names': ['joy', 'love', 'anger'],
            'banking_multilabel_names': ['change', 'card', 'pin'], 'splits': splits,
            'sources': {'multilabel': {}, 'banking_multilabel': {}}}


def test_expanded_views_and_request_seed_cover_both_multilabel_tasks(mixed_bundle):
    cfg = {'seed': 42, 'options_min': 2, 'options_max': 3, 'eval': {'validation_views': 5}}
    expanded = requests_for(mixed_bundle, 'validation', cfg)
    assert len(expanded) == 20
    assert {r['validation_view'] for r in expanded} == set(range(5))
    for domain, names in (('multilabel', mixed_bundle['multilabel_names']),
                          ('banking_multilabel', mixed_bundle['banking_multilabel_names'])):
        questions = [q for r in expanded for q in r['questions'] if q['type'] == domain]
        assert len(questions) == 5
        assert all(set(q['options']) <= set(names) for q in questions)
        assert all(sum(q['target']) == 2 for q in questions)
    assert requests_for(mixed_bundle, 'test', cfg) == requests_for(
        mixed_bundle, 'test', {**cfg, 'seed': 44, 'request_seed': 42})
    assert len(requests_for(mixed_bundle, 'test', cfg)) == 4


def test_warm_start_sentiment_selection_preserves_multihead_calibration(model, mixed_bundle, tmp_path):
    from jevlet.train import train
    cfg = load_config(ROOT / 'config.yaml')
    model.save(tmp_path/'initial')
    cfg.update(initial_checkpoint=str(tmp_path/'initial'), checkpoint_metric='sentiment_accuracy',
               augment_questions=True, multilabel=True, banking_multilabel=True,
               device='cpu', cpu_threads=1, options_min=2, options_max=3)
    cfg['train'].update(epochs=2, patience=2, batch_size=2, accumulation=1, lr=0.)
    cfg['eval'].update(batch_size=4, validation_views=5)
    original_save = Jevlet.save
    saves = []
    def record_save(self, directory):
        saves.append(str(directory))
        return original_save(self, directory)
    metrics = [({'score': {'accuracy': .4}}, []), ({'score': {'accuracy': .6}}, [])]
    with patch.object(Jevlet, 'save', record_save), patch('jevlet.train.score_records', side_effect=metrics):
        trained, _ = train(cfg, tmp_path/'refined', mixed_bundle)
    history = json.loads((tmp_path/'refined/training.json').read_text())['history']
    assert [e['validation_metrics']['score']['accuracy'] for e in history] == [.4, .6]
    assert history[0]['validation_loss'] == pytest.approx(history[1]['validation_loss'])
    assert len(saves) == 3  # Two accuracy-selected epochs, then the calibrated checkpoint.
    calibration = json.loads((tmp_path/'refined/calibration.json').read_text())
    assert all(calibration['fit'][h]['status'] == 'fitted' for h in trained.heads)
    assert len(Jevlet.load(tmp_path/'refined').temperatures) == 5


def test_saved_comparison_report_runs_without_cuda(tmp_path, monkeypatch, capsys):
    from jevlet import gpu_compare
    monkeypatch.setattr(gpu_compare, 'ROOT', tmp_path)
    monkeypatch.setattr(gpu_compare, 'require_cuda', lambda: pytest.fail('Saved reporting must not require CUDA'))
    checkpoint = tmp_path/'model'
    checkpoint.mkdir()
    (checkpoint/'training.json').write_text(json.dumps({
        'history': [{'epoch': 1, 'train_loss': .3, 'validation_loss': .4}],
        'training_seconds': 12., 'peak_gpu_bytes': 2**30}))
    summary = {'models': {'jevlet_softmax': {'checkpoint': str(checkpoint), 'calibrated': {
        'choice': {'accuracy': .9}, 'score': {'accuracy': .5, 'macro_f1': .45}, 'yesno': {'accuracy': .95}}}}}
    gpu_compare.print_report(summary, ['jevlet_softmax'], history=True)
    output = capsys.readouterr().out
    assert 'SAVED TEST RESULTS' in output and 'RECORDED TRAINING HISTORY' in output
    assert 'train_loss=0.3000' in output and 'validation_loss=0.4000' in output
