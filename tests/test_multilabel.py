import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertTokenizerFast, DistilBertConfig, DistilBertModel

from jevlet.config import load_config
from jevlet.data import TOKENS, collate, pack_request, requests_for, validate_questions
from jevlet.metrics import fit_temperatures, fit_multilabel_threshold, score_records
from jevlet.model import Jevlet, training_loss
from jevlet.multilabel import reviewed_rows, sample_multilabel_request
from jevlet.server import DecisionRequest, build_questions, create_app


@pytest.fixture
def model(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / 'vocab.txt'
    path.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'joy', 'love', 'anger', 'text', 'what', '?']))
    tok = BertTokenizerFast(vocab_file=str(path))
    tok.add_special_tokens({'additional_special_tokens': list(TOKENS)})
    obj = Jevlet(DistilBertModel(DistilBertConfig(vocab_size=len(tok), dim=16, hidden_dim=32,
        n_layers=1, n_heads=2, dropout=0., attention_dropout=0.)))
    obj.tokenizer, obj.max_length, obj.source_revision = tok, 64, 'offline-test'
    return obj.eval()


def question(target=None):
    q = {'id': 'emotions', 'type': 'multilabel', 'text': 'what ?', 'options': ['joy', 'love', 'anger']}
    if target is not None:
        q['target'] = target
    return q


def test_legacy_checkpoint_stays_legacy(model, tmp_path):
    model.save(tmp_path / 'old')
    loaded = Jevlet.load(tmp_path / 'old')
    assert loaded.heads == ('choice', 'score', 'yesno')
    with pytest.raises(ValueError, match='no trained multi-label'):
        loaded.predict('text', [question()])


def test_multiple_or_zero_answers_and_one_encoder_pass(model, tmp_path):
    model.enable_multilabel()
    with torch.no_grad():
        model.multilabel_head.weight.zero_()
        model.multilabel_head.bias.fill_(2.)
    with patch.object(model.encoder, 'forward', wraps=model.encoder.forward) as encoder:
        result = model.predict('text', [question()])
    assert encoder.call_count == 1
    answer = result['answers'][0]
    assert answer['answer'] == ['joy', 'love', 'anger']
    assert sum(answer['probabilities']) > 2
    assert 'confidence' not in answer
    assert model.predict('text', [{**question(), 'threshold': .99}])['answers'][0]['answer'] == []
    model.multilabel_threshold = .37
    model.temperatures[-1] = 1.7
    model.save(tmp_path / 'new')
    loaded = Jevlet.load(tmp_path / 'new')
    assert loaded.multilabel_threshold == .37
    assert loaded.predict('text', [question()]) == model.predict('text', [question()])


def test_unknown_annotations_and_padding_have_no_gradient(model):
    model.enable_multilabel()
    req = {'text': 'text', 'questions': [question([1, -1, 0])]}
    batch = collate([pack_request(model.tokenizer, req, 64)], model.tokenizer.pad_token_id)
    outputs = model(batch)
    outputs['multilabel'].retain_grad()
    loss = training_loss(model, outputs, batch, .5)
    loss.backward()
    grad = outputs['multilabel'].grad
    assert torch.isfinite(loss)
    assert grad[0, 0, 0] < 0 and grad[0, 0, 2] > 0
    assert grad[0, 0, 1] == 0 and grad[0, 0, 3:].abs().sum() == 0
    assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('override', [{'target': [1]}, {'target': [-1, -1, -1]},
    {'target': [1, 2, 0]}, {'threshold': 0}, {'threshold': 1}, {'threshold': float('nan')}])
def test_invalid_multilabel_targets_and_thresholds(override):
    with pytest.raises(ValueError):
        validate_questions([{**question(), **override}])


def records():
    return [dict(type='multilabel', logits=[4., -4., 3.], target=[1, 0, 1], option_labels=[0, 1, 2], gold_labels=[0, 2]),
            dict(type='multilabel', logits=[-4., 3.], target=[0, 1], option_labels=[0, 1], gold_labels=[1]),
            dict(type='multilabel', logits=[-2., -3.], target=[0, 0], option_labels=[1, 2], gold_labels=[0])]


def test_set_metrics_variable_candidates_and_candidate_recall():
    metrics, predictions = score_records(records())
    m = metrics['multilabel']
    assert m['micro_f1'] == m['macro_f1'] == m['exact_set_accuracy'] == 1
    assert m['hamming_loss'] == 0
    assert m['candidate_recall'] == .75
    assert predictions[-1]['predicted'] == [0, 0]
    assert all('confidence' not in r for r in predictions)


def test_partial_annotations_do_not_count_as_wrong_or_exact_sets():
    row = {**records()[0], 'target': [1, -1, 1]}
    m = score_records([row])[0]['multilabel']
    assert m['micro_f1'] == 1 and m['observed_label_decisions'] == 2
    assert m['exact_set_accuracy'] is None


def test_calibration_and_validation_threshold():
    rows = records()
    temps, details = fit_temperatures(rows)
    assert details['multilabel']['nll_after'] <= details['multilabel']['nll_before'] + 1e-6
    threshold, fit = fit_multilabel_threshold(rows, temps['multilabel'])
    assert 0 < threshold < 1 and fit['split'] == 'validation'
    with pytest.raises(ValueError, match='positive'):
        fit_multilabel_threshold([{**rows[-1], 'target': [0, 0]}], 1.)


def test_catalogue_sampling_and_missing_positive_cases():
    cfg = {'seed': 42, 'options_min': 2, 'options_max': 4}
    names = ['joy', 'love', 'anger', 'fear', 'neutral']
    row = {'id': 'a', 'text': 'text', 'labels': [0, 1], 'domain': 'multilabel', 'data_kind': 'human_annotated'}
    req = sample_multilabel_request(row, names, cfg)
    assert req == sample_multilabel_request(row, names, cfg)
    assert sum(req['questions'][0]['target']) == 2
    empty = sample_multilabel_request({**row, 'candidate_mode': 'none'}, names, cfg)
    assert sum(empty['questions'][0]['target']) == 0
    bundle = {'label_names': ['bank a', 'bank b'], 'multilabel_names': names, 'splits': {'test': [row]}}
    assert set(requests_for(bundle, 'test', cfg)[0]['questions'][0]['options']) <= set(names)


def test_reviewed_data_rejects_text_and_source_leakage(tmp_path):
    names = ['a', 'b', 'c']
    base = {'splits': {split: [] for split in ('train', 'validation', 'calibration', 'test')}}
    rows = [dict(id=split, split=split, text=split, labels=['a', 'b'], annotation_complete=True)
            for split in base['splits']]
    path = tmp_path / 'data.jsonl'
    def save(): path.write_text('\n'.join(json.dumps(r) for r in rows))
    save()
    assert len(reviewed_rows(path, names, base)['train']) == 1
    rows[-1]['text'] = 'train'; save()
    with pytest.raises(ValueError, match='overlaps'):
        reviewed_rows(path, names, base)
    rows[-1]['text'] = 'test'
    rows[0]['source_ids'] = ['reused']; rows[-1]['source_ids'] = ['reused']; save()
    with pytest.raises(ValueError, match='source ID overlaps'):
        reviewed_rows(path, names, base)


def test_api_multilabel_contract_and_old_checkpoint_rejection(model):
    from fastapi.testclient import TestClient
    class Engine:
        labels = ['card arrival', 'lost card']
        multilabel_labels = ['joy', 'love', 'anger']
        info = {'labels': labels, 'multilabel_labels': multilabel_labels}
        calls = []
        def predict(self, text, questions):
            self.calls.append((text, questions))
            return model.predict(text, questions)
    model.enable_multilabel()
    engine = Engine()
    payload = dict(case='emotions', text='text', options=engine.multilabel_labels, threshold=.4)
    with TestClient(create_app(engine=engine)) as client:
        response = client.post('/api/predict', json=payload)
        assert response.status_code == 200
        assert response.json()['answers'][0]['threshold'] == .4
        for override in ({'options': ['joy', 'invented']}, {'checks': ['joy']}, {'threshold': 0}):
            assert client.post('/api/predict', json={**payload, **override}).status_code == 422
        assert client.get('/cases/emotions').status_code == 200
        engine.multilabel_labels = []
        assert client.post('/api/predict', json=payload).status_code == 422


def test_training_calibration_baseline_and_reload(model, tmp_path):
    from jevlet import ROOT
    from jevlet.train import train, collect_records
    from jevlet.multilabel import baseline
    cfg = load_config(ROOT / 'config.yaml')
    cfg.update(multilabel=True, device='cpu', cpu_threads=1, options_min=2, options_max=4)
    cfg['train'].update(epochs=1, batch_size=2, accumulation=1)
    cfg['eval']['batch_size'] = 2
    names = ['joy', 'love', 'anger', 'fear']
    splits = {}
    for split in ('train', 'validation', 'calibration', 'test'):
        splits[split] = [
            {'id': split+'bank', 'domain': 'banking', 'text': 'text', 'label': 0},
            {'id': split+'score', 'domain': 'sst', 'text': 'text', 'label': 2},
            {'id': split+'multi', 'domain': 'multilabel', 'text': 'joy love', 'labels': [0, 1], 'data_kind': 'human_annotated'},
            {'id': split+'other', 'domain': 'multilabel', 'text': 'anger', 'labels': [2], 'data_kind': 'human_annotated'}]
    bundle = {'label_names': names, 'multilabel_names': names, 'splits': splits,
              'sources': {'multilabel': {'kind': 'human_annotated'}}}
    with patch.object(Jevlet, 'pretrained', return_value=model):
        trained, _ = train(cfg, tmp_path / 'run', bundle)
    loaded = Jevlet.load(tmp_path / 'run')
    assert 0 < loaded.multilabel_threshold < 1
    calibration = json.loads((tmp_path / 'run/calibration.json').read_text())
    assert calibration['fit']['multilabel']['status'] == 'fitted'
    rows = collect_records(loaded, requests_for(bundle, 'test', cfg), cfg)
    assert set(score_records(rows)[0]) == {'choice', 'score', 'yesno', 'multilabel'}
    assert baseline(bundle, cfg, tmp_path / 'run')['metrics']['samples'] == 2


def test_banking_and_emotion_heads_train_independently_and_reload(model, tmp_path):
    model.enable_banking_multilabel()
    assert model.heads == ('choice', 'score', 'yesno', 'multilabel', 'banking_multilabel')
    assert model.multilabel_head.weight.data_ptr() != model.banking_multilabel_head.weight.data_ptr()
    questions = [question([1, 0, 1]), {**question([0, 1, -1]), 'id': 'banking', 'type': 'banking_multilabel'}]
    batch = collate([pack_request(model.tokenizer, {'text': 'text', 'questions': questions}, 64)], model.tokenizer.pad_token_id)
    outputs = model(batch)
    outputs['banking_multilabel'].retain_grad()
    loss = training_loss(model, outputs, batch, .5)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.multilabel_head.weight.grad.abs().sum() > 0
    assert model.banking_multilabel_head.weight.grad.abs().sum() > 0
    assert outputs['banking_multilabel'].grad[0, 1, 2] == 0
    model.multilabel_threshold, model.banking_multilabel_threshold = .25, .75
    model.temperatures.copy_(torch.tensor([1., 1., 1., 1.2, 1.7]))
    with patch.object(model.encoder, 'forward', wraps=model.encoder.forward) as encoder:
        expected = model.predict('text', questions)
    assert encoder.call_count == 1
    assert [a['threshold'] for a in expected['answers']] == [.25, .75]
    model.save(tmp_path / 'combined')
    assert json.loads((tmp_path / 'combined/jevlet.json').read_text())['format_version'] == 3
    assert Jevlet.load(tmp_path / 'combined').predict('text', questions) == expected


def test_banking_calibration_metrics_and_threshold_are_separate():
    emotion = records()[0]
    banking = {**emotion, 'type': 'banking_multilabel', 'target': [0, 1, 0]}
    metrics, _ = score_records([emotion, banking])
    assert metrics['multilabel']['micro_f1'] == 1
    assert metrics['banking_multilabel']['micro_f1'] == 0
    temperatures, details = fit_temperatures([emotion, banking])
    for head in ('multilabel', 'banking_multilabel'):
        assert details[head]['nll_after'] <= details[head]['nll_before'] + 1e-6
        _, tuning = fit_multilabel_threshold([emotion, banking], temperatures[head], head)
        assert tuning['samples'] == 1


def test_banking_api_catalogues_and_legacy_checkpoint_rejection():
    payload = dict(case='banking_multi', text='Please change my PIN', options=['change', 'pin'], threshold=.6)
    request = DecisionRequest(**payload)
    q = build_questions(request, ['card arrival'], ['joy', 'love'], ['change', 'pin'])[0]
    assert q['type'] == 'banking_multilabel' and q['threshold'] == .6
    with pytest.raises(ValueError, match='no trained'):
        build_questions(request, ['card arrival'], ['joy', 'love'])
    with pytest.raises(ValueError, match='catalogue'):
        build_questions(DecisionRequest(**{**payload, 'options': ['joy', 'love']}), [], ['joy', 'love'], ['change', 'pin'])
    with pytest.raises(ValueError):
        DecisionRequest(**{**payload, 'checks': ['change']})


def test_nlupp_partitions_preserve_source_folds_and_remove_duplicate_text():
    import io
    import zipfile
    from jevlet.banking_multilabel import REVISION, FOLDS, bundle_from_archive
    source = io.BytesIO()
    prefix = f'task-specific-datasets-{REVISION}/nlupp/data/'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr(prefix+'ontology.json', json.dumps({'intents': {
            'change': {'domain': ['general'], 'description': 'change something'},
            'pin': {'domain': ['banking'], 'description': 'a PIN'},
            'hotel': {'domain': ['hotels'], 'description': 'a hotel'}}}))
        for fold in range(20):
            archive.writestr(prefix+f'banking/fold{fold}.json', json.dumps([
                {'text': f'unique fold {fold}', 'intents': ['change', 'pin']},
                {'text': 'shared duplicate', 'intents': ['pin']}]))
    base = {'label_names': ['bank'], 'multilabel_names': ['joy', 'love'], 'spec': {},
            'sources': {'multilabel': {'id': 'goemotions'}},
            'splits': {s: [{'id': s, 'domain': 'multilabel', 'text': s, 'labels': [0]}] for s in FOLDS}}
    base['splits']['calibration'][0]['text'] = 'unique fold 1'
    with zipfile.ZipFile(source) as archive:
        bundle = bundle_from_archive(base, {'seed': 42}, archive)
    assert bundle['banking_multilabel_names'] == ['change', 'pin']
    seen_text, seen_ids = set(), set()
    for split, rows in bundle['splits'].items():
        rows = [r for r in rows if r['domain'] == 'banking_multilabel']
        texts, ids = {r['text'] for r in rows}, {r['id'] for r in rows}
        assert not seen_text & texts and not seen_ids & ids
        seen_text.update(texts); seen_ids.update(ids)
        assert all(int(r['id'].split('/')[2][4:]) in FOLDS[split] for r in rows)
    assert any(r['text'] == 'shared duplicate' for r in bundle['splits']['test'])
    assert len(bundle['sources']['banking_multilabel']['source_file_sha256']) == 20
    assert bundle['sources']['banking_multilabel']['removed_cross_domain_rehearsal_duplicates'] == {'calibration/multilabel': 1}
    assert not {r['text'] for r in bundle['splits']['train']} & {r['text'] for r in bundle['splits']['calibration']}


def test_combined_training_fits_both_temperatures_and_thresholds(model, tmp_path):
    from jevlet import ROOT
    from jevlet.train import train
    cfg = load_config(ROOT / 'config.yaml')
    cfg.update(multilabel=True, banking_multilabel=True, device='cpu', cpu_threads=1, options_min=2, options_max=3)
    cfg['train'].update(epochs=1, batch_size=2, accumulation=1)
    cfg['eval']['batch_size'] = 2
    names = ['joy', 'love', 'anger']
    splits = {}
    for split in ('train', 'validation', 'calibration', 'test'):
        splits[split] = [
            {'id': split+'bank', 'domain': 'banking', 'text': 'text', 'label': 0},
            {'id': split+'score', 'domain': 'sst', 'text': 'text', 'label': 2},
            {'id': split+'emotion', 'domain': 'multilabel', 'text': 'joy love', 'labels': [0, 1], 'data_kind': 'human_annotated'},
            {'id': split+'bankmulti', 'domain': 'banking_multilabel', 'text': 'text', 'labels': [0, 2], 'data_kind': 'human_annotated'}]
    bundle = {'label_names': names, 'multilabel_names': names, 'banking_multilabel_names': ['change', 'card', 'pin'],
              'splits': splits, 'sources': {'multilabel': {}, 'banking_multilabel': {}}}
    with patch.object(Jevlet, 'pretrained', return_value=model):
        trained, _ = train(cfg, tmp_path / 'combined', bundle)
    calibration = json.loads((tmp_path / 'combined/calibration.json').read_text())
    assert all(calibration['fit'][h]['status'] == 'fitted' for h in trained.heads)
    assert calibration['threshold_fit']['split'] == calibration['banking_threshold_fit']['split'] == 'validation'
    assert len(Jevlet.load(tmp_path / 'combined').temperatures) == 5
