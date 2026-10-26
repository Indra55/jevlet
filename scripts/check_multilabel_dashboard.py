"""Verify the live GoEmotions dashboard using real checkpoint predictions."""
import argparse
import ast
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--case', choices=('emotions', 'banking_multi'), default='emotions')
    args = parser.parse_args()
    banking = args.case == 'banking_multi'
    args.output = args.output or Path('.cache/banking-multilabel-dashboard' if banking else '.cache/multilabel-dashboard')
    args.output.mkdir(parents=True, exist_ok=True)
    errors, requests, results = [], [], {}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1050})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append((request.method, request.url)))
        page.goto(args.url)
        page.locator(f'.case-card.{args.case}').wait_for()
        assert page.locator('.case-card').count() == 5
        page.locator(f'.case-card.{args.case}').click()

        def predict():
            with page.expect_response(lambda r: r.url.endswith('/api/predict') and r.request.method == 'POST') as pending:
                page.locator('#run-button').click()
            response = pending.value
            assert response.status == 200, response.text()
            result = response.json()
            answer = result['answers'][0]
            assert result['encoder_passes'] == 1 and answer['type'] == ('banking_multilabel' if banking else 'multilabel')
            assert 'confidence' not in answer
            assert answer['answer'] == [label for label, value in zip(answer['options'], answer['probabilities'])
                                        if value >= answer['threshold']]
            assert all(0 <= value <= 1 for value in answer['probabilities'])
            page.locator('.answer-title').wait_for()
            return result

        results['default'] = predict()
        posts = sum(method == 'POST' for method, _ in requests)
        for threshold in (.05, .95):
            page.locator('#threshold').fill(str(threshold))
            count = sum(value >= threshold for value in results['default']['answers'][0]['probabilities'])
            assert page.locator('.prob-row.winner').count() == count
        assert sum(method == 'POST' for method, _ in requests) == posts
        page.locator('#threshold').fill(str(round(results['default']['answers'][0]['threshold'], 2)))
        page.get_by_role('tab', name='Integrate', exact=True).click()
        ast.parse(page.locator('#integration-code').inner_text())
        page.get_by_role('tab', name='Evaluation notes', exact=True).click()
        assert ('NLU++' if banking else 'GoEmotions') in page.locator('#guide-body').inner_text()
        assert 'Candidate micro-F1' in page.locator('#guide-body').inner_text()
        page.screenshot(path=str(args.output / f'{args.case}-desktop.png'), full_page=True, animations='disabled')
        page.locator('#add-option').select_option('loan' if banking else 'surprise')
        assert 'Input changed' in page.locator('#stale-notice').inner_text()
        results['changed_candidates'] = predict()
        page.locator('#text').fill(('My transfer is not working. ' if banking else 'Thank you for helping me. ') * 200)
        results['truncation'] = predict()
        assert results['truncation']['context_tokens_dropped'] > 0
        assert page.locator('.warning').is_visible()
        page.route('**/api/predict', lambda route: route.fulfill(status=503,
            content_type='application/json', body='{"detail":"Test: server unavailable"}'))
        page.locator('#run-button').click()
        page.get_by_role('alert').get_by_text('Test: server unavailable').wait_for()
        page.unroute('**/api/predict')
        page.locator('[data-example="1"]').click()
        results['recovery'] = predict()
        page.set_viewport_size({'width': 390, 'height': 844})
        page.goto(args.url + '/cases/' + args.case)
        page.locator('#run-button').wait_for()
        results['mobile'] = predict()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(args.output / f'{args.case}-mobile.png'), full_page=True, animations='disabled')
        page.goto(args.url)
        page.locator(f'.case-card.{args.case}').wait_for()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(args.output / 'home-mobile.png'), full_page=True, animations='disabled')
        assert not errors, errors
        assert all(url.startswith(args.url) or url.startswith('data:') for _, url in requests)
        browser.close()
    (args.output / 'verification.json').write_text(json.dumps({'passed': True, 'responses': results,
        'checks': ['real model predictions', 'independent probabilities', 'one encoder pass',
                   'zero-or-many selection', 'threshold does not rerun model', 'editable candidates',
                   'visible truncation', 'API error recovery', 'Python integration syntax',
                   'desktop/mobile layout', 'no browser errors or external assets']}, indent=2) + '\n')
    print(f'Live multi-label browser checks passed: {args.output}')


if __name__ == '__main__':
    main()
