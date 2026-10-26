"""Check the live dashboard against a real checkpoint; no mocked model responses.

Optional tooling: uv pip install --python .venv/bin/python playwright
PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright .venv/bin/python -m playwright install chromium
Start make demo, then run this script with the same PLAYWRIGHT_BROWSERS_PATH.
Screenshots and the verification summary go to .cache/dashboard-checks/.
"""

import argparse
import ast
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    output = Path(".cache/dashboard-checks")
    output.mkdir(parents=True, exist_ok=True)
    errors, requests, responses = [], [], {}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append((request.method, request.url)))

        def no_overflow():
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Horizontal overflow"

        def predict(expected_questions):
            with page.expect_response(lambda r: r.url.endswith("/api/predict") and r.request.method == "POST") as pending:
                page.locator("#run-button").click()
            response = pending.value
            assert response.status == 200, response.text()
            result = response.json()
            assert result["encoder_passes"] == 1
            assert len(result["answers"]) == expected_questions
            assert all(abs(sum(a["probabilities"]) - 1) < 1e-5 for a in result["answers"])
            assert all(a["confidence"] == max(a["probabilities"]) for a in result["answers"])
            page.locator(".answer-title").wait_for()
            assert page.locator("#run-button").is_enabled()
            return result

        page.goto(args.url)
        page.locator(".case-card").first.wait_for()
        assert page.locator(".case-card").count() == 5
        assert "Ready" in page.locator("#connection").inner_text()
        no_overflow()
        page.screenshot(path=str(output / "home-desktop.png"), full_page=True, animations="disabled")
        page.locator(".case-card.banking").click()
        responses["banking"] = predict(3)
        assert responses["banking"]["answers"][0]["answer"] in responses["banking"]["answers"][0]["options"]
        page.screenshot(path=str(output / "banking-desktop.png"), full_page=True, animations="disabled")
        page.locator("#add-option").select_option("activate my card")
        assert "Input changed" in page.locator("#stale-notice").inner_text()
        assert page.locator(".option-chip").count() == 4
        page.locator('[data-remove="3"]').click()
        page.get_by_role("tab", name="Integrate", exact=True).click()
        ast.parse(page.locator("#integration-code").inner_text())
        page.get_by_role("tab", name="How to use", exact=True).click()
        page.keyboard.press("ArrowRight")
        assert page.get_by_role("tab", name="How it works", exact=True).get_attribute("aria-selected") == "true"
        page.get_by_role("tab", name="Evaluation notes", exact=True).click()
        assert "77" in page.locator("#guide-body").inner_text()
        page.locator(".back").click()
        page.locator(".case-card.reviews").click()
        responses["reviews"] = predict(1)
        assert 1 <= responses["reviews"]["answers"][0]["expected_score"] <= 5
        page.screenshot(path=str(output / "reviews-desktop.png"), full_page=True, animations="disabled")
        page.locator(".back").click()
        page.locator(".case-card.routing").click()
        responses["routing"] = predict(3)
        count = sum(method == "POST" for method, _ in requests)
        page.locator("#threshold").fill("0.5")
        assert "Would route" in page.locator(".route-outcome").inner_text()
        page.locator("#threshold").fill("1")
        assert "human review" in page.locator(".route-outcome").inner_text()
        assert sum(method == "POST" for method, _ in requests) == count, "Slider reran the model"
        page.screenshot(path=str(output / "routing-desktop.png"), full_page=True, animations="disabled")
        page.get_by_role("tab", name="Integrate", exact=True).click()
        ast.parse(page.locator("#integration-code").inner_text())
        # Real oversized context: check visible truncation without changing questions.
        page.locator("#text").fill("My card was charged twice. " * 200)
        truncated = predict(3)
        assert truncated["context_tokens_dropped"] > 0
        assert page.locator(".warning").is_visible()
        # Explicit transport-failure test: verify visible error, then retry the real API.
        page.route("**/api/predict", lambda route: route.fulfill(
            status=503, content_type="application/json", body='{"detail":"Test: server unavailable"}'))
        page.locator("#run-button").click()
        page.get_by_role("alert").get_by_text("Test: server unavailable").wait_for()
        assert page.locator("#run-button").is_enabled()
        page.unroute("**/api/predict")
        page.locator('[data-example="0"]').click()
        predict(3)
        page.set_viewport_size({"width": 390, "height": 844})
        for case in ("banking", "reviews", "routing"):
            page.goto(f"{args.url}/cases/{case}")
            page.locator("#run-button").wait_for()
            predict(1 if case == "reviews" else 3)
            no_overflow()
            page.screenshot(path=str(output / f"{case}-mobile.png"), full_page=True, animations="disabled")
        page.goto(args.url)
        page.locator(".case-card").first.wait_for()
        no_overflow()
        page.screenshot(path=str(output / "home-mobile.png"), full_page=True, animations="disabled")
        assert not errors, errors
        assert all(url.startswith(args.url) or url.startswith("data:") for _, url in requests), "External dashboard request"
        browser.close()
    (output / "verification.json").write_text(json.dumps({"passed": True,
        "checks": ["real predictions for all three cases", "one encoder pass per request",
                   "probability normalization", "slider reuses prediction", "visible truncation",
                   "input controls", "keyboard tabs", "integration Python syntax",
                   "visible error and recovery", "desktop and mobile", "no external asset requests"],
        "responses": responses}, indent=2) + "\n")
    print(f"Dashboard checks passed. Screenshots and real responses: {output}")


if __name__ == "__main__":
    main()
