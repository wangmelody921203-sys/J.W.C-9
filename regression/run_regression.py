import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = os.environ.get("AGENT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
AUTH_TOKEN = os.environ.get("AGENT_BEARER_TOKEN", "").strip()
CASES_PATH = Path(__file__).with_name("agent_regression_cases.json")


def post_generate(payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

    req = urllib.request.Request(f"{BASE_URL}/generate", data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"error": f"http_{exc.code}", "raw": body[:300]}
        return exc.code, parsed


def assert_common(case_id: str, status_code: int, data: dict) -> list[str]:
    errors = []
    if status_code >= 500 and data.get("error") not in {"groq_error", "groq_unavailable"}:
        errors.append(f"{case_id}: unexpected server error status={status_code} error={data.get('error')}")

    if "observability" not in data:
        errors.append(f"{case_id}: missing observability")
        return errors

    obs = data.get("observability", {})
    required_keys = {
        "used_memories",
        "used_memory_count",
        "used_tools",
        "used_tool_count",
        "latency_ms",
        "fallback_used",
        "fallback_reason",
        "safety_mode",
        "crisis_level",
        "crisis_phase",
    }
    missing = [k for k in sorted(required_keys) if k not in obs]
    if missing:
        errors.append(f"{case_id}: observability missing keys {missing}")

    if "reply" not in data and "fallback" not in data:
        errors.append(f"{case_id}: neither reply nor fallback present")

    return errors


def run_case(case: dict) -> tuple[bool, list[str], bool]:
    case_id = str(case.get("id", "unknown"))
    requires_auth = bool(case.get("requires_auth", False))
    if requires_auth and not AUTH_TOKEN:
        return True, [f"{case_id}: skipped (requires auth token)"], True

    messages = []
    for item in case.get("history", []):
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": str(item.get("content", ""))})
    messages.append({"role": "user", "content": str(case.get("message", ""))})

    payload = {
        "emotion": str(case.get("emotion", "unknown")),
        "persona": "courage_coach",
        "messages": messages,
    }

    status_code, data = post_generate(payload)
    errors = assert_common(case_id, status_code, data)

    expect = case.get("expect", {}) if isinstance(case.get("expect"), dict) else {}
    obs = data.get("observability", {}) if isinstance(data.get("observability"), dict) else {}

    expected_safety = str(expect.get("safety_mode", "")).strip()
    if expected_safety and str(obs.get("safety_mode", "")).strip() != expected_safety:
        errors.append(f"{case_id}: expected safety_mode={expected_safety}, got {obs.get('safety_mode')}")

    expected_phase = str(expect.get("crisis_phase", "")).strip()
    if expected_phase and str(obs.get("crisis_phase", "")).strip() != expected_phase:
        errors.append(f"{case_id}: expected crisis_phase={expected_phase}, got {obs.get('crisis_phase')}")

    expected_tools = expect.get("tool_names_any", [])
    if isinstance(expected_tools, list) and expected_tools:
        tool_names = {str(item.get("name", "")).strip() for item in obs.get("used_tools", []) if isinstance(item, dict)}
        if not all(tool in tool_names for tool in expected_tools):
            errors.append(f"{case_id}: expected tools {expected_tools}, got {sorted(tool_names)}")

    reply_text = str(data.get("reply") or data.get("fallback") or "")
    expected_reply_words = expect.get("reply_contains_any", [])
    if isinstance(expected_reply_words, list) and expected_reply_words:
        if not any(str(word) in reply_text for word in expected_reply_words):
            errors.append(f"{case_id}: reply missing any of {expected_reply_words}")

    return len(errors) == 0, errors, False


def main() -> int:
    if not CASES_PATH.exists():
        print(f"Cases file not found: {CASES_PATH}")
        return 2

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        print("Invalid cases format: expected list")
        return 2

    passed = 0
    failed = 0
    skipped = 0

    for case in cases:
        ok, details, is_skipped = run_case(case)
        if is_skipped:
            skipped += 1
            print(f"[SKIP] {details[0]}")
            continue
        if ok:
            passed += 1
            print(f"[PASS] {case.get('id', 'unknown')}")
        else:
            failed += 1
            print(f"[FAIL] {case.get('id', 'unknown')}")
            for line in details:
                print(f"       - {line}")

    print("\n=== Regression Summary ===")
    print(f"BASE_URL={BASE_URL}")
    print(f"TOTAL={len(cases)} PASS={passed} FAIL={failed} SKIP={skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
