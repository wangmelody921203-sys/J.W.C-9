import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


BASE_URL = os.environ.get("AGENT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
BEARER_TOKEN = os.environ.get("AGENT_BEARER_TOKEN", "").strip()
DAILY_REVIEW_DAYS = os.environ.get("AGENT_DAILY_REVIEW_DAYS", "1").strip()


def _build_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    return headers


def _request_json(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=_build_headers(),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except Exception:
            parsed = {"raw": raw[:300]}
        return exc.code, parsed


def run_daily_review() -> tuple[bool, str]:
    try:
        days = max(1, min(7, int(DAILY_REVIEW_DAYS)))
    except ValueError:
        days = 1

    query = urllib.parse.urlencode({"days": str(days)})
    status, data = _request_json("GET", f"/agent/daily-review?{query}")
    if status != 200 or not data.get("ok"):
        return False, f"daily-review failed status={status} data={data}"

    review = data.get("review", {}) if isinstance(data.get("review"), dict) else {}
    summary_lines = review.get("summary_lines", []) if isinstance(review.get("summary_lines"), list) else []
    top_emotion = review.get("top_emotion", "unknown")
    recommended_step = review.get("recommended_step", "")

    print(f"[DAILY-REVIEW] top_emotion={top_emotion}")
    for line in summary_lines[:4]:
        print(f"  - {line}")
    if recommended_step:
        print(f"  - 建議下一步: {recommended_step}")
    return True, ""


def run_next_day_followup() -> tuple[bool, str]:
    status, data = _request_json("POST", "/agent/followups/next-day", payload={})
    if status not in (200, 201) or not data.get("ok"):
        return False, f"next-day-followup failed status={status} data={data}"

    created = bool(data.get("created", False))
    reason = str(data.get("reason", "")).strip()
    task = data.get("task", {}) if isinstance(data.get("task"), dict) else {}
    task_id = str(task.get("id", "")).strip()

    if created:
        print(f"[FOLLOWUP] created task_id={task_id}")
    else:
        print(f"[FOLLOWUP] skipped reason={reason or 'not_created'} task_id={task_id}")
    return True, ""


def main() -> int:
    if not BEARER_TOKEN:
        print("AGENT_BEARER_TOKEN is required.")
        return 2

    print(f"BASE_URL={BASE_URL}")
    review_ok, review_err = run_daily_review()
    followup_ok, followup_err = run_next_day_followup()

    if review_ok and followup_ok:
        print("Daily jobs completed successfully.")
        return 0

    if not review_ok:
        print(review_err)
    if not followup_ok:
        print(followup_err)
    return 1


if __name__ == "__main__":
    sys.exit(main())
