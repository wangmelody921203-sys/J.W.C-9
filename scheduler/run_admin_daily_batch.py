import json
import os
import sys
import urllib.error
import urllib.request


BASE_URL = os.environ.get("AGENT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ADMIN_TOKEN = os.environ.get("AGENT_ADMIN_TOKEN", "").strip()
LOOKBACK_DAYS = os.environ.get("AGENT_BATCH_LOOKBACK_DAYS", "2").strip()
MAX_USERS = os.environ.get("AGENT_BATCH_MAX_USERS", "100").strip()
DRY_RUN = os.environ.get("AGENT_BATCH_DRY_RUN", "false").strip().lower() in {"1", "true", "yes", "y"}


def _post_json(path: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Agent-Admin-Token": ADMIN_TOKEN,
    }
    req = urllib.request.Request(f"{BASE_URL}{path}", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except Exception:
            parsed = {"raw": raw[:500]}
        return exc.code, parsed


def main() -> int:
    if not ADMIN_TOKEN:
        print("AGENT_ADMIN_TOKEN is required.")
        return 2

    try:
        lookback_days = max(1, min(30, int(LOOKBACK_DAYS)))
    except ValueError:
        lookback_days = 2
    try:
        max_users = max(1, min(500, int(MAX_USERS)))
    except ValueError:
        max_users = 100

    payload = {
        "lookback_days": lookback_days,
        "max_users": max_users,
        "dry_run": DRY_RUN,
        "create_followups": True,
    }
    print(f"POST {BASE_URL}/agent/jobs/daily-batch payload={payload}")
    status, data = _post_json("/agent/jobs/daily-batch", payload)

    if status != 200 or not data.get("ok"):
        print(f"[ERROR] status={status} data={data}")
        return 1

    print("[OK] Daily batch completed")
    print(f"user_count={data.get('user_count', 0)}")
    print(f"followups_created={data.get('followups_created', 0)}")
    print(f"followups_existing={data.get('followups_existing', 0)}")
    print(f"errors_count={data.get('errors_count', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
