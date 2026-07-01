"""
Test harness: replays C1-C10 against a running API server.
Start the server first, then run: python3 run_tests.py
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8320"
TRACES_DIR = Path(__file__).parent / "GenAI_SampleConversations"


def parse_trace(path: Path) -> list[dict]:
    text = path.read_text()
    turns = []
    current_role = None
    current_content = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "**User**":
            if current_role and current_content:
                turns.append({"role": current_role, "content": " ".join(current_content).strip()})
            current_role = "user"
            current_content = []
        elif stripped == "**Agent**":
            if current_role and current_content:
                turns.append({"role": current_role, "content": " ".join(current_content).strip()})
            current_role = "assistant"
            current_content = []
        elif stripped.startswith("> "):
            current_content.append(stripped[2:])
    if current_role and current_content:
        turns.append({"role": current_role, "content": " ".join(current_content).strip()})
    return [t for t in turns if t["content"]]


def run_trace(turns: list[dict], trace_name: str) -> dict:
    messages = []
    results = {"user_turns": [], "schema_issues": [], "warnings": []}

    for turn in turns:
        if turn["role"] != "user":
            continue
        messages.append({"role": "user", "content": turn["content"]})
        payload = {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}

        try:
            resp = httpx.post(f"{BASE_URL}/chat", json=payload, timeout=15)
            data = resp.json()
        except Exception as e:
            results["user_turns"].append({
                "input": turn["content"][:80],
                "error": str(e),
            })
            continue

        reply = data.get("reply", "")
        recommendations = data.get("recommendations", [])
        eoc = data.get("end_of_conversation", False)

        # Schema checks
        issues = []
        if not isinstance(reply, str):
            issues.append("reply not string")
        if not isinstance(recommendations, list):
            issues.append("recommendations not list")
        elif len(recommendations) > 10:
            issues.append(f"{len(recommendations)} recs > 10")
        else:
            for i, rec in enumerate(recommendations):
                if not isinstance(rec, dict):
                    issues.append(f"rec[{i}] not dict")
                    continue
                if not rec.get("name"):
                    issues.append(f"rec[{i}] missing name")
                if not rec.get("url"):
                    issues.append(f"rec[{i}] missing url")
                if not isinstance(rec.get("test_type"), list):
                    issues.append(f"rec[{i}] test_type not list")
        if not isinstance(eoc, bool):
            issues.append("end_of_conversation not bool")

        if issues:
            results["schema_issues"].append({"turn": len(messages), "issues": issues})

        # Check for hallucinated URLs (empty URLs)
        for rec in recommendations:
            if rec.get("url") and "shl.com" not in rec["url"]:
                results["warnings"].append(f"Non-SHL URL for '{rec.get('name')}'")

        results["user_turns"].append({
            "input": turn["content"][:80],
            "reply_preview": reply[:100],
            "rec_count": len(recommendations),
            "eoc": eoc,
            "has_issues": bool(issues),
        })

        messages.append({"role": "assistant", "content": reply})

        if eoc:
            break

    return results


def main():
    traces = sorted(TRACES_DIR.glob("C*.md"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    print(f"Found {len(traces)} traces\n")

    all_issues = []
    all_warnings = []
    stats = {"total_turns": 0, "total_recs": 0, "schema_errors": 0}

    for path in traces:
        name = path.stem
        turns = parse_trace(path)
        print(f"  {name}: {len(turns)} turns ({sum(1 for t in turns if t['role']=='user')} user)")

        results = run_trace(turns, name)
        user_turns = results["user_turns"]
        all_issues.extend(results["schema_issues"])
        all_warnings.extend(results["warnings"])
        stats["total_turns"] += len(user_turns)
        stats["schema_errors"] += len(results["schema_issues"])

        for ut in user_turns:
            if "error" in ut:
                print(f"    ! T{user_turns.index(ut)+1}: input='{ut['input'][:60]}' → ERROR: {ut['error']}")
                continue
            stats["total_recs"] += ut["rec_count"]
            marker = "⚠ " if ut["has_issues"] else "  "
            print(f"    {marker}T{user_turns.index(ut)+1}: input='{ut['input'][:60]}' → "
                  f"{ut['rec_count']} recs, eoc={ut['eoc']}")

        if results["schema_issues"]:
            for issue in results["schema_issues"]:
                print(f"    ✗ Turn {issue['turn']} schema: {issue['issues']}")
        if results["warnings"]:
            for w in results["warnings"]:
                print(f"    ! {w}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(traces)} traces, {stats['total_turns']} turns, "
          f"{stats['total_recs']} total recommendations")
    print(f"Schema issues: {stats['schema_errors']}")
    print(f"Warnings: {len(all_warnings)}")

    if stats["schema_errors"]:
        print("\n❌ Schema errors found - fix before submission")
        sys.exit(1)
    else:
        print("\n✅ All schema checks pass")


if __name__ == "__main__":
    main()
