"""
Automated test harness for SHL Assessment Advisor.
Parses C1-C10.md traces and replays them against the API.
"""

import itertools
import json
import re
import sys
import time
from pathlib import Path

import httpx
import yaml

BASE_URL = "http://localhost:8319"
TRACES_DIR = Path(__file__).resolve().parent.parent / "GenAI_SampleConversations"


def parse_traces() -> list[dict]:
    """Parse C1-C10.md files into structured conversation dicts."""
    traces = []
    md_files = sorted(TRACES_DIR.glob("C*.md"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    for path in md_files:
        text = path.read_text()
        turns = []
        current_role = None
        current_content = []

        for line in text.split("\n"):
            if line.startswith("**User**"):
                if current_role and current_content:
                    turns.append({"role": current_role, "content": "\n".join(current_content).strip()})
                current_role = "user"
                current_content = []
            elif line.startswith("**Agent**"):
                if current_role and current_content:
                    turns.append({"role": current_role, "content": "\n".join(current_content).strip()})
                current_role = "assistant"
                current_content = []
            elif line.startswith("> "):
                current_content.append(line[2:])
            elif line.strip() and current_role and not line.startswith("_") and not line.startswith("|") and not line.startswith("#"):
                pass  # skip metadata lines

        if current_role and current_content:
            turns.append({"role": current_role, "content": "\n".join(current_content).strip()})

        traces.append({
            "name": path.stem,
            "turns": [t for t in turns if t["content"]],
        })
    return traces


def replay_trace(trace: dict, base_url: str) -> list[dict]:
    """Replay a trace turn-by-turn and collect responses."""
    responses = []
    messages = []

    for turn in trace["turns"]:
        if turn["role"] == "user":
            messages.append(turn)
            payload = {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}
            try:
                resp = httpx.post(f"{base_url}/chat", json=payload, timeout=10)
                data = resp.json()
                responses.append({
                    "turn": len(messages),
                    "input": turn["content"],
                    "output": data,
                    "status": resp.status_code,
                })
                if data.get("reply"):
                    messages.append({"role": "assistant", "content": data["reply"]})
            except Exception as e:
                responses.append({
                    "turn": len(messages),
                    "input": turn["content"],
                    "error": str(e),
                    "output": None,
                    "status": 0,
                })
    return responses


def check_schema(data: dict) -> list[str]:
    """Validate response schema. Returns list of issues."""
    issues = []
    if not isinstance(data, dict):
        return ["Response is not a dict"]
    if "reply" not in data:
        issues.append("Missing 'reply' field")
    elif not isinstance(data["reply"], str):
        issues.append("'reply' must be string")
    if "recommendations" not in data:
        issues.append("Missing 'recommendations' field")
    elif not isinstance(data["recommendations"], list):
        issues.append("'recommendations' must be list")
    elif len(data["recommendations"]) > 10:
        issues.append(f"'recommendations' has {len(data['recommendations'])} items (max 10)")
    else:
        for i, rec in enumerate(data["recommendations"]):
            if not isinstance(rec, dict):
                issues.append(f"recommendations[{i}] is not a dict")
                continue
            if "name" not in rec or not rec["name"]:
                issues.append(f"recommendations[{i}] missing 'name'")
            if "url" not in rec or not rec["url"]:
                issues.append(f"recommendations[{i}] missing 'url'")
            if "test_type" not in rec:
                issues.append(f"recommendations[{i}] missing 'test_type'")
    if "end_of_conversation" not in data:
        issues.append("Missing 'end_of_conversation' field")
    elif not isinstance(data["end_of_conversation"], bool):
        issues.append("'end_of_conversation' must be bool")
    return issues


def main():
    traces = parse_traces()
    print(f"Found {len(traces)} traces: {[t['name'] for t in traces]}")

    all_issues = []
    for trace in traces:
        print(f"\n{'='*60}")
        print(f"Replaying {trace['name']} ({len(trace['turns'])} turns)")
        print(f"{'='*60}")

        responses = replay_trace(trace, BASE_URL)

        for i, resp in enumerate(responses):
            turn_num = resp["turn"]
            if resp.get("error"):
                print(f"  Turn {turn_num}: ERROR - {resp['error']}")
                continue

            issues = check_schema(resp.get("output", {}))
            if issues:
                print(f"  Turn {turn_num}: SCHEMA ISSUES: {issues}")
                all_issues.append({"trace": trace["name"], "turn": turn_num, "issues": issues})
            else:
                rec_count = len(resp["output"].get("recommendations", []))
                eoc = resp["output"].get("end_of_conversation", False)
                print(f"  Turn {turn_num}: ✓ {rec_count} recs, end={eoc}")

                # Check for hallucination: URLs must be non-empty
                for rec in resp["output"].get("recommendations", []):
                    if not rec.get("url"):
                        print(f"    WARNING: '{rec.get('name')}' has empty URL")

    # Summary
    print(f"\n{'='*60}")
    print(f"SCHEMA ISSUES: {len(all_issues)}")
    for issue in all_issues:
        print(f"  {issue['trace']} T{issue['turn']}: {issue['issues']}")

    if all_issues:
        sys.exit(1)
    print("\nAll schema checks passed!")


if __name__ == "__main__":
    main()
