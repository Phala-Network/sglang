"""Serial, finite, no-retry protocol gate. Credential arrives only as stdin JSON."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

from fixtures import canonical, sha, wire


def now():
    return datetime.now(timezone.utc).isoformat()


def strict_loads(value):
    def invalid_constant(text):
        raise ValueError("non-JSON numeric constant: " + text)

    return json.loads(value, parse_constant=invalid_constant)


def verify_manifest(manifest):
    payload = dict(manifest)
    expected = payload.pop("payload_sha256")
    assert sha(payload) == expected, "manifest payload hash mismatch"
    cases = manifest["cases"]
    assert manifest["request_order"] == [c["case_id"] for c in cases]
    assert len({c["case_id"] for c in cases}) == len(cases)
    for case in cases:
        assert sha(case["body"]) == case["body_sha256"], case["case_id"]
        assert (
            hashlib.sha256(wire(case["body"]).encode()).hexdigest()
            == case["body_wire_sha256"]
        ), "wire ordering/hash mismatch"
    conflict = [c for c in cases if c["group"] == "conflict35"]
    assert len(conflict) == 35
    assert Counter(c["cohort"] for c in conflict) == {"fixed15": 15, "random20": 20}
    for c in conflict:
        b = c["body"]
        assert set(b) == {
            "model",
            "temperature",
            "max_tokens",
            "reasoning_effort",
            "response_format",
            "messages",
        }
        assert (
            b["temperature"] == 0
            and b["max_tokens"] == 2000
            and b["reasoning_effort"] == "none"
        )
        assert len(b["messages"]) == 1 and b["messages"][0]["role"] == "user"
        assert b["response_format"]["json_schema"]["schema"] == c["expect"]["schema"]


def notable_prefix(content):
    """Count complete strings only in the root notable array; never repair JSON."""
    decoder = json.JSONDecoder()
    text, i = content, 0
    values = []

    def skip(p):
        while p < len(text) and text[p].isspace():
            p += 1
        return p

    i = skip(i)
    if i >= len(text) or text[i] != "{":
        return {"count": None, "kind": "unknown_not_root_object"}
    i += 1
    try:
        while True:
            i = skip(i)
            if text[i] == "}":
                return {"count": None, "kind": "notable_missing"}
            key, i = decoder.raw_decode(text, i)
            if not isinstance(key, str):
                return {"count": None, "kind": "malformed_key"}
            i = skip(i)
            if text[i] != ":":
                return {"count": None, "kind": "malformed_colon"}
            i = skip(i + 1)
            if key != "notable":
                _, i = decoder.raw_decode(text, i)
                i = skip(i)
                if text[i] == ",":
                    i += 1
                    continue
                return {"count": None, "kind": "notable_missing"}
            if text[i] != "[":
                return {"count": None, "kind": "notable_not_array"}
            i += 1
            while True:
                i = skip(i)
                if i < len(text) and text[i] == "]":
                    return {
                        "count": len(values),
                        "kind": "complete_array",
                        "partial_item": False,
                        "duplicate_complete_items": sum(
                            n - 1 for n in Counter(values).values()
                        ),
                    }
                try:
                    value, end = decoder.raw_decode(text, i)
                except (ValueError, IndexError):
                    return {
                        "count": len(values),
                        "kind": "lower_bound",
                        "partial_item": i < len(text),
                        "duplicate_complete_items": sum(
                            n - 1 for n in Counter(values).values()
                        ),
                    }
                if not isinstance(value, str):
                    return {"count": len(values), "kind": "non_string_item"}
                values.append(value)
                i = skip(end)
                if i == len(text):
                    return {
                        "count": len(values),
                        "kind": "lower_bound",
                        "partial_item": False,
                        "duplicate_complete_items": sum(
                            n - 1 for n in Counter(values).values()
                        ),
                    }
                if text[i] == ",":
                    i += 1
                elif text[i] != "]":
                    return {"count": len(values), "kind": "malformed_delimiter"}
    except (ValueError, IndexError):
        return {"count": None, "kind": "prefix_unavailable"}


def reasoning_fields(message):
    result = {}
    for name in ("reasoning", "reasoning_content", "reasoning_details"):
        if name not in message:
            result[name] = {"present": False, "type": "missing", "characters": None}
            continue
        value = message[name]
        item = {
            "present": True,
            "type": type(value).__name__,
            "characters": len(value) if isinstance(value, str) else None,
        }
        if isinstance(value, list):
            item["text_entries"] = [
                {"index": i, "field": key, "characters": len(v)}
                for i, entry in enumerate(value)
                if isinstance(entry, dict)
                for key, v in entry.items()
                if key in ("text", "content") and isinstance(v, str)
            ]
        result[name] = item
    return result


def visible_reasoning(fields):
    return any(
        (item["characters"] or 0) > 0
        or any(e["characters"] > 0 for e in item.get("text_entries", []))
        for item in fields.values()
    )


def merge_sse(events, allowed_names):
    choices, usage, failures, response_id = {}, None, [], None
    for event in events:
        if "error" in event:
            failures.append("stream_error_envelope")
        response_id = event.get("id", response_id)
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            index = choice.get("index")
            if not isinstance(index, int):
                failures.append("choice_index_invalid")
                continue
            state = choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"content": ""},
                    "finish_reason": None,
                    "_tools": {},
                },
            )
            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice["finish_reason"]
            if choice.get("native_finish_reason") is not None:
                state["native_finish_reason"] = choice["native_finish_reason"]
            delta = choice.get("delta") or {}
            for name in ("content", "reasoning", "reasoning_content"):
                if isinstance(delta.get(name), str):
                    state["message"][name] = (
                        state["message"].get(name, "") + delta[name]
                    )
            if "reasoning_details" in delta:
                value = delta["reasoning_details"]
                if isinstance(value, list):
                    state["message"].setdefault("reasoning_details", []).extend(value)
                else:
                    state["message"]["reasoning_details"] = value
            for part in delta.get("tool_calls") or []:
                ti = part.get("index")
                if not isinstance(ti, int):
                    failures.append("tool_index_invalid")
                    continue
                call = state["_tools"].setdefault(
                    ti,
                    {
                        "index": ti,
                        "id": "",
                        "type": "",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                for key in ("id", "type"):
                    if part.get(key):
                        if call[key] and call[key] != part[key]:
                            failures.append("tool_" + key + "_changed")
                        call[key] = part[key]
                function = part.get("function") or {}
                if function.get("name"):
                    new, prior = function["name"], call["function"]["name"]
                    call["function"]["name"] = prior if new == prior else prior + new
                arguments = function.get("arguments")
                if arguments is not None:
                    if not isinstance(arguments, str):
                        failures.append("arguments_wire_type_invalid")
                    else:
                        if arguments and call["function"]["name"] not in allowed_names:
                            failures.append("orphan_tool_argument_delta")
                        call["function"]["arguments"] += arguments
    output = []
    for index in sorted(choices):
        state = choices[index]
        tool_map = state.pop("_tools")
        if tool_map:
            state["message"]["tool_calls"] = [tool_map[i] for i in sorted(tool_map)]
        output.append(state)
    return {"id": response_id, "choices": output, "usage": usage}, failures


def schema_errors(value, schema):
    from jsonschema import Draft202012Validator

    return [error.message for error in Draft202012Validator(schema).iter_errors(value)]


def evaluate(case, envelope, stream_failures=()):
    expect, body = case["expect"], case["body"]
    failures = list(stream_failures)
    if not isinstance(envelope, dict):
        return {
            "failure_classes": failures + ["response_envelope_invalid"],
            "choices": [],
        }
    if envelope.get("error"):
        failures.append("error_envelope")
    choices = envelope.get("choices") or []
    if not isinstance(choices, list) or any(
        not isinstance(choice, dict) for choice in choices
    ):
        return {
            "failure_classes": failures + ["choices_envelope_invalid"],
            "choices": [],
            "usage": envelope.get("usage"),
        }
    if len(choices) != body.get("n", 1):
        failures.append("choice_count_mismatch")
    if sorted(c.get("index", i) for i, c in enumerate(choices)) != list(
        range(body.get("n", 1))
    ):
        failures.append("choice_index_mismatch")
    usage = envelope.get("usage")
    counters = {}
    if isinstance(usage, dict):
        if "reasoning_tokens" in usage:
            counters["reasoning_tokens"] = usage["reasoning_tokens"]
        detail = usage.get("completion_tokens_details")
        if isinstance(detail, dict) and "reasoning_tokens" in detail:
            counters["completion_tokens_details.reasoning_tokens"] = detail[
                "reasoning_tokens"
            ]
    rows = []
    defined = {
        t["function"]["name"]: t["function"]["parameters"]
        for t in body.get("tools", [])
    }
    for index, choice in enumerate(choices):
        message = choice.get("message") or {}
        content, finish = message.get("content"), choice.get("finish_reason")
        fields = reasoning_fields(message)
        row = {
            "index": choice.get("index", index),
            "finish_reason": finish,
            "native_finish_reason": choice.get("native_finish_reason"),
            "content_type": type(content).__name__,
            "content_characters": len(content) if isinstance(content, str) else None,
            "reasoning_fields": fields,
            "json_parse_ok": None,
            "json_schema_ok": None,
            "semantic_match": None,
        }
        mode = expect.get("reasoning")
        if mode in ("off", "exclude") and visible_reasoning(fields):
            failures.append("reasoning_visible_when_" + mode)
        if mode == "off" and any(
            isinstance(v, (float, int)) and v > 0 for v in counters.values()
        ):
            failures.append("positive_reasoning_tokens_when_off")
        if mode == "on" and not visible_reasoning(fields):
            failures.append("reasoning_inclusion_missing")
        if expect["kind"] == "tools":
            calls = message.get("tool_calls") or []
            if finish != "tool_calls":
                failures.append("finish_reason_mismatch")
            if len(calls) != len(expect["calls"]):
                failures.append("tool_call_count_mismatch")
            ids = [call.get("id") for call in calls]
            if any(not isinstance(i, str) or not i for i in ids) or len(
                set(ids)
            ) != len(ids):
                failures.append("tool_id_missing_or_collision")
            actual = []
            for call in calls:
                function = call.get("function") or {}
                name, arguments = function.get("name"), function.get("arguments")
                if call.get("type") != "function":
                    failures.append("tool_call_type_invalid")
                if name not in defined:
                    failures.append("unknown_tool")
                try:
                    if not isinstance(arguments, str):
                        raise ValueError("arguments is not a JSON string")
                    parsed = strict_loads(arguments)
                    if name in defined and schema_errors(parsed, defined[name]):
                        failures.append("arguments_schema_mismatch")
                    actual.append({"name": name, "arguments": parsed})
                except (ValueError, TypeError):
                    failures.append("arguments_invalid_json")
            row["tool_count"] = len(calls)
            row["semantic_match"] = sorted(map(canonical, actual)) == sorted(
                map(canonical, expect["calls"])
            )
            if not row["semantic_match"]:
                failures.append("tool_semantic_mismatch")
        else:
            if finish != "stop":
                failures.append("finish_reason_mismatch")
            if message.get("tool_calls"):
                failures.append("unexpected_tool_call")
            if not isinstance(content, str) or not content:
                failures.append("content_missing")
            elif expect["kind"] == "json":
                try:
                    value = strict_loads(content)
                    row["json_parse_ok"] = True
                    errors = schema_errors(value, expect["schema"])
                    row["json_schema_ok"], row["schema_errors"] = not errors, errors
                    if errors:
                        failures.append("schema_mismatch")
                    if "value" in expect:
                        row["semantic_match"] = canonical(value) == canonical(
                            expect["value"]
                        )
                        if not row["semantic_match"]:
                            failures.append("semantic_mismatch")
                except (ValueError, TypeError) as exc:
                    row["json_parse_ok"] = False
                    row["parse_error"] = str(exc)
                    failures.append("invalid_json")
            else:
                if "text" in expect:
                    got, expected = content.strip(), expect["text"]
                    if expect.get("casefold"):
                        got, expected = got.casefold(), expected.casefold()
                    row["semantic_match"] = got == expected
                else:
                    row["semantic_match"] = all(
                        s in content for s in expect.get("contains", [])
                    )
                if not row["semantic_match"]:
                    failures.append("semantic_mismatch")
        if finish == "length":
            row["length_diagnostics"] = {
                "first_300": content[:300] if isinstance(content, str) else content,
                "last_300": content[-300:] if isinstance(content, str) else content,
                "notable": notable_prefix(content)
                if isinstance(content, str)
                else {"count": None, "kind": "content_not_string"},
            }
            failures.append(
                "finish_length_valid_json" if row["json_parse_ok"] else "truncation"
            )
        rows.append(row)
    return {
        "failure_classes": sorted(set(failures)),
        "choices": rows,
        "usage": usage,
        "reasoning_token_counters": counters,
        "reasoning_accounting_available": bool(counters),
        "response_id": envelope.get("id"),
    }


def exchange(target, token, body=None, timeout=300, cancel_after=None):
    connection = http.client.HTTPConnection(
        target.hostname, target.port, timeout=timeout
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": "Bearer " + token,
    }
    started = time.monotonic()
    raw, events, failures, data_lines = bytearray(), [], [], []
    done = cancelled = False
    status, safe_headers = None, {}
    semantic_events = 0
    try:
        path = target.path if body is not None else "/v1/models"
        connection.request(
            "POST" if body is not None else "GET",
            path,
            body=wire(body).encode() if body is not None else None,
            headers=headers,
        )
        response = connection.getresponse()
        status = response.status
        safe_headers = {
            k.lower(): v
            for k, v in response.getheaders()
            if k.lower() in ("content-type", "x-request-id", "request-id")
        }
        if body is None or not body.get("stream") or status != 200:
            raw.extend(response.read())
        else:
            if "text/event-stream" not in safe_headers.get("content-type", ""):
                failures.append("sse_content_type_invalid")
            while True:
                if time.monotonic() - started > timeout:
                    raise TimeoutError("request total timeout")
                line = response.readline()
                if not line:
                    if data_lines:
                        failures.append("sse_event_without_blank_terminator")
                    break
                raw.extend(line)
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError("response exceeds finite protocol evidence bound")
                stripped = line.rstrip(b"\r\n")
                if stripped.startswith(b"data:"):
                    data_lines.append(stripped[5:].lstrip())
                elif stripped == b"" and data_lines:
                    payload = b"\n".join(data_lines)
                    data_lines.clear()
                    if payload == b"[DONE]":
                        done = True
                        break
                    try:
                        event = strict_loads(payload.decode("utf-8"))
                        if not isinstance(event, dict):
                            raise ValueError("SSE event not an object")
                        events.append(event)
                        semantic = any(
                            any(
                                (c.get("delta") or {}).get(key)
                                for key in (
                                    "content",
                                    "reasoning_content",
                                    "reasoning",
                                    "tool_calls",
                                )
                            )
                            for c in event.get("choices", [])
                        )
                        semantic_events += int(semantic)
                        if cancel_after is not None and semantic_events >= cancel_after:
                            cancelled = True
                            break
                    except (ValueError, UnicodeError):
                        failures.append("sse_parse_error")
        return {
            "http_status": status,
            "response_headers": safe_headers,
            "raw": bytes(raw),
            "events": events,
            "sse_done": done,
            "client_cancelled": cancelled,
            "semantic_events": semantic_events,
            "sse_failures": failures,
            "latency_s": time.monotonic() - started,
        }
    except Exception as exc:
        partial = getattr(exc, "partial", None)
        if isinstance(partial, bytes):
            raw.extend(partial)
        return {
            "http_status": status,
            "response_headers": safe_headers,
            "raw": bytes(raw),
            "events": events,
            "sse_done": done,
            "client_cancelled": cancelled,
            "semantic_events": semantic_events,
            "sse_failures": failures,
            "latency_s": time.monotonic() - started,
            "transport_exception": type(exc).__name__,
            "transport_error": str(exc),
        }
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--group", choices=("all", "conflict35", "p0"), default="all")
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--runtime-source", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--pacing", type=float, default=0.2)
    parser.add_argument("--static-check", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_manifest(manifest)
    from jsonschema import Draft202012Validator

    for case in manifest["cases"]:
        if "schema" in case["expect"]:
            Draft202012Validator.check_schema(case["expect"]["schema"])
        for tool in case["body"].get("tools", []):
            Draft202012Validator.check_schema(tool["function"]["parameters"])
    if args.static_check:
        print(
            canonical(
                {
                    "manifest": "PASS",
                    "groups": dict(Counter(c["group"] for c in manifest["cases"])),
                    "payload_sha256": manifest["payload_sha256"],
                }
            )
        )
        return 0
    target = urlsplit(manifest["endpoint"])
    assert (
        target.scheme == "http"
        and target.hostname == "127.0.0.1"
        and target.port == 18301
        and target.path == "/v1/chat/completions"
        and not target.query
    )
    credentials = json.loads(sys.stdin.read())
    token = credentials.get("token")
    if not isinstance(token, str) or not token or "\n" in token or "\r" in token:
        raise ValueError(
            "stdin must contain a nonempty token field; no fallback credential source"
        )
    del credentials
    args.output.mkdir(parents=True, exist_ok=False)
    rawdir = args.output / "raw"
    rawdir.mkdir()

    def save(path, value):
        text = json.dumps(value, ensure_ascii=False, indent=2).replace(
            token, "[REDACTED_CREDENTIAL]"
        )
        path.write_text(text + "\n", encoding="utf-8")

    cases = [
        c for c in manifest["cases"] if args.group == "all" or c["group"] == args.group
    ]
    save(
        args.output / "run-inputs.json",
        {
            "arm": args.arm,
            "runtime_image": args.runtime_image,
            "runtime_source": args.runtime_source,
            "endpoint": manifest["endpoint"],
            "endpoint_layer": "direct localhost backend",
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "manifest_payload_sha256": manifest["payload_sha256"],
            "group": args.group,
            "planned": len(cases),
            "timeout_s": args.timeout,
            "pacing_s": args.pacing,
            "automatic_retries": 0,
            "credential_source": "parent root TOKEN via stdin; value not retained",
            "started_at": now(),
        },
    )
    save(args.output / "manifest.json", manifest)
    rows, stopped = [], None
    for phase in ("before",):
        try:
            health = exchange(target, token, timeout=min(args.timeout, 30))
            save(
                args.output / f"health-{phase}.json",
                {
                    "http_status": health["http_status"],
                    "body": health["raw"].decode("utf-8", "replace"),
                },
            )
            if health["http_status"] != 200:
                stopped = "readiness_or_auth_failure"
        except Exception as exc:
            save(
                args.output / f"health-{phase}.json",
                {"exception": type(exc).__name__, "error": str(exc)},
            )
            stopped = "readiness_transport_error"
    for ordinal, case in enumerate(cases, 1):
        if stopped:
            break
        row = {
            "ordinal": ordinal,
            "case_id": case["case_id"],
            "group": case["group"],
            "cohort": case.get("cohort"),
            "city": case.get("city"),
            "suffix": case.get("suffix"),
            "timestamp": now(),
            "request_sha256": case["body_sha256"],
            "request_wire_sha256": case["body_wire_sha256"],
            "schema_sha256": sha(case["expect"]["schema"])
            if "schema" in case["expect"]
            else None,
            "stream": case["body"].get("stream", False),
            "retry_count": 0,
        }
        raw = b""
        try:
            result = exchange(
                target,
                token,
                case["body"],
                args.timeout,
                case["expect"].get("cancel_after_semantic_events"),
            )
            raw = result.pop("raw")
            events = result.pop("events")
            row.update(result)
            if "transport_error" in row:
                row["failure_classes"] = ["transport_error"]
                stopped = "transport_availability_failure"
            elif row["http_status"] != 200:
                row["failure_classes"] = [
                    "rate_limited" if row["http_status"] == 429 else "http_error"
                ]
                if row["http_status"] in (401, 403) or row["http_status"] >= 500:
                    stopped = "auth_or_server_availability_failure"
            elif case["expect"]["kind"] == "cancel":
                row["failure_classes"] = result["sse_failures"] + (
                    [] if row["client_cancelled"] else ["cancel_not_exercised"]
                )
                row["backend_cleanup_verified"] = False
            else:
                if case["body"].get("stream"):
                    names = {
                        t["function"]["name"] for t in case["body"].get("tools", [])
                    }
                    envelope, errors = merge_sse(events, names)
                    errors += result["sse_failures"]
                    if not result["sse_done"]:
                        errors.append("sse_done_missing")
                else:
                    envelope, errors = strict_loads(raw.decode("utf-8")), []
                row.update(evaluate(case, envelope, errors))
                save(
                    rawdir / f"{ordinal:03d}-{case['case_id']}.envelope.json", envelope
                )
        except (ValueError, UnicodeError) as exc:
            row["failure_classes"] = ["response_json_invalid"]
            row["error"] = str(exc)
        except Exception as exc:
            row["failure_classes"] = ["transport_error"]
            row["exception_type"] = type(exc).__name__
            row["error"] = str(exc)
            stopped = "transport_availability_failure"
        redacted = raw.replace(token.encode(), b"[REDACTED_CREDENTIAL]")
        rawpath = (
            rawdir
            / f"{ordinal:03d}-{case['case_id']}.{'sse' if row['stream'] else 'json'}"
        )
        rawpath.write_bytes(redacted)
        row.update(
            raw_response_path=str(rawpath.relative_to(args.output)),
            raw_response_sha256=hashlib.sha256(redacted).hexdigest(),
            credential_redacted=(redacted != raw),
        )
        row["status"] = (
            "FAIL"
            if row["failure_classes"]
            else (
                "CANCEL_SENT_REQUIRES_BACKEND_CHECK"
                if case["expect"]["kind"] == "cancel"
                else "PASS"
            )
        )
        row = json.loads(canonical(row).replace(token, "[REDACTED_CREDENTIAL]"))
        rows.append(row)
        with (args.output / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(canonical(row) + "\n")
        choices = row.get("choices") or []
        completion = (row.get("usage") or {}).get("completion_tokens")
        print(
            f"{ordinal:02d} | {case['case_id']} {case.get('city', '')} {case.get('suffix', '')} | HTTP {row.get('http_status')} | finish={[c['finish_reason'] for c in choices]} | completion={completion} | reasoning={[c['reasoning_fields'] for c in choices]} | content_chars={[c['content_characters'] for c in choices]} | json={[c['json_parse_ok'] for c in choices]} | {row['status']}",
            flush=True,
        )
        for choice in choices:
            if "length_diagnostics" in choice:
                print(
                    "LENGTH_DIAGNOSTICS " + canonical(choice["length_diagnostics"]),
                    flush=True,
                )
        time.sleep(max(0, args.pacing))
    try:
        health = exchange(target, token, timeout=min(args.timeout, 30))
        save(
            args.output / "health-after.json",
            {
                "http_status": health["http_status"],
                "body": health["raw"].decode("utf-8", "replace"),
            },
        )
        if health["http_status"] != 200:
            stopped = stopped or "post_run_readiness_failed"
    except Exception as exc:
        save(
            args.output / "health-after.json",
            {"exception": type(exc).__name__, "error": str(exc)},
        )
        stopped = stopped or "post_run_readiness_transport_error"
    summary = {
        "finished_at": now(),
        "planned": len(cases),
        "attempted": len(rows),
        "unattempted": len(cases) - len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "failure_classes": dict(Counter(f for r in rows for f in r["failure_classes"])),
        "stop_reason": stopped,
        "cohorts": {
            name: {
                "attempted": sum(r.get("cohort") == name for r in rows),
                "wire_pass": sum(
                    r.get("cohort") == name and r["status"] == "PASS" for r in rows
                ),
                "reasoning_accounting_unknown": sum(
                    r.get("cohort") == name
                    and not r.get("reasoning_accounting_available", False)
                    for r in rows
                ),
            }
            for name in ("fixed15", "random20")
        },
        "finite_subset_only": True,
        "not_proven": [
            "full corpus/repetition pass",
            "provider-chain acceptance",
            "backend cancellation cleanup without external counters",
            "generation-disabled reasoning when accounting is absent",
            "load throughput/capacity",
        ],
    }
    save(args.output / "summary.json", summary)
    print(canonical(summary), flush=True)
    return int(
        bool(
            stopped
            or len(rows) != len(cases)
            or any(r["status"] == "FAIL" for r in rows)
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
