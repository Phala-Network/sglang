import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import fixtures
import run_protocol as runner


class Assets(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(
            (Path(__file__).parent / "manifest.json").read_text(encoding="utf-8")
        )

    def test_frozen_manifest_and_exact_35(self):
        runner.verify_manifest(self.manifest)
        self.assertEqual(len(self.manifest["cases"]), 54)
        fixed = self.manifest["cases"][:15]
        self.assertEqual(
            fixed[0]["body"]["messages"][0]["content"],
            "Think about it carefully, step by step, and reason through what you know about Osaka before answering. Give facts about Osaka. [r217058]",
        )
        self.assertEqual(fixed[5]["city"], "Cordoba")
        self.assertEqual(fixed[10]["city"], "Tallinn")
        for case in self.manifest["cases"][:35]:
            self.assertNotIn("stream", case["body"])
            self.assertNotIn("seed", case["body"])
            self.assertNotIn("maxItems", fixtures.canonical(case["body"]))

    def test_wire_preserves_original_schema_order(self):
        value = self.manifest["cases"][0]["body"]
        sent = json.loads(fixtures.wire(value))
        self.assertEqual(list(sent["response_format"]["json_schema"]["schema"]["properties"]), ["city", "country", "population", "notable"])
        sorted_manifest = json.loads(fixtures.canonical(self.manifest))
        self.assertEqual(fixtures.sha(sorted_manifest), fixtures.sha(self.manifest))
        with self.assertRaisesRegex(AssertionError, "wire ordering"):
            runner.verify_manifest(sorted_manifest)

    def test_manifest_rejects_mutation(self):
        value = copy.deepcopy(self.manifest)
        value["cases"][0]["body"]["max_tokens"] = 3000
        with self.assertRaises(AssertionError):
            runner.verify_manifest(value)

    def test_exchange_sends_the_bound_wire_bytes(self):
        case = self.manifest['cases'][0]
        with patch.object(runner.http.client, 'HTTPConnection') as connection:
            response = connection.return_value.getresponse.return_value
            response.status = 200
            response.getheaders.return_value = []
            response.read.return_value = b'{"choices":[]}'
            runner.exchange(urlsplit(fixtures.ENDPOINT), 'offline-fixture-only', case['body'])
            sent = connection.return_value.request.call_args.kwargs['body']
        self.assertEqual(hashlib.sha256(sent).hexdigest(), case['body_wire_sha256'])
        properties = json.loads(sent)['response_format']['json_schema']['schema']['properties']
        self.assertEqual(list(properties), ['city', 'country', 'population', 'notable'])

    def test_notable_partial_parser_handles_escapes_and_nested_false_keys(self):
        text = '{"other":{"notable":["not-root"]},"notable":["a,]b","escaped \\" quote","unfinished'
        result = runner.notable_prefix(text)
        self.assertEqual(result["count"], 2)
        self.assertTrue(result["partial_item"])
        self.assertEqual(
            runner.notable_prefix('{"notable":["x","x"]}')["duplicate_complete_items"],
            1,
        )

    def test_sse_tools_merge_and_detect_orphan(self):
        events = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location":',
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"Paris"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ]
        envelope, errors = runner.merge_sse(events, {"get_weather"})
        self.assertEqual(errors, [])
        self.assertEqual(
            envelope["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            '{"location":"Paris"}',
        )
        del events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        self.assertIn(
            "orphan_tool_argument_delta", runner.merge_sse(events, {"get_weather"})[1]
        )

    def test_nan_is_not_json(self):
        with self.assertRaises(ValueError):
            runner.strict_loads('{"x":NaN}')

    def test_reasoning_fields_remain_separate_and_missing_unknown(self):
        result = runner.reasoning_fields(
            {
                "reasoning_content": "abc",
                "reasoning": None,
                "reasoning_details": [{"text": "de"}],
            }
        )
        self.assertEqual(result["reasoning_content"]["characters"], 3)
        self.assertIsNone(result["reasoning"]["characters"])
        self.assertEqual(
            result["reasoning_details"]["text_entries"][0]["characters"], 2
        )
        self.assertFalse(runner.reasoning_fields({})["reasoning_content"]["present"])

    def test_length_is_failure_even_if_json_valid(self):
        case = self.manifest["cases"][0]
        value = {"city": "Osaka", "country": "Japan", "population": 1, "notable": ["a"]}
        response = {
            "choices": [
                {
                    "index": 0,
                    "message": {"content": json.dumps(value)},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "completion_tokens": 2000,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }
        with patch.object(runner, "schema_errors", return_value=[]):
            result = runner.evaluate(case, response)
        self.assertIn("finish_length_valid_json", result["failure_classes"])
        self.assertEqual(
            result["choices"][0]["length_diagnostics"]["notable"]["count"], 1
        )


if __name__ == "__main__":
    unittest.main()
