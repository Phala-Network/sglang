"""Exercise finite two-operation capture through actual D/store/ACK methods."""

import copy
import hashlib
import json
import unittest
from unittest.mock import Mock

from test_shared_cache_seed_callsite_cpu import SeedCallsiteFixture


class SharedCacheSeedMultiOperationTests(SeedCallsiteFixture):
    def setUp(self):
        super().setUp()
        self.config.update(
            schema="phala.shared-cache.seed-config.v2",
            page_range={"start": 2, "end": 4},
            operation_ranges=[{"start": 2, "end": 3}, {"start": 3, "end": 4}],
        )
        self.write_config()

    def submit_two(self):
        self.submit(page_start=2, prior="first")
        self.submit(page_start=3, prior="second")
        self.assertEqual(len(self.native_calls), 4)
        self.assertTrue(all(op.shared_cache_seed_selected for op in self.operations))

    def assert_unsealed(self):
        self.assertFalse((self.directory / "seed.seed.json").exists())

    def manifest(self):
        return json.loads((self.directory / "seed.seed.json").read_text())

    def test_independent_operations_seal_exact_union_and_load_allowlist(self):
        self.submit_two()
        self.ack()
        self.assert_unsealed()
        self.ack()
        document = self.manifest()
        self.assertEqual(document["schema"], "phala.shared-cache.seed-manifest.v2")
        self.assertEqual(document["page_range"], {"start": 2, "end": 4})
        self.assertEqual(len(document["operations"]), 2)
        self.assertEqual([operation.id for operation in self.operations], [0, 1])
        self.assertEqual(
            [operation["operation_id"] for operation in document["operations"]], [0, 1]
        )
        self.assertTrue(
            all(
                type(operation["operation_id"]) is int
                for operation in document["operations"]
            )
        )
        self.assertEqual({key["operation_id"] for key in document["keys"]}, {0, 1})
        self.assertTrue(
            all(type(key["operation_id"]) is int for key in document["keys"])
        )
        self.assertNotIn("operation_id", document)
        self.assertNotIn("backup_ack", document)
        for operation, actual in zip(document["operations"], self.operations):
            self.assertEqual(operation["operation_id"], actual.id)
            self.assertEqual(operation["tokens"], len(actual.token_ids))
            self.assertEqual(operation["tokens"], operation["expected_tokens"])
            self.assertIs(operation["complete"], True)
            self.assertEqual(len(operation["key_ids"]), 2)
        self.assertEqual(
            {(key["page_index"], key["component"]) for key in document["keys"]},
            {
                (page, component)
                for page in (2, 3)
                for component in self.config["required_components"]
            },
        )
        self.assertTrue(self.capture_namespace["_multi_seed_evidence_valid"](document))
        collector = self.capture_namespace["SharedCacheDiagnostics"](
            enabled=False, log=Mock()
        )
        self.assertTrue(collector.arm_from_manifest(self.capture.sealed_path))
        self.assertEqual(len(collector._key_ids), 4)
        original = (self.directory / "seed.seed.json").read_bytes()
        self.submit(page_start=4, prior="outside")
        self.ack()
        self.assertEqual((self.directory / "seed.seed.json").read_bytes(), original)
        self.assertFalse(self.operations[-1].shared_cache_seed_selected)
        self.assertIsNone(self.capture._failed)

    def test_reverse_ack_order_waits_for_both_real_acks(self):
        self.submit_two()
        first = self.controller.ack_backup_queue.get()
        second = self.controller.ack_backup_queue.get()
        self.controller.ack_backup_queue.put(second)
        self.controller.ack_backup_queue.put(first)
        self.ack()
        self.assert_unsealed()
        self.assertTrue(self.capture._operations[second.id]["complete"])
        self.assertFalse(self.capture._operations[first.id]["complete"])
        self.ack()
        self.assertEqual(
            [op["operation_id"] for op in self.manifest()["operations"]],
            [first.id, second.id],
        )

    def test_missing_second_operation_stays_provisional(self):
        self.submit(page_start=2, prior="first")
        self.ack()
        self.assert_unsealed()
        self.assertEqual(len(self.capture._operations), 1)
        self.assertTrue(self.capture._operations[self.operations[0].id]["complete"])

    def test_one_failed_native_put_never_seals_or_retries_business(self):
        self.submit(page_start=2, prior="first")
        self.results = [0, -1]
        self.submit(page_start=3, prior="second")
        self.ack()
        self.ack()
        self.assert_unsealed()
        self.assertEqual(len(self.operations), 2)
        self.assertEqual(len(self.native_calls), 4)
        self.assertIsNotNone(self.capture._failed)

    def test_existing_key_is_not_positive_new_put_evidence(self):
        self.submit(page_start=2, prior="first")
        self.exists = [1, 1]
        self.submit(page_start=3, prior="second")
        self.ack()
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "put_not_new_success")
        self.assertEqual(len(self.native_calls), 2)

    def test_duplicate_range_does_not_change_business_enqueue(self):
        self.submit(page_start=2, prior="first")
        self.submit(page_start=2, prior="duplicate-range")
        self.ack()
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "invalid_or_duplicate_operation")
        self.assertEqual(len(self.native_calls), 4)

    def test_duplicate_operation_id_fails_closed(self):
        self.submit(page_start=2, prior="first")
        self.assertFalse(
            self.capture.bind_operation(
                self.req.rid,
                self.operations[0].id,
                page_range={"start": 3, "end": 4},
                expected_tokens=2,
            )
        )
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "invalid_or_duplicate_operation")

    def test_duplicate_key_across_operations_cannot_seal(self):
        self.submit(page_start=2, prior="same")
        self.submit(page_start=3, prior="same")
        self.ack()
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "duplicate_key")
        self.assertEqual(len(self.native_calls), 4)

    def test_missing_page_component_cannot_seal(self):
        self.submit_two()
        second_id = self.operations[1].id
        key_id = next(
            key_id
            for key_id, entry in self.capture._entries.items()
            if entry["operation_id"] == second_id
        )
        self.capture._entries.pop(key_id)
        self.ack()
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "operation_put_or_components_incomplete")

    def test_duplicate_ack_before_other_operation_is_terminal(self):
        self.submit_two()
        self.ack()
        first = self.operations[0]
        self.assertFalse(
            self.capture.backup_ack(
                request_id=self.req.rid,
                operation_id=first.id,
                complete=True,
                tokens=2,
                expected_tokens=2,
            )
        )
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "backup_incomplete_or_duplicate")

    def test_late_arm_bind_and_ack_cannot_seal(self):
        self.submit(page_start=2, prior="first")
        self.capture._started -= 31
        self.submit(page_start=3, prior="late")
        self.ack()
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "duration_exceeded")
        self.assertEqual(len(self.native_calls), 4)

    def test_native_result_after_deadline_cannot_seal(self):
        put = self.store._put_batch_zero_copy_impl

        def late_put(*args):
            result = put(*args)
            self.capture._started -= 31
            return result

        self.store._put_batch_zero_copy_impl = late_put
        self.submit(page_start=2, prior="first")
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "duration_exceeded")

    def test_ack_after_deadline_cannot_seal(self):
        self.submit_two()
        self.ack()
        self.capture._started -= 31
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "duration_exceeded")

    def test_changed_worker_identity_cannot_join_capture(self):
        self.submit(page_start=2, prior="first")
        self.manager.shared_cache_d_worker_id = "other-D-incarnation"
        self.submit(page_start=3, prior="second")
        self.ack()
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "operation_identity_mismatch")

    def test_invalid_plans_and_unbounded_product_cannot_arm(self):
        for ranges in (
            [{"start": 2, "end": 3}, {"start": 2, "end": 4}],
            [{"start": 2, "end": 3}, {"start": 4, "end": 5}],
            [{"start": 2, "end": 3}],
            [{"start": 2, "end": 3}, {"start": 3, "end": 1000000}],
        ):
            with self.subTest(ranges=ranges):
                self.config["operation_ranges"] = ranges
                self.config["page_range"] = {"start": 2, "end": ranges[-1]["end"]}
                self.write_config()
                self.submit(page_start=2, prior=str(ranges))
                self.ack()
                self.assertIsNone(self.capture._config)

    def test_forged_merged_ack_and_changed_proof_are_rejected(self):
        self.submit_two()
        self.ack()
        self.ack()
        original = self.manifest()
        for mutate in (
            lambda doc: doc.update(backup_ack={"complete": True}),
            lambda doc: doc["operations"][1].update(complete=False),
            lambda doc: doc["operations"][1].update(
                operation_id=doc["operations"][0]["operation_id"]
            ),
            lambda doc: doc["keys"][0].update(operation_id=999),
            lambda doc: doc["put_results"][0].update(already_present=True),
            lambda doc: doc["keys"][0].update(operation_id=True),
            lambda doc: doc["keys"][0].update(rank=False),
            lambda doc: doc["keys"][0].update(
                component=doc["keys"][1]["component"],
                page_index=doc["keys"][1]["page_index"],
            ),
            lambda doc: doc["operations"][0].update(expected_tokens=True),
            lambda doc: doc.pop("model_revision"),
            lambda doc: (
                doc["page_range"].update(end=10**18),
                doc["operation_ranges"][1].update(end=10**18),
            ),
        ):
            document = copy.deepcopy(original)
            mutate(document)
            document.pop("manifest_sha256")
            document["manifest_sha256"] = hashlib.sha256(
                json.dumps(
                    document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
            ).hexdigest()
            path = self.directory / "tampered.seed.json"
            path.write_text(json.dumps(document))
            path.chmod(0o600)
            collector = self.capture_namespace["SharedCacheDiagnostics"](
                enabled=False, log=Mock()
            )
            self.assertFalse(collector.arm_from_manifest(str(path)))

    def test_revision_must_be_present_and_null_is_exact(self):
        self.config.pop("model_revision")
        self.write_config()
        self.submit(page_start=2, prior="missing-revision")
        self.ack()
        self.assertIsNone(self.capture._config)

    def test_boolean_ack_id_or_expected_count_is_rejected(self):
        self.submit_two()
        first = self.operations[0]
        self.assertFalse(
            self.capture.backup_ack(
                request_id=self.req.rid,
                operation_id=first.id,
                complete=True,
                tokens=2,
                expected_tokens=True,
            )
        )
        self.ack()
        self.ack()
        self.assert_unsealed()

    def test_boolean_operation_ack_id_is_rejected(self):
        self.submit_two()
        self.assertFalse(
            self.capture.backup_ack(
                request_id=self.req.rid,
                operation_id=True,
                complete=True,
                tokens=2,
                expected_tokens=2,
            )
        )
        self.ack()
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "backup_incomplete_or_duplicate")

    def test_unfinished_puts_still_consume_the_byte_budget(self):
        self.config["max_logical_bytes"] = 8
        self.write_config()
        real_complete = self.capture.complete_batch
        self.capture.complete_batch = lambda *args, **kwargs: None
        self.submit(page_start=2, prior="first")
        self.capture.complete_batch = real_complete
        self.ack()
        self.assert_unsealed()
        self.assertEqual(self.capture._failed, "byte_budget_exceeded")
        self.assertEqual(len(self.native_calls), 2)

    def test_unknown_existence_and_coerced_native_results_are_not_success(self):
        for existed, result in (
            (-1, 0),
            (None, 0),
            (False, 0),
            (0.0, 0),
            ("0", 0),
            (0, False),
            (0, 0.5),
            (0, "0"),
            (0, None),
        ):
            with self.subTest(existed=existed, result=result):
                case = SharedCacheSeedMultiOperationTests(
                    "test_missing_second_operation_stays_provisional"
                )
                case.setUp()
                try:
                    case.exists = [existed, existed]
                    case.results = [result, result]
                    case.submit(page_start=2, prior="first")
                    case.submit(page_start=3, prior="second")
                    case.ack()
                    case.ack()
                    case.assert_unsealed()
                    case.assertEqual(case.capture._failed, "put_not_new_success")
                    case.assertEqual(len(case.operations), 2)
                    case.assertEqual(len(case.native_calls), 4)
                finally:
                    case.doCleanups()


if __name__ == "__main__":
    unittest.main()
