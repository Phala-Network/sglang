# Bounded state and finite-state JSON string matching

An unbounded repetition with a minimum above 128 was expanded as `expr*`
followed by a counted mandatory suffix. Although the accepted language is the
same, the Earley matcher starts another possible suffix at each input position.
The active state set and token-mask latency can consequently grow with output.
This affects JSON strings with `minLength > 128` and other affected repetitions.

The patch moves the existing unbounded tail after the existing mandatory
prefix: `expr{minimum} expr*`. It preserves the counted prefix, lookahead,
bounded ranges, decoded Unicode/escape counting, and token rollback. It does
not disable grammar enforcement, reduce context or generation budgets, change
model/KV precision, or rewrite generated responses.

On the prior r6 image with the actual Nemotron tokenizer, the synthetic
minLength 129/200/1000 cases grew to 3.7-4.0 seconds per mask by 128 characters.
Through the dev PIG/HAProxy chain, the minLength 200 reproducer emitted mask
warnings of 3.3, 11.5, 16.7, 23.3, and 29.4 seconds; a 90-second client deadline
ended with incomplete JSON. This establishes the failure class, not the exact
schema of an unavailable production customer request.

The prefix-order correction alone keeps the state bounded through 4096
characters, but does not remove the cost of checking uncertain tokens against
single-character CFG rules. That cost also appears at minLength 1: about
40-64 milliseconds per mask with the actual Nemotron tokenizer, amplified by
EAGLE's speculative checks. A valid 826-token garden description took 132.153
seconds through the dev PIG/HAProxy chain. A prefix-order-only build is not
the selected repair.

The second patch compiles strings with a positive minLength and no maxLength
into finite-state character blocks using GrammarBuilder::AddRegex. The
pattern explicitly handles JSON escapes, so its json_string exclusion flag is
false. Each block consumes 128 decoded JSON characters, followed
by the exact remainder and a free finite-state tail. A raw Unicode scalar, a
short escape, a BMP Unicode escape, or a complete surrogate escape pair each
counts as one character. Existing rejection of isolated surrogates and raw
control characters is retained. Strings with maxLength retain their existing
bounded matcher path. Pattern plus length remains explicitly unsupported as
documented in the previous release, rather than silently weakening the schema.

The two patches pass 91 language/boundary/Unicode/token-mask/rollback
regressions. With the actual Nemotron tokenizer, all 39 minLength 129/200/1000
measurement points through 4096 characters stay below 0.203 milliseconds per
mask in the debug image. This CPU measurement does not substitute for actual
generation. Serving-chain completion, latency, mixed-model load, cancellation,
final-image qualification and production results are recorded separately.

No context, output budget, weight/KV precision, sampling parameter, model
artifact, or speculative-decoding setting is changed. The repair does not
rewrite a model response or classify truncated JSON as successful. Semantic
answer accuracy remains separate from grammar validity.

Upstream references reviewed:

- XGrammar issue [805](https://github.com/mlc-ai/xgrammar/issues/805) matches
  the repeated-prefix growth path in pinned source 0.2.6.
- Issue [873](https://github.com/mlc-ai/xgrammar/issues/873) and draft PR
  [883](https://github.com/mlc-ai/xgrammar/pull/883), head
  `4e4d55b17c22fe8101b000d80d8de71c9c3aa96b`, address a separate
  `PushBackIndirect` reserve/copy path absent from the pinned 0.2.6 source.
  They are not applied here.

Focused regression command in the built image:

```sh
python3 -m pytest -q \
  test/registered/unit/constrained/test_xgrammar_unbounded_prefix.py \
  test/registered/unit/constrained/test_xgrammar_string_fsm_blocks.py
```

The immutable XGrammar sdist, patch SHA-256, final changed-file manifest and
package version are specified in `xgrammar-source.lock.json` and verified by
the Dockerfile. The final runtime uses the same official SGLang base and
full-source build contract as this release profile.
