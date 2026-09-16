# Muse single-card format alignment

The r3 serving adapter constrains required/named Muse tool output to a JSON
array, while its baked chat template instructs the model to generate ATEM.
The original single-H200 qualification retained incorrect one-call parallel
outputs on XGrammar, plus JSON-object semantic failures. Removing DFLASH
reproduced the same three failures; permitting unrestricted JSON whitespace
also exposed a depth-10 whitespace runaway. These original responses remain
in the task's evidence directory.

This overlay passes the actual selected JSON tool schema and response format
to the opt-in Muse template. Required/named calls describe the enforced array;
normal automatic calls retain ATEM. Structured answers describe JSON mode and
the supplied schema. The request's messages, schemas, desired call count,
sampling, token budget, reasoning controls, and output parser are unchanged.
No constant values or calls are invented after generation. Other model
families receive no added template metadata.

Upstream issue #34631 concerns channel-boundary grammar activation, which is
already repaired in the r3 base; it does not align these prompt instructions.
The current issue search found no exact upstream fix for this mismatch.

Unit tests cover format visibility, normal chat/media/native-tool preservation,
reasoning mode and system-message preservation. Runtime protocol, image-input,
long-context, cancellation and AgentX qualification remain separate gates.

Only serving_chat.py and the baked Muse template change. The immutable r3 base
provides all dependencies; Dockerfile guards before and after hashes. This
branch is local and is not pushed as part of the model rollout authorization.

Runtime acceptance reproduced the r3 cancellation lifecycle defect: closing the stream left its scheduler request running beyond 60 seconds. The candidate additionally adopts the exact scoped tokenizer-manager/scheduler changes and behavior regressions from ec6f1c44c8 (upstream #35255 adaptation), including related parallel parent-placeholder cleanup. No other framework changes are imported.
