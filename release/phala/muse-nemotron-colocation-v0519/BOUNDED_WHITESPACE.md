# Opt-in JSON whitespace bound

Nemotron Structured Output qualification uses XGrammar with flexible JSON
whitespace and `--constrained-json-max-whitespace-cnt 64`. Remove the conflicting
`--constrained-json-disable-any-whitespace` flag for this candidate only.

The new argument accepts positive integers, bounds syntactic whitespace rather
than JSON string contents, and is deliberately opt-in. An unset value preserves
the existing backend behavior, including Muse's separate llguidance settings.
The builtin unrestricted-JSON path also observes an explicit bound. Unsupported
grammar backends and tokenizer fallback cannot silently ignore the setting.

This fixes neither arbitrary model semantic mistakes nor all JSON Schema
limitations. XGrammar 0.2.1 still has an independently reproduced optional-field
type issue when `additionalProperties` is enabled. Its bare-CR rejection also
predates this patch. Neither boundary is hidden by the regression suite.

CPU regressions must be followed by installed-image GPU tests, reasoning and
streaming variants, neighbouring tool controls, and co-located Muse/ingress
checks. A local build is not a registry release or a production deployment.

The release Dockerfile uses the pinned official SGLang base and rebuilds the
complete Python runtime and all four Rust extensions from the frozen tree.
Pre-build inputs and post-build attestations are separate artifacts; final
image/SBOM/provenance/reproducibility results must not be embedded back into
their own subject. Publication and production deployment require authorization.
