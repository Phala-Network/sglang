"""Source-integrated adapter, converted from dsv41_reasoning_effort.py.
No import hooks or external runtime code paths.
"""

_TARGET = "sglang.srt.entrypoints.openai.encoding_dsv41"


REASONING_EFFORT_MAPPINGS = {
    "minimal": 25,
    "low": 50,
    "medium": 62,
    "high": 75,
    "xhigh": 90,
    "max": 100,
}


def apply(module) -> None:
    module.REASONING_EFFORT_MAPPINGS.clear()
    module.REASONING_EFFORT_MAPPINGS.update(REASONING_EFFORT_MAPPINGS)
    module.DEFAULT_REASONING_EFFORT = "high"
