"""Self-contained synthetic JSON fixtures shared by release token-mask tests."""


def cases():
    props = {"foo": {"type": "boolean"}, "barbaz": {"type": "string"}}
    optional = {"type": "object", "properties": props, "additionalProperties": True}
    required = dict(optional, required=["foo", "barbaz"])
    return [
        ("optional_reordered", optional, {"barbaz": "x", "foo": True}),
        ("optional_extra_first", optional, {"zzz": 1, "barbaz": "x", "foo": True}),
        ("required_extra_first", required, {"zzz": 1, "barbaz": "x", "foo": True}),
        (
            "optional_vision_metadata_shape",
            {
                "type": "object",
                "properties": {
                    "has_data_bearing_chart": {"type": "boolean"},
                    "chart_kinds": {"type": "array", "items": {"type": "string"}},
                    "figures": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
            {"figures_found": ["5"], "has_data_bearing_chart": False, "chart_kinds": [], "figures": ["5"]},
        ),
    ]
