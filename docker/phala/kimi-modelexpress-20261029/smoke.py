"""Focused installed-image dependency and CPU import checks."""

import importlib
import importlib.metadata as md
import json

from modelexpress.gds_transfer import is_gds_available
from packaging.requirements import Requirement

expected = {
    "sglang": "0.5.21",
    "modelexpress": "0.8.0+phala.mx1",
    "torch": "2.13.0+cu130",
    "nixl": "1.5.0",
    "mooncake-transfer-engine-cuda13": "0.3.13",
    "protobuf": "7.36.2",
}
for name, version in expected.items():
    assert md.version(name) == version, (name, md.version(name))
try:
    md.version("fastsafetensors")
except md.PackageNotFoundError:
    pass
else:
    raise AssertionError("fastsafetensors must not be installed")
for value in md.requires("modelexpress"):
    requirement = Requirement(value)
    if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
        assert md.version(requirement.name) in requirement.specifier, value
assert not [
    e
    for e in md.distribution("modelexpress").entry_points
    if e.name == "modelexpress_rl"
]
for module in [
    "modelexpress.engines.sglang.loader",
    "modelexpress.p2p_pb2_grpc",
    "modelexpress.ucx_utils",
    "mooncake.engine",
    "nixl",
]:
    importlib.import_module(module)
assert not is_gds_available(), "unexpected default GDS strategy"
print(
    json.dumps(
        {
            "versions": expected,
            "imports": "passed",
            "required_dependencies": "passed",
            "fastsafetensors": "absent",
            "gds": "unavailable",
        }
    )
)
