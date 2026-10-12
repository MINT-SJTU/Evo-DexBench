from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping


BUILTIN_TASK_MODULES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "m01_v1": "dex_benchmark.sim.tasks.button_driven.m01",
        "m02_v1": "dex_benchmark.sim.tasks.button_driven.m02",
        "m03_v1": "dex_benchmark.sim.tasks.button_driven.m03",
        "m04_v1": "dex_benchmark.sim.tasks.button_driven.m04",
        "m05_v1": "dex_benchmark.sim.tasks.button_driven.m05",
        "m06_v1": "dex_benchmark.sim.tasks.rotational_manipulation.m06",
        "m07_v1": "dex_benchmark.sim.tasks.rotational_manipulation.m07",
        "m08_v1": "dex_benchmark.sim.tasks.rotational_manipulation.m08",
        "m09_v1": "dex_benchmark.sim.tasks.rotational_manipulation.m09",
        "m10_v1": "dex_benchmark.sim.tasks.rotational_manipulation.m10",
        "m11_v1": "dex_benchmark.sim.tasks.access_and_transfer.m11",
        "m12_v1": "dex_benchmark.sim.tasks.access_and_transfer.m12",
        "m13_v1": "dex_benchmark.sim.tasks.access_and_transfer.m13",
        "m14_v1": "dex_benchmark.sim.tasks.access_and_transfer.m14",
        "m15_v1": "dex_benchmark.sim.tasks.access_and_transfer.m15",
        "b01_v1": "dex_benchmark.sim.tasks.structured_configuration.b01",
        "b02_v1": "dex_benchmark.sim.tasks.structured_configuration.b02",
        "b03_v1": "dex_benchmark.sim.tasks.structured_configuration.b03",
        "b04_v1": "dex_benchmark.sim.tasks.structured_configuration.b04",
        "b05_v1": "dex_benchmark.sim.tasks.structured_configuration.b05",
        "b06_v1": "dex_benchmark.sim.tasks.assembly.b06",
        "b07_v1": "dex_benchmark.sim.tasks.assembly.b07",
        "b08_v1": "dex_benchmark.sim.tasks.assembly.b08",
        "b09_v1": "dex_benchmark.sim.tasks.assembly.b09",
        "b10_v1": "dex_benchmark.sim.tasks.assembly.b10",
        "b11_v1": "dex_benchmark.sim.tasks.articulated_configuration_control.b11",
        "b12_v1": "dex_benchmark.sim.tasks.articulated_configuration_control.b12",
        "b13_v1": "dex_benchmark.sim.tasks.articulated_configuration_control.b13",
        "b14_v1": "dex_benchmark.sim.tasks.articulated_configuration_control.b14",
        "b15_v1": "dex_benchmark.sim.tasks.articulated_configuration_control.b15",
    }
)
