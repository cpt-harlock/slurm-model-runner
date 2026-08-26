"""Paths and model-spec loading.

Every mutable path is overridable by environment variable, because on Leonardo
$HOME is a 50 GB quota that must hold code only -- weights, caches and logs all
have to live elsewhere or jobs fail in confusing ways.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MR_ROOT = Path(os.environ.get("MR_ROOT", Path(__file__).resolve().parents[2]))

# Runtime state: registry, specs, rendered sbatch, logs. Shared FS, must be
# visible from both login and compute nodes.
MR_STATE = Path(os.environ.get("MR_STATE", Path(os.environ["SCRATCH"]) / "model-runner"))

# Model weights. Flash tier by default -- there is no node-local disk, so Lustre
# read bandwidth is literally the cold-start time.
MR_WEIGHTS = Path(os.environ.get("MR_WEIGHTS", Path(os.environ["FAST"]) / "weights"))

# Singularity image built by scripts/build-container.sh
MR_SIF = Path(os.environ.get("MR_SIF", MR_STATE / "containers" / "vllm.sif"))

REGISTRY_DIR = MR_STATE / "registry"
SPEC_DIR = MR_STATE / "specs"
LOG_DIR = MR_STATE / "logs"
MODELS_DIR = MR_ROOT / "config" / "models"


@dataclass
class SlurmSpec:
    account: str
    partition: str = "boost_usr_prod"
    qos: str = "boost_qos_lprod"
    nodes: int = 1
    time: str = "4-00:00:00"
    cpus_per_task: int = 32
    gpus_per_node: int = 4


@dataclass
class EngineSpec:
    tensor_parallel_size: int = 4
    pipeline_parallel_size: int = 1
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    # NOTE: never add --kv-cache-dtype fp8 here. A100 is compute capability 8.0;
    # FP8 needs 8.9+. It will either refuse to start or silently fall back.
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ModelSpec:
    name: str
    hf_repo: str
    slurm: SlurmSpec
    engine: EngineSpec
    served_name: str = ""
    weights_dir: Path = Path()
    idle_timeout_s: int = 1800
    # Lead time for launching the successor job before this one hits its wall.
    # Placeholder until Phase 0 measures a real Lustre load time (see docs/architecture.md §6).
    handoff_lead_s: int = 3600

    def __post_init__(self) -> None:
        self.served_name = self.served_name or self.name
        self.weights_dir = Path(self.weights_dir or MR_WEIGHTS / self.hf_repo.split("/")[-1])

    @property
    def world_size(self) -> int:
        return self.engine.tensor_parallel_size * self.engine.pipeline_parallel_size


def load(name: str) -> ModelSpec:
    # TOML, not YAML: tomllib is stdlib in 3.11+, which keeps the control plane
    # dependency-free on a login node where we do not control the environment.
    path = MODELS_DIR / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(f"no model config {path}; try `mrctl models`")
    raw = tomllib.loads(path.read_text())
    return ModelSpec(
        name=raw.get("name", name),
        hf_repo=raw["hf_repo"],
        served_name=raw.get("served_name", ""),
        weights_dir=raw.get("weights_dir", ""),
        idle_timeout_s=raw.get("idle_timeout_s", 1800),
        handoff_lead_s=raw.get("handoff_lead_s", 3600),
        slurm=SlurmSpec(**raw.get("slurm", {})),
        engine=EngineSpec(**raw.get("engine", {})),
    )


def available() -> list[str]:
    return sorted(p.stem for p in MODELS_DIR.glob("*.toml"))


def ensure_dirs() -> None:
    for d in (REGISTRY_DIR, SPEC_DIR, LOG_DIR, MR_SIF.parent, MR_WEIGHTS):
        d.mkdir(parents=True, exist_ok=True)
