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

# Which cluster we are on: config/sites/<MR_SITE>.toml holds everything that
# differs between clusters (Slurm account/partition/qos/walltime, NCCL
# interfaces, weights tier, container tag). Defaults to the original target.
MR_SITE = os.environ.get("MR_SITE", "leonardo")
SITES_DIR = MR_ROOT / "config" / "sites"


@dataclass
class Site:
    name: str
    weights: str = ""
    bind_roots: list[str] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    slurm: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)


def load_site(name: str) -> Site:
    path = SITES_DIR / f"{name}.toml"
    if not path.exists():
        known = ", ".join(sorted(p.stem for p in SITES_DIR.glob("*.toml")))
        raise FileNotFoundError(f"no site profile {path} (MR_SITE={name!r}); known: {known}")
    raw = tomllib.loads(path.read_text())
    return Site(
        name=name,
        weights=raw.get("weights", ""),
        bind_roots=raw.get("bind_roots", []),
        image=raw.get("image", {}),
        slurm=raw.get("slurm", {}),
        env={k: str(v) for k, v in raw.get("env", {}).items()},
    )


def _expand(value: str, what: str) -> Path:
    out = os.path.expandvars(value)
    if "$" in out:
        raise RuntimeError(
            f"{what} = {value!r} needs an unset environment variable "
            f"(site {MR_SITE!r}); set it, or MR_SITE / MR_WEIGHTS for this cluster"
        )
    return Path(out)


SITE = load_site(MR_SITE)

# Runtime state: registry, specs, rendered sbatch, logs. Shared FS, must be
# visible from both login and compute nodes.
MR_STATE = Path(os.environ.get("MR_STATE", Path(os.environ["SCRATCH"]) / "model-runner"))

# Model weights: the site's fastest shared tier (there is no node-local disk on
# Leonardo, so Lustre read bandwidth is literally the cold-start time).
MR_WEIGHTS = Path(os.environ.get("MR_WEIGHTS") or _expand(SITE.weights, "site weights"))

# Singularity image built by scripts/build-container.sh
MR_SIF = Path(os.environ.get("MR_SIF", MR_STATE / "containers" / "vllm.sif"))

REGISTRY_DIR = MR_STATE / "registry"
SPEC_DIR = MR_STATE / "specs"
LOG_DIR = MR_STATE / "logs"
MODELS_DIR = MR_ROOT / "config" / "models"


@dataclass
class SlurmSpec:
    # No site-specific defaults here: they come from config/sites/<MR_SITE>.toml
    # [slurm], which load() merges under the model's own [slurm] keys.
    account: str
    partition: str
    qos: str
    time: str
    nodes: int = 1
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
    # Which Claude Code alias this model should answer to -- "primary" for
    # sonnet/opus, "small" for haiku (architecture.md §7). Explicit, not
    # inferred: mr.gateway used to pick models[0] (alphabetical) as primary
    # and match "32b" in the name as small, which only ever worked because
    # qwen3-32b was the only model configured. Adding a real flagship that
    # doesn't happen to sort first would have silently routed sonnet/opus to
    # the small model instead -- caught before it ever shipped, not live.
    role: str = ""
    # Real "READY after Ns" from the most recent bring-up, recorded by hand
    # (README's Phase 0 pattern) after each measurement. 0 means "never
    # measured yet". Used for the waker's queue-aware ETA (mr.gateway) --
    # separate from handoff_lead_s, which is a derived scheduling margin, not
    # this raw number.
    measured_load_s: int = 0
    # How long the job's readiness loop (vllm-server.sbatch) waits for the
    # server to answer before giving up. One cluster-wide hour was fine for
    # models up to ~600 GB, but Kimi-K3's 1454 GiB was still at 68% of its
    # shards after 59 minutes on Leonardo's Lustre (job 63571100), and a job
    # that outlives the loop is never published READY, so the gateway never
    # routes to it. Per model, and per site via [sites.<MR_SITE>], because it
    # follows weight size and filesystem bandwidth.
    ready_timeout_s: int = 3600
    # Static per-request params the gateway should inject into every LiteLLM
    # call for this model (rendered into litellm_params, forwarded straight
    # through hosted_vllm/ to the OpenAI-compatible request body). Exists for
    # deepseek-v3.1-terminus's chat template: its default (thinking=false)
    # bakes a literal "</think>" into the *prompt* itself rather than letting
    # the model generate one, so vLLM's reasoning parser (which only inspects
    # generated tokens) never sees a closing tag and dumps the entire answer
    # into `reasoning` with content: null -- confirmed by inspecting
    # vllm.reasoning.basic_parsers.BaseThinkingReasoningParser.extract_reasoning
    # (falls back to "assume everything is reasoning" when no end token is
    # found in the output) and the checkpoint's own chat_template.jinja
    # (`{% if not thinking %}{{'</think>'}}{% else %}{{'<think>'}}{% endif %}`
    # at the generation-prompt boundary). Passing {"thinking": true} here
    # makes the template emit `<think>` instead, so the model's own
    # </think> lands in its generated output where the parser can find it --
    # confirmed live: content splits correctly, real reasoning preserved.
    chat_template_kwargs: dict = field(default_factory=dict)
    # Per-model container image. Empty means the global MR_SIF. A bare file
    # name resolves next to MR_SIF (the containers dir), so a TOML can say
    # sif = "vllm-0.32.sif" without hardcoding $SCRATCH. Exists so a newer vLLM
    # (needed for e.g. DeepSeek-V3.2's sparse attention) can be adopted per
    # model, while proven models stay on the image they were validated with.
    sif: Path = Path()

    def __post_init__(self) -> None:
        self.served_name = self.served_name or self.name
        # weights_dir may name an env var ("$SCRATCH/weights/X") so a TOML that
        # needs a bigger tier than the site default stays portable.
        self.weights_dir = (_expand(str(self.weights_dir), f"{self.name} weights_dir")
                            if str(self.weights_dir) not in ("", ".")
                            else MR_WEIGHTS / self.hf_repo.split("/")[-1])
        self.sif = resolve_sif(self.sif)

    @property
    def world_size(self) -> int:
        return self.engine.tensor_parallel_size * self.engine.pipeline_parallel_size


def resolve_sif(value: str | Path) -> Path:
    value = str(value or "")
    if value in ("", "."):
        return MR_SIF
    p = Path(value)
    return p if p.is_absolute() else MR_SIF.parent / p


def load(name: str) -> ModelSpec:
    # TOML, not YAML: tomllib is stdlib in 3.11+, which keeps the control plane
    # dependency-free on a login node where we do not control the environment.
    path = MODELS_DIR / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(f"no model config {path}; try `mrctl models`")
    raw = tomllib.loads(path.read_text())
    # Precedence, lowest first: site profile < model TOML < the model's
    # [sites.<MR_SITE>] table. [slurm]/[engine] merge key by key; any other key
    # in the per-site table replaces the model's (e.g. sif, weights_dir).
    over = raw.get("sites", {}).get(MR_SITE, {})
    slurm = {**SITE.slurm, **raw.get("slurm", {}), **over.get("slurm", {})}
    engine = {**raw.get("engine", {}), **over.get("engine", {})}
    raw = {**raw, **{k: v for k, v in over.items() if k not in ("slurm", "engine")}}
    return ModelSpec(
        name=raw.get("name", name),
        hf_repo=raw["hf_repo"],
        served_name=raw.get("served_name", ""),
        weights_dir=raw.get("weights_dir", ""),
        idle_timeout_s=raw.get("idle_timeout_s", 1800),
        handoff_lead_s=raw.get("handoff_lead_s", 3600),
        role=raw.get("role", ""),
        measured_load_s=raw.get("measured_load_s", 0),
        ready_timeout_s=raw.get("ready_timeout_s", 3600),
        chat_template_kwargs=raw.get("chat_template_kwargs", {}),
        sif=raw.get("sif", ""),
        slurm=SlurmSpec(**slurm),
        engine=EngineSpec(**engine),
    )


def available() -> list[str]:
    return sorted(p.stem for p in MODELS_DIR.glob("*.toml"))


def ensure_dirs() -> None:
    for d in (REGISTRY_DIR, SPEC_DIR, LOG_DIR, MR_SIF.parent, MR_WEIGHTS):
        d.mkdir(parents=True, exist_ok=True)
