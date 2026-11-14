"""LiteLLM proxy hook: rewrite `reasoning_effort` per model.

Loaded by LiteLLM itself, not by mr: mr.gateway copies this file next to
litellm.config.yaml and names it under `litellm_settings.callbacks`
(LiteLLM resolves "litellm_hooks.proxy_handler_instance" relative to the
config file's directory). It runs on LiteLLM's interpreter, so it imports
nothing from mr; its only input is the JSON sidecar mr.gateway writes beside
it, {model_name_or_pattern: {client_effort: model_effort}}, built from each
model's `reasoning_effort_map` TOML table.

Why it exists: the Qwen3.8 chat templates accept only xhigh/medium/low and
raise on anything else, so vLLM answers HTTP 400 to the "high" that most
clients send. Worse, Claude Code never sends an effort at all: it sends
Anthropic `thinking.budget_tokens`, and LiteLLM's hosted_vllm transformation
buckets that into an OpenAI-style effort (>=4096 "high", >=2048 "medium",
>=1024 "low", below that "minimal") -- and only does so when no `reasoning_effort` is
already set. So for a budget we compute the same bucket ourselves, map it,
and set `reasoning_effort` first; LiteLLM then leaves it alone. Verified
live 2026-11-14 on qwen3.8-2.4t-mxfp4: budget 6000 and 16000 both 400'd
with "Unexpected reasoning effort high" before this hook.
"""

from __future__ import annotations

import fnmatch
import json
import os

try:  # present when LiteLLM loads us; absent in mr's own unit tests
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover - exercised only outside LiteLLM
    CustomLogger = object

MAPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litellm_hooks.json")

# LiteLLM's thresholds (litellm.constants DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET,
# env-overridable there too); read lazily so the unit tests need no LiteLLM.
_DEFAULT_BUDGETS = {"high": 4096, "medium": 2048, "low": 1024}


def effort_from_budget(budget: int) -> str:
    try:
        from litellm.litellm_core_utils.reasoning_effort_utils import reasoning_effort_from_thinking_budget
        return reasoning_effort_from_thinking_budget(budget)
    except ImportError:
        for effort in ("high", "medium", "low"):
            if budget >= _DEFAULT_BUDGETS[effort]:
                return effort
        return "minimal"


def map_for(model: str, maps: dict[str, dict[str, str]]) -> dict[str, str]:
    """The effort map for a requested model name: exact match, else the first
    matching wildcard row (the claude-*sonnet* aliases)."""
    if model in maps:
        return maps[model]
    for pattern, m in maps.items():
        if fnmatch.fnmatchcase(model, pattern):
            return m
    return {}


def rewrite(data: dict, maps: dict[str, dict[str, str]]) -> dict:
    """Return `data` with reasoning_effort mapped for its model (in place)."""
    effort_map = map_for(str(data.get("model", "")), maps)
    if not effort_map:
        return data
    effort = data.get("reasoning_effort")
    if effort is None:
        thinking = data.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            effort = effort_from_budget(int(thinking.get("budget_tokens") or 0))
    if isinstance(effort, str):
        data["reasoning_effort"] = effort_map.get(effort, effort)
    return data


class _Cache:
    mtime = -1.0
    maps: dict = {}


def load_maps() -> dict:
    try:
        mtime = os.stat(MAPS_PATH).st_mtime
        if mtime != _Cache.mtime:
            with open(MAPS_PATH) as f:
                _Cache.maps = json.load(f)
            _Cache.mtime = mtime
    except (OSError, ValueError):
        _Cache.maps, _Cache.mtime = {}, -1.0
    return _Cache.maps


class EffortMapper(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            return rewrite(data, load_maps())
        except Exception:  # never fail a request because of this hook
            return data


proxy_handler_instance = EffortMapper()
