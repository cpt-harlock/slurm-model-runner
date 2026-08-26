"""Render LiteLLM's config from the registry and restart the proxy on change.

LiteLLM is the single process that gives us both wire formats over one vLLM
backend: `/v1/messages` for Claude Code and `/v1/chat/completions` for Avante.
This module renders LiteLLM's config and also runs the "waker" stub (below)
that a cold model's requests land on -- otherwise it never proxies traffic
itself.

Runs on a login node. Backends are addressed directly by hostname because
login->compute is open on arbitrary ports (verified, architecture.md §1).
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import signal
import subprocess
import threading
import time

from . import config, demand, registry, slurm

log = logging.getLogger("mr.gateway")

CONFIG_PATH = config.MR_STATE / "litellm.config.yaml"
PID_PATH = config.MR_STATE / "litellm.pid"
PORT = int(os.environ.get("MR_GATEWAY_PORT", "4000"))
WAKER_PORT = int(os.environ.get("MR_WAKER_PORT", "4100"))
POLL_S = 15


class _Shutdown(Exception):
    """Raised from the SIGTERM handler to unwind run()'s try/finally.

    Python installs no default SIGTERM handler (unlike SIGINT, which raises
    KeyboardInterrupt) -- the OS just kills the process immediately, skipping
    every `finally` block. Without this, `kill`/`pkill` on the gateway process
    orphans its litellm child instead of terminating it.
    """


def _on_sigterm(signum, frame) -> None:
    raise _Shutdown()


def render(model_status: dict[str, registry.Backend | None]) -> str:
    """Build a LiteLLM config. Hand-rolled YAML to stay dependency-free.

    model_status maps every configured model name to its active Backend, or
    None if nothing is serving it right now (idle-reaped with demand still 0,
    or never started -- see mr.demand). A None entry still gets a real
    model_list row, routed to the local waker stub instead of a vLLM URL, so
    a request for a cold model gets a clear "warming up" response (and bumps
    demand) instead of LiteLLM's opaque "Invalid model name" 400.
    """
    if not model_status:
        # A bare "model_list:" key parses as YAML null, not []. LiteLLM seems
        # to tolerate that at request time (it 400s cleanly), but an explicit
        # empty list is the honest representation of "no backends yet".
        return "model_list: []\n\ngeneral_settings:\n  telemetry: false\n"

    # hosted_vllm/<name> isn't in LiteLLM's built-in cost map, so it has no
    # idea what context window it's serving -- get_model_info()/get_max_tokens()
    # both return None for it. Without model_info, Claude Code's real default
    # (max_tokens=64000) sails straight through: vLLM 400s with "max_tokens=
    # 64000 cannot be greater than max_model_len=32768" the first time anyone
    # actually uses this. max_output_tokens is a quarter of the model's context
    # -- a generous completion budget while leaving room for a real prompt;
    # tune per-model if that ever binds.
    max_len_by_model = {m: config.load(m).engine.max_model_len for m in model_status}

    def entry(model_name: str, served_name: str, url: str, backend_model: str) -> list[str]:
        max_len = max_len_by_model.get(backend_model)
        lines = [
            f"  - model_name: {model_name}",
            "    litellm_params:",
            # hosted_vllm/ tells LiteLLM to speak OpenAI to the upstream while
            # still accepting Anthropic-format requests from Claude Code.
            f"      model: hosted_vllm/{served_name}",
            f"      api_base: {url}/v1",
            "      api_key: none",
        ]
        if max_len:
            lines += [
                "    model_info:",
                f"      max_input_tokens: {max_len}",
                f"      max_output_tokens: {max_len // 4}",
            ]
        return lines

    def route_for(model: str) -> tuple[str, str, str]:
        """(served_name, url, backend_model) for whichever is real right now."""
        b = model_status[model]
        if b:
            return b.served_name, b.url, b.model
        return model, f"http://127.0.0.1:{WAKER_PORT}/{model}", model

    lines = ["model_list:"]
    for model in model_status:
        served, url, backend_model = route_for(model)
        lines += entry(served, served, url, backend_model)

    # Claude Code resolves `--model sonnet/opus/haiku` to a full model ID
    # *client-side* before the request ever reaches us -- confirmed empirically
    # against the real CLI: `--model sonnet` sent "claude-sonnet-5", not
    # "sonnet". A model_alias_map keyed on the short alias (tried first) never
    # matched real traffic; it 400'd with "Invalid model name passed in
    # model=claude-sonnet-5". Wildcard model_list entries match on LiteLLM's
    # own pattern router instead (`*` -> `.*`, matched anywhere in the model
    # string -- see router_utils/pattern_match_deployments.py in the installed
    # package), so this survives model version churn: any Claude Code release,
    # any Sonnet/Opus/Haiku snapshot, without re-pinning exact IDs here.
    # Which model answers to sonnet/opus ("primary") vs haiku ("small") is an
    # explicit per-model `role` (architecture.md §7), not inferred: picking
    # models[0] (alphabetical) and matching "32b" in the name only ever
    # worked because qwen3-32b was the only model configured. Falls back to
    # that old heuristic if nothing sets `role` explicitly, so a single-model
    # deployment that predates this still works unchanged.
    models = list(model_status)
    roles = {m: config.load(m).role for m in models}
    primary_model = next((m for m in models if roles[m] == "primary"), models[0])
    small_model = next(
        (m for m in models if roles[m] == "small"),
        next((m for m in models if "32b" in m), primary_model),
    )
    for pattern, target_model in (
        ("claude-*sonnet*", primary_model),
        ("claude-*opus*", primary_model),
        ("claude-*haiku*", small_model),
        # Bare short forms too, in case something sends the alias literally
        # (older CLI versions, ANTHROPIC_MODEL=sonnet-style env overrides).
        ("sonnet", primary_model),
        ("opus", primary_model),
        ("haiku", small_model),
    ):
        served, url, backend_model = route_for(target_model)
        lines += entry(pattern, served, url, backend_model)

    lines += [
        "",
        "litellm_settings:",
        "  drop_params: true",   # vLLM rejects Anthropic-only params it doesn't know
        "  modify_params: true", # required for model_info's max_output_tokens to
                                  # actually clamp an oversized client max_tokens
                                  # instead of just describing the model
        "",
        "general_settings:",
        "  telemetry: false",
    ]
    return "\n".join(lines) + "\n"


def sync_once() -> bool:
    """Re-render if the active backend set changed. Returns True if it did."""
    model_status = {m: registry.active(m) for m in config.available()}
    new = render(model_status)
    old = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else ""
    if new == old:
        return False

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(new)
    os.replace(tmp, CONFIG_PATH)
    log.info("config changed: %s", {m: (b.url if b else "cold") for m, b in model_status.items()})
    return True


def _wake_eta_message(model: str) -> str:
    """Best-effort human-readable ETA for the waker's response.

    Queue-aware, not a fixed guess (architecture.md/README Phase 5): real
    numbers when there's something real to look at (an already-queued or
    -loading job's own Slurm/registry state), honest uncertainty otherwise
    (nothing exists yet -- the supervisor hasn't even run its next tick).
    Never raises: this is best-effort UX text, not something that should be
    able to break the waker's response.
    """
    try:
        m = config.load(model)
        now = time.time()
        live = {j.job_id: j for j in slurm.jobs() if j.name == f"mr-{model}"}
        backends = [b for b in registry.for_model(model, include_stale=True) if b.job_id in live]
        load_s = m.measured_load_s or 300

        loading = next((b for b in backends if b.state in (registry.QUEUED, registry.LOADING)), None)
        if loading and loading.started_at and m.measured_load_s:
            remaining = max(0, m.measured_load_s - (now - loading.started_at))
            return f"already loading, ~{int(remaining)}s left (measured {m.measured_load_s}s last time)"
        if loading:
            return "already loading, should be ready soon"

        pending = next((j for j in live.values() if j.state == "PENDING"), None)
        if pending:
            eta = slurm.start_estimate(pending.job_id)
            if eta:
                wait = max(0, eta - now)
                return f"queued (Slurm estimates start in ~{int(wait / 60)}m), then ~{int(load_s)}s to load"
            return f"queued for resources (no Slurm estimate yet); ~{int(load_s)}s to load once it starts"

        return f"a start will be requested within 60s; ~{int(load_s)}s to load once Slurm grants resources"
    except Exception:
        log.exception("wake ETA estimation failed for %s", model)
        return "a start has been triggered; retry in a few minutes"


class _WakerHandler(http.server.BaseHTTPRequestHandler):
    """Answers requests LiteLLM routes to a cold model's waker URL.

    The model name travels in the URL path (api_base was rendered as
    http://127.0.0.1:{WAKER_PORT}/{model}), not the JSON body -- LiteLLM
    appends the real endpoint (.../chat/completions) itself, so this needs no
    body parsing at all: just the first path segment.
    """

    protocol_version = "HTTP/1.1"

    def _handle(self) -> None:
        # Drain and discard any request body first -- good hygiene for an
        # HTTP/1.1 handler regardless of what else is going on here.
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)

        model = self.path.strip("/").split("/")[0] or "unknown"
        try:
            demand.wake(model)
        except Exception:
            log.exception("wake failed for %s", model)
        eta = _wake_eta_message(model)
        body = json.dumps({
            "error": {
                "message": f"model '{model}' is cold; {eta}",
                "type": "model_warming_up",
                "code": "400",
            }
        }).encode()
        # 400, not 503: LiteLLM classifies 503 as ServiceUnavailableError and
        # auto-retries it (2 retries, real backoff) before ever returning
        # anything to the client -- caught live as a ~90+ second hang with
        # zero log output in between. A 400 (BadRequestError to LiteLLM,
        # correctly -- this isn't a transient infra hiccup we want retried,
        # it's "come back later") returns to the client immediately, same as
        # the pre-existing "Invalid model name" 400 already did.
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    do_GET = _handle
    do_POST = _handle

    def log_message(self, fmt: str, *args) -> None:
        log.debug("waker: " + fmt, *args)


def start_waker() -> http.server.ThreadingHTTPServer:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", WAKER_PORT), _WakerHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("waker on 127.0.0.1:%d", WAKER_PORT)
    return server


def watch_only() -> None:
    """Keep the config file current without owning a litellm process.

    For running `litellm` by hand in another session while debugging.
    """
    from .supervisor import raise_cpu_limit

    raise_cpu_limit()  # login node RLIMIT_CPU is 600s soft
    start_waker()
    log.info("gateway sync-only watching %s", config.REGISTRY_DIR)
    while True:
        try:
            sync_once()
        except Exception:
            log.exception("sync failed")
        time.sleep(POLL_S)


def restart_litellm(proc: subprocess.Popen) -> subprocess.Popen:
    """Replace the running proxy with one that picks up the new config on boot.

    Decided against a live reload: the installed LiteLLM's bare (non-gunicorn)
    server installs no SIGHUP handler, so the OS default -- killing the
    process outright -- would fire instead of any reload, and its
    /config/update admin endpoint requires a Postgres-backed `prisma_client`
    we don't want to run just for this. A plain restart is a few hundred ms of
    dropped in-flight requests, and only happens when the backend set changes
    (a model going up/down, or a handoff cutover) -- not per request.
    """
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    return start_litellm()


def run() -> None:
    """Own the litellm subprocess: launch it, keep its config current, restart
    it whenever the backend set changes. Runs until killed."""
    from .supervisor import raise_cpu_limit

    raise_cpu_limit()  # login node RLIMIT_CPU is 600s soft
    signal.signal(signal.SIGTERM, _on_sigterm)
    start_waker()  # daemon thread; dies with the process, no explicit shutdown needed
    proc = start_litellm()
    log.info("gateway watching %s", config.REGISTRY_DIR)
    try:
        while True:
            time.sleep(POLL_S)
            try:
                if sync_once():
                    log.info("backend set changed; restarting litellm")
                    proc = restart_litellm(proc)
            except Exception:
                log.exception("sync failed")
    except _Shutdown:
        log.info("gateway shutting down")
    finally:
        proc.terminate()


def start_litellm() -> subprocess.Popen:
    """Launch the proxy itself. Expects `litellm` on PATH (pipx or a venv)."""
    sync_once()
    proc = subprocess.Popen(
        ["litellm", "--config", str(CONFIG_PATH), "--port", str(PORT), "--host", "127.0.0.1"],
        stdout=open(config.LOG_DIR / "litellm.out", "ab"),
        stderr=subprocess.STDOUT,
    )
    PID_PATH.write_text(str(proc.pid))
    log.info("litellm on 127.0.0.1:%d (pid %d)", PORT, proc.pid)
    return proc
