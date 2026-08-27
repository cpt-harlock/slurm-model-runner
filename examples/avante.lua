-- Example lazy.nvim plugin spec wiring Avante.nvim to a model-runner gateway
-- instead of Copilot. The gateway is LiteLLM, OpenAI-compatible, listening on
-- 127.0.0.1:4000 by default (mr.gateway.PORT / $MR_GATEWAY_PORT).
--
-- Running on the same login node the gateway binds on (e.g. this file lives
-- under ~/.config/nvim on the cluster itself): no tunnel needed, use
-- 127.0.0.1 directly, as below.
--
-- Running off-cluster instead: open the same SSH tunnel described in the
-- README's Usage section first --
--   ssh -L 4000:localhost:4000 <user>@login.leonardo.cineca.it
-- -- and the endpoints below need no other change, since the tunnel makes
-- 127.0.0.1:4000 on your laptop reach the gateway.
--
-- No API key: the gateway has no LITELLM_MASTER_KEY set by default (see
-- README's Setup section), so api_key_name = "" turns off Avante's "API key
-- is not set" check entirely (see avante.providers.env.require_api_key: true
-- only when api_key_name is a non-empty string). If you *do* set a
-- LiteLLM virtual key, give it a real env var name here instead.
--
-- Two providers, matching the primary/small-fast split `mrctl models` lists
-- (docs/architecture.md §7, mr.config.ModelSpec.role) -- swap the `model`
-- fields for whatever `config/models/*.toml` you actually have staged and
-- up. Switch between them with <leader>ap / :AvanteSwitchProvider.
--
-- Cold start: mr.supervisor idle-reaps an unused backend, and the gateway's
-- waker stub answers a request for a cold model with an immediate error (not
-- a hang) while it triggers a real start in the background. Avante doesn't
-- retry that automatically -- the first request after a period of no use may
-- show an error; resend it once `mrctl status` shows the backend `ready`.
return {
  {
    "yetone/avante.nvim",
    opts = {
      provider = "model-runner",
      providers = {
        -- Primary/flagship model -- real work.
        ["model-runner"] = {
          __inherited_from = "openai",
          endpoint = "http://127.0.0.1:4000/v1",
          model = "qwen3-coder-480b-awq", -- match a `name` from `mrctl models`
          api_key_name = "",
          timeout = 300000, -- 5 min: a cold-start wake can take a few minutes on its own
          context_window = 65536, -- match that model's config `max_model_len`
          use_response_api = false, -- vLLM/LiteLLM here speak Chat Completions, not the Response API
          support_previous_response_id = false,
          extra_request_body = {
            temperature = 0.7,
            max_tokens = 8192,
          },
        },
        -- Small/fast model -- quick edits, cheaper to keep warm.
        ["model-runner-fast"] = {
          __inherited_from = "openai",
          endpoint = "http://127.0.0.1:4000/v1",
          model = "qwen3-32b",
          api_key_name = "",
          timeout = 300000,
          context_window = 32768,
          use_response_api = false,
          support_previous_response_id = false,
          extra_request_body = {
            temperature = 0.7,
            max_tokens = 8192,
          },
        },
      },
    },
  },
}
