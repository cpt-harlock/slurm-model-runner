# Guida allo studio di `model-runner`

*Per il maintainer (sysadmin CINECA). Snapshot a HEAD `39f853f` (2026-08-28), lettura senza modifiche ai file. I riferimenti `file:riga` sono relativi alla root del repo.*

---

## 1. Cosa fa e architettura

`model-runner` è uno strato di "colla" che porta su Leonardo Booster un servizio di inferenza LLM self-hosted, usabile da Claude Code (API Anthropic `/v1/messages`) e da Avante.nvim (API OpenAI `/v1/chat/completions`). Il codice di inferenza non è suo: usa vLLM, LiteLLM, `ray symmetric-run` e `hf download` così come sono (architecture.md:41-55).

Requisiti originali (CLAUDE.md:1-15):
- Slurm e Singularity;
- inferenza distribuita su più nodi GPU NVIDIA;
- endpoint sul login node, perché i compute node non hanno Internet;
- riavvio automatico allo scadere del walltime.

Il piano di controllo è separato dal piano dati. Il supervisor non sta mai sul percorso delle richieste e il gateway non parla mai con Slurm, salvo la stima ETA del waker (architecture.md:105-107; gateway.py:210).

```
  LAPTOP                         LOGIN NODE (es. login02)                       COMPUTE NODES (Booster, A100)
 ┌──────────────┐  ssh -L      ┌─────────────────────────────────────────┐    ┌─────────────────────────────────────┐
 │ Claude Code  │  4000:local  │  mrctl gateway  (gateway.py)            │    │  job Slurm  "mr-<model>"            │
 │ ANTHROPIC_   │  host:4000   │   ├─ litellm  127.0.0.1:4000  ──────────┼─HTTP──▶ vLLM 0.0.0.0:(8000+jobid%1000)  │
 │  BASE_URL    │─────────────▶│   │   (/v1/messages, /v1/chat/compl.)   │    │   singularity exec --nv vllm.sif    │
 │ Avante.nvim  │              │   └─ waker    127.0.0.1:4100 (modelli   │    │   1 nodo: vllm serve TP=4           │
 │ OPENAI_      │              │        freddi → 400 "warming up")       │    │   N nodi: srun + ray symmetric-run  │
 │  BASE_URL    │              │                                         │    │           TP=4 (NVLink) x PP=N (IB) │
 └──────────────┘              │  mrctl supervise (supervisor.py)        │    │                                     │
                               │   tick 60s: squeue --me + registry      │    │  vllm-server.sbatch:                │
                               │   → sbatch / scancel / drain / reap     │    │   publish loading→ready|failed      │
                               │   → scrape /metrics dei backend ────────┼─HTTP──▶ heartbeat ogni 30s            │
                               └──────────────┬──────────────────────────┘    │   trap cleanup → evict record       │
                                              │                               └──────────────┬──────────────────────┘
                                              ▼                                              ▼
                          ┌─────────────────────────────────────────────────────────────────────────────┐
                          │ Lustre condiviso  $MR_STATE (default $SCRATCH/model-runner, config.py:19)   │
                          │   registry/<model>.<jobid>.json   ← service discovery (host, port, state)   │
                          │   demand/<model>.json {desired}   ← flag scale-to-zero                      │
                          │   specs/<model>.json, logs/, litellm.config.yaml, containers/vllm.sif       │
                          │ $FAST/weights/<repo>  (pesi, striping 16x4M)                                │
                          └─────────────────────────────────────────────────────────────────────────────┘
```

I componenti principali:

- **Gateway** (`src/mr/gateway.py`). Non implementa endpoint HTTP. Genera `litellm.config.yaml` dal registry e supervisiona un processo `litellm` che usa il provider `hosted_vllm/` (gateway.py:1-11, 93-97). Fa girare anche un piccolo server HTTP, il "waker", su 4100 (gateway.py:197-292).
- **Supervisor** (`src/mr/supervisor.py`). È un reconcile loop senza stato, con tick da 60 s (supervisor.py:27, 217-241). A ogni tick ricostruisce lo stato da `squeue --me` e dal registry.
- **Registry** (`src/mr/registry.py`). È la service discovery, fatta di file JSON su Lustre e scritti in modo atomico (registry.py:81-87). La scelta dei file è motivata dal fatto che la connettività compute→login non è verificata (registry.py:6-8). Il gateway invece si collega direttamente a `hostname:port` del compute node, perché login→compute è aperto su qualunque porta (architecture.md:21, 31-33).
- **Job batch** (`slurm/vllm-server.sbatch`). Avvia vLLM nel SIF ed è l'unico processo che scrive l'indirizzo del backend nel registry (vllm-server.sbatch:93-94).

---

## 2. Flusso end-to-end di una richiesta

### 2.1 Modello "caldo" (backend READY)

1. Il client invia la richiesta a `http://127.0.0.1:4000` sul laptop, e l'SSH tunnel la porta a `127.0.0.1:4000` del login node (README.md:83-90; examples/avante.lua:40, 63).
2. LiteLLM sceglie una riga della `model_list` in base al nome del modello:
   - Avante usa il nome del config, per esempio `qwen3-coder-480b-awq` (avante.lua:38-74);
   - Claude Code invia ID completi come `claude-sonnet-5`, che vengono intercettati dalle righe wildcard `claude-*sonnet*`, `claude-*opus*` e `claude-*haiku*`, più le righe bare `sonnet`/`opus`/`haiku` (gateway.py:131-165).
   - Il primario è il modello con `role="primary"`, altrimenti `models[0]` in ordine alfabetico. Il modello small è quello con `role="small"`, altrimenti il primo nome che contiene "32b", altrimenti il primario (gateway.py:141-153).
   - Oggi `qwen3-coder-480b-awq` è primary (qwen3-coder-480b-awq.toml:10) e `qwen3-32b` è small (qwen3-32b.toml:5). DeepSeek non ha role, quindi non ha alias.
3. Una riga calda contiene `model_name=served_name`, `api_base=http://<host>:<port>/v1`, `model=hosted_vllm/<served>` e `api_key: none` (gateway.py:119-129, 93-97). Host e porta vengono da `registry.active()`, che preferisce READY a DRAINING e poi il record con `expires_at` più alto (registry.py:124-133).
4. Budget di contesto. Il `max_model_len` di vLLM è un budget unico prompt+completion. Il gateway pubblica quindi `max_output_tokens = max_len//4` e `max_input_tokens = max_len - max_output` (gateway.py:65-116). Esempi: 262144 → 196608 in / 65536 out; 32768 → 24576 / 8192.
   - `drop_params: true` scarta i parametri solo-Anthropic.
   - `modify_params: true` fa sì che LiteLLM riduca un `max_tokens` troppo grande (gateway.py:167-173).
5. Reasoning. `chat_template_kwargs` viene inserito in `litellm_params` e inoltrato a ogni richiesta (gateway.py:99-109). DeepSeek usa `{thinking=true}` (deepseek-v3.1-terminus.toml:42-43). La separazione tra reasoning e content, lo streaming e la traduzione Anthropic↔OpenAI li fanno vLLM e LiteLLM: il gateway non ha codice proprio per questo.
6. vLLM risponde e LiteLLM riporta la risposta al client.

### 2.2 Modello "freddo" (nessun backend attivo) e cold start

1. Una riga fredda punta al waker: `model_name=<nome config>`, `api_base=http://127.0.0.1:4100/<model>/v1` (gateway.py:119-124). Il modello resta quindi nella `model_list` e il client non riceve l'opaco "Invalid model name" (gateway.py:52-57). Anche gli alias puntano all'URL waker del modello target: `claude-*haiku*` sveglia qwen3-32b.
2. LiteLLM chiama `POST /<model>/chat/completions` sul waker. Il waker:
   - legge e scarta il body (`Content-Length`);
   - prende il primo segmento del path come nome del modello;
   - chiama `demand.wake(model)`, cioè `desired=1`, ignorando eventuali errori (gateway.py:246-279);
   - risponde **400** `{error:{type:"model_warming_up", code:"400"}}` con `Connection: close`.

   Usa 400 e non 503 perché LiteLLM ritenta i 503 per circa 90 s (gateway.py:266-272; tests/test_gateway.py:124-136).
3. Il testo dell'ETA (gateway.py:207-232) prova questi casi in ordine:
   1. backend QUEUED/LOADING con `measured_load_s`: secondi residui;
   2. backend in caricamento senza misura: "should be ready soon";
   3. job `mr-<model>` PENDING: stima `squeue %S` (slurm.py:91-97);
   4. altrimenti: "a start will be requested within 60s".
4. Al tick successivo (≤60 s) il supervisor vede `desired=1` senza backend READY né pending ed esegue `launch` (supervisor.py:129-131, 235). Il job pubblica `loading`, poi `ready`. Entro `POLL_S=15` s il gateway nota il cambiamento, riscrive la config in modo atomico (gateway.py:183-192) e **riavvia** LiteLLM (gateway.py:313-330, 345-349). Non c'è hot reload: LiteLLM non gestisce SIGHUP e `/config/update` richiede prisma (gateway.py:316-322).
5. Il client deve reinviare la richiesta a mano (avante.lua:26-30).

### 2.3 Idle reap (scale-to-zero)

1. `refresh_activity` legge `/metrics` di ogni backend READY/DRAINING non stale (supervisor.py:162-184). Somma `vllm:num_requests_running`, `vllm:num_requests_waiting` e `vllm:request_success_total` (supervisor.py:51-73). Se c'è attività, o se il contatore dei completati è cambiato, aggiorna `last_active_at` nel registry.
2. `decide` emette `reap_idle` quando `now - last_active_at` (o `ready_at`) supera `idle_timeout_s` (supervisor.py:150-157). `idle_timeout_s=0` disabilita il reap (supervisor.py:155).
3. `apply` esegue `scancel` e poi `demand.set_desired(model, 0)` (supervisor.py:199-207). Il secondo passo è stato aggiunto dopo 22 cicli notturni di reap e relaunch.
4. Il trap del job rimuove il record (vllm-server.sbatch:83-91). `active()` restituisce None, il gateway rende il modello freddo e LiteLLM riparte. Il flusso torna al §2.2.

---

## 3. Ciclo di vita dei job Slurm e restart

### 3.1 Sottomissione (`mrctl up` o `apply('launch'|'launch_successor')`)

`supervisor.apply` chiama `cli.cmd_up` con `force=True`, passando un oggetto duck-typed (supervisor.py:211-214). `cmd_up` (cli.py:49-97) procede così:

1. Verifica che SIF e pesi esistano (cli.py:52-57).
2. Verifica che `TP*PP == nodes*gpus_per_node` (cli.py:58-65).
3. Salta se ci sono già backend non stale e non draining, a meno di `--force` (cli.py:67-70).
4. Chiama `demand.wake` (cli.py:76).
5. Scrive `SPEC_DIR/<model>.json` (cli.py:24-44). Il file è **uno per modello**, non uno per job.
6. Chiama `sbatch --parsable --export=ALL,MR_SPEC=…,MR_ROOT=…` con `-A -p --qos -N --ntasks-per-node=1 -c --gres=gpu:N -t -J mr-<model> -o/-e` (cli.py:80-91; slurm.py:80-84).

### 3.2 Esecuzione sul nodo di testa (`vllm-server.sbatch`)

1. Legge lo spec con one-liner python (:19-30). `HEAD_NODE` è il primo nodo di `scontrol show hostnames` (:33) e `PORT = 8000 + jobid%1000` (:37).
2. Ricava `EXPIRES` dall'`EndTime` concesso, letto con `scontrol show job` (:79-81).
3. Installa `trap cleanup EXIT TERM` (:91), pubblica `state=loading` (:93-94) e avvia una subshell di heartbeat ogni 30 s (:103).
4. Avvia vLLM:
   - **1 nodo**: `VLLM_HOST_IP=$(hostname -i)`, poi `singularity exec --nv … vllm serve --host 0.0.0.0 --port $PORT -tp -pp …` in background (:107-124).
   - **N nodi**: genera `specs/<model>.<jobid>.worker.sh`, che ricava `VLLM_HOST_IP` per nodo ed esegue `ray symmetric-run --address HEAD:6379 --min-nodes N -- vllm serve … --distributed-executor-backend ray` (:126-148). Lo script è lanciato con `srun --ntasks-per-node=1` (:150). NCCL è vincolato a `mlx5_1,mlx5_4` / `ib0`.
5. Readiness: esegue `curl -sf http://127.0.0.1:$PORT/v1/models` ogni 15 s, per al massimo 240 tentativi, cioè 60 min (:157-165).
   - Se risponde, pubblica `ready` e `ready_at` e stampa `READY after $SECONDS s`, che è la misura del tempo di caricamento.
   - Se il PID di vLLM muore, pubblica `failed` e fa `exit 1` (:163).

   Poi esegue `wait $SERVER` (:167).

### 3.3 Il tick del supervisor (supervisor.py:217-241)

All'avvio alza RLIMIT_CPU dal soft all'hard limit, perché il login node uccide i processi a 600 CPU-s (supervisor.py:76-83). Poi, in ciclo, per ogni modello:

1. `refresh_activity`;
2. `decide(model, demand.get_desired(model))`;
3. `apply` delle decisioni.

Infine esegue `registry.sweep(job vivi)` e dorme 60 s.

Regole di `decide`, in ordine (supervisor.py:86-159):

| # | Condizione | Azione |
|---|---|---|
| a | DRAINING da più di `DRAIN_GRACE_S=300` | cancel |
| b | `desired==0` | start_drain dei READY, cancel dei pending |
| c | nessun READY e nessun pending | launch |
| d | READY con `seconds_left < handoff_lead_s` e nessun pending | launch_successor |
| e | `len(ready) > desired` | start_drain del READY con `expires_at` minimo |
| f | idle oltre `idle_timeout_s` | reap_idle |

`pending_ids` è calcolato come job vivi in squeue meno gli ID READY/DRAINING (supervisor.py:100-108). Così conta anche un job PENDING che non ha ancora un record, e questo è il fix del triplo launch (test_supervisor.py:144-151).

### 3.4 Restart da walltime (rolling handoff)

1. A `EndTime - handoff_lead_s` il supervisor lancia un successore. `handoff_lead_s` vale 3000 s per qwen3-32b, 3300 per il 480B e 3600 di default.
2. Mentre il successore è in PENDING o LOADING è in `pending_ids`, che blocca i duplicati.
3. Quando il successore è READY, `active()` lo sceglie subito perché ha `expires_at` maggiore. Il gateway passa al successore e riavvia LiteLLM.
4. Con `len(ready)=2 > 1` l'incumbent va in DRAINING (supervisor.py:188-199). Viene cancellato quando le richieste in volo scendono a 0 (supervisor.py:175-178), oppure dopo 300 s.

> **Attenzione:** questo percorso non è mai stato eseguito dal vivo (architecture.md:235; i log del supervisor mostrano solo launch e reap). Ha inoltre un bug di doppio launch confermato (§7, R1).

### 3.5 Crash

Non esiste un percorso dedicato. Se vLLM muore in load, il job pubblica `failed` ed esce, e il trap rimuove il record. Se muore dopo READY, `wait` ritorna e il trap fa lo stesso. In entrambi i casi il job esce da squeue e al tick successivo `decide` esegue un `launch` normale. Non c'è backoff (§7, R5). Il supervisor si può riavviare in sicurezza perché non ha stato.

---

## 4. Modelli configurati

| Modello (TOML) | Checkpoint | Role | Nodi | TP×PP | ctx | gpu_util | KV pin | idle_timeout_s | measured_load_s | handoff_lead_s | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3-32b` | Qwen/Qwen3-32B (qwen3-32b.toml:4) | small (haiku) | 1 | 4×1 | 32768 | 0.90 | 38745599181 B | 1800 | 244 | 3000 | parser hermes / qwen3 |
| `qwen3-coder-480b-awq` | TechxGenus/…-AWQ INT4 (qwen3-coder-480b-awq.toml:9) | primary (sonnet/opus) | 2 | 4×2 | 262144 (tetto nativo) | 0.85 | 15404725044 B | **7200** (:12) | 348 | 3300 | parser qwen3_coder, CUDA graph attivi |
| `deepseek-v3.1-terminus` | INC4AI/…-int4-AutoRound (deepseek-v3.1-terminus.toml:28) | — (nessun alias) | 2 | 4×2 | 32768 | 0.85 | no | 1800 | 0 (non misurato) | 3600 (placeholder) | `thinking=true`, `--enforce-eager` (crash Triton MLA) |
| `kimi-k2-thinking` | moonshotai/Kimi-K2-Thinking INT4 (kimi-k2-thinking.toml:27) | — | 4 | 4×4 | 32768 | 0.85 | no | 1800 | 0 | 3600 | messo in secondo piano: bug del parser upstream; **pesi cancellati** (commit 8e51c3f) |

Vincoli comuni:
- su A100 (SM80) niente FP8, solo INT4 weight-only via Marlin (config.py:51-52);
- il container deve usare un tag `-cu12x`, perché il driver è R535 (build-container.sh:13-28, TAG `v0.28.0-cu129`);
- `walltime` è 4 giorni, partizione `boost_usr_prod`, QoS `boost_qos_lprod`.

---

## 5. Mappa dei file

| Path | Ruolo |
|---|---|
| `CLAUDE.md` | Brief originale con requisiti e vincoli, invariato da 89f4c6c |
| `README.md` | Pagina pubblica: stato dei modelli, setup (env `MR_ROOT/MR_STATE/MR_WEIGHTS/PYTHONPATH`, :54-64), uso di `mrctl`, tunnel client (:83-95) |
| `docs/architecture.md` | Design e diario di ingegneria (923 righe). §8 piano a fasi, §9 i 30 "rischi" (post-mortem con job ID), §10 portabilità |
| `examples/avante.lua` | Spec lazy.nvim con due provider (`model-runner` 262144, `model-runner-fast` 32768), `max_tokens` 8192, nessuna key |
| `src/mr/__init__.py` | Docstring e `__version__='0.1.0'`. Non c'è pyproject: `mrctl` è un alias di `python3 -m mr.cli` (README.md:60-61) |
| `src/mr/config.py` | Path da env (MR_ROOT, MR_STATE, MR_WEIGHTS, MR_SIF; :15-31), dataclass `SlurmSpec/EngineSpec/ModelSpec`, `load()`, `available()` (stem ordinati, :131-132), `ensure_dirs()` |
| `src/mr/registry.py` | `Backend`, stati queued/loading/ready/draining/failed (:26-30), `publish` atomico, `read_all` forward-compatible (scarta le chiavi ignote, :100-112), staleness 90 s (:34), `active()`, `sweep/evict`, `heartbeat_from_job` (read-modify-write, :150-163) |
| `src/mr/cli.py` | `mrctl models/up/status/down/stage/gateway/supervise` (:192-226), `_write_spec`, validazioni pre-submit |
| `src/mr/slurm.py` | Wrapper di squeue/sbatch/scancel/scontrol (`jobs`, `submit`, `cancel`, `start_estimate`, `node_hostnames`) |
| `src/mr/supervisor.py` | Reconcile loop: `decide`, `refresh_activity`, `apply`, `run`; `TICK_S=60`, `DRAIN_GRACE_S=300` |
| `src/mr/gateway.py` | Render della config LiteLLM, `sync_once`, waker, ciclo di vita di litellm |
| `src/mr/demand.py` | `$MR_STATE/demand/<model>.json {desired}`; se il file manca o è corrotto vale 1 (:36-41) |
| `slurm/vllm-server.sbatch` | Job sul nodo di testa: spec, cache fuori da $HOME, NCCL, registry, heartbeat, vLLM a 1 o N nodi, readiness, trap |
| `scripts/build-container.sh` | Build del SIF dal sandbox scrivibile `docker://vllm/vllm-openai`, più `pip install ray[default]` |
| `scripts/stage-model.sh` | `lfs setstripe -c 16 -S 4M` sulla destinazione nuova, poi `hf download --max-workers 16` |
| `config/models/*.toml` | Un TOML per modello (§4) |
| `tests/test_{supervisor,gateway,demand}.py` | 14 + 13 + 5 = 32 test unittest (stdlib) |

---

## 6. Decisioni di design

1. **Colla, non motore.** LiteLLM gestisce entrambi i formati wire e il gateway si limita a generarne la config (gateway.py:1-7).
2. **Riconciliazione senza stato** da squeue e registry, senza FSM né event log. Riavviare il supervisor è sempre sicuro (supervisor.py:1-12).
3. **"Pending" calcolato da Slurm** e non dal registry, per coprire la finestra tra il ritorno di sbatch e il primo publish (supervisor.py:100-108).
4. **Service discovery su file Lustre** con tmp+`os.replace`, invece di callback HTTP compute→login (registry.py:6-11).
5. **Lettore del registry tollerante** alle chiavi sconosciute, dopo un outage di circa 34 min di "no backends" seguito a un cambio di schema (registry.py:100-112).
6. **Il drain è solo un cambio di stato** nel registry. `active()` preferisce READY, quindi il gateway non ha bisogno di segnali (supervisor.py:188-199).
7. **Handoff sovrapposto.** Il costo del cold start, legato alla banda Lustre e all'assenza di disco locale sul nodo, è quello che guida il design (architecture.md:35-37, 125-141).
8. **Deadline dall'`EndTime` concesso** e non dal `-t` richiesto (vllm-server.sbatch:79-81).
9. **Idle detection dalle metriche di vLLM** e non dall'heartbeat. Una `/metrics` non raggiungibile vale "sconosciuto", mai "idle" (supervisor.py:51-57, 148-152).
10. **`reap_idle` imposta `desired=0`.** Il modello resta spento finché non torna una domanda reale (supervisor.py:200-207).
11. **Waker con 400 e non 503**, e il nome del modello passa nel path URL (gateway.py:238-241, 266-272).
12. **Restart completo di LiteLLM a ogni cambio di config**, accettando la perdita delle richieste in volo (gateway.py:313-323).
13. **Alias wildcard e `role` esplicito** invece di dedurre il ruolo dall'ordine o dal nome (gateway.py:131-153; config.py:68-75).
14. **Output budget fisso a 1/4 di `max_model_len`**, più `modify_params` (gateway.py:65-84, 171-173).
15. **`chat_template_kwargs` lato server**, così il comportamento di DeepSeek non dipende dal client (gateway.py:99-109).
16. **KV cache fissata** con `--kv-cache-memory-bytes` dopo un OOM in fase di CUDA graph (qwen3-32b.toml:31-41).
17. **Solo stdlib sul login node**: TOML con `tomllib`, YAML scritto a mano (config.py:110-111; gateway.py:50).
18. **Dati fuori da $HOME**: stato su $SCRATCH, pesi su $FAST. Striping impostato prima di scrivere (stage-model.sh:21-27).
19. **`ulimit -t unlimited` negli script e RLIMIT_CPU alzato nei loop**, per il limite di 600 CPU-s del login node (build-container.sh:7-11; stage-model.sh:7-12; supervisor.py:76-83).
20. **Sandbox scrivibile e pip invece di `--fakeroot`**, che non è disponibile per mancanza di subuid (build-container.sh:42-54).

---

## 7. Rischi e bug

Legenda: **[C]** confermato (dal codice o in simulazione), **[I]** incerto o plausibile. Ho tolto i claim smentiti dal cross-check: la fd leak in `start_litellm`, KillWait troppo corto (sul sito vale 180 s) e "EndTime mancante abortisce il job".

### 7.1 Gravi

- **R1 [C, simulato] Ogni handoff lancia un job in più.** Con l'incumbent a 1000 s dalla fine e il successore READY, `decide` restituisce `[launch_successor, start_drain]` nello stesso tick. Il motivo è che `pending_ids` è vuoto quando il successore è READY (supervisor.py:108), quindi la guardia `not pending_ids` a :136 passa di nuovo. Conseguenze:
  - parte un terzo job;
  - quando diventa READY, viene messo in drain il successore, che ha `expires_at` minimo (:145);
  - LiteLLM riparte due volte.

  Per il 480B significa 6 nodi contemporanei, contro un tetto QoS di 8 nodi (architecture.md:20). Il test `test_successor_ready_drains_oldest_incumbent` (test_supervisor.py:88-97) usa solo `assertIn` e non se ne accorge.
- **R2 [C] SIGTERM può essere ingoiato dal gateway.** `_Shutdown` eredita da `Exception` (gateway.py:35). Viene quindi catturato da `except Exception: log.exception("sync failed")` (gateway.py:350) se arriva durante `sync_once()` o `restart_litellm()`, che possono durare fino a 15 s. Il `kill` sembra non avere effetto.
- **R3 [C] Crash di litellm non rilevato.** `run()` non chiama mai `proc.poll()` (gateway.py:343-351). Se litellm muore, l'API resta giù finché non cambia il set di backend, senza nessun log.
- **R4 [C] Il control plane non ha nessuno che lo riavvii, e al momento è giù.** Non ci sono processi gateway o supervisor, il registry è vuoto e l'ultimo log del supervisor è del 2026-08-28. README.md:78-81 delega a tmux, systemd o cron. In più `registry.sweep(...)` è fuori da ogni try (supervisor.py:240): un errore transitorio di squeue termina il supervisor.
- **R5 [C] Crash loop senza backoff.** Se un job fallisce in load, pubblica `failed`, esce e il record viene rimosso (vllm-server.sbatch:163, 83-91). `decide` non legge mai FAILED e rilancia a ogni tick (supervisor.py:129-131). Anche il codice di ritorno di `cmd_up` viene ignorato (supervisor.py:214): un SIF o dei pesi mancanti producono un errore silenzioso ripetuto ogni 60 s.
- **R6 [C] Readiness timeout senza esito.** Se `/v1/models` non risponde entro 60 min ma il PID resta vivo (per esempio un hang di NCCL o Ray), il loop finisce senza `failed` e il job resta in `wait` (vllm-server.sbatch:157-167). Il job rimane LOADING e in `pending_ids`, quindi blocca i relaunch fino al walltime di 4 giorni.
- **R7 [C, simulato] Un backend READY bloccato (stale) disabilita il modello.**
  - `decide` conta anche i record stale (`include_stale=True`, supervisor.py:96), quindi non lancia nulla.
  - `refresh_activity` li salta (supervisor.py:168), quindi `last_active_at` resta fermo.
  - Dopo `idle_timeout_s` arriva il reap e `desired=0`.
  - Nel frattempo il gateway, il cui `active()` esclude gli stale (registry.py:130), mostra il modello come freddo e il waker promette "a start will be requested within 60s" mentre `decide` restituisce `none`.
- **R8 [C] Nessuna autenticazione.**
  - vLLM ascolta su `0.0.0.0:$PORT` senza key (vllm-server.sbatch:115), quindi qualunque utente del cluster può chiamare `lrdnXXXX:80xx` direttamente.
  - LiteLLM e il waker sono su 127.0.0.1 di un login node condiviso con `api_key none` (gateway.py:97).
  - Se il gateway è giù, chiunque può occupare la porta 4100 prima di lui.
  - Il waker scrive `demand/<segmento>.json` per qualunque segmento di path (gateway.py:253-255; demand.py:32-33).

### 7.2 Medi

- **R9 [C] Ogni restart di LiteLLM interrompe gli stream di tutti i modelli**, non solo di quello cambiato, perché c'è un unico processo (gateway.py:345-349). Basta un reap di qwen3-32b, un cold start di deepseek o un heartbeat oltre 90 s (registry.py:34) per tagliare gli stream del flagship. Il criterio di uscita della Phase 3 ("nessuna richiesta fallita al walltime") non è raggiungibile così (architecture.md:125).
- **R10 [C] Il tunnel del README può finire sul login node sbagliato.** README.md:87 usa l'alias round-robin `login.leonardo.cineca.it`, ma LiteLLM è legato a 127.0.0.1 di un nodo specifico (login02 secondo architecture.md:78).
- **R11 [C] Race di lost-update sul registry.** `heartbeat_from_job` è un read-modify-write senza lock (registry.py:150-163). Un heartbeat del job che si sovrappone a uno `start_drain` del supervisor può riportare il record a `ready` e cancellare `draining_since`, il che fa ripartire il drain e può innescare di nuovo R1. Il commento a supervisor.py:192-196 presuppone che questo non accada.
- **R12 [C] Record fantasma dopo l'evict.** Il trap non termina `$HEARTBEAT` (vllm-server.sbatch:83-91, 168). Un heartbeat successivo ricrea il record come `queued`, `host=<hostname>`, `port=0` (registry.py:153-159). Finché il job è in squeue, il waker dice "already loading" (gateway.py:214-219). Il trap gira inoltre due volte: una su TERM e una su EXIT. Anche `cmd_down` rimuove il record a job ancora vivo, con lo stesso effetto (cli.py:139-147).
- **R13 [C] EndTime vuoto produce una scadenza nel passato.** Con `set -euo pipefail`, `date -d ""` ha successo e restituisce la mezzanotte locale (vllm-server.sbatch:81). Risultato: `seconds_left=0`, un successore subito dopo READY e quindi R1.
- **R14 [C] Lo spec è per modello e non per job** (cli.py:26). Un job ancora PENDING legge l'ultimo spec scritto, mentre i flag sbatch restano quelli della sua sottomissione: TP/PP e `-N` possono non coincidere.
- **R15 [C] Job tenuti in coda contano come pending per sempre.** `decide` non guarda mai `j.state` (supervisor.py:95, 108), quindi un job in hold o REQUEUE_HOLD blocca ogni launch.
- **R16 [C] Finestra tra reap e config fredda.** Tra lo scancel (supervisor.py:206), l'evict e il sync successivo (≤15 s più il restart), le richieste arrivano a un host:port morto e non chiamano mai `wake`.
- **R17 [C] Un TOML malformato blocca il gateway all'avvio.** `sync_once()` in `start_litellm` è fuori da ogni try (gateway.py:360). Nel loop invece l'errore viene solo loggato e la config resta congelata (gateway.py:350-351). `cmd_models` va in crash allo stesso modo (cli.py:183-189).
- **R18 [C] Successore in ritardo.** Se coda più load superano `handoff_lead_s` (supervisor.py:136), c'è un buco. Il margine di coda è stimato (qwen3-32b.toml:9-12) e il tetto QoS durante la sovrapposizione peggiora la situazione: flagship (2) + deepseek (2) + qwen3-32b (1) fa già 5 nodi.
- **R19 [C] `squeue --me`** (slurm.py:53-73) vede solo i job dell'operatore. Un secondo membro dello staff che fa `mrctl up` sullo stesso account `cin_staff` è invisibile al supervisor.
- **R20 [C] Matching wildcard di LiteLLM ancorato all'inizio.** Nella 1.98.0 installata la regex è `re.match`, quindi `anthropic/claude-sonnet-5` non corrisponde, contrariamente a quanto dice gateway.py:137. LiteLLM non è pinnato (gateway.py:359) e `ray[default]` nemmeno (build-container.sh:71).
- **R21 [I] Guasto multi-nodo dopo READY.** `srun` a vllm-server.sbatch:150 non usa `--kill-on-bad-exit`. Se un worker muore e il server API sul nodo di testa resta su, `/v1/models` continua a rispondere ma le richieste falliscono. Mai testato.
- **R22 [I] `chat_template_kwargs` passthrough.** Nel codice di LiteLLM 1.98 `hosted_vllm` non ha una gestione esplicita, ma config.py:97 lo dà per "confirmed live".

### 7.3 Minori o latenti

- **[C]** Il nome del modello cambia tra caldo e freddo se `served_name != name` (gateway.py:123 vs :124). È latente perché nessun TOML imposta `served_name`.
- **[C]** Il `config` legge `os.environ['SCRATCH']` e `['FAST']` subito, anche quando `MR_STATE`/`MR_WEIGHTS` sono impostati (config.py:19, 23), e dà KeyError all'import.
- **[C]** `ModelSpec.weights_dir=Path()` è sempre truthy, quindi la costruzione diretta produce `PosixPath('.')` (config.py:63, 102).
- **[C]** Il nome nel TOML e lo stem del file possono divergere (config.py:117 vs :132), e la validazione è sbilanciata: le chiavi top-level sconosciute vengono ignorate, quelle di `[slurm]`/`[engine]` danno TypeError (config.py:116-128). I default sono duplicati (config.py:64-81 vs :121-125).
- **[C]** I file tmp contengono solo il PID e non l'host (registry.py:85; demand.py:46; gateway.py:190), e i tmp orfani non vengono mai rimossi.
- **[C]** Il controllo duplicati di `cmd_up` non vede i job pending senza record (cli.py:67), quindi due `mrctl up` manuali possono sottomettere due volte.
- **[C]** Nel percorso single-node `$VLLM_ARGS` non è quotato e subisce word-splitting (vllm-server.sbatch:124). Latente.
- **[C]** Porta `8000+jobid%1000` senza controllo di disponibilità, porta Ray fissa a 6379 (vllm-server.sbatch:37, 147). In pratica conta poco perché i nodi sono interi.
- **[C]** Il waker esegue squeue in modo sincrono a ogni richiesta fredda (gateway.py:210); ETA fisso a "60s" (:229); fallback a 300 s quando `measured_load_s=0` (:212); `code` è la stringa "400" (:263).
- **[C]** `chat_template_kwargs` è emesso in YAML senza quoting né nesting (gateway.py:106-109). Va bene con i config attuali (le 10 voci si parsano con `yaml.safe_load`), ma si rompe con `:` o `#`.
- **[C]** Il fallback del primary è `models[0]` in ordine alfabetico (gateway.py:149).
- **[C]** Con budget fisso max_len/4, sui modelli a 32768 l'output è limitato a 8192 e i prompt sopra 24576 ricevono un 400 da vLLM (gateway.py:82-84).
- **[C]** Il `finally` termina il processo senza `wait` e può lasciare orfano il nuovo litellm (gateway.py:355). Finestra stretta.
- **[I]** Una wake può perdersi tra `get_desired` (supervisor.py:235) e `set_desired(0)` (:207). Finestra molto stretta.
- **[C]** `_poll_activity` richiede tutte e tre le metriche (supervisor.py:71-72): se vLLM rinomina una metrica, tutti i backend diventano "unknown".
- **[C]** `decide` dovrebbe essere "pura" ma fa I/O (supervisor.py:94-96). Non supporta `desired>1` (supervisor.py:127-146).
- **[C]** È rimasto un file di demand orfano, `$MR_STATE/demand/qwen3-32b-4n.json`, e nessuno pulisce la directory.
- **[C]** Con `mrctl supervise` senza argomenti viene incluso anche kimi (cli.py:179). Con un `MR_STATE` nuovo il default `desired=1` (demand.py:40) proverebbe ad avviare tutti i modelli.
- **[C]** deepseek e kimi non hanno il pin della KV cache, che è proprio la modalità di OOM documentata in qwen3-32b.toml:31-41.
- **[C]** `stage-model.sh` applica lo striping solo se la destinazione non esiste (:23-27), non gestisce token HF e ha un echo FP8 obsoleto. `build-container.sh` ricostruisce il sandbox a ogni esecuzione (:53, :75).

### 7.4 Doc e codice che non coincidono [C]

| Documento | Codice reale |
|---|---|
| Readiness "/health + real completion" (architecture.md:138; vllm-server.sbatch:155-156) | solo `GET /v1/models` (:158) |
| Porta 8080 (architecture.md:92, 208, 214) | 4000 (gateway.py:30; README.md:87-90) |
| `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL=qwen3-coder-480b` (architecture.md:209-210) | nessuna key; quel nome non corrisponde a nessuna riga; il routing passa dagli alias `claude-*` |
| `mrctl logs` (:65), `up --pin` (:161), `models gc` (:198), `runai_streamer` (:195), adapter SGLang (:49) | non esistono (cli.py:197-223) |
| Cold model "503 + Retry-After" (:147, :159) | 400 senza Retry-After (gateway.py:266-273) |
| Stato GONE (:119) | non esiste; FAILED manca dal diagramma (registry.py:26-30) |
| Record stale "evicted" (tabella dei guasti) | solo filtrati (registry.py:115); sweep basato sull'assenza in squeue (:140-147) |
| Il registry è "solo letto" dal supervisor (registry.py:3-4) | il supervisor scrive (supervisor.py:182, 197) |
| Kimi "staged" (README.md:27; architecture.md:650-653) | pesi cancellati in 8e51c3f |
| Idle del flagship 1800 s (architecture.md:237, 848, 875) | 7200 (qwen3-coder-480b-awq.toml:12, commit 39f853f) |
| `max_output_tokens` "currently 8192" (architecture.md:331) | 65536 per il flagship (gateway.py:111) |
| Nota su LITELLM_MASTER_KEY nel README (avante.lua:15-16) | non esiste; `mrctl models` non stampa il role (avante.lua:21-22 vs cli.py:183-189) |
| Elenco dei punti che alzano RLIMIT_CPU (README.md:141-142; §10) | manca stage-model.sh (architecture.md:263-265) |
| Docstring di cli.py (:1-8) | mancano `gateway` e `supervise` |

---

## 8. Copertura dei test e lacune

Si eseguono con `PYTHONPATH=src python3 -m unittest tests.test_supervisor tests.test_gateway tests.test_demand`. pytest non è installato e servono `SCRATCH` e `FAST` nell'ambiente. Tutti i 32 test passano (14 supervisor + 13 gateway + 5 demand).

**Cosa è coperto:**
- `decide()`, in 12 casi: cold start, steady state, `launch_successor`, successore pending che blocca il duplicato, drain dell'incumbent, grace del drain, idle reap, `desired=0`, record fuori da squeue e la regressione del triplo launch.
- `apply` per `reap_idle` e per `cancel`.
- `render()`: set vuoto, routing caldo, URL del waker, split del budget di contesto (rischio 29), `chat_template_kwargs`, alias per role.
- Waker via HTTP reale (400, drain del body) e i rami dell'ETA.
- `demand`.

**Cosa non è coperto:**
- `config.py`, `registry.py`, `cli.py` e `slurm.py`, del tutto: parsing di squeue, `_epoch`, `active()`, `sweep`, la race in `heartbeat_from_job`, il controllo TP×PP.
- `refresh_activity` e `_poll_activity`, gli `apply` di `launch` e `start_drain`, la gestione errori di `run()`, FAILED e backoff, i record stale.
- **Il tick di handoff completo**, che avrebbe rilevato R1, perché il test usa solo `assertIn`.
- `sync_once`, `start/restart_litellm`, `run`, `watch_only`, la semantica dello shutdown (avrebbe rilevato R2) e il crash di litellm.
- Validità del YAML: l'output non viene mai parsato né caricato in LiteLLM. Nessun test end-to-end passa da LiteLLM (streaming, traduzione `/v1/messages`).
- Lo script sbatch e gli script shell, verificati solo dal vivo (architecture.md, job 54343864, 54381732, 54706174).
- Allineamento della documentazione: nessun controllo che `context_window` in avante.lua coincida con i TOML, né che i comandi citati nei doc esistano.

**Fragilità dei test:**
- In test_gateway.py:37-45 `config.load` è sostituito da un MagicMock. Il MagicMock è truthy, quindi `if ctk:` passa e viene emesso un `chat_template_kwargs:` vuoto (YAML null) senza che nessun test lo verifichi.
- In test_supervisor.py:22 `ModelSpec` costruito direttamente eredita il bug di `weights_dir=Path()`.

---

## 9. Domande aperte

1. **Chi tiene vivo il control plane?** Bisogna decidere tra systemd `--user` sul login node (consentito dalla policy?), cron di respawn e tmux, e se servono watchdog per litellm (R3, R4).
2. **Quale login node è quello canonico?** Va fissato nel README (`login02…`?) o va resa possibile la migrazione, tenendo conto del limite RSS di 1 GB del login node (architecture.md:22), non misurato per LiteLLM (R10).
3. **Autenticazione.** Le opzioni sono una master key LiteLLM più `--api-key` su vLLM, oppure un bind restrittivo. È accettabile che vLLM sia raggiungibile da tutto il cluster (R8)?
4. **Correzione di R1.** Vuoi che `launch_successor` si attivi solo quando c'è un solo READY e nessun pending, con un test che verifichi l'intera lista di decisioni?
5. **Backoff dei crash.** Serve un contatore di FAILED per modello, con desired=0 dopo N fallimenti consecutivi (R5)? E un `failed` esplicito al timeout di readiness (R6)?
6. **Record stale.** `decide` dovrebbe trattare uno stale READY come guasto (cancel e relaunch) invece di contarlo come sano (R7)?
7. **Restart di LiteLLM.** Si può fare un restart per modello, per esempio con un processo LiteLLM per modello o un router diverso, o rinviarlo finché non ci sono stream attivi (R9)? È accettabile per l'obiettivo di "zero downtime"?
8. **QoS e account condiviso.** Il tetto di 8 nodi regge flagship + deepseek + handoff? Serve vedere i job `mr-*` di tutto l'account e non solo `--me` (R18, R19)?
9. **Kimi.** Rimuoverlo da `config/models` o escluderlo da `supervise`, visto che i pesi non ci sono e lo spazio su $FAST non basta per ripristinarli?
10. **Pinning.** Va fissata la versione di LiteLLM sul login node e quella di `ray` nel SIF? La semantica del wildcard (R20) e il passthrough di `chat_template_kwargs` (R22) dipendono dalla versione.
11. **Documentazione.** Correggere architecture.md sul posto (§7, §8, tabella guasti, rischio 30) invece di aggiungere voci in fondo? Spostare il diario dei rischi in un file separato?
12. **Multi-nodo dopo READY.** Aggiungere `srun --kill-on-bad-exit` e un health check periodico con una completion reale (R21)?