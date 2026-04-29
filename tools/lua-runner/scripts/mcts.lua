-- mcts.lua — Monte Carlo Tree Search token selection for llama-lua
--
-- Uses token-level MCTS: at each generation step, the search tree is rooted
-- at the current sequence state and explores a fixed set of candidate tokens.
-- Each simulation copies the KV cache into a scratch slot, decodes the
-- candidate, rolls out greedily for a few steps, and scores the result by
-- the mean log-probability over the rollout. After the simulation budget the
-- token with the most visits is committed to the main sequence and output.
--
-- KV-cache layout (n_seq_max = 2):
--   slot 0 — the committed ("main") sequence
--   slot 1 — scratch slot reused for every simulation rollout
--
-- Usage:
--   llama-lua --script mcts.lua --model model.gguf \
--     -- "The meaning of life is" 64 32 8 16
--                                  ^   ^   ^  ^
--                  max_tokens -----+   |   |  +-- top_k candidates
--                  n_simul  ---------+   |
--                  rollout_len  ---------+
--
-- Parameters (all positional, passed after --):
--   arg[1]  prompt text                              (default: "The meaning of life is")
--   arg[2]  max_tokens  tokens to generate           (default 64)
--   arg[3]  n_simul     MCTS simulations per step    (default 32)
--   arg[4]  rollout_len greedy rollout depth         (default 8)
--   arg[5]  top_k       candidate tokens at root     (default 16)

local model_path  = arg.model or error("no model provided (use --model)")
local prompt      = arg[1]    or "The meaning of life is"
local max_tokens  = tonumber(arg[2]) or 64
local n_simul     = tonumber(arg[3]) or 32
local rollout_len = tonumber(arg[4]) or 8
local top_k       = tonumber(arg[5]) or 16
local UCB_C       = 1.414   -- UCB1 exploration constant

-- ---------------------------------------------------------------------------
-- Model / context
-- ---------------------------------------------------------------------------
local model, merr = llama.model_load(model_path, -1)
if not model then error(merr) end

local ctx, cerr = llama.context_init(model, {
    n_ctx     = 4096,
    n_batch   = 512,
    n_seq_max = 2,    -- slot 0 = main, slot 1 = scratch
})
if not ctx then error(cerr) end

local eos   = llama.token_eos(model)
local batch = llama.batch_init(512 + rollout_len)

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------

-- Fast greedy step in the scratch sequence (slot 1).
-- Returns (token_id, log_prob) using argmax_logits_ith for speed.
local function greedy_step_scratch(pos)
    local tok, _, lp = llama.argmax_logits_ith(ctx, -1)
    return tok, lp
end

-- ---------------------------------------------------------------------------
-- MCTS per-step function
--
-- Precondition:  slot 0 holds the committed sequence up to position n_past-1.
--               The logits from the last committed decode are still valid.
-- Returns:       the 0-based token id to commit next.
-- ---------------------------------------------------------------------------
local function mcts_select(n_past)
    -- Step 1: get candidate tokens from the current logits (still valid).
    -- make_candidates returns them sorted by logit (highest first).
    local logits = llama.get_logits_ith(ctx, -1)
    local cands  = llama.make_candidates(logits, top_k)

    -- Per-candidate MCTS statistics: visits and cumulative value (sum of
    -- mean log-probs over rollouts — higher is better).
    local visits = {}
    local values = {}
    for i = 1, cands.n do
        visits[i] = 0
        values[i] = 0.0
    end
    local total_visits = 0

    -- Step 2: run simulations.
    for _ = 1, n_simul do
        -- UCB1 selection: pick the candidate with the best exploration score.
        local best_ci = 1
        local best_score = -math.huge
        for ci = 1, cands.n do
            local score
            if visits[ci] == 0 then
                score = math.huge   -- try every candidate at least once
            else
                score = (values[ci] / visits[ci])
                      + UCB_C * math.sqrt(math.log(total_visits) / visits[ci])
            end
            if score > best_score then
                best_score = score
                best_ci    = ci
            end
        end

        local cand_tok = cands.data[best_ci].id   -- 0-based token id

        -- Copy main sequence into scratch slot.
        llama.seq_rm(ctx, 1, 0, -1)
        llama.seq_cp(ctx, 0, 1, 0, -1)

        -- Decode the candidate token in the scratch slot.
        llama.batch_clear(batch)
        llama.batch_add(batch, cand_tok, n_past, {1}, true)
        local sim_pos = n_past + 1

        local rollout_sum = 0.0
        local rollout_steps = 0

        local ok = llama.decode(ctx, batch)
        if ok then
            -- Greedy rollout: decode rollout_len more tokens in scratch slot.
            for r = 1, rollout_len do
                local next_tok, lp = greedy_step_scratch(sim_pos - 1)
                rollout_sum   = rollout_sum + lp
                rollout_steps = rollout_steps + 1

                if next_tok == eos then break end

                llama.batch_clear(batch)
                llama.batch_add(batch, next_tok, sim_pos, {1}, true)
                sim_pos = sim_pos + 1

                local ok2 = llama.decode(ctx, batch)
                if not ok2 then break end
            end
        end

        -- Backpropagate: record normalized rollout value.
        -- Use a large negative penalty when the candidate itself fails to decode
        -- (e.g. context overflow), so it is not chosen again.
        local NO_ROLLOUT_PENALTY = -1e6
        local mean_lp = rollout_steps > 0 and (rollout_sum / rollout_steps) or NO_ROLLOUT_PENALTY
        visits[best_ci] = visits[best_ci] + 1
        values[best_ci] = values[best_ci] + mean_lp
        total_visits     = total_visits + 1
    end

    -- Step 3: commit to the candidate with the most visits
    -- (most visited = most reliably good according to MCTS).
    local best_ci = 1
    local best_v  = -1
    for ci = 1, cands.n do
        if visits[ci] > best_v then
            best_v  = visits[ci]
            best_ci = ci
        end
    end

    return cands.data[best_ci].id   -- 0-based token id
end

-- ---------------------------------------------------------------------------
-- Prefill: encode the prompt into main sequence (slot 0)
-- ---------------------------------------------------------------------------
local tokens = llama.tokenize(ctx, prompt, true)
local n_past = 0

llama.batch_clear(batch)
for i, tok in ipairs(tokens) do
    llama.batch_add(batch, tok, n_past, {0}, i == #tokens)
    n_past = n_past + 1
end

local ok0, pe = llama.decode(ctx, batch)
if not ok0 then error("prefill failed: " .. tostring(pe)) end

io.write(prompt)
io.flush()

-- ---------------------------------------------------------------------------
-- Main generation loop: MCTS selects each token, then we commit it.
-- ---------------------------------------------------------------------------
for step = 1, max_tokens do
    local next_tok = mcts_select(n_past)

    if next_tok == eos then break end

    io.write(llama.token_to_piece(ctx, next_tok))
    io.flush()

    -- Commit the chosen token to the main sequence.
    llama.batch_clear(batch)
    llama.batch_add(batch, next_tok, n_past, {0}, true)
    n_past = n_past + 1

    local ok1, de = llama.decode(ctx, batch)
    if not ok1 then error("decode failed at step " .. step .. ": " .. tostring(de)) end
end

io.write("\n")

io.stderr:write(string.format(
    "[mcts] steps=%d, simulations/step=%d, rollout=%d, top_k=%d\n",
    max_tokens, n_simul, rollout_len, top_k))
