-- beam_search.lua — beam search decoding driven entirely from Lua
--
-- This script demonstrates that complex multi-sequence decoding strategies
-- (impossible via the sampler hook) are straightforward when Lua controls
-- the full decode loop.
--
-- Usage:
--   llama-lua --script beam_search.lua --model model.gguf \
--             -- "The capital of France is" [beam_width] [max_tokens]
--
-- How it works:
--   1. The prompt is decoded once into sequence slot 0.
--   2. seq_cp duplicates that KV-cache into slots 1..B-1.
--   3. Each step decodes one token per beam in a single batched decode call.
--   4. We score all beam × vocab combinations, keep the top-B, compact the
--      KV cache, and continue until every beam has emitted EOS or max_tokens
--      is reached.
--   5. The best (highest log-prob) beam is printed.

local model_path  = arg.model or error("no model provided (use --model)")
local prompt      = arg[1]    or "The capital of France is"
local beam_width  = tonumber(arg[2]) or 4
local max_tokens  = tonumber(arg[3]) or 64

-- ---------------------------------------------------------------------------
-- Model / context setup
-- ---------------------------------------------------------------------------
local model, err = llama.model_load(model_path, -1)
if not model then error(err) end

-- n_seq_max must be >= beam_width so we can hold all beams in the KV cache.
local ctx, err2 = llama.context_init(model, {
    n_ctx     = 2048,
    n_batch   = 512,
    n_seq_max = beam_width,
})
if not ctx then error(err2) end

local eos     = llama.token_eos(model)
local n_vocab = llama.n_vocab(model)

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------

-- Natural log of softmax denominator (log-sum-exp) — used to convert logits
-- to log-probs in a numerically stable way.
local function log_softmax(logits)
    local max_l = -math.huge
    for _, v in ipairs(logits) do
        if v > max_l then max_l = v end
    end
    local sum = 0.0
    for _, v in ipairs(logits) do
        sum = sum + math.exp(v - max_l)
    end
    local log_denom = max_l + math.log(sum)
    local lp = {}
    for id, v in ipairs(logits) do
        lp[id] = v - log_denom   -- log P(token = id)
    end
    return lp
end

-- Top-K selection: returns {{token_id=..., log_prob=...}, ...} (1-based token ids)
local function top_k(log_probs, k)
    local cands = {}
    for id = 1, n_vocab do
        cands[#cands + 1] = { token_id = id - 1, log_prob = log_probs[id] }
    end
    table.sort(cands, function(a, b) return a.log_prob > b.log_prob end)
    local result = {}
    for i = 1, math.min(k, #cands) do result[i] = cands[i] end
    return result
end

-- ---------------------------------------------------------------------------
-- Beam state
--   beam = {
--     seq_id   : int (slot in the KV cache),
--     tokens   : {token_id, ...}  (tokens generated so far, not including prompt),
--     log_prob : float            (cumulative log-probability),
--     done     : bool             (hit EOS),
--   }
-- ---------------------------------------------------------------------------

local batch  = llama.batch_init(beam_width + 512)

-- Step 1: encode the prompt into sequence slot 0 --------------------------
local prompt_tokens = llama.tokenize(ctx, prompt, true)
local n_prompt = #prompt_tokens

llama.batch_clear(batch)
for i, tok in ipairs(prompt_tokens) do
    llama.batch_add(batch, tok, i - 1, {0}, i == n_prompt)
end

local ok, de = llama.decode(ctx, batch)
if not ok then error(de) end

local init_logits = llama.get_logits_ith(ctx, -1)  -- -1 = last output (last prompt token)
local init_lp     = log_softmax(init_logits)
local top_init    = top_k(init_lp, beam_width)

-- Step 2: seed beam_width beams from the top-beam_width initial tokens -----
-- Copy the prompt KV cache into slots 1..beam_width-1.
for b = 1, beam_width - 1 do
    llama.seq_cp(ctx, 0, b, 0, -1)  -- -1 = copy all positions
end

local beams = {}
for b = 1, beam_width do
    beams[b] = {
        seq_id   = b - 1,
        tokens   = { top_init[b].token_id },
        log_prob = top_init[b].log_prob,
        done     = (top_init[b].token_id == eos),
    }
end

-- ---------------------------------------------------------------------------
-- Main beam-search loop
-- ---------------------------------------------------------------------------
local n_past = n_prompt  -- position of the first generated token

for step = 1, max_tokens do
    -- Check if all beams are finished
    local all_done = true
    for _, beam in ipairs(beams) do
        if not beam.done then all_done = false end
    end
    if all_done then break end

    -- Decode one token per active beam in a single batched call ------------
    llama.batch_clear(batch)
    local active = {}  -- indices into beams[] for non-finished beams
    local batch_idx = 0  -- 0-based index into the output logits

    for i, beam in ipairs(beams) do
        if not beam.done then
            local last_token = beam.tokens[#beam.tokens]
            llama.batch_add(batch, last_token, n_past, {beam.seq_id}, true)
            active[#active + 1] = { beam_idx = i, batch_idx = batch_idx }
            batch_idx = batch_idx + 1
        end
    end

    local ok2, de2 = llama.decode(ctx, batch)
    if not ok2 then error(de2) end

    -- Collect candidates: for each active beam, top-beam_width extensions --
    local candidates = {}
    for _, entry in ipairs(active) do
        local beam   = beams[entry.beam_idx]
        local logits = llama.get_logits_ith(ctx, entry.batch_idx)
        local lp     = log_softmax(logits)
        local top    = top_k(lp, beam_width)
        for _, cand in ipairs(top) do
            candidates[#candidates + 1] = {
                parent_idx = entry.beam_idx,
                token_id   = cand.token_id,
                log_prob   = beam.log_prob + cand.log_prob,
            }
        end
    end

    -- Keep top-beam_width candidates ---------------------------------------
    table.sort(candidates, function(a, b) return a.log_prob > b.log_prob end)

    local new_beams   = {}
    local used_slots  = {}  -- which seq_ids are re-used

    for rank = 1, beam_width do
        local cand   = candidates[rank]
        local parent = beams[cand.parent_idx]
        local slot   = rank - 1  -- target seq_id slot (0..beam_width-1)

        -- Copy the parent's KV cache into the target slot.
        -- If parent.seq_id == slot, no copy is needed.
        if parent.seq_id ~= slot then
            -- Remove any existing entries in the target slot first.
            llama.seq_rm(ctx, slot, 0, -1)
            llama.seq_cp(ctx, parent.seq_id, slot, 0, -1)  -- copy all positions
        end
        used_slots[slot] = true

        local new_tokens = {}
        for _, t in ipairs(parent.tokens) do new_tokens[#new_tokens + 1] = t end
        new_tokens[#new_tokens + 1] = cand.token_id

        new_beams[rank] = {
            seq_id   = slot,
            tokens   = new_tokens,
            log_prob = cand.log_prob,
            done     = (cand.token_id == eos),
        }
    end

    -- Remove KV entries for slots that are no longer used -----------------
    for slot = 0, beam_width - 1 do
        if not used_slots[slot] then
            llama.seq_rm(ctx, slot, 0, -1)
        end
    end

    beams  = new_beams
    n_past = n_past + 1
end

-- ---------------------------------------------------------------------------
-- Output: pick the beam with the highest cumulative log-prob
-- ---------------------------------------------------------------------------
table.sort(beams, function(a, b) return a.log_prob > b.log_prob end)
local best = beams[1]

io.write(prompt)
for _, tok in ipairs(best.tokens) do
    if tok == eos then break end
    io.write(llama.token_to_piece(ctx, tok))
end
io.write("\n")

io.write(string.format("\n[beam width=%d, tokens=%d, log_prob=%.4f]\n",
    beam_width, #best.tokens, best.log_prob))
