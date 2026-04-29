-- greedy.lua — simple greedy decoding driven entirely from Lua
--
-- Demonstrates two ways to pick the next token inside llama-lua:
--
--   1. Inline greedy: directly use llama.argmax_logits_ith() (fast, no overhead).
--   2. Sampler-script bridge: use llama.make_candidates() + apply(candidates)
--      so that scripts from the samplers/ directory work unchanged here.
--      Pass the sampler path as arg[3] to activate the bridge.
--
-- Usage (inline greedy):
--   llama-lua --script greedy.lua --model model.gguf -- "Hello, my name is"
--
-- Usage (bridge to an external sampler script):
--   llama-lua --script greedy.lua --model model.gguf \
--     -- "Hello, my name is" 128 samplers/temperature.lua

local model_path   = arg.model or error("no model provided (use --model)")
local prompt       = arg[1]    or "Hello, my name is"
local max_tokens   = tonumber(arg[2]) or 128
local sampler_path = arg[3]    -- optional: path to a samplers/*.lua script

-- Load model and create context (1 sequence slot is enough for greedy)
local model, err = llama.model_load(model_path, -1)
if not model then error(err) end

local ctx, err2 = llama.context_init(model, { n_ctx = 2048, n_batch = 512 })
if not ctx then error(err2) end

local eos = llama.token_eos(model)

-- Tokenize prompt
local tokens = llama.tokenize(ctx, prompt, true)
local n_past = 0

-- Allocate a reusable batch
local batch = llama.batch_init(512)

-- ---------------------------------------------------------------------------
-- Token-selection function
-- ---------------------------------------------------------------------------
-- Option 1 (default): inline greedy — pick token with highest logit.
local function pick_next_token()
    -- argmax_logits_ith scans logits in C++ and returns (token_id, logit, log_prob).
    -- This avoids building a full n_vocab Lua table on every step.
    local tok = llama.argmax_logits_ith(ctx, -1)
    return tok
end

-- Option 2 (sampler bridge): load an external sampler script that defines
-- apply(candidates) in the same format as the llama_sampler_i hook.
if sampler_path then
    local ok_load, load_err = pcall(dofile, sampler_path)
    if not ok_load then
        error("failed to load sampler script: " .. tostring(load_err))
    end
    if type(apply) ~= "function" then
        error("sampler script must define a global apply(candidates) function")
    end

    -- Override pick_next_token to use the loaded apply() function.
    -- make_candidates is called with top_k=50 to limit the Lua table to the
    -- most likely candidates; raise this limit if your sampler script needs
    -- access to lower-ranked tokens (e.g. nucleus/top-p over a wide tail).
    local BRIDGE_TOP_K = 50
    pick_next_token = function()
        -- make_candidates builds the candidates table (sampler-hook format).
        local logits  = llama.get_logits_ith(ctx, -1)
        local cands   = llama.make_candidates(logits, BRIDGE_TOP_K)
        local sel_idx = apply(cands)
        if type(sel_idx) ~= "number" then
            error("apply() must return a 1-based index")
        end
        return cands.data[sel_idx].id  -- 0-based token id
    end

    io.stderr:write(string.format("[llama-lua] using sampler: %s\n", sampler_path))
end

-- ---------------------------------------------------------------------------
-- Prefill: process all prompt tokens in one batch
-- ---------------------------------------------------------------------------
io.write(prompt)
io.flush()

llama.batch_clear(batch)
for i, tok in ipairs(tokens) do
    llama.batch_add(batch, tok, n_past, {0}, i == #tokens)
    n_past = n_past + 1
end

local ok3, decerr = llama.decode(ctx, batch)
if not ok3 then error(decerr) end

-- ---------------------------------------------------------------------------
-- Decode loop
-- ---------------------------------------------------------------------------
local next_tok = pick_next_token()

for _ = 1, max_tokens do
    if next_tok == eos then break end

    io.write(llama.token_to_piece(ctx, next_tok))
    io.flush()

    llama.batch_clear(batch)
    llama.batch_add(batch, next_tok, n_past, {0}, true)
    n_past = n_past + 1

    local ok4, decerr2 = llama.decode(ctx, batch)
    if not ok4 then error(decerr2) end

    next_tok = pick_next_token()
end

io.write("\n")
