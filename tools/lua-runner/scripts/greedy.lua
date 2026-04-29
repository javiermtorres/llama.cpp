-- greedy.lua — simple greedy decoding driven entirely from Lua
--
-- Usage:
--   llama-lua --script greedy.lua --model model.gguf -- "Hello, my name is"

local model_path = arg.model or error("no model provided (use --model)")
local prompt     = arg[1]    or "Hello, my name is"
local max_tokens = tonumber(arg[2]) or 128

-- Load model and create context
local model, err = llama.model_load(model_path, -1)  -- -1 = all layers to GPU
if not model then error(err) end

local ctx, err2 = llama.context_init(model, { n_ctx = 2048, n_batch = 512 })
if not ctx then error(err2) end

local eos     = llama.token_eos(model)
local n_vocab = llama.n_vocab(model)

-- Tokenize prompt
local tokens  = llama.tokenize(ctx, prompt, true)
local n_past  = 0

-- Allocate a reusable batch
local batch = llama.batch_init(512)

-- Greedy helper: pick the token with the highest logit
local function greedy(logits)
    local best_id, best_val = 0, -math.huge
    for id = 1, #logits do
        if logits[id] > best_val then
            best_id  = id
            best_val = logits[id]
        end
    end
    return best_id - 1  -- convert to 0-based token id
end

io.write(prompt)

-- Prefill: process the prompt tokens
llama.batch_clear(batch)
for i, tok in ipairs(tokens) do
    local want_logits = (i == #tokens)  -- only need logits for the last token
    llama.batch_add(batch, tok, n_past, {0}, want_logits)
    n_past = n_past + 1
end

local ok, decerr = llama.decode(ctx, batch)
if not ok then error(decerr) end

local logits = llama.get_logits_ith(ctx, -1)  -- -1 = last output token (last prompt token)
local next_tok = greedy(logits)

-- Decode loop
for _ = 1, max_tokens do
    if next_tok == eos then break end

    local piece = llama.token_to_piece(ctx, next_tok)
    io.write(piece)
    io.flush()

    llama.batch_clear(batch)
    llama.batch_add(batch, next_tok, n_past, {0}, true)
    n_past = n_past + 1

    local ok2, decerr2 = llama.decode(ctx, batch)
    if not ok2 then error(decerr2) end

    logits   = llama.get_logits_ith(ctx, 0)
    next_tok = greedy(logits)
end

io.write("\n")
