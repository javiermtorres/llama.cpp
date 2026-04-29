-- Temperature sampler: applies softmax with a configurable temperature and
-- samples a token proportional to the resulting probabilities.
--
-- Set the temperature before requiring this script, or edit the value below.
-- temperature = 0.8  -- lower = more deterministic, higher = more random
--
-- Usage (via llama_sampler_init_lua):
--   llama_sampler_chain_add(chain, llama_sampler_init_lua("samplers/temperature.lua"));

local temperature = 0.8

-- Seed the RNG once on startup.
math.randomseed(os.time())

function apply(candidates)
    local n    = candidates.n
    local data = candidates.data

    -- Find max logit for numerical stability.
    local max_logit = data[1].logit
    for i = 2, n do
        if data[i].logit > max_logit then
            max_logit = data[i].logit
        end
    end

    -- Compute softmax with temperature scaling.
    local probs = {}
    local sum   = 0.0
    for i = 1, n do
        local p = math.exp((data[i].logit - max_logit) / temperature)
        probs[i] = p
        sum = sum + p
    end

    -- Sample proportionally.
    local r      = math.random() * sum
    local cumsum = 0.0
    for i = 1, n do
        cumsum = cumsum + probs[i]
        if cumsum >= r then
            return i
        end
    end

    return n  -- fallback to last candidate
end
