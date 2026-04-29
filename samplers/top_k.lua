-- Top-K sampler: samples uniformly from the K candidates with the highest
-- logit values.
--
-- Usage (via llama_sampler_init_lua):
--   llama_sampler_chain_add(chain, llama_sampler_init_lua("samplers/top_k.lua"));

local k = 50  -- number of top candidates to consider

math.randomseed(math.floor(os.time() + os.clock() * 1e6))

function apply(candidates)
    local n    = candidates.n
    local data = candidates.data

    -- Collect and sort candidates by logit descending.
    local sorted = {}
    for i = 1, n do
        sorted[i] = {idx = i, logit = data[i].logit}
    end
    table.sort(sorted, function(a, b) return a.logit > b.logit end)

    -- Clamp k to the number of available candidates.
    local actual_k = math.min(k, n)

    -- Sample uniformly from the top-k.
    local pick = sorted[math.random(actual_k)]
    return pick.idx
end
