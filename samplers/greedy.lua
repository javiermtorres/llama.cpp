-- Greedy sampler: always selects the token with the highest logit.
--
-- Usage (via llama_sampler_init_lua):
--   llama_sampler_chain_add(chain, llama_sampler_init_lua("samplers/greedy.lua"));

function apply(candidates)
    local best_idx   = 1
    local best_logit = candidates.data[1].logit

    for i = 2, candidates.n do
        if candidates.data[i].logit > best_logit then
            best_idx   = i
            best_logit = candidates.data[i].logit
        end
    end

    return best_idx
end
