// llama-lua: run a Lua script that drives llama.cpp inference.
//
// The tool exposes a "llama" global table in Lua with the following API:
//
//   -- Model / context lifecycle
//   local model = llama.model_load(path, n_gpu_layers)
//   llama.model_free(model)
//   local ctx   = llama.context_init(model, opts)  -- opts: {n_ctx, n_batch, n_seq_max, n_threads}
//   llama.context_free(ctx)
//
//   -- Vocab helpers
//   local n     = llama.n_vocab(model)
//   local bos   = llama.token_bos(model)
//   local eos   = llama.token_eos(model)
//   local piece = llama.token_to_piece(ctx, token_id)  -- returns string
//
//   -- Tokenization
//   local tokens = llama.tokenize(ctx, text, add_bos)  -- returns 1-based table
//
//   -- Batch management
//   local batch = llama.batch_init(max_tokens [, n_seq_max])
//   llama.batch_free(batch)
//   llama.batch_clear(batch)
//   llama.batch_add(batch, token_id, pos, seq_ids, want_logits)
//     -- seq_ids: 1-based table of integer sequence ids
//
//   -- Decode
//   local ok = llama.decode(ctx, batch)  -- true on success
//
//   -- Logits / probabilities (valid until next decode call)
//   local logits    = llama.get_logits_ith(ctx, i)        -- output index i (0-based, or -1 for last)
//     -- returns 1-based table of n_vocab floats
//   local probs     = llama.softmax(logits)               -- numerically stable softmax
//     -- returns 1-based table of n_vocab floats
//   local cands     = llama.make_candidates(logits [, k]) -- sampler-hook-format candidates table
//     -- if k > 0, only the top-k candidates are included (sorted by logit descending)
//     -- cands.n, cands.sorted, cands.data[i].{id, logit, p}
//   local tok, logit, lp = llama.argmax_logits_ith(ctx, i)  -- fast greedy pick (no Lua table)
//     -- returns 0-based token id, raw logit, and log-probability
//
//   -- Memory / KV-cache management
//   llama.seq_cp(ctx, src_seq_id, dst_seq_id [, p0, p1])  -- p1=-1 means "all"
//   llama.seq_rm(ctx, seq_id [, p0, p1])
//   llama.seq_keep(ctx, seq_id)
//
// Usage:
//   llama-lua --script my_script.lua --model model.gguf [-- extra args passed to Lua as arg table]

#include "llama.h"

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Userdata tag names
// ---------------------------------------------------------------------------
static const char * MODEL_TAG = "llama.model";
static const char * CTX_TAG   = "llama.context";
static const char * BATCH_TAG = "llama.batch";

// ---------------------------------------------------------------------------
// Helper: push a formatted error and return the number of results (0)
// ---------------------------------------------------------------------------
static int push_error(lua_State * L, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    lua_pushnil(L);
    lua_pushvfstring(L, fmt, ap);
    va_end(ap);
    return 2; // nil, errmsg
}

// ---------------------------------------------------------------------------
// Model userdata
// ---------------------------------------------------------------------------
static llama_model ** check_model(lua_State * L, int idx) {
    return (llama_model **) luaL_checkudata(L, idx, MODEL_TAG);
}

static int l_model_free(lua_State * L) {
    llama_model ** pp = check_model(L, 1);
    if (*pp) {
        llama_model_free(*pp);
        *pp = nullptr;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Context userdata
// ---------------------------------------------------------------------------
static llama_context ** check_ctx(lua_State * L, int idx) {
    return (llama_context **) luaL_checkudata(L, idx, CTX_TAG);
}

static int l_context_free(lua_State * L) {
    llama_context ** pp = check_ctx(L, 1);
    if (*pp) {
        llama_free(*pp);
        *pp = nullptr;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Batch userdata
//
// We store both the llama_batch and the max_tokens so we can do bounds checks.
// ---------------------------------------------------------------------------
struct lua_batch {
    llama_batch batch;
    int32_t     max_tokens;
    int32_t     n_seq_max;   // max seq_ids per token slot
};

static lua_batch * check_batch(lua_State * L, int idx) {
    return (lua_batch *) luaL_checkudata(L, idx, BATCH_TAG);
}

static int l_batch_free(lua_State * L) {
    lua_batch * lb = check_batch(L, 1);
    if (lb->max_tokens > 0) {
        llama_batch_free(lb->batch);
        lb->max_tokens = 0;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// llama.model_load(path, n_gpu_layers) -> model | nil, errmsg
// ---------------------------------------------------------------------------
static int l_model_load(lua_State * L) {
    const char * path          = luaL_checkstring(L, 1);
    int32_t      n_gpu_layers  = (int32_t) luaL_optinteger(L, 2, 0);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        return push_error(L, "failed to load model: %s", path);
    }

    llama_model ** pp = (llama_model **) lua_newuserdata(L, sizeof(llama_model *));
    *pp = model;
    luaL_setmetatable(L, MODEL_TAG);
    return 1;
}

// ---------------------------------------------------------------------------
// llama.context_init(model, opts) -> ctx | nil, errmsg
//   opts table keys (all optional):
//     n_ctx, n_batch, n_ubatch, n_seq_max, n_threads, n_threads_batch
// ---------------------------------------------------------------------------
static int l_context_init(lua_State * L) {
    llama_model ** pp = check_model(L, 1);
    if (!*pp) {
        return push_error(L, "model is already freed");
    }

    llama_context_params cparams = llama_context_default_params();

    if (lua_istable(L, 2)) {
        auto get_int = [&](const char * key, uint32_t & out) {
            lua_getfield(L, 2, key);
            if (lua_isinteger(L, -1)) {
                out = (uint32_t) lua_tointeger(L, -1);
            }
            lua_pop(L, 1);
        };
        auto get_int32 = [&](const char * key, int32_t & out) {
            lua_getfield(L, 2, key);
            if (lua_isinteger(L, -1)) {
                out = (int32_t) lua_tointeger(L, -1);
            }
            lua_pop(L, 1);
        };

        get_int  ("n_ctx",           cparams.n_ctx);
        get_int  ("n_batch",         cparams.n_batch);
        get_int  ("n_ubatch",        cparams.n_ubatch);
        get_int  ("n_seq_max",       cparams.n_seq_max);
        get_int32("n_threads",       cparams.n_threads);
        get_int32("n_threads_batch", cparams.n_threads_batch);
    }

    llama_context * ctx = llama_init_from_model(*pp, cparams);
    if (!ctx) {
        return push_error(L, "failed to create context");
    }

    llama_context ** cpp = (llama_context **) lua_newuserdata(L, sizeof(llama_context *));
    *cpp = ctx;
    luaL_setmetatable(L, CTX_TAG);
    return 1;
}

// ---------------------------------------------------------------------------
// Vocab helpers
// ---------------------------------------------------------------------------
static int l_n_vocab(lua_State * L) {
    llama_model ** pp   = check_model(L, 1);
    const llama_vocab * vocab = llama_model_get_vocab(*pp);
    lua_pushinteger(L, llama_vocab_n_tokens(vocab));
    return 1;
}

static int l_token_bos(lua_State * L) {
    llama_model ** pp   = check_model(L, 1);
    const llama_vocab * vocab = llama_model_get_vocab(*pp);
    lua_pushinteger(L, llama_vocab_bos(vocab));
    return 1;
}

static int l_token_eos(lua_State * L) {
    llama_model ** pp   = check_model(L, 1);
    const llama_vocab * vocab = llama_model_get_vocab(*pp);
    lua_pushinteger(L, llama_vocab_eos(vocab));
    return 1;
}

static int l_token_to_piece(lua_State * L) {
    llama_context ** pp = check_ctx(L, 1);
    llama_token tok = (llama_token) luaL_checkinteger(L, 2);

    const llama_model * model = llama_get_model(*pp);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    char buf[512];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf) - 1, 0, true);
    if (n < 0) {
        lua_pushstring(L, "");
    } else {
        buf[n] = '\0';
        lua_pushstring(L, buf);
    }
    return 1;
}

// ---------------------------------------------------------------------------
// llama.tokenize(ctx, text, add_bos) -> 1-based table of token ids
// ---------------------------------------------------------------------------
static int l_tokenize(lua_State * L) {
    llama_context ** pp = check_ctx(L, 1);
    const char *     text    = luaL_checkstring(L, 2);
    bool             add_bos = lua_toboolean(L, 3); // default false

    const llama_model * model = llama_get_model(*pp);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int text_len = (int) strlen(text);

    // First pass: determine the exact number of tokens needed.
    int n_tokens = llama_tokenize(vocab, text, text_len, nullptr, 0, add_bos, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens; // llama_tokenize returns -required_size on overflow
    }
    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(vocab, text, text_len,
                              tokens.data(), (int32_t) tokens.size(),
                              add_bos, true);
    if (n_tokens < 0) {
        return push_error(L, "tokenize failed unexpectedly after sizing pass");
    }
    tokens.resize(n_tokens);

    lua_createtable(L, n_tokens, 0);
    for (int i = 0; i < n_tokens; ++i) {
        lua_pushinteger(L, tokens[i]);
        lua_rawseti(L, -2, i + 1); // 1-based
    }
    return 1;
}

// ---------------------------------------------------------------------------
// llama.batch_init(max_tokens [, n_seq_max]) -> batch
//   n_seq_max: max number of seq_ids per token slot (default 1)
// ---------------------------------------------------------------------------
static int l_batch_init(lua_State * L) {
    int32_t max_tokens = (int32_t) luaL_checkinteger(L, 1);
    int32_t n_seq_max  = (int32_t) luaL_optinteger(L, 2, 1);
    if (max_tokens <= 0) {
        return luaL_error(L, "max_tokens must be > 0");
    }
    if (n_seq_max <= 0) {
        return luaL_error(L, "n_seq_max must be > 0");
    }

    lua_batch * lb = (lua_batch *) lua_newuserdata(L, sizeof(lua_batch));
    lb->batch      = llama_batch_init(max_tokens, 0, n_seq_max);
    lb->max_tokens = max_tokens;
    lb->n_seq_max  = n_seq_max;
    luaL_setmetatable(L, BATCH_TAG);
    return 1;
}

static int l_batch_clear(lua_State * L) {
    lua_batch * lb = check_batch(L, 1);
    lb->batch.n_tokens = 0;
    return 0;
}

// ---------------------------------------------------------------------------
// llama.batch_add(batch, token_id, pos, seq_ids, want_logits)
//   seq_ids: 1-based table  {0, 1, 2, ...}   (integer sequence ids)
// ---------------------------------------------------------------------------
static int l_batch_add(lua_State * L) {
    lua_batch * lb         = check_batch(L, 1);
    llama_token token      = (llama_token) luaL_checkinteger(L, 2);
    llama_pos   pos        = (llama_pos)   luaL_checkinteger(L, 3);
    luaL_checktype(L, 4, LUA_TTABLE);
    bool        want_logits = lua_toboolean(L, 5);

    if (lb->batch.n_tokens >= lb->max_tokens) {
        return luaL_error(L, "batch is full (max_tokens=%d)", lb->max_tokens);
    }

    // collect seq_ids from the Lua table
    int n_seq = (int) lua_rawlen(L, 4);
    if (n_seq <= 0 || n_seq > lb->n_seq_max) {
        return luaL_error(L, "seq_ids table has %d entries but batch n_seq_max=%d", n_seq, lb->n_seq_max);
    }

    int32_t idx = lb->batch.n_tokens;
    lb->batch.token   [idx] = token;
    lb->batch.pos     [idx] = pos;
    lb->batch.n_seq_id[idx] = n_seq;
    for (int i = 0; i < n_seq; ++i) {
        lua_rawgeti(L, 4, i + 1);
        lb->batch.seq_id[idx][i] = (llama_seq_id) lua_tointeger(L, -1);
        lua_pop(L, 1);
    }
    lb->batch.logits[idx] = want_logits ? 1 : 0;
    lb->batch.n_tokens++;
    return 0;
}

// ---------------------------------------------------------------------------
// llama.decode(ctx, batch) -> true | nil, errmsg
// ---------------------------------------------------------------------------
static int l_decode(lua_State * L) {
    llama_context ** pp = check_ctx(L, 1);
    lua_batch * lb      = check_batch(L, 2);

    int ret = llama_decode(*pp, lb->batch);
    if (ret != 0) {
        return push_error(L, "llama_decode failed (ret=%d)", ret);
    }
    lua_pushboolean(L, 1);
    return 1;
}

// ---------------------------------------------------------------------------
// llama.get_logits_ith(ctx, batch_index) -> 1-based table of n_vocab floats
//   batch_index is 0-based (matches the order tokens were added to the batch
//   with want_logits=true)
// ---------------------------------------------------------------------------
static int l_get_logits_ith(lua_State * L) {
    llama_context ** pp = check_ctx(L, 1);
    int32_t          i  = (int32_t) luaL_checkinteger(L, 2);

    const float * logits = llama_get_logits_ith(*pp, i);
    if (!logits) {
        return push_error(L, "get_logits_ith(%d) returned NULL", i);
    }

    const llama_model * model = llama_get_model(*pp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    lua_createtable(L, n_vocab, 0);
    for (int32_t v = 0; v < n_vocab; ++v) {
        lua_pushnumber(L, (lua_Number) logits[v]);
        lua_rawseti(L, -2, v + 1); // 1-based
    }
    return 1;
}

// ---------------------------------------------------------------------------
// llama.softmax(logits) -> 1-based table of n_vocab probabilities
//   Numerically stable softmax: P[i] = exp(logits[i] - max) / sum
// ---------------------------------------------------------------------------
static int l_softmax(lua_State * L) {
    luaL_checktype(L, 1, LUA_TTABLE);
    int n = (int) lua_rawlen(L, 1);
    if (n <= 0) {
        lua_createtable(L, 0, 0);
        return 1;
    }

    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        lua_rawgeti(L, 1, i + 1);
        v[i] = (float) lua_tonumber(L, -1);
        lua_pop(L, 1);
    }

    float max_v = *std::max_element(v.begin(), v.end());
    float sum   = 0.0f;
    for (int i = 0; i < n; ++i) {
        v[i]  = std::exp(v[i] - max_v);
        sum  += v[i];
    }

    lua_createtable(L, n, 0);
    for (int i = 0; i < n; ++i) {
        lua_pushnumber(L, (lua_Number)(v[i] / sum));
        lua_rawseti(L, -2, i + 1);
    }
    return 1;
}

// ---------------------------------------------------------------------------
// llama.make_candidates(logits [, top_k]) -> candidates table
//
// Returns a candidates table in the same format used by the llama_sampler_i
// hook, so that sampler scripts that define apply(candidates) can be used
// unchanged inside llama-lua scripts:
//
//   cands.n          -- number of candidates (n_vocab, or top_k if k > 0)
//   cands.sorted     -- true if k > 0 (sorted by logit descending)
//   cands.data[i]    -- 1-based
//     .id            -- 0-based token id
//     .logit         -- raw logit
//     .p             -- softmax probability
//
// Example bridge to an existing sampler script:
//   dofile("samplers/greedy.lua")          -- defines apply(candidates)
//   local logits = llama.get_logits_ith(ctx, -1)
//   local cands  = llama.make_candidates(logits)
//   local sel    = apply(cands)            -- returns 1-based index
//   local tok    = cands.data[sel].id     -- 0-based token id
//
// Note: building a full n_vocab table on every step carries overhead.
// When only the argmax is needed (e.g. greedy rollout), prefer
// llama.argmax_logits_ith() instead.
// ---------------------------------------------------------------------------
static int l_make_candidates(lua_State * L) {
    luaL_checktype(L, 1, LUA_TTABLE);
    int n      = (int) lua_rawlen(L, 1);
    int top_k  = (int) luaL_optinteger(L, 2, 0); // 0 = all
    if (n <= 0) {
        return push_error(L, "make_candidates: logits table is empty");
    }

    std::vector<float> logits(n);
    for (int i = 0; i < n; ++i) {
        lua_rawgeti(L, 1, i + 1);
        logits[i] = (float) lua_tonumber(L, -1);
        lua_pop(L, 1);
    }

    // softmax
    float max_l = *std::max_element(logits.begin(), logits.end());
    float sum   = 0.0f;
    std::vector<float> probs(n);
    for (int i = 0; i < n; ++i) {
        probs[i]  = std::exp(logits[i] - max_l);
        sum      += probs[i];
    }
    for (int i = 0; i < n; ++i) probs[i] /= sum;

    // build index list, optionally top-k sorted
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    bool sorted = (top_k > 0);
    if (sorted) {
        if (top_k > n) top_k = n;
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
            [&](int a, int b){ return logits[a] > logits[b]; });
        idx.resize(top_k);
    }

    int out_n = (int) idx.size();

    // build Lua table: {n=..., sorted=..., data={...}}
    lua_createtable(L, 0, 3);

    lua_pushinteger(L, out_n);
    lua_setfield(L, -2, "n");

    lua_pushboolean(L, sorted ? 1 : 0);
    lua_setfield(L, -2, "sorted");

    lua_createtable(L, out_n, 0);
    for (int i = 0; i < out_n; ++i) {
        int ti = idx[i];         // token index (0-based id)
        lua_createtable(L, 0, 3);

        lua_pushinteger(L, ti);
        lua_setfield(L, -2, "id");

        lua_pushnumber(L, (lua_Number) logits[ti]);
        lua_setfield(L, -2, "logit");

        lua_pushnumber(L, (lua_Number) probs[ti]);
        lua_setfield(L, -2, "p");

        lua_rawseti(L, -2, i + 1); // 1-based
    }
    lua_setfield(L, -2, "data");

    return 1;
}

// ---------------------------------------------------------------------------
// llama.argmax_logits_ith(ctx, output_index) -> token_id, logit, log_prob
//
// Fast greedy pick: scans logits in C++ and returns the single best token.
// output_index follows the same convention as get_logits_ith (0-based among
// tokens that requested logits, or -1 for the last output).
//
// Use this inside simulation rollouts where building a full n_vocab Lua table
// on every step would be too slow.
// ---------------------------------------------------------------------------
static int l_argmax_logits_ith(lua_State * L) {
    llama_context ** pp = check_ctx(L, 1);
    int32_t          i  = (int32_t) luaL_checkinteger(L, 2);

    const float * logits = llama_get_logits_ith(*pp, i);
    if (!logits) {
        return push_error(L, "argmax_logits_ith(%d) returned NULL", i);
    }

    const llama_model * model   = llama_get_model(*pp);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    int32_t             n_vocab = llama_vocab_n_tokens(vocab);

    // argmax
    int32_t best_id   = 0;
    float   best_logit = logits[0];
    for (int32_t v = 1; v < n_vocab; ++v) {
        if (logits[v] > best_logit) {
            best_logit = logits[v];
            best_id    = v;
        }
    }

    // log-prob of the best token via numerically-stable softmax:
    //   P(best) = exp(best_logit - best_logit) / sum_i exp(logit[i] - best_logit)
    //           = 1 / sum
    //   log P(best) = -log(sum)
    float sum = 0.0f;
    for (int32_t v = 0; v < n_vocab; ++v) {
        sum += std::exp(logits[v] - best_logit);
    }
    float log_prob = -(float) std::log((double) sum);

    lua_pushinteger(L, best_id);
    lua_pushnumber(L, (lua_Number) best_logit);
    lua_pushnumber(L, (lua_Number) log_prob);
    return 3;
}

// ---------------------------------------------------------------------------
// KV-cache / sequence management
// ---------------------------------------------------------------------------
static int l_seq_cp(lua_State * L) {
    llama_context ** pp = check_ctx(L, 1);
    llama_seq_id src = (llama_seq_id) luaL_checkinteger(L, 2);
    llama_seq_id dst = (llama_seq_id) luaL_checkinteger(L, 3);
    llama_pos    p0  = (llama_pos)    luaL_optinteger(L, 4, 0);
    llama_pos    p1  = (llama_pos)    luaL_optinteger(L, 5, -1);
    llama_memory_seq_cp(llama_get_memory(*pp), src, dst, p0, p1);
    return 0;
}

static int l_seq_rm(lua_State * L) {
    llama_context ** pp = check_ctx(L, 1);
    llama_seq_id seq = (llama_seq_id) luaL_checkinteger(L, 2);
    llama_pos    p0  = (llama_pos)    luaL_optinteger(L, 3, 0);
    llama_pos    p1  = (llama_pos)    luaL_optinteger(L, 4, -1);
    llama_memory_seq_rm(llama_get_memory(*pp), seq, p0, p1);
    return 0;
}

static int l_seq_keep(lua_State * L) {
    llama_context ** pp = check_ctx(L, 1);
    llama_seq_id seq = (llama_seq_id) luaL_checkinteger(L, 2);
    llama_memory_seq_keep(llama_get_memory(*pp), seq);
    return 0;
}

// ---------------------------------------------------------------------------
// Register the "llama" module
// ---------------------------------------------------------------------------
static const luaL_Reg llama_funcs[] = {
    { "model_load",          l_model_load        },
    { "context_init",        l_context_init      },
    { "n_vocab",             l_n_vocab           },
    { "token_bos",           l_token_bos         },
    { "token_eos",           l_token_eos         },
    { "token_to_piece",      l_token_to_piece    },
    { "tokenize",            l_tokenize          },
    { "batch_init",          l_batch_init        },
    { "batch_clear",         l_batch_clear       },
    { "batch_add",           l_batch_add         },
    { "decode",              l_decode            },
    { "get_logits_ith",      l_get_logits_ith    },
    { "softmax",             l_softmax           },
    { "make_candidates",     l_make_candidates   },
    { "argmax_logits_ith",   l_argmax_logits_ith },
    { "seq_cp",              l_seq_cp            },
    { "seq_rm",              l_seq_rm            },
    { "seq_keep",            l_seq_keep          },
    { nullptr, nullptr }
};

static void register_metatables(lua_State * L) {
    // model
    luaL_newmetatable(L, MODEL_TAG);
    lua_pushcfunction(L, l_model_free);
    lua_setfield(L, -2, "__gc");
    lua_pop(L, 1);

    // context
    luaL_newmetatable(L, CTX_TAG);
    lua_pushcfunction(L, l_context_free);
    lua_setfield(L, -2, "__gc");
    lua_pop(L, 1);

    // batch
    luaL_newmetatable(L, BATCH_TAG);
    lua_pushcfunction(L, l_batch_free);
    lua_setfield(L, -2, "__gc");
    lua_pop(L, 1);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s --script <script.lua> [--model <model.gguf>] [-- arg1 arg2 ...]\n", prog);
    fprintf(stderr, "  --script  Path to the Lua script to run\n");
    fprintf(stderr, "  --model   Optional default model path (also available as arg.model in Lua)\n");
    fprintf(stderr, "  --        All arguments after this are passed to the Lua script as the 'arg' table\n");
}

int main(int argc, char ** argv) {
    const char * script_path = nullptr;
    const char * model_path  = nullptr;
    int          lua_arg_start = argc; // index of first Lua arg

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--script") == 0 && i + 1 < argc) {
            script_path = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--") == 0) {
            lua_arg_start = i + 1;
            break;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!script_path) {
        fprintf(stderr, "error: --script is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    lua_State * L = luaL_newstate();
    luaL_openlibs(L);
    register_metatables(L);

    // Register llama module as global table
    luaL_newlib(L, llama_funcs);
    lua_setglobal(L, "llama");

    // Populate the 'arg' table (similar to standalone Lua interpreter):
    //   arg[0] = script name, arg[1]..arg[n] = extra CLI args
    //   arg.model = model_path (if provided)
    lua_createtable(L, argc - lua_arg_start, 2);
    lua_pushstring(L, script_path);
    lua_rawseti(L, -2, 0);
    for (int i = lua_arg_start; i < argc; ++i) {
        lua_pushstring(L, argv[i]);
        lua_rawseti(L, -2, i - lua_arg_start + 1);
    }
    if (model_path) {
        lua_pushstring(L, model_path);
        lua_setfield(L, -2, "model");
    }
    lua_setglobal(L, "arg");

    int ret = 0;
    if (luaL_dofile(L, script_path) != LUA_OK) {
        fprintf(stderr, "error: %s\n", lua_tostring(L, -1));
        ret = 1;
    }

    lua_close(L);
    llama_backend_free();
    return ret;
}
