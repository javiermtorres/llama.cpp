#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef LLAMA_USE_LUA

// Write a Lua script string to a temporary file and return its path.
static std::string write_lua_script(const char * name, const char * code) {
    std::string path = std::string("/tmp/") + name;
    FILE * f = fopen(path.c_str(), "w");
    assert(f != nullptr);
    fputs(code, f);
    fclose(f);
    return path;
}

// Build a small candidate array for testing.
static std::vector<llama_token_data> make_candidates(const std::vector<float> & logits) {
    std::vector<llama_token_data> out;
    out.reserve(logits.size());
    for (llama_token id = 0; id < (llama_token)logits.size(); id++) {
        out.push_back({id, logits[id], 0.0f});
    }
    return out;
}

// Test: greedy selection returns the index with the highest logit.
static void test_lua_greedy() {
    const char * script =
        "function apply(candidates)\n"
        "    local best = 1\n"
        "    local best_logit = candidates.data[1].logit\n"
        "    for i = 2, candidates.n do\n"
        "        if candidates.data[i].logit > best_logit then\n"
        "            best = i\n"
        "            best_logit = candidates.data[i].logit\n"
        "        end\n"
        "    end\n"
        "    return best\n"
        "end\n";

    std::string path = write_lua_script("test_greedy.lua", script);

    llama_sampler * smpl = llama_sampler_init_lua(path.c_str());
    assert(smpl != nullptr);

    // Logits: token 2 has the highest value.
    auto cur = make_candidates({1.0f, 2.0f, 5.0f, 3.0f, 0.5f});
    llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};

    llama_sampler_apply(smpl, &cur_p);

    // Expect index 2 (0-based), i.e. the third token.
    assert(cur_p.selected == 2);

    llama_sampler_free(smpl);
    printf("test_lua_greedy passed\n");
}

// Test: accept and reset callbacks are invoked without errors.
static void test_lua_callbacks() {
    static int accept_calls = 0;
    static int reset_calls  = 0;
    (void)accept_calls;
    (void)reset_calls;

    const char * script =
        "accept_calls = 0\n"
        "reset_calls  = 0\n"
        "function apply(candidates)\n"
        "    return 1\n"
        "end\n"
        "function accept(token)\n"
        "    accept_calls = accept_calls + 1\n"
        "end\n"
        "function reset()\n"
        "    reset_calls = reset_calls + 1\n"
        "end\n";

    std::string path = write_lua_script("test_callbacks.lua", script);

    llama_sampler * smpl = llama_sampler_init_lua(path.c_str());
    assert(smpl != nullptr);

    auto cur = make_candidates({1.0f, 2.0f, 3.0f});
    llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};

    llama_sampler_apply(smpl, &cur_p);
    assert(cur_p.selected == 0);

    llama_sampler_accept(smpl, 42);
    llama_sampler_reset(smpl);

    llama_sampler_free(smpl);
    printf("test_lua_callbacks passed\n");
}

// Test: missing 'apply' function returns nullptr.
static void test_lua_missing_apply() {
    const char * script = "-- no apply function defined\n";

    std::string path = write_lua_script("test_missing_apply.lua", script);

    llama_sampler * smpl = llama_sampler_init_lua(path.c_str());
    assert(smpl == nullptr);

    printf("test_lua_missing_apply passed\n");
}

// Test: invalid Lua syntax returns nullptr.
static void test_lua_syntax_error() {
    const char * script = "this is not valid lua !!!\n";

    std::string path = write_lua_script("test_syntax_error.lua", script);

    llama_sampler * smpl = llama_sampler_init_lua(path.c_str());
    assert(smpl == nullptr);

    printf("test_lua_syntax_error passed\n");
}

// Test: clone creates an independent sampler that behaves the same way.
static void test_lua_clone() {
    const char * script =
        "function apply(candidates)\n"
        "    return candidates.n  -- always select the last candidate\n"
        "end\n";

    std::string path = write_lua_script("test_clone.lua", script);

    llama_sampler * orig = llama_sampler_init_lua(path.c_str());
    assert(orig != nullptr);

    llama_sampler * cloned = llama_sampler_clone(orig);
    assert(cloned != nullptr);

    auto cur = make_candidates({1.0f, 2.0f, 3.0f});
    llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};

    llama_sampler_apply(cloned, &cur_p);
    // Expected: last candidate index = n-1 = 2 (0-based).
    assert(cur_p.selected == 2);

    llama_sampler_free(orig);
    llama_sampler_free(cloned);
    printf("test_lua_clone passed\n");
}

int main() {
    test_lua_greedy();
    test_lua_callbacks();
    test_lua_missing_apply();
    test_lua_syntax_error();
    test_lua_clone();

    printf("All Lua sampler tests passed.\n");
    return 0;
}

#else // LLAMA_USE_LUA

int main() {
    printf("Lua support not compiled in; skipping tests.\n");
    return 0;
}

#endif // LLAMA_USE_LUA
