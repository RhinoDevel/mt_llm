
// Marcel Timm, RhinoDevel, 2024aug21

#include <cmath>
#include <cstdint>
#include <cassert>
#include <vector>
#include <cstdio>
#include <cstring>

#include "log.h"
#include "common.h"
#include "llama.h"
#include "chat.h"

#include "mt_llm.h"
#include "mt_llm_p.h"
#include "mt_llm_model.h"
#include "mt_llm_ctx.h"
#include "mt_llm_s.h"
#include "mt_llm_log.h"
#include "mt_llm_state.h"

#include "mt_llm_tok_type.h"

// mt_llm_snapshot is also hard-coded to two LLM slots!
static struct mt_llm_s * s_slots[] = { nullptr, nullptr };

static bool s_is_common_init = false;
static bool s_is_backend_init = false;

static void clear_llama_memory(llama_context const * const ctx)
{
    llama_memory_t kv = llama_get_memory(ctx);

    if(kv != nullptr)
    {
        llama_memory_clear(kv, true);
        kv = nullptr;
    }
}

static void init_common_if_necessary(bool const enable_llama_cpp_log)
{
    if(!s_is_common_init)
    {
        common_init();

        if(!enable_llama_cpp_log)
        {
            common_log_pause(common_log_main());
            //
            //static void llama_log_callback_null(ggml_log_level level, const char * text, void * user_data) { (void) level; (void) text; (void) user_data; }
            //llama_log_set(llama_log_callback_null, NULL);
        }

        s_is_common_init = true;
    }
}

static void init_backend_if_necessary(void)
{
    if(!s_is_backend_init)
    {
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED); // Unnecessary this way.

        s_is_backend_init = true;
    }
}

/** Calculate the probability of each token's logit in the given vector via
 *  softmax.
 *
 * - The index equals the token ID.
 * - Original source: llama.cpp/tools/server/utils.hpp/get_token_probabilities()
 */
static std::vector<float> get_probabilities(
    std::vector<float> const & logits, float const max)
{
    assert(!logits.empty());
    assert(*std::max_element(logits.begin(), logits.end()) == max);

    int const len = static_cast<int>(logits.size());
    std::vector<float> ret_val;
    float sum = 0.0f;

    ret_val.resize(len);

    for(int i = 0; i < len; ++i)
    {
        ret_val[i] = expf(logits[i] - max);

        sum += ret_val[i];
    }

    for(int i = 0; i < len; ++i)
    {
        ret_val[i] /= sum;
    }

    return ret_val;
}

/** Get a copy of the last logits and fill given reference with maximum logit
 *  value found.
 * 
 * - The index equals the token ID.
 * - Returns an empty vector, if no last logits available.
 * - Original source: llama.cpp/tools/server/utils.hpp/get_token_probabilities()
 */
static std::vector<float> get_last_logits(mt_llm_s const &s, float& max)
{
    std::vector<float> ret_val;

    float const * const logits = llama_get_logits_ith(s.ctx, -1);

    if(logits == nullptr)
    {
        assert(ret_val.empty());
        return ret_val;
    }

    int32_t const n_vocab = llama_vocab_n_tokens(
        llama_model_get_vocab(s.model->model));

    assert(0 < n_vocab);

    ret_val.resize(static_cast<size_t>(n_vocab));

    ret_val[0] = logits[0];
    max = ret_val[0];
    for(int i = 1; i < n_vocab; ++i) // i == Token ID.
    {
        ret_val[i] = logits[i];

        if(max < ret_val[i])
        {
            max = ret_val[i];
        }
    }
    return ret_val;
}

static std::vector<float> get_token_group_probabilities(
    std::vector<std::vector<int>> const & token_groups,
    std::vector<float> const & token_probabilities)
{
    size_t const n_groups = token_groups.size(); // Count of token groups.
    std::vector<float> ret_val(n_groups);

    for(int i = 0; i < static_cast<int>(n_groups); ++i)
    {
        for(int j = 0; j < static_cast<int>(token_groups[i].size()); ++j)
        {
            ret_val[i] += token_probabilities[token_groups[i][j]];
        }
    }
    return ret_val;
}

/**
 * - Returns true for an empty string given. 
 */
static bool is_whitespace_only(char const * const c)
{
    assert(c != nullptr);

    char const * ptr = c;

    while(*ptr != '\0')
    {
        if(!std::isspace(static_cast<int>(static_cast<unsigned char>(*ptr))))
        {
            return false;
        }
        ++ptr;
    }
    return true;
}

static bool callback_handler(
    llama_token const tok,
    std::string const & piece,
    std::vector<float> const & dig_probs,
    mt_llm_s const &s)
{
    assert(0 < s.last_tok_type);

    if(piece.empty()) // Token is omitted by llama.cpp => Also omit here.
    {
        MT_LOG("Warning: Retrieved token with ID %d having empty string representation.", tok);
        return false; // <=> No interruption.
    }
    return s.mt_p->callback(
        static_cast<int>(tok),
        piece.c_str(),
        s.last_tok_type,
        dig_probs.empty() ? nullptr : dig_probs.data());
}

/** Add tokens to context.
 *
 *  - "Decode" as in using the decoder of the LLM architecture to add to its
 *    context.
 *  - Slot's token count will hold the correct value on exit, even, if the
 *    decode failed (e.g. not all tokens could be decoded, because of full
 *    context).
 */
static bool decode_tokens(
    std::vector<llama_token>& tokens,
    int const tok_type,
    mt_llm_s& s,
    std::vector<float> const & dig_probs,
    bool& irq)
{
    int decoded_tok_cnt = 0;

    s.last_tok_type = tok_type;

    bool const ret_val = mt_llm_ctx_decode(
        s,
        tokens,
        decoded_tok_cnt,
        callback_handler,
        dig_probs,
        irq);

    // Will be correct on error, too:
    assert(0 <= decoded_tok_cnt);
    // Works, because function called above insert a BOS token into the vector,
    // if necessary:
    assert(!ret_val || decoded_tok_cnt == static_cast<int>(tokens.size()));
    s.tok_cnt += decoded_tok_cnt;

    if(!ret_val)
    {
        MT_LOG_ERR("Decoding!\n");
    }
    return ret_val;
}

/** Add token representation of given string to context.
 * 
 *  - "Decode" as in using the decoder of the LLM architecture to add to its
 *    context.
 *  - Slot's token count will hold the correct value on exit, even, if the
 *    decode failed (e.g. not all tokens could be decoded, because of full
 *    context).
 */
static bool decode_str(char const * const str, int const tok_type, mt_llm_s& s)
{
    bool irq_dummy; // IRQ is ignored, here.
    std::vector<llama_token> tokens;

    if(str[0] == '\0')
    {
        MT_LOG("Warning: Received empty string.");
    }

    // Tokenize the given string:
    tokens = mt_llm_ctx_tokenize(
        *s.ctx,
        str,
        false); // No adding of BOS and/or EOS [is both model-dependent].

//#ifndef NDEBUG
//    // E.g. to compare with:
//    //     llama-completion.exe --model D:\LLMs\Qwen3.5-9B-Q4_K_M.gguf --verbose --jinja
//    MT_LOG("Tokenized \"%s\" into:\n", str);
//    for(int i = 0; i < static_cast<int>(tokens.size()); ++i)
//    {
//        MT_LOG("Index %d = %d\n", i, tokens[i]);
//    }
//#endif //NDEBUG

    return decode_tokens(tokens, tok_type, s, std::vector<float>(), irq_dummy);
}

static bool decode_some_prompt_end_delim_with_thinking(
    char const * const some_prompt_end_delim, mt_llm_s& s)
{
    assert(s.mt_p->think_end_delim[0] != '\0');

    std::string str_some_prompt_end_delim = some_prompt_end_delim;
    std::string const str_newline = "\n";
    std::string const str_think_beg_delim = s.mt_p->think_beg_delim;
    std::string const str_think_end_delim = s.mt_p->think_end_delim;

    int decode_trailing_newlines = 0;
    bool decode_think_end_delim = false;
    int decode_middle_newlines = 0;
    bool decode_think_beg_delim = false;

    while(string_ends_with(str_some_prompt_end_delim, str_newline))
    {
        ++decode_trailing_newlines;

        string_remove_suffix(str_some_prompt_end_delim, str_newline);
    }

    if(string_ends_with(str_some_prompt_end_delim, str_think_end_delim))
    {
        decode_think_end_delim = true;

        string_remove_suffix(str_some_prompt_end_delim, str_think_end_delim);
    }

    while(string_ends_with(str_some_prompt_end_delim, str_newline))
    {
        ++decode_middle_newlines;

        string_remove_suffix(str_some_prompt_end_delim, str_newline);
    }

    if(string_ends_with(str_some_prompt_end_delim, str_think_beg_delim))
    {
        decode_think_beg_delim = true;

        string_remove_suffix(str_some_prompt_end_delim, str_think_beg_delim);
    }

    if(!decode_str(str_some_prompt_end_delim.c_str(), MT_TOK_TYPE_DELIM, s))
    {
        MT_LOG_ERR("Decoding some prompt end delimiter!\n");
        return false;
    }
    if(decode_think_beg_delim)
    {
        if(!decode_str(s.mt_p->think_beg_delim, MT_TOK_TYPE_THINK_BEGIN, s))
        {
            MT_LOG_ERR("Decoding thinking prompt begin delimiter after some prompt!\n");
            return false;
        }
    }
    if(decode_middle_newlines != 0)
    {
        std::string str_newline_buf = "";

        // We do not want to separate the newlines (sometimes multiple newlines
        // are getting merged into a single token):

        for(int i = 0; i < decode_middle_newlines; ++i)
        {
            str_newline_buf += str_newline;
        }

        if(!decode_str(
                str_newline_buf.c_str(),
                decode_think_beg_delim
                    ? MT_TOK_TYPE_THINK_TEXT
                    : MT_TOK_TYPE_DELIM,
                s))
        {
            MT_LOG_ERR("Decoding some prompt middle newlines!\n");
            return false;
        }
    }
    if(decode_think_end_delim)
    {
        if(!decode_str(s.mt_p->think_end_delim, MT_TOK_TYPE_THINK_END, s))
        {
            MT_LOG_ERR("Decoding thinking prompt end delimiter after some prompt!\n");
            return false;
        }
    }
    if(decode_trailing_newlines != 0)
    {
        std::string str_newline_buf = "";

        // We do not want to separate the newlines (sometimes multiple newlines
        // are getting merged into a single token):

        for(int i = 0; i < decode_trailing_newlines; ++i)
        {
            str_newline_buf += str_newline;
        }

        if(!decode_str(
                str_newline_buf.c_str(),
                decode_think_beg_delim && !decode_think_end_delim
                    ? MT_TOK_TYPE_THINK_TEXT
                    : MT_TOK_TYPE_DELIM,
                s))
        {
            MT_LOG_ERR("Decoding some prompt end delimiter's trailing newlines!\n");
            return false;
        }
    }

    return true;
}

/**
 * - Automatically detects optional "jumpstart" or disabling thinking/reasoning
 *   tokens at the end of the system prompt end delimiter.
 */
static bool decode_sys_prompt_end_delim(mt_llm_s& s)
{
    assert(s.mt_p->sys_prompt[0] != '\0');

    if(s.mt_p->think_beg_delim[0] == '\0')
    {
        // Simple case, where there are no thinking/reasoning delimiters.
        assert(s.mt_p->think_end_delim[0] == '\0');

        if(!decode_str(s.mt_p->sys_prompt_end_delim, MT_TOK_TYPE_DELIM, s))
        {
            MT_LOG_ERR("Decoding system prompt end delimiter (1)!\n");
            return false;
        }
        return true;
    }
    return decode_some_prompt_end_delim_with_thinking(
        s.mt_p->sys_prompt_end_delim, s);
}

static bool decode_initial_query(
    char const * const prompt,
    mt_llm_s& s,
    bool const skip_sys_prompt_end_delim)
{
    assert(s.mt_p->sys_prompt[0] != '\0');
    assert(prompt != nullptr && prompt[0] != '\0');

    if(!decode_str(s.mt_p->sys_prompt_beg_delim, MT_TOK_TYPE_DELIM, s))
    {
        MT_LOG_ERR("Decoding system prompt begin delimiter!\n");
        return false;
    }
    if(!decode_str(s.mt_p->sys_prompt, MT_TOK_TYPE_SYS_PROMPT, s))
    {
        MT_LOG_ERR("Decoding system prompt!\n");
        return false;
    }
    if(!decode_str(s.mt_p->sys_prompt_mid_delim, MT_TOK_TYPE_DELIM, s))
    {
        MT_LOG_ERR("Decoding system prompt middle delimiter!\n");
        return false;
    }
    if(!decode_str(prompt, MT_TOK_TYPE_PROMPT, s))
    {
        MT_LOG_ERR("Decoding prompt!\n");
        return false;
    }

    if(!skip_sys_prompt_end_delim)
    {
        if(!decode_sys_prompt_end_delim(s))
        {
            return false; // (called function logs on error)
        }
    }
    //
    // Otherwise: E.g. useful to decode system prompt and first part of a prompt
    //            one (e.g. a text to later query the LLM about). Then take a
    //            snapshot of that state. After that, it is possible to test
    //            multiple ends of that first prompt (would be the actual query
    //            about the text, in the example) faster, by recovering from the
    //            snapshot for each query-version instead of re-decoding the
    //            whole date (text) the LLM shall be questioned about.
    //
    //            Also see decode_prompt_and_sys_prompt_end_delim() & its usage.

    return true;
}

/**
 * - Also see comment in decode_initial_query() about skip_sys_prompt_end_delim.
 */
static bool decode_prompt_and_sys_prompt_end_delim(
    char const * const prompt, mt_llm_s& s)
{
    assert(s.mt_p->sys_prompt[0] != '\0');
    assert(prompt != nullptr && prompt[0] != '\0');

    if(!decode_str(prompt, MT_TOK_TYPE_PROMPT, s))
    {
        MT_LOG_ERR("Decoding prompt!\n");
        return false;
    }
    if(!decode_sys_prompt_end_delim(s))
    {
        return false; // (called function logs on error)
    }
    return true;
}

/**
 * - Automatically detects optional "jumpstart" or disabling thinking/reasoning
 *   tokens at the end of the prompt end delimiter.
 */
static bool decode_follow_up_query(char const * const prompt, mt_llm_s& s)
{
    assert(prompt != nullptr && prompt[0] != '\0');

    if(!decode_str(s.mt_p->prompt_beg_delim, MT_TOK_TYPE_DELIM, s))
    {
        MT_LOG_ERR("Decoding prompt begin delimiter!\n");
        return false;
    }
    if(!decode_str(prompt, MT_TOK_TYPE_PROMPT, s))
    {
        MT_LOG_ERR("Decoding prompt!\n");
        return false;
    }

    if(s.mt_p->think_beg_delim[0] == '\0')
    {
        // Simple case, where there are no thinking/reasoning delimiters.
        assert(s.mt_p->think_end_delim[0] == '\0');

        if(!decode_str(s.mt_p->prompt_end_delim, MT_TOK_TYPE_DELIM, s))
        {
            MT_LOG_ERR("Decoding prompt end delimiter (1)!\n");
            return false;
        }
        return true;
    }
    return decode_some_prompt_end_delim_with_thinking(
        s.mt_p->prompt_end_delim, s);
}

/** Return the EOG token's string representation (the EOG token that is used by
 *  the model after a response).
 */
static std::string get_response_tok_eog_str(mt_llm_s const & s)
{
    llama_vocab const * const vocab = llama_model_get_vocab(s.model->model);

    // TODO: On Android, for the following models, this should be the
    //       other way around, as it seems (try EOS first, then EOT):
    //       - EXAONE 3.0 7.8B Instruct
    //
    // See: https://github.com/ggerganov/llama.cpp/pull/8296

    llama_token const tok_eot = llama_vocab_eot(vocab);

    assert(tok_eot != -1 || llama_vocab_eos(vocab) != -1);
    
    llama_token const tok_eog = tok_eot == -1
        ? llama_vocab_eos(vocab)
        : tok_eot;

    return mt_llm_ctx_get_piece_from(*s.ctx, tok_eog);
}

/**
 * - To be used by inference().
 */
static void decode_irq_tokens(mt_llm_s& s)
{
    // Adding EOG token, only (e.g. using "..." plus EOG token can trigger that
    // the LLM also uses "..." in its responses).

    if(!decode_str(
            // Overdone token -> string -> token pipeline, but to be able to use
            // decode_str() function, only:
            get_response_tok_eog_str(s).c_str(), MT_TOK_TYPE_IRQ, s))
    {
        MT_LOG_ERR("Decoding IRQ EOG token!\n");
    }
}

/**
 * - To be called by inference().
 */
static int get_sampled_tok_type(
    mt_llm_s const & s, llama_token const new_tok_id, std::string const &piece)
{
    llama_vocab const * const vocab = llama_model_get_vocab(s.model->model);

    if(llama_vocab_is_eog(vocab, new_tok_id))
    {
        return MT_TOK_TYPE_SAMPLED_EOG;
    }
    if(llama_vocab_is_control(vocab, new_tok_id))
    {
        return MT_TOK_TYPE_SAMPLED_CONTROL_NON_EOG;
    }

    // Can the model used think/reason?
    bool const is_thinker = s.mt_p->think_beg_delim[0] != '\0';

    // TODO: BUG: This won't work, if the think begin and/or end
    //            delimiters consist of multiple tokens (see
    //            mt_llm_model.cpp, e.g s_gemma4.think_beg_delim!
    //
    // "Think" tokens don't seem to be ctrl. tokens.
    if(strncmp(
        piece.c_str(),
        s.mt_p->think_beg_delim,
        MT_LLM_P_LEN_THINK_BEG_DELIM) == 0)
    {
        return MT_TOK_TYPE_THINK_BEGIN;
    }
    if(strncmp(
        piece.c_str(),
        s.mt_p->think_end_delim,
        MT_LLM_P_LEN_THINK_END_DELIM) == 0)
    {
        return MT_TOK_TYPE_THINK_END;
    }

    // These are the ones to be visible to the end user
    // [although tokens with attribution "unknown" are also
    // included here, see llama_vocab.cpp, token_to_piece() in
    // comparance to llama_vocab_is_control() & ...THINK_BEGIN/...THINK_END]:

    // Also see decoding of system and "normal" prompt end delim.
    if(s.last_tok_type == MT_TOK_TYPE_THINK_BEGIN
        || s.last_tok_type == MT_TOK_TYPE_THINK_TEXT)
    {
        return MT_TOK_TYPE_THINK_TEXT; // Currently thinking/reasoning.
    }

    return MT_TOK_TYPE_SAMPLED_NON_EOG_NON_CONTROL;
}

/**
 * - To be called by inference().
 */
static std::vector<float> create_dig_probs(mt_llm_s const & s)
{
    std::vector<float> ret_val;
    float max = 0.0f;

    // TODO: Do just once during initialization:
    std::vector<std::vector<int>> const dig_toks =
        mt_llm_model_get_digit_tokens(*s.model);

    std::vector<float> const logits = get_last_logits(s, max);
    std::vector<float> const probs = get_probabilities(logits, max);

    ret_val = get_token_group_probabilities(dig_toks, probs);

    //{
    //    float prob_sum = 0.0f;
    //
    //    for(int i = 0; i < static_cast<int>(ret_val.size()); ++i)
    //    {
    //        MT_LOG("  %d: %6.2f%%\n", i, 100.0f * ret_val[i]);
    //
    //        prob_sum += ret_val[i];
    //    }
    //    MT_LOG("Sum: %6.2f%%\n", 100.0f * prob_sum);
    //}

    return ret_val;
}

static bool inference(mt_llm_s& s)
{
    // For performance measurement:
    int64_t const t_main_start = ggml_time_us();
    int const initial_tok_cnt = s.tok_cnt;

    bool irq = false; // Callback sets this to true, if interrupt is requested.
    std::vector<float> dig_probs;

    // Maximum count of tokens the context can hold.
    int const max_tok_cnt = static_cast<int>(llama_n_ctx(s.ctx));

    while(s.tok_cnt < max_tok_cnt)
    {
        bool cur_irq = false;
        std::vector<llama_token> tokens;

        // Space for at least one token is left in context, if getting here.

        if(irq // <- An interrupt of the inference was requested.
            // Only interrupt actual response to the user (also see below).
            && s.last_tok_type == MT_TOK_TYPE_SAMPLED_NON_EOG_NON_CONTROL)
        {
            decode_irq_tokens(s); // Called function logs on error.
            break;
        }

        llama_token const new_tok_id = llama_sampler_sample(
            s.sampler, s.ctx, -1);
        std::string const piece = mt_llm_ctx_get_piece_from(*s.ctx, new_tok_id);

        s.last_tok_type = get_sampled_tok_type(s, new_tok_id, piece);

        // Calculate probabilities of all digits for first sampled non-EOG,
        // non-control, non-whitespace, non-thinking, non-empty-piece token
        // (assumes that the sampling of all former whitespaces was "correct",
        // which is kind of wrong, but OK in practice):
        if(s.last_tok_type == MT_TOK_TYPE_SAMPLED_NON_EOG_NON_CONTROL
            && dig_probs.empty() // <=> No non-whitespace sampled, yet.
            && !piece.empty()
            && !is_whitespace_only(piece.c_str()))
        {
            dig_probs = create_dig_probs(s);
        }

        // (do NOT use the piece for this to make sure, that the exact same
        // token is added to the context)
        tokens.push_back(new_tok_id);
        if(!decode_tokens(
                tokens,
                s.last_tok_type,
                s,
                dig_probs,
                cur_irq))
        {
            MT_LOG_ERR("Decoding inferred token!\n");
            break;
        }
        irq = irq || cur_irq; // Don't overwrite alr. received IRQ (see above).

        // Break, if some kind of EOG token was generated.
        if (s.last_tok_type == MT_TOK_TYPE_SAMPLED_EOG)
        {
            break;
        }
    }

    {
        float const t_decode_seconds =
            static_cast<float>(ggml_time_us() - t_main_start) / 1000000.0f;
        int const n_decode = s.tok_cnt - initial_tok_cnt;

        MT_LOG(
            "Decoded %d tokens in %.2fs, speed: %.2f t/s.\n",
            n_decode,
            t_decode_seconds,
            static_cast<float>(n_decode) / t_decode_seconds);
    }

    return s.last_tok_type == MT_TOK_TYPE_SAMPLED_EOG;
}

/**
 * - To be called by mt_llm_init().
 * - Caller takes ownership.
 */
static llama_sampler* create_sampler(
    llama_vocab const * const vocab, mt_llm_p const &p)
{
    static size_t const min_keep = 0; // TODO: Is this the best option?
    static char const * const grammar_root = "root"; // TODO: Is this correct?

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();

    assert(sp.no_perf);

    llama_sampler * const ret_val = llama_sampler_chain_init(sp);

    if(p.grammar[0] != '\0') // <- Does not seem to be necessary.
    {
        // TODO: Test, if this actually works this way!

        llama_sampler_chain_add(
            ret_val,
            llama_sampler_init_grammar(vocab, p.grammar, grammar_root));
    }

    llama_sampler_chain_add(ret_val, llama_sampler_init_top_k(p.top_k));

    llama_sampler_chain_add(
        ret_val, llama_sampler_init_top_p(p.top_p, min_keep));

    llama_sampler_chain_add(
        ret_val, llama_sampler_init_min_p(p.min_p, min_keep));

    llama_sampler_chain_add(ret_val, llama_sampler_init_temp(p.temp));

    assert(LLAMA_DEFAULT_SEED == static_cast<uint32_t>(-1));
    llama_sampler_chain_add(ret_val, llama_sampler_init_dist(p.seed));

    return ret_val;
}

/**
 * - Returns an empty vector on error.
 * - Works for pooling type "rank", only.
 */
static std::vector<float> rerank_batch_decode(
    llama_batch& batch, mt_llm_s const & s)
{
    std::vector<float> ret_val;

    ret_val.clear(); // Necessary?

    clear_llama_memory(s.ctx);

    if(llama_decode(s.ctx, batch) != 0)
    {
        assert(ret_val.size() == 0);
        return ret_val;
    }

    for(int i = 0; i < batch.n_tokens; i++)
    {
        assert(batch.logits[i] != 0); // Must be configured this way.

        int const embd_pos = batch.seq_id[i][0];
        float const * const embd = llama_get_embeddings_seq(s.ctx, embd_pos);

        if(embd == nullptr)
        {
            assert(false); // Should not get here.
            ret_val.clear();
            return ret_val;
        }

        if(static_cast<int>(ret_val.size()) != embd_pos)
        {
            assert(ret_val.back() == embd[0]);
            continue;
        }
        ret_val.push_back(embd[0]);
    }
    return ret_val;
}

MT_EXPORT_LLM_API void __stdcall mt_llm_free(void * const ptr)
{
    free(ptr);
}

MT_EXPORT_LLM_API int mt_llm_get_token_count(
    char const * const text, bool const add_special, int const slot_index)
{
    if(slot_index != 0 && slot_index != 1)
    {
        MT_LOG_ERR("Invalid slot index given!\n");
        return -3;
    }
    if(s_slots[slot_index] == nullptr)
    {
        MT_LOG_ERR("Not intialized!\n");
        return -1;
    }

    mt_llm_s const &s = *s_slots[slot_index];

    if(text == nullptr)
    {
        MT_LOG_ERR("NULL given!\n");
        return -2;
    }
    if(s.mt_p->emb_or_rerank != 0)
    {
        MT_LOG_ERR("Configured for embeddings creation or reranking usage!\n");
        return -4;
    }

    std::vector<llama_token> const tokens = mt_llm_ctx_tokenize(
        *s.ctx, text, add_special);

    return static_cast<int>(tokens.size());
}

MT_EXPORT_LLM_API struct mt_llm_state * __stdcall mt_llm_state_create(
    int const slot_index)
{
    struct mt_llm_state * state = nullptr;

    if(slot_index != 0 && slot_index != 1)
    {
        MT_LOG_ERR("Invalid slot index given!\n");
        return nullptr;
    }
    if(s_slots[slot_index] == nullptr)
    {
        MT_LOG_ERR("Not intialized!\n");
        return nullptr;
    }

    mt_llm_s const &s = *s_slots[slot_index];

    assert(s.ctx != nullptr);

    size_t const state_size = llama_state_get_size(s.ctx);

    MT_LOG("Serialized state size would be: %zu bytes\n", state_size);

    state = static_cast<mt_llm_state*>(malloc(sizeof *state));
    if(state == nullptr)
    {
        MT_LOG_ERR("Failed to allocate state object!\n");
        return nullptr;
    }

    state->state = static_cast<uint8_t*>(malloc(state_size));
    if(state->state == nullptr)
    {
        MT_LOG_ERR("Failed to allocate %zu bytes!\n", state_size);
        free(state);
        state = nullptr;
        return nullptr;
    }

    size_t const written = llama_state_get_data(
        s.ctx, state->state, state_size);

    if(written != state_size)
    {
        MT_LOG_ERR("Failed to write all %zu bytes!\n", state_size);
        free(state->state);
        state->state = nullptr;
        free(state);
        state = nullptr;
        return nullptr;
    }

    state->size = state_size;
    state->last_tok_type = s.last_tok_type;
    state->tok_cnt = s.tok_cnt;
    MT_LOG("Successfully created %zu state bytes from LLM memory (tok. count: %d, last tok. type: %d, slot index: %d).\n", state->size, state->tok_cnt, state->last_tok_type, slot_index);
    return state; // Caller takes ownership!
}

MT_EXPORT_LLM_API bool __stdcall mt_llm_state_restore(
    struct mt_llm_state const * const state, int const slot_index)
{
    assert(state != nullptr);
    assert(state->state != nullptr);
    assert(0 < state->size);

    if(slot_index != 0 && slot_index != 1)
    {
        MT_LOG_ERR("Invalid slot index given!\n");
        return false;
    }
    if(s_slots[slot_index] == nullptr)
    {
        MT_LOG_ERR("Not intialized!\n");
        return false;
    }

    mt_llm_s &s = *s_slots[slot_index];

    assert(s.ctx != nullptr);
    
    size_t const read = llama_state_set_data(s.ctx, state->state, state->size);

    if(read != state->size)
    {
        MT_LOG_ERR("Failed to read exactly %zu bytes!\n", state->size);
        return false;
    }

    s.last_tok_type = state->last_tok_type;
    s.tok_cnt = state->tok_cnt;
    MT_LOG("Successfully restored %zu bytes from state to LLM memory (tok. count: %d, last tok. type: %d, slot index: %d).\n", state->size, state->tok_cnt, state->last_tok_type, slot_index);
    return true;
}

MT_EXPORT_LLM_API bool __stdcall mt_llm_decode_response(
    char const * const response, int const slot_index)
{
    if(response == nullptr)
    {
        MT_LOG_ERR("No response given (is NULL), doing nothing.\n");
        return false;
    }
    if(slot_index != 0 && slot_index != 1)
    {
        MT_LOG_ERR("Unsupported slot index given, doing nothing.\n");
        return false;
    }
    if(s_slots[slot_index] == nullptr)
    {
        MT_LOG_ERR("Not intialized!\n");
        return false;
    }

    mt_llm_s& s = *s_slots[slot_index];

    if(s.mt_p->emb_or_rerank != 0)
    {
        MT_LOG_ERR("Configured for embeddings creation or reranking usage!\n");
        return false;
    }
    assert(s.mt_p != nullptr);
    assert(s.model != nullptr);
    assert(s.ctx != nullptr);
    assert(s.sampler != nullptr); // Alth. not needed, here..

    if(!decode_str(
            response,
            // Not really sampled here..
            MT_TOK_TYPE_SAMPLED_NON_EOG_NON_CONTROL,
            s))
    {
        MT_LOG_ERR("Decoding (given) response failed!\n");
        return false;
    }

    if(!decode_str(
            get_response_tok_eog_str(s).c_str(),
            MT_TOK_TYPE_SAMPLED_EOG, // Not really sampled here..
            s))
    {
        MT_LOG_ERR("Decoding (given) response's EOG token failed!\n");
        return false;
    }

    return true;
}

MT_EXPORT_LLM_API bool __stdcall mt_llm_query(
    char const * const prompt,
    int const slot_index,
    bool const skip_sys_prompt_end_delim_and_inference,
    bool const follow_up_decode_prompt_and_sys_prompt_end_delim)
{
    if(slot_index != 0 && slot_index != 1)
    {
        MT_LOG("Error: Unsupported slot index given, doing nothing.\n");
        return false;
    }
    if(s_slots[slot_index] == nullptr)
    {
        MT_LOG_ERR("Not intialized!\n");
        return false;
    }

    mt_llm_s& s = *s_slots[slot_index];

    if(s.mt_p->emb_or_rerank != 0)
    {
        MT_LOG_ERR("Configured for embeddings creation or reranking usage!\n");
        return false;
    }

    assert(s.mt_p != nullptr);
    assert(s.model != nullptr);
    assert(s.ctx != nullptr);
    assert(s.sampler != nullptr);
    
    if(s.tok_cnt == 0 && s.mt_p->sys_prompt[0] != '\0')
    {
        // Although it would be no problem, just not intended this way.
        assert(!follow_up_decode_prompt_and_sys_prompt_end_delim);

        if(!decode_initial_query(
                prompt, s, skip_sys_prompt_end_delim_and_inference))
        {
            return false; // (called function logs on error)
        }
    }
    else
    {
        // Although it would be no problem, just not intended this way.
        assert(!skip_sys_prompt_end_delim_and_inference);

        if(follow_up_decode_prompt_and_sys_prompt_end_delim)
        {
            if(!decode_prompt_and_sys_prompt_end_delim(prompt, s))
            {
                return false; // (called function logs on error)
            }
        }
        else
        {
            if(!decode_follow_up_query(prompt, s))
            {
                return false; // (called function logs on error)
            }
        }
    }

    if(!skip_sys_prompt_end_delim_and_inference)
    {
        if(!inference(s))
        {
            return false; // (called function logs on error)
        }
    }

    MT_LOG("Token count: %d.\n", s.tok_cnt);
    return true;
}

MT_EXPORT_LLM_API float* __stdcall mt_llm_create_embeddings(
    char const * const prompt, int const slot_index, int * const out_count)
{
    //llama-embedding.exe -m snowflake-arctic-embed-s-q8_0.gguf -p "Ich habe einen Kumpel, der ist Maurer." --pooling cls --verbose-prompt --no-warmup --embd-normalize 2 -ngl 0

    std::vector<llama_token> inp;
    llama_batch batch;

    // *************************************************************************
    // *** Check input arguments and (context) configuration:                ***
    // *************************************************************************

    assert(out_count != nullptr);

    if(prompt == nullptr
        || (strnlen(prompt, 65535) == 65535)) // <- Hard-coded limit.
    {
        MT_LOG_ERR("Prompt is not given or it is no or a too long C-string!\n");
        return nullptr;
    }

    if(slot_index != 0 && slot_index != 1)
    {
        MT_LOG_ERR("Unsupported slot index given, doing nothing.\n");
        return nullptr;
    }
    if(s_slots[slot_index] == nullptr)
    {
        MT_LOG_ERR("Not intialized!\n");
        return nullptr;
    }

    mt_llm_s const &s = *s_slots[slot_index];

    if(s.mt_p->emb_or_rerank != 1) // Hard-coded
    {
        MT_LOG_ERR("NOT configured for embeddings creation!\n");
        return nullptr;
    }

    assert(s.mt_p != nullptr);
    assert(s.model != nullptr);
    assert(s.ctx != nullptr);
    //assert(s.sampler != nullptr); // Does not matter.

    // No support for other pooling types (e.g. for a reranking model), here:
    assert( // See mt_llm_ctx_create().
        llama_pooling_type(s.ctx) == LLAMA_POOLING_TYPE_CLS
            || llama_pooling_type(s.ctx) == LLAMA_POOLING_TYPE_MEAN
            || llama_pooling_type(s.ctx) == LLAMA_POOLING_TYPE_LAST);

    uint32_t const n_ctx = llama_n_ctx(s.ctx);

    // *************************************************************************
    // *** Clear the memory (K/V cache):                                     ***
    // *************************************************************************

    clear_llama_memory(s.ctx);

    // *************************************************************************
    // *** Get token representation of given prompt/text:                    ***
    // *************************************************************************

    //MT_LOG("Prompt: \"%s\"\n", prompt);

    inp = common_tokenize(
            s.ctx,
            prompt, // <- Implicit conversion.
            true, // Add specials.
            true); // Parse specials.
    if(inp.size() == 0)
    {
        MT_LOG_ERR("No tokens generated from given prompt!\n");
        return nullptr;
    }

    const llama_vocab * const vocab = llama_model_get_vocab(s.model->model);

    if(inp.back() != llama_vocab_sep(vocab)
        && inp.back() != llama_vocab_eos(vocab))
    {
        MT_LOG("Warning: Last token in the prompt is neither SEP nor EOS.\n");

        // tokenizer.ggml.add_eos_token should be set to true in the GGUF
        // header.
    }

    //MT_LOG("Number of tokens in prompt: %zu\n", inp.size());

#ifndef NDEBUG
    for(int i = 0; i < static_cast<int>(inp.size()); ++i)
    {
        MT_LOG(
            "%6d => \"%s\"\n",
            inp[i],
            common_token_to_piece(s.ctx, inp[i]).c_str());
    }
#endif //NDEBUG

    if(n_ctx < inp.size())
    {
        MT_LOG_ERR(
            "Input prompt token count is %zu, but context length is smaller (%u)!\n",
            inp.size(),
            n_ctx);
        return nullptr;
    }

    // *************************************************************************
    // *** Fill model's decoder with the generated tokens:                   ***
    // *************************************************************************

    batch = llama_batch_init(
        static_cast<int32_t>(inp.size()),

        // No embeddings, because these would be INPUT TOKEN embeddings to
        // use instead of creating them from actual input tokens (to speed
        // up the process):
        0,

        1); // Single sequence.

    for(int i = 0; i < static_cast<int>(inp.size()); ++i)
    {
        common_batch_add(
            batch,
            inp[i],
            i,
            { 0 }, // Single sequence with ID 0.

            // It seems to be expected/necessary to request embeddings for ALL
            // tokens, not just the last one (because of NON-causal model
            // support?):
            true);
    }

    int32_t const decode_result = llama_decode(s.ctx, batch);

    llama_batch_free(batch);

    if(decode_result != 0)
    {
        MT_LOG_ERR("Failed to decode!\n");
        
        return nullptr;
    }

    // *************************************************************************
    // *** Get embedding vector:                                             ***
    // *************************************************************************

    float const * const embd = llama_get_embeddings_seq(
        s.ctx, 0); // (0 is the sole sequence ID used)

    if(embd == nullptr)
    {
        MT_LOG_ERR("Failed to get sequence embeddings!\n");
        return nullptr;
    }

    // *************************************************************************
    // *** Create (Euclidean) normalized copy of the embedding vector:       ***
    // *************************************************************************

    int const n_embd = llama_model_n_embd(s.model->model);

    assert(0 < n_embd);

    float * const emb = static_cast<float*>(malloc(n_embd * sizeof *emb));

    assert(emb != nullptr);

    common_embd_normalize(
        embd,
        emb,
        n_embd,
        2); // <- Euclidean normalization.

    // *************************************************************************
    // *** Free some memory, set/return result:                              ***
    // *************************************************************************

    *out_count = n_embd;
    return emb; // Caller takes ownership.
}

MT_EXPORT_LLM_API float* __stdcall mt_llm_rerank(
    char const * const query,
    char const * const * documents,
    int const doc_count,
    int const slot_index)
{
    //llama-embedding.exe -m E:\Rer\bge-reranker-v2-m3-q5_k_m.gguf -p "What is a whale?\tButterflies eat flowers.\nWhat is a whale?\tA large mammal that lives in water.\nWhat is a whale?\tDogs want food." --pooling rank --verbose-prompt --embd-normalize -1

    std::vector<std::vector<llama_token>> inputs;
    llama_batch batch;
    int sb = 0;
    std::vector<float> scores;

    // *************************************************************************
    // *** Check input arguments and (context) configuration:                ***
    // *************************************************************************

    if(query == nullptr
        || (strnlen(query, 65535) == 65535)) // <- Hard-coded limit.
    {
        MT_LOG_ERR("Query is not given or it is no or a too long C-string!\n");
        return nullptr;
    }

    if(documents == nullptr)
    {
        MT_LOG_ERR("No documents given!\n");
        return nullptr;
    }
    if(doc_count < 1)
    {
        MT_LOG_ERR("Invalid document count given (0 or lower)!\n");
        return nullptr;
    }
    for(int i = 0; i < doc_count; ++i)
    {
        if(documents[i] == nullptr)
        {
            MT_LOG_ERR("Document at index %d is \"unset\" (null)!\n", i);
            return nullptr;
        }
        if(strnlen(documents[i], 65535) == 65535) // <- Hard-coded limit.
        {
            MT_LOG_ERR("Document at index %d is a too long C-string!\n", i);
            return nullptr;
        }
    }

    if(slot_index != 0 && slot_index != 1)
    {
        MT_LOG_ERR("Unsupported slot index given, doing nothing.\n");
        return nullptr;
    }
    if(s_slots[slot_index] == nullptr)
    {
        MT_LOG_ERR("Not intialized!\n");
        return nullptr;
    }

    mt_llm_s const &s = *s_slots[slot_index];

    if(s.mt_p->emb_or_rerank != 2) // Hard-coded
    {
        MT_LOG_ERR("NOT configured for reranking usage!\n");
        return nullptr;
    }

    assert(s.mt_p != nullptr);
    assert(s.model != nullptr);
    assert(s.ctx != nullptr);
    //assert(s.sampler != nullptr); // Does not matter.

    // No support for other pooling types, here:
    assert( // See mt_llm_ctx_create().
        llama_pooling_type(s.ctx) == LLAMA_POOLING_TYPE_RANK);

    if(llama_model_chat_template(s.model->model, "rerank") != nullptr)
    {
        // See llama.cpp's embedding.cpp example for implementation.
        MT_LOG_ERR("Reranking model has a chat template, this is currently not supported!\n");
        return nullptr;
    }

    if(llama_model_n_cls_out(s.model->model) != 1)
    {
        // See llama.cpp's embedding.cpp example for implementation.
        MT_LOG_ERR("Reranking model has more than one classifier output, this is currently not supported!\n");
        return nullptr;
    }

    uint32_t const n_ctx = llama_n_ctx(s.ctx);

    uint32_t const n_batch = llama_n_batch(s.ctx);

    // *************************************************************************
    // *** Get token representation of given query and documents:            ***
    // *************************************************************************

    //MT_LOG("Query: \"%s\", document count: %d\n", query, doc_count);

    const llama_vocab * const vocab = llama_model_get_vocab(s.model->model);

    // Get add-SEP and add-EOS token, if there are any:
    std::string const add_sep_tok_str =
        llama_vocab_get_add_sep(vocab)
            ? llama_vocab_get_text(vocab, llama_vocab_sep(vocab))
            : "";
    std::string const add_eos_tok_str =
        llama_vocab_get_add_eos(vocab)
            ? llama_vocab_get_text(vocab, llama_vocab_eos(vocab))
            : "";

    // For warnings, below:
    llama_token const sep_tok = llama_vocab_sep(vocab);
    llama_token const eos_tok = llama_vocab_eos(vocab);

    // Currently works for reranking models without a chat template, only
    // (see check, above).

    for(int i = 0; i < doc_count; ++i)
    {
        std::vector<llama_token> inp;
        std::string prompt;

        prompt += query; // Implicit conversion.
        if(!add_eos_tok_str.empty())
        {
            prompt += add_eos_tok_str;
        }
        if(!add_sep_tok_str.empty())
        {
            prompt += add_sep_tok_str;
        }
        prompt += documents[i]; // Implicit conversion.

        inp = common_tokenize(
            s.ctx,
            prompt,
            true, // Add specials.
            true); // Parse specials.
        if(inp.size() == 0)
        {
            MT_LOG_ERR("No tokens generated from prompt at index %d!\n", i);
            return nullptr;
        }
        if(n_batch < inp.size())
        {
            MT_LOG_ERR(
                "Token count of prompt at index %d exceeds max. token count per batch, which is %d (increase and re-run?)\n",
                i,
                static_cast<int>(n_batch));
            return nullptr;
        }

        //MT_LOG("Number of tokens in prompt at index &d: %zu\n", i, inp.size());
#ifndef NDEBUG
        for(int j = 0; j < static_cast<int>(inp.size()); ++j)
        {
            MT_LOG(
                "%6d => \"%s\"\n",
                inp[j],
                common_token_to_piece(s.ctx, inp[j]).c_str());
        }
#endif //NDEBUG

        if(inp.back() != sep_tok && inp.back() != eos_tok)
        {
            MT_LOG(
                "Warning: Last token in the prompt at index %d is neither SEP nor EOS.\n",
                i);

            // tokenizer.ggml.add_eos_token should be set to true in the GGUF
            // header.
        }

        inputs.push_back(inp);
    }

    // *************************************************************************
    // *** Get scores for each query/document pair via batch processing:     ***
    // *************************************************************************

    batch = llama_batch_init(
        n_batch, // Using the maximum number of tokens allowed per batch.
        0, // No (input) embeddings.
        1); // Single sequence per token.

    int const n_seq_max = static_cast<int>(llama_max_parallel_sequences());

    scores.clear(); // Necessary?
    sb = 0; // Number of prompts in current batch.
    for(int k = 0; k < static_cast<int>(inputs.size()); ++k)
    {
        std::vector<llama_token>& inp = inputs[k];
        uint64_t const n_toks = static_cast<uint64_t>(inp.size());

        if(n_batch < batch.n_tokens + n_toks || n_seq_max <= sb)
        {
            // No more capacity left, decode & add remember current scores:

            std::vector<float> const cur_scores = rerank_batch_decode(batch, s);

            if(cur_scores.empty())
            {
                // Indicates an error.
                MT_LOG_ERR(
                    "Batch decode for reranking (index %d) failed (1)!\n",
                    k);
                llama_batch_free(batch);
                return nullptr;
            }
            for(int i = 0; i < cur_scores.size(); ++i)
            {
                scores.push_back(cur_scores[i]);
            }

            sb = 0;
            common_batch_clear(batch);
        }
            
        // Add to batch:
        for(int i = 0; i < static_cast<int>(inp.size()); ++i)
        {
            common_batch_add(batch, inp[i], i, { sb }, true);
        }

        sb += 1;
    }

    std::vector<float> const final_scores = rerank_batch_decode(batch, s);

    llama_batch_free(batch);

    if(final_scores.empty())
    {
        // Indicates an error.
        MT_LOG_ERR("Final batch decode for reranking failed!\n");
        return nullptr;
    }
    for(int i = 0; i < static_cast<int>(final_scores.size()); ++i)
    {
        scores.push_back(final_scores[i]);
    }

    assert(doc_count == static_cast<int>(inputs.size()));
    assert(scores.size() == inputs.size());
    
    // *************************************************************************
    // *** Copy scores from vectors to heap array:                           ***
    // *************************************************************************

    float * const raw_scores =
        static_cast<float*>(malloc(scores.size() * sizeof *raw_scores));

    assert(raw_scores != nullptr);

    for(int i = 0; i < static_cast<int>(scores.size()); ++i)
    {
        raw_scores[i] = scores[i];
    }

    return raw_scores; // Caller takes ownership.
}

MT_EXPORT_LLM_API void __stdcall mt_llm_reset(
    char const * const sys_prompt, int const slot_index)
{
    if(slot_index != 0 && slot_index != 1)
    {
        MT_LOG("Warning: Unsupported slot index given, doing nothing.");
        return; // Just do nothing.
    }

    if(s_slots[slot_index] == nullptr)
    {
        return; // Cannot do anything.
    }

    mt_llm_s& s = *s_slots[slot_index];

    assert(s.mt_p != nullptr);
    assert(s.model != nullptr);
    assert(s.ctx != nullptr);
    assert(s.sampler != nullptr);

    clear_llama_memory(s.ctx);

    llama_sampler_reset(s.sampler);

    s.last_tok_type = 0;
    s.tok_cnt = 0;

    if(sys_prompt != NULL)
    {
        strncpy(s.mt_p->sys_prompt, sys_prompt, MT_LLM_P_LEN_SYS_PROMPT - 1);
        s.mt_p->sys_prompt[MT_LLM_P_LEN_SYS_PROMPT - 1] = '\0';

        MT_LOG("sys_prompt" ": " "\"%s\"" "\n", s.mt_p->sys_prompt);
    }
}

MT_EXPORT_LLM_API void __stdcall mt_llm_deinit(int const slot_index)
{
    mt_llm_s* s = nullptr;

    if(slot_index != 0 && slot_index != 1)
    {
        MT_LOG("Warning: Unsupported slot index given, doing nothing.");
        return; // Just do nothing.
    }

    s = s_slots[slot_index];

    if(s == nullptr)
    {
        return; // Just do nothing.
    }

    if(s->mt_p != nullptr)
    {
        mt_llm_p_free(s->mt_p);
        s->mt_p = nullptr;
    }
    if(s->ctx != nullptr)
    {
        llama_free(s->ctx);
        s->ctx = nullptr;
    }
    if(s->sampler != nullptr)
    {
        llama_sampler_free(s->sampler);
        s->sampler = nullptr;
    }
    if(s->model != nullptr)
    {
        mt_llm_model_free(s->model);
        s->model = nullptr;
    }

    if(s_slots[0] == nullptr
        && s_slots[1] == nullptr
        && s_is_backend_init)
    {
        llama_backend_free();
        s_is_backend_init = false;
    }

    free(s);
    s = nullptr;
    s_slots[slot_index] = nullptr;
}

MT_EXPORT_LLM_API bool __stdcall mt_llm_reinit(
    struct mt_llm_p const * const mt_p, int const slot_index)
{
    if(slot_index != 0 && slot_index != 1)
    {
        MT_LOG_ERR("Unsupported slot index given!\n");
        return false;
    }

    init_common_if_necessary(mt_p->enable_llama_cpp_log != 0);

    if(s_slots[slot_index] != nullptr)
    {
        mt_llm_deinit(slot_index);
    }
    assert(s_slots[slot_index] == nullptr);

    if(mt_p->callback == nullptr && mt_p->emb_or_rerank == 0)
    {
        MT_LOG_ERR("Callback is not set (only allowed for embedding vector generation or reranking usage)!\n");
        //mt_llm_deinit(slot_index);
        return false;
    }

    s_slots[slot_index] = static_cast<struct mt_llm_s *>(
        malloc(sizeof *s_slots[slot_index]));
    if(s_slots[slot_index] == nullptr)
    {
        MT_LOG_ERR("Failed to allocate memory for settings!\n");
        //mt_llm_deinit(slot_index);
        return false;
    }

    mt_llm_s& s = *s_slots[slot_index];

    s.mt_p = mt_llm_p_create_copy(*mt_p);
    if(s.mt_p == nullptr)
    {
        MT_LOG_ERR("Failed to deep-copy parameters!\n");
        mt_llm_deinit(slot_index);
        return false;
    }
    //
    // Do not use mt_p from here on!

    // If not given, automatically set the thread count:
    if(s.mt_p->threads == 0)
    {
        s.mt_p->threads = (uint32_t)common_cpu_get_num_physical_cores();
        assert(0 < s.mt_p->threads);
    }

    // Do not change s_slots[slot_index]->mt_p properties from here on, with
    // the exception of prompts (see below)!

    // Initialize the inference engine:
    init_backend_if_necessary();

    // TODO: If the other slot is already initialized and all model-related
    //       parameters are equal, we can(?) reuse the already initialized
    //       model (not the context!), here!
    // 
    // Initialize the model:
    s.model = mt_llm_model_create(*s.mt_p);
    if (s.model == nullptr)
    {
        MT_LOG_ERR("Unable to load model!\n");
        mt_llm_deinit(slot_index);
        return false;
    }

    // Initialize the sampling (unnecessary for embeddings creation and
    // reranking usage):
    s.sampler = create_sampler(llama_model_get_vocab(s.model->model), *s.mt_p);
    if(s.sampler == nullptr)
    {
        MT_LOG_ERR("Unable to create sampler!\n");
        mt_llm_deinit(slot_index);
        return false;
    }

    // Support for models with an encoder is currently not implemented (for
    // embeddings and reranking, models with encoder AND decoder are not
    // supported, anyway):
    if(llama_model_has_encoder(s.model->model))
    {
        MT_LOG_ERR("Model has an encoder, that is currently not supported!\n");
        mt_llm_deinit(slot_index);
        return false;
    }

    // Modify prompt strings by model (name), if wanted (maybe unnecessary for
    // embeddings creation and reranking usage):
    if(s.mt_p->try_prompts_by_model != 0)
    {
        mt_llm_model_try_set_prompts(*s.model, *s.mt_p);
        // Return value ignored, as called function logs (and this is no error).
    }
    else
    {
        MT_LOG("Trying to use default prompts for model is not wanted.\n");
    }
    
    // Do not change any s_slots[slot_index]->mt_p properties from here on!
    mt_llm_p_print(*s.mt_p);

    // Initialize the context:

    s.ctx = mt_llm_ctx_create(*s.mt_p, *s.model);
    if (s.ctx == nullptr)
    {
        MT_LOG_ERR("Creating context!\n");
        mt_llm_deinit(slot_index);
        return false;
    }

    {
        int32_t const n_ctx_train = llama_model_n_ctx_train(s.model->model),
            n_ctx_ctx = static_cast<int32_t>(llama_n_ctx(s.ctx)); // Bold cast?

        if(n_ctx_train < n_ctx_ctx)
        {
            // For non-embedding, non-reranking, interpreted as error here (by
            // definition):
            if(s.mt_p->emb_or_rerank == 0)
            {
                assert(
                    s.mt_p->n_ctx == 0
                        || static_cast<int32_t>(s.mt_p->n_ctx) == n_ctx_ctx);

                MT_LOG_ERR(
                    "Model was trained on %d tokens (wanted %d tokens)!\n",
                    n_ctx_train,
                    n_ctx_ctx);
                mt_llm_deinit(slot_index);
                return false;
            }

            // For embedding and reranking, just issue a warning:
            MT_LOG(
                "Warning: Model was trained on %d tokens (using %d tokens).\n",
                n_ctx_train,
                n_ctx_ctx);
        }
    }

    MT_LOG("System info: %s\n", llama_print_system_info());

    s.last_tok_type = 0;
    s.tok_cnt = 0;

    if(llama_model_chat_template(s.model->model, nullptr) != nullptr)
    {
        bool const use_jinja = true;
        std::string chat_template; // Empty
        std::map<std::string, std::string> template_kwargs; // Empty

        common_chat_templates_ptr const chat_templates =
            common_chat_templates_init(s.model->model, chat_template);

        std::string const chat_format_example = common_chat_format_example(
            chat_templates.get(), use_jinja, template_kwargs);

        assert(use_jinja);
        MT_LOG(
            "Example using model's default chat template (using Jinja): \"%s\"\n",
            chat_format_example.c_str());
    }
    else
    {
        MT_LOG("No default chat template found in model.\n");
    }

    return true;
}
