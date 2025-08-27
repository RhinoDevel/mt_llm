
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

#include "mt_llm.h"
#include "mt_llm_p.h"
#include "mt_llm_model.h"
#include "mt_llm_ctx.h"
#include "mt_llm_s.h"
#include "mt_llm_log.h"
#include "mt_llm_state.h"

#include "mt_llm_tok_type.h"

static struct mt_llm_s * s = nullptr;

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
static std::vector<float> get_last_logits(float& max)
{
    assert(s != nullptr);
    assert(s->model != nullptr);

    std::vector<float> ret_val;

    float const * const logits = llama_get_logits_ith(s->ctx, -1);

    if(logits == nullptr)
    {
        assert(ret_val.empty());
        return ret_val;
    }

    int32_t const n_vocab = llama_vocab_n_tokens(
        llama_model_get_vocab(s->model));

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
    std::vector<float> const & dig_probs)
{
    assert(s != nullptr);
    assert(0 < s->last_tok_type);

    if(piece.empty()) // Token is omitted by llama.cpp => Also omit here.
    {
        return false; // <=> No interruption.
    }

    return s->mt_p->callback(
        static_cast<int>(tok),
        piece.c_str(),
        s->last_tok_type,
        dig_probs.empty() ? nullptr : dig_probs.data());
}

/** Add token representation of given string to context. Let the callback know
 *  that the tokens of this string are of given type. Increase overall token
 *  count.
 * 
 * - "Decode" as in using the decoder of the LLM architecture to add to its
 *   context.
 */
static bool decode(char const * const str, int const tok_type)
{
    s->last_tok_type = tok_type;

    int const str_tok_cnt = mt_llm_ctx_decode(
            *s->ctx,
            *s->sampler,
            s->tok_cnt,
            str,
            callback_handler);

    if(str_tok_cnt < 0)
    {
        MT_LOG_ERR("Decoding!\n");
        return false;
    }
    s->tok_cnt += str_tok_cnt;
    return true;
}

/**
 * - Just assumes that the context length is always long enough to hold the
 *   prompt to be decoded, here (no check..).
 */
static bool decode_initial_query(char const * const prompt)
{
    assert(s->mt_p->sys_prompt[0] != '\0');
    assert(prompt != nullptr && prompt[0] != '\0');

    if(!decode(s->mt_p->sys_prompt_beg_delim, MT_TOK_TYPE_DELIM))
    {
        MT_LOG_ERR("Decoding system prompt begin delimiter!");
        return false;
    }
    if(!decode(s->mt_p->sys_prompt, MT_TOK_TYPE_SYS_PROMPT))
    {
        MT_LOG_ERR("Decoding system prompt!");
        return false;
    }
    if(!decode(s->mt_p->sys_prompt_mid_delim, MT_TOK_TYPE_DELIM))
    {
        MT_LOG_ERR("Decoding system prompt middle delimiter!");
        return false;
    }
    if(!decode(prompt, MT_TOK_TYPE_PROMPT))
    {
        MT_LOG_ERR("Decoding prompt!");
        return false;
    }
    if(!decode(s->mt_p->sys_prompt_end_delim, MT_TOK_TYPE_DELIM))
    {
        MT_LOG_ERR("Decoding system prompt end delimiter!");
        return false;
    }
    return true;
}

/**
 * - Just assumes that the context length is always long enough to hold the
 *   prompt to be decoded, here (no check..).
 */
static bool decode_follow_up_query(char const * const prompt)
{
    assert(prompt != nullptr && prompt[0] != '\0');

    if(!decode(s->mt_p->prompt_beg_delim, MT_TOK_TYPE_DELIM))
    {
        MT_LOG_ERR("Decoding prompt begin delimiter!");
        return false;
    }
    if(!decode(prompt, MT_TOK_TYPE_PROMPT))
    {
        MT_LOG_ERR("Decoding prompt!");
        return false;
    }
    if(!decode(s->mt_p->prompt_end_delim, MT_TOK_TYPE_DELIM))
    {
        MT_LOG_ERR("Decoding prompt end delimiter!");
        return false;
    }
    return true;
}

static bool inference()
{
    // TODO: Improve to avoid sending reverse prompt twice by always waiting for
    //       the count of tokens the reverse prompt has before calling the
    //       callback (with the exception if EOG)!

    assert(s != nullptr);

    llama_batch batch;
    int n_cur = 0,
        
        // If reverse prompt is given, this will hold the index in the
        // last_chars "ring buffer" where the last character was added.
        last_chars_index = -1;

    bool irq = false,
        is_thinking = false,

        // Set to true, if reverse prompt is given and the count of characters
        // of the reverse prompt was added to the last_chars "ring buffer".
        last_chars_filled = false;

    // To hold the token sequence to be used, if callback requests an interrupt
    // (e.g., if the user wants to interrupt the "chatbot"). Will hold the
    // reverse prompt tokens, if a reverse prompt is given.
    std::vector<int> irq_tokens;

    // Used, if reverse prompt is given. Will be able to hold the count of
    // characters necessary for the reverse prompt (w/o \0), then.
    char* last_chars = nullptr;

    int const rev_prompt_len = static_cast<int>(strlen(s->mt_p->rev_prompt));
    llama_vocab const * const vocab = llama_model_get_vocab(s->model);

    std::vector<float> dig_probs;

    // Prepare irq_tokens. Also prepare "ring buffer" that holds the last
    // characters received, if reverse prompt is given / to be used:
    //
    // At least, if SPM vocabulary is used and to-be-tokenized string is not
    // empty, the tokenizer may adds a space character as prefix before the
    // created tokens.
    // Since the interrupt can happen at each position of the LLM's response,
    // that should not be a problem, here.
    //
    if(rev_prompt_len == 0) // Use magic (or empty) str. & EOT (or EOS), only.
    {
        irq_tokens = mt_llm_ctx_tokenize(
            *s->ctx,
            "", // E.g. "..." can cause an LLM to also use "..." just "for fun"!
            false); // No adding of BOS and/or EOS [is both model-dependent].

        // TODO: On Android, for the following models, this should be the other
        //       way around, as it seems (try EOS first, then EOT):
        //       - EXAONE 3.0 7.8B Instruct
        //
        // See: https://github.com/ggerganov/llama.cpp/pull/8296
        //
        llama_token const tok_eot = llama_vocab_eot(vocab);
        //
        assert(tok_eot != -1 || llama_vocab_eos(vocab) != -1);
        irq_tokens.push_back(tok_eot == -1 ? llama_vocab_eos(vocab) : tok_eot);
    }
    else // Use reverse prompt given.
    {
        assert(last_chars_index == -1);
        assert(!last_chars_filled);

        irq_tokens = mt_llm_ctx_tokenize(
            *s->ctx,
            s->mt_p->rev_prompt,
            false); // No adding of BOS and/or EOS [is both model-dependent].

        last_chars = static_cast<char*>(
            malloc(rev_prompt_len * sizeof *last_chars));
        assert(last_chars != nullptr);
    }

    int64_t const t_main_start = ggml_time_us();

    batch = llama_batch_init(1, 0, 1); // Needs to be freed!

    // E.g.:
    //
    // Existing token count: 30 <=> Indices  0...29 => First new token index: 30
    //
    int const first_new_tok_index = s->tok_cnt;
    int const n_ctx = static_cast<int>(llama_n_ctx(s->ctx));

    bool const is_thinker = s->mt_p->think_beg_delim[0] != '\0';

    for(n_cur = first_new_tok_index; n_cur < n_ctx; ++n_cur)
    {
        // Break, if (optional) reverse prompt was sampled last:
        if(last_chars_filled)
        {
            assert(0 <= last_chars_index && last_chars_index < rev_prompt_len);

            //  0123456
            // "Master:"
            // <=> rev_prompt_len = 7

            bool rev_prompt_found = true; // TRUE by default.
            
            for(int i = rev_prompt_len - 1; 0 <= i; --i)
            {
                int const cur_last_chars_index =
                        (i + last_chars_index + 1) % rev_prompt_len;

                assert(
                    0 <= cur_last_chars_index
                        && cur_last_chars_index < rev_prompt_len);

                if(s->mt_p->rev_prompt[i] != last_chars[cur_last_chars_index])
                {
                    rev_prompt_found = false;
                    break;
                }
            }

            if(rev_prompt_found)
            {
                s->last_tok_type = MT_TOK_TYPE_REV_PROMPT;

                // "Hack":
                //
                // The reverse prompt got already sampled and send to the
                // client code as non-reverse-prompt token type and now gets
                // send the second time AS reverse-prompt token type.
                //
                // The client code must be aware of this and take action!
                //
                s->mt_p->callback(
                    0,
                    s->mt_p->rev_prompt, // TODO: What about a possible leading space (would be "missing" here)? Problem??
                    s->last_tok_type,
                    nullptr);
                break;
            }
        }

        if(irq)
        {
            s->last_tok_type = MT_TOK_TYPE_IRQ;

            if(!mt_llm_ctx_decode(
                    *s->ctx,
                    *s->sampler,
                    n_cur,
                    irq_tokens,
                    callback_handler))
            {
                MT_LOG_ERR("Decoding IRQ tokens!\n");
                llama_batch_free(batch);
                free(last_chars);
                last_chars = nullptr;
                return false;
            }
            n_cur += static_cast<int>(irq_tokens.size()); // TODO: NOT caring about maximum count..!
            break;
        }

        llama_token const new_tok_id = llama_sampler_sample(
            s->sampler, s->ctx, -1);

        bool const new_tok_is_eog = llama_vocab_is_eog(vocab, new_tok_id);

        std::string const piece = mt_llm_ctx_get_piece_from(
            *s->ctx, new_tok_id);

        if(is_thinker && !is_thinking)
        {
            if(strncmp(
                piece.c_str(),
                s->mt_p->think_beg_delim,
                MT_LLM_P_LEN_THINK_BEG_DELIM) == 0)
            {
                is_thinking = true; // BEFORE calling callback.
            }
        }
        //
        // Otherwise: The model is not a thinker.

        if(new_tok_is_eog)
        {
            s->last_tok_type = MT_TOK_TYPE_SAMPLED_EOG; // (causes stop, below)
        }
        else
        {
            if(llama_vocab_is_control(vocab, new_tok_id))
            {
                s->last_tok_type = MT_TOK_TYPE_SAMPLED_CONTROL_NON_EOG;
            }
            else
            {
                // These are the ones to be visible to the end user
                // [although tokens with attribution "unknown" are also
                // included here, see llama_vocab.cpp, token_to_piece() in
                // comparance to llama_vocab_is_control()]:

                if(is_thinking)
                {
                    s->last_tok_type = MT_TOK_TYPE_SAMPLED_THINK;
                }
                else
                {
                    s->last_tok_type = MT_TOK_TYPE_SAMPLED_NON_EOG_NON_CONTROL;

                    // Calculate probabilities of all digits for first sampled
                    // non-EOG, non-control, non-whitespace, non-thinking,
                    // non-empty-piece token (assumes that the sampling of all
                    // former whitespaces was "correct", which is kind of wrong,
                    // but OK in practice):
                    //
                    if(dig_probs.empty() // <=> No non-whitespace sampled, yet.
                        && !piece.empty()
                        && !is_whitespace_only(piece.c_str()))
                    {
                        float max = 0.0f;

                        // TODO: Do just once during initialization:
                        //
                        std::vector<std::vector<int>> const dig_toks =
                            mt_llm_model_get_digit_tokens(*s->model);

                        std::vector<float> const logits = get_last_logits(max);
                        std::vector<float> const probs = get_probabilities(logits, max);
                        
                        dig_probs = get_token_group_probabilities(
                            dig_toks, probs);

                        //{
                        //    float prob_sum = 0.0f;
                        //
                        //    for(int i = 0; i < static_cast<int>(dig_probs.size()); ++i)
                        //    {
                        //        MT_LOG("  %d: %6.2f%%\n", i, 100.0f * dig_probs[i]);
                        //
                        //        prob_sum += dig_probs[i];
                        //    }
                        //    MT_LOG("Sum: %6.2f%%\n", 100.0f * prob_sum);
                        //}
                    }
                }
            }
        }

    	irq = callback_handler(new_tok_id, piece, dig_probs);

        if(is_thinker && is_thinking)
        {
            if(strncmp(
                piece.c_str(),
                s->mt_p->think_end_delim,
                MT_LLM_P_LEN_THINK_END_DELIM) == 0)
            {
                is_thinking = false; // AFTER calling callback.
            }
        }
        //
        // Otherwise: The model is not a thinker.

        // Current/single token per "batch":

        common_batch_clear(batch);
        common_batch_add(batch, new_tok_id, n_cur, { 0 }, true);

        int32_t const llama_decode_res = llama_decode(s->ctx, batch);

        if (llama_decode_res != 0)
        {
            MT_LOG_ERR(
                "Decoding current \"batch\" (error code %d)!\n",
                static_cast<int>(llama_decode_res));
            llama_batch_free(batch);
            free(last_chars);
            last_chars = nullptr;
            return false;
        }

        llama_sampler_accept(s->sampler, new_tok_id);

        // Break, if some kind of EOG token was generated:
        //
        if (new_tok_is_eog)
        {
            ++n_cur;
            break;
        }

        if(last_chars != nullptr) // => Reverse prompt is used and ring buffer.
        {
            char const * piece_chars = piece.c_str();

            while(*piece_chars != '\0') // Add current piece to ringbuffer.
            {
                ++last_chars_index;

                if(last_chars_index == rev_prompt_len)
                {
                    last_chars_index = 0;
                }

                last_chars[last_chars_index] = *piece_chars;

                if(last_chars_index == rev_prompt_len - 1)
                {
                    // Ring buffer is filled <=> Holds enough characters to
                    // compare with the reverse prompt. Signalize this:

                    last_chars_filled = true; // (setting once would be enough)
                }

                ++piece_chars;
            }
        }
    }
    llama_batch_free(batch);
    free(last_chars);
    last_chars = nullptr;

    if(n_ctx <= n_cur)
    {
        MT_LOG_ERR("Last token was no EOG (ctx. length reached?)\n");
        return false;
    }

    int64_t const t_main_end = ggml_time_us();
    int const n_decode = n_cur - s->tok_cnt;
    MT_LOG(
        "Decoded %d tokens in %.2fs, speed: %.2f t/s.\n",
        n_decode,
        static_cast<float>(t_main_end - t_main_start) / 1000000.0f,
        static_cast<float>(n_decode)
            / (static_cast<float>(t_main_end - t_main_start) / 1000000.0f));

    s->tok_cnt = n_cur;
    return true;
}

/**
 * - To be called by mt_llm_init().
 * - Caller takes ownership.
 */
static llama_sampler* create_sampler(llama_vocab const * const vocab)
{
    assert(s != nullptr && s->mt_p != nullptr);

    static size_t const min_keep = 0; // TODO: Is this the best option?
    static char const * const grammar_root = "root"; // TODO: Is this correct?

    llama_sampler_chain_params p = llama_sampler_chain_default_params();

    assert(p.no_perf);

    llama_sampler * const ret_val = llama_sampler_chain_init(p);

    if(s->mt_p->grammar[0] != '\0') // <- Does not seem to be necessary.
    {
        // TODO: Test, if this actually works this way!

        llama_sampler_chain_add(
            ret_val,
            llama_sampler_init_grammar(
                vocab, s->mt_p->grammar, grammar_root));
    }

    // Not used here: llama_sampler_init_logit_bias()

    // Not used here: llama_sampler_init_penalties()

    // Not used here: llama_sampler_init_dry()

    llama_sampler_chain_add(
        ret_val, llama_sampler_init_top_k(s->mt_p->top_k));

    // Not used here: llama_sampler_init_typical()

    llama_sampler_chain_add(
        ret_val, llama_sampler_init_top_p(s->mt_p->top_p, min_keep));

    llama_sampler_chain_add(
        ret_val, llama_sampler_init_min_p(s->mt_p->min_p, min_keep));

    // Not used here: llama_sampler_init_xtc()

    llama_sampler_chain_add( // TODO: Better use llama_sampler_init_temp_ext()?
        ret_val, llama_sampler_init_temp(s->mt_p->temp));

    assert(LLAMA_DEFAULT_SEED == static_cast<uint32_t>(-1));
    llama_sampler_chain_add(ret_val, llama_sampler_init_dist(s->mt_p->seed));

    return ret_val;
}

MT_EXPORT_LLM_API int mt_llm_get_token_count(
    char const * const text, bool const add_special)
{
    if(s == nullptr)
    {
        MT_LOG_ERR("Not intialized!\n");
        return -1;
    }
    if(text == nullptr)
    {
        MT_LOG_ERR("NULL given!\n");
        return -2;
    }

    std::vector<int> const tokens = mt_llm_ctx_tokenize(
        *s->ctx, text, add_special);

    return static_cast<int>(tokens.size());
}

MT_EXPORT_LLM_API struct mt_llm_state * __stdcall mt_llm_state_create()
{
    struct mt_llm_state * state = nullptr;

    if(s == nullptr)
    {
        MT_LOG_ERR("Not intialized!\n");
        return nullptr;
    }

    assert(s->ctx != nullptr);

    size_t const state_size = llama_state_get_size(s->ctx);

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
        s->ctx, state->state, state_size);

    if(written != state_size)
    {
        MT_LOG_ERR("Failed to write all %zu bytes!\n", state_size);
        free(state->state);
        state->state = nullptr;
        free(state);
        state = nullptr;
        return nullptr;
    }

    MT_LOG("Successfully copied %zu state bytes to memory.\n", state_size);
    state->size = state_size;
    state->last_tok_type = s->last_tok_type;
    state->tok_cnt = s->tok_cnt;
    return state; // Caller takes ownership!
}

MT_EXPORT_LLM_API bool __stdcall mt_llm_state_restore(
    struct mt_llm_state const * const state)
{
    assert(state != nullptr);
    assert(state->state != nullptr);
    assert(0 < state->size);

    if(s == nullptr)
    {
        MT_LOG_ERR("Not intialized!\n");
        return false;
    }

    assert(s->ctx != nullptr);
    
    size_t const read = llama_state_set_data(s->ctx, state->state, state->size);

    if(read != state->size)
    {
        MT_LOG_ERR("Filed to read exactly %zu bytes!\n", state->size);
        return false;
    }
    s->last_tok_type = state->last_tok_type;
    s->tok_cnt = state->tok_cnt;
    return true;
}

MT_EXPORT_LLM_API bool __stdcall mt_llm_query(char const * const prompt)
{
    if(s == nullptr)
    {
        MT_LOG_ERR("Not intialized!\n");
        return false;
    }

    assert(s->mt_p != nullptr);
    assert(s->model != nullptr);
    assert(s->ctx != nullptr);
    assert(s->sampler != nullptr);
    
    if(s->tok_cnt == 0 && s->mt_p->sys_prompt[0] != '\0')
    {
        if(!decode_initial_query(prompt))
        {
            return false; // (called function logs on error)
        }
    }
    else
    {
        if(!decode_follow_up_query(prompt))
        {
            return false; // (called function logs on error)
        }
    }

    if(!inference())
    {
        return false; // (called function logs on error)
    }

    MT_LOG("Token count: %d.\n", s->tok_cnt);
    return true;
}

MT_EXPORT_LLM_API void __stdcall mt_llm_reset()
{
    if(s == nullptr)
    {
        return; // Cannot do anything.
    }

    assert(s->mt_p != nullptr);
    assert(s->model != nullptr);
    assert(s->ctx != nullptr);
    assert(s->sampler != nullptr);

    {
        llama_memory_t kv = llama_get_memory(s->ctx);

        if(kv != nullptr)
        {
            llama_memory_clear(kv, true);
            kv = nullptr;
        }
    }

    llama_sampler_reset(s->sampler);

    s->last_tok_type = 0;
    s->tok_cnt = 0;
}

MT_EXPORT_LLM_API void __stdcall mt_llm_deinit()
{
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
        llama_model_free(s->model);
        s->model = nullptr;
    }
    llama_backend_free(); // Assuming(!) that this is OK, if not initialized..

    free(s);
    s = nullptr;
}

MT_EXPORT_LLM_API bool __stdcall mt_llm_reinit(
    struct mt_llm_p const * const mt_p)
{
    //common_init(); // Not calling this, seems to work anyway..

    common_log_pause(common_log_main());
    //
    //static void llama_log_callback_null(ggml_log_level level, const char * text, void * user_data) { (void) level; (void) text; (void) user_data; }
    //llama_log_set(llama_log_callback_null, NULL);

    if(s != nullptr)
    {
        mt_llm_deinit();
    }
    assert(s == nullptr);

    if(mt_p->callback == nullptr)
    {
        MT_LOG_ERR("Callback is not set!\n");
        //mt_llm_deinit();
        return false;
    }

    s = static_cast<struct mt_llm_s *>(malloc(sizeof *s));
    if(s == nullptr)
    {
        MT_LOG_ERR("Failed to allocate memory for settings!");
        //mt_llm_deinit();
        return false;
    }

    s->mt_p = mt_llm_p_create_copy(*mt_p);
    if(s->mt_p == nullptr)
    {
        MT_LOG_ERR("Failed to deep-copy parameters!\n");
        mt_llm_deinit();
        return false;
    }
    //
    // Do not use mt_p from here on!

    // If not given, automatically set the thread count:
    //
    if(s->mt_p->threads == 0)
    {
        s->mt_p->threads = (uint32_t)cpu_get_num_physical_cores();
        assert(0 < s->mt_p->threads);
    }

    // Do not change s->mt_p properties from here on, with the exception of
    // prompts (see below)!

    // Initialize the LLM:
    //
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED); // Unnecessary this way.

    // Initialize the model:
    //
    s->model = mt_llm_model_create(*s->mt_p);
    if (s->model == nullptr)
    {
        MT_LOG_ERR("Unable to load model!\n");
        mt_llm_deinit();
        return false;
    }

    // Initialize the sampling:
    //
    s->sampler = create_sampler(llama_model_get_vocab(s->model));
    if(s->sampler == nullptr)
    {
        MT_LOG_ERR("Unable to create sampler!\n");
        mt_llm_deinit();
        return false;
    }

    // Support for models with an encoder should be easy to add, but is
    // currently not implemented:
    //
    if(llama_model_has_encoder(s->model))
    {
        MT_LOG_ERR("Model has an encoder, that is currently not supported!\n");
        mt_llm_deinit();
        return false;
    }

    // Modify prompt strings by model (name), if wanted:
    //
    if(s->mt_p->try_prompts_by_model != 0)
    {
        mt_llm_model_try_set_prompts(*s->model, *s->mt_p);
        //
        // Return value ignored, as called function logs (and this is no error).
    }
    else
    {
        MT_LOG("Trying to use default prompts for model is not wanted.\n");
    }
    
    // Do not change any s->mt_p properties from here on!
    //
    mt_llm_p_print(*s->mt_p);

    // Initialize the context:

    s->ctx = mt_llm_ctx_create(*s->mt_p, *s->model);
    if (s->ctx == nullptr)
    {
        MT_LOG_ERR("Creating context!\n");
        mt_llm_deinit();
        return false;
    }

    {
        int32_t const n_ctx_train = llama_model_n_ctx_train(s->model),
            n_ctx_ctx = static_cast<int32_t>(llama_n_ctx(s->ctx)); // Bold cast?

        // Interpreted as error here, by definition:
        //
        assert(
            s->mt_p->n_ctx == 0
                || static_cast<int32_t>(s->mt_p->n_ctx) == n_ctx_ctx);
        if(n_ctx_train < n_ctx_ctx)
        {
            MT_LOG_ERR(
                "Model was trained on %d tokens (wanted %d tokens)!\n",
                n_ctx_train,
                n_ctx_ctx);
            mt_llm_deinit();
            return false;
        }
    }

    MT_LOG("System info: %s\n", llama_print_system_info());

    s->last_tok_type = 0;
    s->tok_cnt = 0;

    return true;
}
