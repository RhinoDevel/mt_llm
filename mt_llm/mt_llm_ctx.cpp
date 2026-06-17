
// Marcel Timm, RhinoDevel, 2024aug21

#include <cstdio>
#include <cassert>

#include "llama.h"
#include "common.h"
#include "llama-model.h"

#include "mt_llm_p.h"
#include "mt_llm_ctx.h"
#include "mt_llm_log.h"
#include "mt_llm_model.h"
#include "mt_llm_s.h"

static llama_context_params get_ctx_params(
    mt_llm_p const & mt_p, llama_model const & model)
{
    llama_context_params ret_val = llama_context_default_params();

    //mt_p.seed; // Used for the sampler somewhere else.
    assert(0 < mt_p.threads);
    ret_val.n_threads = mt_p.threads;
    ret_val.n_threads_batch = ret_val.n_threads;

    ret_val.n_seq_max = 1; // Max. number of sequences.

    ret_val.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    if(mt_p.emb_or_rerank == 0)
    {
        ret_val.n_ctx = mt_p.n_ctx;

        // TODO: Add parameters for the batch sizes to mt_llm_p!
        //ret_val.n_batch // Logical max. batch size.
        //ret_val.n_ubatch // Physical max. batch size.
    }
    else
    {
        // Extract embeddings (together with logits). Correct for rerank., too.
        ret_val.embeddings = true;

        // If we wanted to process multiple sequences at once, we should make
        // use of the parameters .kv_unified and/or .n_parallel, too (see
        // llama.cpp's example). Also used for reranking mode, see below!

        // llama.cpp seems to do pooling for embedding vector calculation over
        // the latest batch / decode call, only (maybe this has something to do
        // with embedding models being almost always NON-causal). This is why
        // the batch size of 1 is not possible for embedding vector creation.

        int32_t const model_n_ctx_train = llama_model_n_ctx_train(&model);
        assert(0 < model_n_ctx_train);

        if(mt_p.n_ctx == 0)
        {
            // Use model's context length.
            
            ret_val.n_ctx = static_cast<uint32_t>(model_n_ctx_train);
        }
        else
        {
            if(mt_p.n_ctx < static_cast<uint32_t>(model_n_ctx_train))
            {
                // Use model's context length (it is the minimum).

                MT_LOG("Warning: Ctx. size of %u tokens wanted, but model train. ctx. size is %d tokens. Using training size..\n",
                    mt_p.n_ctx,
                    model_n_ctx_train);

                ret_val.n_ctx = static_cast<uint32_t>(model_n_ctx_train);
            }
            else
            {
                // Use given context length.
                ret_val.n_ctx = mt_p.n_ctx;
            }
        }
        // Assuming that these maximum batch sizes are OK for the hardware..
        ret_val.n_batch = ret_val.n_ctx; // Logical max. batch size.
        ret_val.n_ubatch = ret_val.n_ctx; // Physical max. batch size.

        // Required for non-causal models:
        assert(ret_val.n_batch == ret_val.n_ubatch);

        // We are trying to use the default pooling type of the model, if known:

        switch(model.hparams.pooling_type)
        {
            case LLAMA_POOLING_TYPE_MEAN: // Falls through.
            case LLAMA_POOLING_TYPE_CLS: // Falls through.
            case LLAMA_POOLING_TYPE_LAST:
            {
                // Should be supported (see llama.cpp's embedding.cpp example
                // for reference..).
                ret_val.pooling_type = model.hparams.pooling_type;
                break;
            }
            
            case LLAMA_POOLING_TYPE_UNSPECIFIED: // Falls through.
            case LLAMA_POOLING_TYPE_NONE: // Falls through.
            case LLAMA_POOLING_TYPE_RANK:
            {
                // At least "rank" pooling type can be supported, see
                // llama.cpp's embedding.cpp sample.
                ret_val.pooling_type = LLAMA_POOLING_TYPE_MEAN;
                MT_LOG(
                    "Warning: Model pooling type is %d, which is not supported. Using %d..\n",
                    static_cast<int>(model.hparams.pooling_type),
                    static_cast<int>(ret_val.pooling_type));
                break;
            }

            default: // Other model pooling type (should not happen, here).
            {
                assert(false);
                ret_val.pooling_type = LLAMA_POOLING_TYPE_MEAN;
                MT_LOG(
                    "Warning: Unsupported model pooling type %d. Using %d..\n",
                    static_cast<int>(model.hparams.pooling_type),
                    static_cast<int>(ret_val.pooling_type));
                break;
            }
        }

        if(mt_p.emb_or_rerank == 2) // Hard-coded (reranking usage).
        {
            // For reranking usage, make sure that pooling type is "rank":
            if(ret_val.pooling_type != LLAMA_POOLING_TYPE_RANK)
            {
                MT_LOG(
                    "Pooling type set is not \"rank\" (%d), but %d. Using \"rank\" instead..\n",
                    static_cast<int>(LLAMA_POOLING_TYPE_RANK),
                    static_cast<int>(ret_val.pooling_type));
                ret_val.pooling_type = LLAMA_POOLING_TYPE_RANK;
            }

            // For reranking, we use batching (could be activated for
            // embeddings, too):
            assert(ret_val.n_seq_max == 1);
            ret_val.kv_unified = true;
            ret_val.n_seq_max = // Per batch.
                static_cast<uint32_t>(llama_max_parallel_sequences());
            MT_LOG(
                "Unified K/V cache is enabled, max. nr. of sequences is %d.\n",
                ret_val.n_seq_max);
        }
    }

    return ret_val;
}

/** Add as many tokens if given to the context as possible.
 *  Inform sampler about the new token.
 *
 *  - Returns the count of tokens successfully added to the context.
 *  - Returns -1 on error!
 *  - Uses 
 *  - Never applies grammar.
 */
static int decode_as_many_tokens_as_possible(
    llama_context * const ctx,
    llama_sampler * const sampler,
    int const existing_token_count,
    std::vector<llama_token>& tokens,
    bool const output_last_logits)
{
    llama_batch batch;

    // We want to work with a copy of the vector to keep the input as it is.
    std::vector<llama_token> buf;

    int const full_token_count = static_cast<int>(tokens.size());

    if(full_token_count == 0)
    {
        return 0; // Nothing to do.
    }

    buf = tokens;

    int const n_ctx = llama_n_ctx(ctx);

    assert( // TODO: We actually do not need existing-token-count parameter!
        llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1
            == existing_token_count);

    int const free_pos_count = n_ctx - existing_token_count;

    if(free_pos_count < full_token_count)
    {
        MT_LOG(
            "Not enough free positions in context t decode %d tokens, trying to decode possible maximum of %d tokens..\n",
            full_token_count,
            free_pos_count);
        buf.erase(buf.end() - full_token_count + free_pos_count, buf.end());
    }

    int const buf_tok_cnt = static_cast<int>(buf.size());

    int const n_batch = static_cast<int>(llama_n_batch(ctx));

    if(n_batch < buf_tok_cnt)
    {
        MT_LOG_ERR(
            "To-be-decoded token count of %d exceeds max. token count per batch, which is %d (increase and re-run?)\n",
            buf_tok_cnt,
            n_batch);
        return -1;
    }

    // Initialize a batch for the tokens to-be-decoded.
    batch = llama_batch_init(buf_tok_cnt, 0, 1);

    // Add all tokens to the batch. If wanted, configure the last token to get
    // its logits output.
    for(int i = 0; i < buf_tok_cnt; ++i)
    {
        common_batch_add(
            batch,
            buf[i],
            existing_token_count + i, // <- The correct position.
            { 0 }, // <- Always the same, single sequence zero.
            output_last_logits && i + 1 == buf_tok_cnt);
    }

    // Try to fill the decoder with the tokens.
    int const decode_result = llama_decode(ctx, batch);

    // Free the batch.
    llama_batch_free(batch);

    if(decode_result != 0)
    {
        assert(false); // Should not get here.
        MT_LOG_ERR("Decoding failed!\n");
        return -1;
    }

    // Inform sampler about the added tokens.
    for(int i = 0; i < buf_tok_cnt; ++i)
    {
        llama_sampler_accept(sampler, buf[i]);
    }

    return buf_tok_cnt;
}

/*
    - Read this out from the GGUF file: tokenizer.ggml.add_bos_token

    // LLAMA_ARG_GRP_ATTN_N
    // common.h: struct common_params
    assert(my_common_params.grp_attn_n == 1); // group-attention factor must be 1 as it seems.

    auto * mem = llama_get_memory(ctx);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    LLAMA_API bool llama_memory_seq_rm(
        llama_memory_t mem,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_add(
        llama_memory_t mem,
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        llama_pos delta);

    // infinite text generation via context shifting
    // if we run out of context:
    // - take the n_keep first tokens from the original prompt (via n_past)
    // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

    if (n_past >= n_ctx)
    {
        const int n_left    = n_past - params.n_keep;
        const int n_discard = n_left/2;

        llama_memory_seq_rm (mem, 0, params.n_keep            , params.n_keep + n_discard);
        llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_past, -n_discard);

        n_past -= n_discard;
    }

    // - Remember index of first 
*/

std::string mt_llm_ctx_get_piece_from(
    llama_context& ctx, llama_token const tok)
{
    return common_token_to_piece(
        &ctx,
        tok,
        true); // Render special tokens, too (unknown or control attr.).
}

bool mt_llm_ctx_decode(
    mt_llm_s const &s,
    std::vector<llama_token>& tokens,
    int& decoded_token_count,
    bool(*callback)(llama_token, std::string const &, std::vector<float> const &, mt_llm_s const &),
    std::vector<float> const & dig_probs,
    bool& irq)
{
    irq = false;

    llama_vocab const * const vocab =
        llama_model_get_vocab(llama_get_model(s.ctx));

    if(s.tok_cnt == 0 && llama_vocab_get_add_bos(vocab))
    {
        // *** AUGMENTS ***

        // A BOS token is wanted at the beginning.
        tokens.insert(tokens.begin(), llama_vocab_bos(vocab));
    }

    int const tok_count = static_cast<int>(tokens.size());

    decoded_token_count = decode_as_many_tokens_as_possible(
        s.ctx, s.sampler, s.tok_cnt, tokens, true);

    if(decoded_token_count != tok_count)
    {
        return false; // (called function logged)
    }

    if(callback != nullptr)
    {
        for(int i = 0; i < tok_count; ++i)
        {
            irq = callback(
                tokens[i],
                mt_llm_ctx_get_piece_from(*s.ctx, tokens[i]),
                dig_probs,
                s);
        }
    }

    return true;
}

std::vector<llama_token> mt_llm_ctx_tokenize(
    llama_context const & ctx, char const * const str, bool const add_special)
{
    // From llama.cpp's llama.h, llama_tokenize():
    //
    // add_special:
    // - Allow to add BOS and EOS tokens if model is configured to do so.
    //
    // parse_special:
    // - Allow tokenizing special and/or control tokens which otherwise are not
    //   exposed and treated as plaintext. Does not insert a leading space.

    return common_tokenize(
        llama_model_get_vocab(llama_get_model(&ctx)),
        str, // Implicit cast/conversion!
        add_special,
        true); // => Always parse string representations of special tokens to
               //    special tokens.
}

llama_context* mt_llm_ctx_create(
    mt_llm_p const & mt_p, mt_llm_model const & model)
{
    return llama_init_from_model(
        model.model, get_ctx_params(mt_p, *model.model));
}
