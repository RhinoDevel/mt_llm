
// Marcel Timm, RhinoDevel, 2024aug21

#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdlib>

#include "mt_llm_p.h"
#include "mt_llm_log.h"

void mt_llm_p_print(struct mt_llm_p const & mt_p)
{
    MT_LOG("n_gpu_layers" ": " "%d" "\n", mt_p.n_gpu_layers);

    MT_LOG("n_ctx" ": " "%u" "\n", mt_p.n_ctx);
    MT_LOG("n_batch" ": " "%u" "\n", mt_p.n_batch);
    MT_LOG("n_ubatch" ": " "%u" "\n", mt_p.n_ubatch);
    MT_LOG("threads" ": " "%u" "\n", mt_p.threads);
    MT_LOG("type_kv_q8_0" ": " "%u" "\n", mt_p.type_kv_q8_0);
    MT_LOG("emb_or_rerank" ": " "%u" "\n", mt_p.emb_or_rerank);

    MT_LOG("seed" ": "  "%u" "\n", mt_p.seed);
    MT_LOG("top_k" ": " "\"%d\"" "\n", mt_p.top_k);
    MT_LOG("top_p" ": " "\"%f\"" "\n", mt_p.top_p);
    MT_LOG("min_p" ": " "\"%f\"" "\n", mt_p.min_p);
    MT_LOG("temp" ": " "\"%f\"" "\n", mt_p.temp);
    MT_LOG("grammar" ": " "\"%s\"" "\n", mt_p.grammar);

    MT_LOG("cpu_moe" ": " "%u" "\n", mt_p.cpu_moe);
    MT_LOG("n_cpu_moe" ": " "%d" "\n", mt_p.n_cpu_moe);
    MT_LOG("model_file_path" ": " "\"%s\"" "\n", mt_p.model_file_path);    
    MT_LOG("sys_prompt" ": " "\"%s\"" "\n", mt_p.sys_prompt);
    MT_LOG("prompt_beg_delim" ": " "\"%s\"" "\n", mt_p.prompt_beg_delim);
    MT_LOG("prompt_end_delim" ": " "\"%s\"" "\n", mt_p.prompt_end_delim);
    MT_LOG(
        "sys_prompt_beg_delim" ": " "\"%s\"" "\n", mt_p.sys_prompt_beg_delim);
    MT_LOG(
        "sys_prompt_mid_delim" ": " "\"%s\"" "\n", mt_p.sys_prompt_mid_delim);
    MT_LOG(
        "sys_prompt_end_delim" ": " "\"%s\"" "\n", mt_p.sys_prompt_end_delim);
    MT_LOG("think_beg_delim" ": " "\"%s\"" "\n", mt_p.think_beg_delim);
    MT_LOG("think_end_delim" ": " "\"%s\"" "\n", mt_p.think_end_delim);

    MT_LOG("try_prompts_by_model" ": " "%u" "\n", mt_p.try_prompts_by_model);
    MT_LOG("enable_thinking" ": " "%u" "\n", mt_p.enable_thinking);
    MT_LOG("enable_llama_cpp_log" ": " "%u" "\n", mt_p.enable_llama_cpp_log);

    MT_LOG(
        "callback" ": " "Is %sset." "\n",
        mt_p.callback == nullptr ? "NOT " : "");
}

void mt_llm_p_free(struct mt_llm_p * const mt_p)
{
    if(mt_p == nullptr)
    {
        return;
    }
    
    // (deallocation of property values must be done here, when necessary)

    free(mt_p);
}

bool mt_llm_p_are_equal(
    struct mt_llm_p const & a,
    struct mt_llm_p const & b, 
    bool const skip_resettable)
{
    if(a.n_gpu_layers != b.n_gpu_layers)
    {
        return false;
    }

    if(a.n_ctx != b.n_ctx)
    {
        return false;
    }
    if(a.n_batch != b.n_batch)
    {
        return false;
    }
    if(a.n_ubatch != b.n_ubatch)
    {
        return false;
    }
    if(a.threads != b.threads)
    {
        return false;
    }
    if(a.type_kv_q8_0 != b.type_kv_q8_0)
    {
        return false;
    }
    if(a.emb_or_rerank != b.emb_or_rerank)
    {
        return false;
    }

    if(a.seed != b.seed)
    {
        return false;
    }

    if(!skip_resettable)
    {
        if(a.top_k != b.top_k)
        {
            return false;
        }
        if(a.top_p != b.top_p)
        {
            return false;
        }
        if(a.min_p != b.min_p)
        {
            return false;
        }
        if(a.temp != b.temp)
        {
            return false;
        }
        if(strncmp(a.grammar, b.grammar, MT_LLM_P_LEN_GRAMMAR) != 0)
        {
            return false;
        }
    }

    if((a.cpu_moe == 0) != (b.cpu_moe == 0))
    {
        return false;
    }
    if(a.n_cpu_moe != b.n_cpu_moe)
    {
        return false;
    }
    if(strncmp(
        a.model_file_path, b.model_file_path, MT_LLM_P_LEN_MODEL_FILE_PATH)
            != 0)
    {
        return false;
    }

    if(!skip_resettable)
    {
        if(strncmp(a.sys_prompt, b.sys_prompt, MT_LLM_P_LEN_SYS_PROMPT) != 0)
        {
            return false;
        }
    }

    if(strncmp(
        a.prompt_beg_delim, b.prompt_beg_delim, MT_LLM_P_LEN_PROMPT_BEG_DELIM)
            != 0)
    {
        return false;
    }
    if(strncmp(
        a.prompt_end_delim, b.prompt_end_delim, MT_LLM_P_LEN_PROMPT_END_DELIM)
            != 0)
    {
        return false;
    }
    if(strncmp(
        a.sys_prompt_beg_delim,
        b.sys_prompt_beg_delim,
        MT_LLM_P_LEN_SYS_PROMPT_BEG_DELIM)
            != 0)
    {
        return false;
    }
    if(strncmp(
        a.sys_prompt_mid_delim,
        b.sys_prompt_mid_delim,
        MT_LLM_P_LEN_SYS_PROMPT_MID_DELIM)
            != 0)
    {
        return false;
    }
    if(strncmp(
        a.sys_prompt_end_delim,
        b.sys_prompt_end_delim,
        MT_LLM_P_LEN_SYS_PROMPT_END_DELIM)
            != 0)
    {
        return false;
    }
    if(strncmp(
        a.think_beg_delim, b.think_beg_delim, MT_LLM_P_LEN_THINK_BEG_DELIM)
            != 0)
    {
        return false;
    }
    if(strncmp(
        a.think_end_delim, b.think_end_delim, MT_LLM_P_LEN_THINK_END_DELIM)
            != 0)
    {
        return false;
    }
    if((a.try_prompts_by_model == 0) != (b.try_prompts_by_model == 0))
    {
        return false;
    }
    if((a.enable_thinking == 0) != (b.enable_thinking == 0))
    {
        return false;
    }
    if((a.enable_llama_cpp_log == 0) != (b.enable_llama_cpp_log == 0))
    {
        return false;
    }

    if(a.callback != b.callback)
    {
        return false;
    }

    return true;
}

struct mt_llm_p * mt_llm_p_create_copy(struct mt_llm_p const & mt_p)
{
    struct mt_llm_p * copy = nullptr;

    copy = (struct mt_llm_p *)malloc(sizeof *copy);
    if(copy == nullptr)
    {
        assert(false); // Must never happen.
        return nullptr;
    }

    copy->n_gpu_layers = mt_p.n_gpu_layers;

    copy->n_ctx = mt_p.n_ctx;
    copy->n_batch = mt_p.n_batch;
    copy->n_ubatch = mt_p.n_ubatch;
    copy->threads = mt_p.threads;
    copy->type_kv_q8_0 = mt_p.type_kv_q8_0;
    copy->emb_or_rerank = mt_p.emb_or_rerank;

    copy->seed = mt_p.seed;
    copy->top_k = mt_p.top_k;
    copy->top_p = mt_p.top_p;
    copy->min_p = mt_p.min_p;
    copy->temp = mt_p.temp;
    strncpy(
        copy->grammar,
        mt_p.grammar,
        MT_LLM_P_LEN_GRAMMAR - 1);
    copy->grammar[MT_LLM_P_LEN_GRAMMAR - 1] = '\0';

    copy->cpu_moe = mt_p.cpu_moe;
    copy->n_cpu_moe = mt_p.n_cpu_moe;

    strncpy(
        copy->model_file_path,
        mt_p.model_file_path,
        MT_LLM_P_LEN_MODEL_FILE_PATH - 1);
    copy->model_file_path[MT_LLM_P_LEN_MODEL_FILE_PATH - 1] = '\0';
    strncpy(
        copy->prompt_beg_delim,
        mt_p.prompt_beg_delim,
        MT_LLM_P_LEN_PROMPT_BEG_DELIM - 1);
    copy->prompt_beg_delim[MT_LLM_P_LEN_PROMPT_BEG_DELIM - 1] = '\0';
    strncpy(
        copy->prompt_end_delim,
        mt_p.prompt_end_delim,
        MT_LLM_P_LEN_PROMPT_END_DELIM - 1);
    copy->prompt_end_delim[MT_LLM_P_LEN_PROMPT_END_DELIM - 1] = '\0';
    strncpy(
        copy->sys_prompt_beg_delim,
        mt_p.sys_prompt_beg_delim,
        MT_LLM_P_LEN_SYS_PROMPT_BEG_DELIM - 1);
    copy->sys_prompt_beg_delim[MT_LLM_P_LEN_SYS_PROMPT_BEG_DELIM - 1] = '\0';
    strncpy(
        copy->sys_prompt_mid_delim,
        mt_p.sys_prompt_mid_delim,
        MT_LLM_P_LEN_SYS_PROMPT_MID_DELIM - 1);
    copy->sys_prompt_mid_delim[MT_LLM_P_LEN_SYS_PROMPT_MID_DELIM - 1] = '\0';
    strncpy(
        copy->sys_prompt_end_delim,
        mt_p.sys_prompt_end_delim,
        MT_LLM_P_LEN_SYS_PROMPT_END_DELIM - 1);
    copy->sys_prompt_end_delim[MT_LLM_P_LEN_SYS_PROMPT_END_DELIM - 1] = '\0';
    strncpy(
        copy->sys_prompt,
        mt_p.sys_prompt,
        MT_LLM_P_LEN_SYS_PROMPT - 1);
    copy->sys_prompt[MT_LLM_P_LEN_SYS_PROMPT - 1] = '\0';
    strncpy(
        copy->think_beg_delim,
        mt_p.think_beg_delim,
        MT_LLM_P_LEN_THINK_BEG_DELIM - 1);
    copy->think_beg_delim[MT_LLM_P_LEN_THINK_BEG_DELIM - 1] = '\0';
    strncpy(
        copy->think_end_delim,
        mt_p.think_end_delim,
        MT_LLM_P_LEN_THINK_END_DELIM - 1);
    copy->think_end_delim[MT_LLM_P_LEN_THINK_END_DELIM - 1] = '\0';

    copy->try_prompts_by_model = mt_p.try_prompts_by_model;
    copy->enable_thinking = mt_p.enable_thinking;
    copy->enable_llama_cpp_log = mt_p.enable_llama_cpp_log;

    copy->callback = mt_p.callback;

    return copy;
}