
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

    MT_LOG("seed" ": "  "%u" "\n", mt_p.seed);
    MT_LOG("n_ctx" ": " "%u" "\n", mt_p.n_ctx);
    MT_LOG("threads" ": " "%u" "\n", mt_p.threads);
    MT_LOG("flash_attn" ": " "%u" "\n", mt_p.flash_attn);

    MT_LOG("top_k" ": " "\"%d\"" "\n", mt_p.top_k);
    MT_LOG("top_p" ": " "\"%f\"" "\n", mt_p.top_p);
    MT_LOG("min_p" ": " "\"%f\"" "\n", mt_p.min_p);
    MT_LOG("temp" ": " "\"%f\"" "\n", mt_p.temp);
    MT_LOG("grammar" ": " "\"%s\"" "\n", mt_p.grammar);

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
    MT_LOG("rev_prompt" ": " "\"%s\"" "\n", mt_p.rev_prompt);
    MT_LOG("think_beg_delim" ": " "\"%s\"" "\n", mt_p.think_beg_delim);
    MT_LOG("think_end_delim" ": " "\"%s\"" "\n", mt_p.think_end_delim);

    MT_LOG("try_prompts_by_model" ": " "%u" "\n", mt_p.try_prompts_by_model);
    
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

    copy->seed = mt_p.seed;
    copy->n_ctx = mt_p.n_ctx;
    copy->threads = mt_p.threads;
    copy->flash_attn = mt_p.flash_attn;

    copy->top_k = mt_p.top_k;
    copy->top_p = mt_p.top_p;
    copy->min_p = mt_p.min_p;
    copy->temp = mt_p.temp;
    strncpy(
        copy->grammar,
        mt_p.grammar,
        MT_LLM_P_LEN_GRAMMAR);

    strncpy(
        copy->model_file_path,
        mt_p.model_file_path,
        MT_LLM_P_LEN_MODEL_FILE_PATH);
    strncpy(
        copy->prompt_beg_delim,
        mt_p.prompt_beg_delim,
        MT_LLM_P_LEN_PROMPT_BEG_DELIM);
    strncpy(
        copy->prompt_end_delim,
        mt_p.prompt_end_delim,
        MT_LLM_P_LEN_PROMPT_END_DELIM);
    strncpy(
        copy->sys_prompt_beg_delim,
        mt_p.sys_prompt_beg_delim,
        MT_LLM_P_LEN_SYS_PROMPT_BEG_DELIM);
    strncpy(
        copy->sys_prompt_mid_delim,
        mt_p.sys_prompt_mid_delim,
        MT_LLM_P_LEN_SYS_PROMPT_MID_DELIM);
    strncpy(
        copy->sys_prompt_end_delim,
        mt_p.sys_prompt_end_delim,
        MT_LLM_P_LEN_SYS_PROMPT_END_DELIM);
    strncpy(
        copy->sys_prompt,
        mt_p.sys_prompt,
        MT_LLM_P_LEN_SYS_PROMPT);
    strncpy(
        copy->rev_prompt,
        mt_p.rev_prompt,
        MT_LLM_P_LEN_REV_PROMPT);
    strncpy(
        copy->think_beg_delim,
        mt_p.think_beg_delim,
        MT_LLM_P_LEN_THINK_BEG_DELIM);
    strncpy(
        copy->think_end_delim,
        mt_p.think_end_delim,
        MT_LLM_P_LEN_THINK_END_DELIM);

    copy->try_prompts_by_model = mt_p.try_prompts_by_model;

    copy->callback = mt_p.callback;

    return copy;
}