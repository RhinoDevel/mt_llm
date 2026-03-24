
// Marcel Timm, RhinoDevel, 2026mar24

// An example about how to use a reranking model with mt_llm.

#include "mt_llm.h"
#include "mt_llm_p.h"

#include <stdio.h>

int main(void)
{
    struct mt_llm_p p;
    
    // *****************************
    // *** Setup the parameters: ***
    // *****************************
    
    p.n_gpu_layers = 0;
    
    p.n_ctx = 0;
    p.threads = 0;
    p.emb_or_rerank = 2;

    p.seed = -1;
    p.top_k = 0;
    p.top_p = 0;
    p.min_p = 0;
    p.temp = 0;
    p.grammar[0] = '\0';
    
    p.cpu_moe = 0;
    p.n_cpu_moe = 0;
    
    strncpy(
        p.model_file_path,
        "bge-reranker-v2-m3-q5_k_m.gguf",
        MT_LLM_P_LEN_MODEL_FILE_PATH);

    strncpy(
        p.sys_prompt,
        "This is just a dummy text, might be OK, if empty..",
        MT_LLM_P_LEN_SYS_PROMPT);
    p.prompt_beg_delim[0] = '\0';
    p.prompt_end_delim[0] = '\0';
    p.sys_prompt_beg_delim[0] = '\0';
    p.sys_prompt_mid_delim[0] = '\0';
    p.sys_prompt_end_delim[0] = '\0';
    p.think_beg_delim[0] = '\0';
    p.think_end_delim[0] = '\0';

    p.try_prompts_by_model = 0;

    p.callback = NULL;

    // **************************
    // *** Initialize mt_llm: ***
    // **************************

    mt_llm_reinit(&p, 0); // Ignoring return value, here..
    
    // ********************************************
    // *** Generate and print reranking scores: ***
    // ********************************************
    
    static char const * query = "What is a whale?";
    static char const * documents[] = {
        "Butterflies eat flowers.",
        "A whale is an animal.",
        "I like to eat fish.",
        "Dogs want food.",
        "A large mammal that lives in water.",
        "A whale shark is a dangerous animal."
    };
    static int const doc_count =
        (int)(sizeof documents / sizeof *documents);
    
    float * const scores = mt_llm_rerank(
        query,
        documents,
        doc_count,
        0);
        
    // Assuming that scores is not NULL (which would indicate an error).
    for(int i = 0; i < doc_count; ++i)
    {
        printf("\"%s\": %f\n", documents[i], scores[i]);
    }
    mt_llm_free(scores);
    
    // ****************************
    // *** Deinitialize mt_llm: ***
    // ****************************

    mt_llm_deinit(0);

    return 0;
}
