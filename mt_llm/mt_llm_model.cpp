
// Marcel Timm, RhinoDevel, 2024aug21

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <vector>

#include "llama.h"
#include "llama-vocab.h"
#include "common.h"

#include "mt_llm_model.h"
#include "mt_llm_log.h"

#define MT_LLM_MODEL_NAME_KEY "general.name"

#define MT_LLM_MODEL_DEFAULT_META_VAL_STR_MAX_LEN (9 + 1) // Trailing '\0' incl. 

// Does not work for all models (see struct prompt_template).
#define MT_LLM_MODEL_CREATE( \
    names, beg, sys, user, bot, end, rev, think_beg, think_end) \
{ \
    .model_names = (names), \
\
    .sys_prompt_beg_delim = beg sys, \
    .sys_prompt_mid_delim = end beg user, \
    .sys_prompt_end_delim = end beg bot, \
\
    .prompt_beg_delim = beg user, \
    .prompt_end_delim = end beg bot, \
\
    .rev_prompt = (rev), \
\
    .think_beg_delim = (think_beg), \
    .think_end_delim = (think_end), \
\
    .n_sys_keep = (int)sizeof(beg sys) - 1 \
}

// *****************************************************************************
// *** MODEL NAMES AND (PROMPT) TEMPLATES                                    ***
// *****************************************************************************

struct prompt_template
{
    char const * const * const model_names;

    char const * const sys_prompt_beg_delim;
    char const * const sys_prompt_mid_delim;
    char const * const sys_prompt_end_delim;

    char const * const prompt_beg_delim;
    char const * const prompt_end_delim;

    char const * const rev_prompt;

    char const * const think_beg_delim;
    char const * const think_end_delim;

    // Does NOT include a maybe necessary BOS (model-dependent) and also not the
    // system prompt itself (as it is not known, here).
    int const n_sys_keep;
};

// *****************************************************************************

static char const * const s_model_names_phi3[] = {
    "Phi3",
    "Phi 3 Mini 4k Instruct",
    "Phi 3.5 Mini Instruct",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
static const struct prompt_template s_phi3 = { // <- Add this to "the" array.
    .model_names = s_model_names_phi3,

    // This is how it is currently implemented here, but see the notes below
    // (uncertain about some newlines, etc.):
    //
    // [SYS. PROMPT BEGIN]<|system|>
    // You are a nice AI assistant.<|end|>
    // <|user|>
    // Hi there, are you a nice AI assistant, or what!?<|end|>
    // <|assistant|>
    // [FIRST INFERENCE BEGIN (see note about newline, below)]Yes, that is correct, I am a nice assistant.<|end|>[FIRST USER PROMPT BEGIN]
    // <|user|>
    // Cool, thanks.<|end|>
    // <|assistant|>
    // [SECOND INFERENCE BEGIN (see note about newline, below)]You're welcome. I am nice like that.<|end|>[SECOND USER PROMPT BEGIN]...

    .sys_prompt_beg_delim = "<|system|>\n",
    .sys_prompt_mid_delim = "<|end|>\n<|user|>\n",

    // NOTE that e.g. at https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
    // the description of the chat template to use says that after the
    // tag <|assistant|> the LLM should add a newline during inference, but
    // (at least "Phi 3.5 Mini Instruct" with mt_llm using llama.cpp) does
    // not do that.
    // =>
    // We are adding a newline after <|assistant|> here. Not sure, if it would
    // be better to not do this (bug in mt_llm?)..

    .sys_prompt_end_delim = "<|end|>\n<|assistant|>" "\n",

    // Also NOTE that I am not quite sure if the newline before <|user|> token
    // below is a good idea. mt_llm currently "shows" a space before the
    // <|user|> token (and elsewhere) - at least with "Phi 3.5 Mini Instruct" -
    // so maybe there is (another?) bug, here..

    .prompt_beg_delim = "\n" "<|user|>\n",
    .prompt_end_delim = "<|end|>\n<|assistant|>" "\n",

    // TODO: Test, if necessary anymore (maybe working now, since 2c35bc457b022a00f65815987583373de246f654)!
    //
    // TODO: This is kind of a workaround, seems not to be necessary on
    //       the PC, but on Android (order of tokens in llama.cpp..?):
    //
    .rev_prompt = "<|end|>",

    .think_beg_delim = "",
    .think_end_delim = "",
 
    .n_sys_keep = -1 // Not supported (use create macro, if wanted).
};

static char const * const s_model_names_phi4[] = {
    "Phi 4",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
static const struct prompt_template s_phi4 = { // <- Add this to "the" array.
    .model_names = s_model_names_phi4,

    // Source: https://huggingface.co/microsoft/phi-4

    // TODO: Add newlines(?)!

    .sys_prompt_beg_delim = "<|im_start|>system<|im_sep|>",
    .sys_prompt_mid_delim = "<|im_end|><|im_start|>user<|im_sep|>",
    .sys_prompt_end_delim = "<|im_end|><|im_start|>assistant<|im_sep|>",

    .prompt_beg_delim = "<|im_start|>user<|im_sep|>",
    .prompt_end_delim = "<|im_end|><|im_start|>assistant<|im_sep|>",

    .rev_prompt = "",

    .think_beg_delim = "",
    .think_end_delim = "",
 
    .n_sys_keep = -1 // Not supported (use create macro, if wanted).
};

static char const * const s_model_names_llama2[] = {
    "LLaMA v2",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
static const struct prompt_template s_llama2 = { // <- Add this to "the" array.
    .model_names = s_model_names_llama2,

    .sys_prompt_beg_delim = /*<s>*/"[INST] <<SYS>>\n",
    // Always answer like a pirate.
    .sys_prompt_mid_delim = "\n<</SYS>>\n\n",
    // Hello! How is the sea today?
    .sys_prompt_end_delim = " [/INST] ",
    // Ahoi! 'Tis very stormy, mate.
    .prompt_beg_delim = /* </s>*/"\n<s>[INST] ",
    // Sounds awesome.
    .prompt_end_delim = " [/INST]",

    .rev_prompt = "",

    .think_beg_delim = "",
    .think_end_delim = "",
 
    .n_sys_keep = -1 // Not supported (use create macro, if wanted).
};

static char const * const s_model_names_llama3[] = {
    "Meta-Llama-3-8B-Instruct",
    "Meta Llama 3 8B Instruct",
    "Llama 3.1 SauerkrautLM 8b Instruct",
    "Meta Llama 3.1 8B Instruct",
    "Llama 3.2 3B Instruct",
    "Llama 3.2 1B Instruct",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
static const struct prompt_template s_llama3 = { // <- Add this to "the" array.
    .model_names = s_model_names_llama3,

    .sys_prompt_beg_delim = "<|start_header_id|>system<|end_header_id|>\n\n",
    .sys_prompt_mid_delim = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
    .sys_prompt_end_delim = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",

    .prompt_beg_delim = /*"<|eot_id|>"*/ "<|start_header_id|>user<|end_header_id|>\n\n",
    .prompt_end_delim = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",

    .rev_prompt = "",

    .think_beg_delim = "",
    .think_end_delim = "",
 
    .n_sys_keep = -1 // Not supported (use create macro, if wanted).
};

static char const * const s_model_names_qwen[] = {
    "Qwen2-0.5B-Instruct",
    "Qwen2-1.5B-Instruct",
    "qwen2-1_5b-instruct",
    "Qwen2.5 0.5B Instruct",
    "Qwen2.5 1.5B Instruct",
    "Qwen2.5 3B Instruct",
    "Qwen2.5 7B Instruct",
    "Qwen3-4B-Instruct-2507",
    "Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-Next-80B-A3B-Instruct",
    "Virtuoso Small v2", // From Arcee AI, based on Qwen2.5 14B.
    "Finalize Slerp", // Lamarck-14B-v0.7
    "MiniCPM3 4B", // By OpenBMB.
    "SmolLM 135M",
    "SmolLM2 135M Instruct",
    "SmolLM2 360M Instruct",
    "SmolLM2 1.7B Instruct",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
// This is what they call "ChatML".
//
// Source: https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
static const struct prompt_template s_qwen = // <- Add this to "the" array.
    MT_LLM_MODEL_CREATE(
        s_model_names_qwen,
        "<|im_start|>",
        "system\n",
        "user\n",
        // NOTE that adding newline after the "assistant" role seems to be the
        // way to go here, as LLM on inference does not seem to begin with a
        // newline.
        "assistant" "\n",
        "<|im_end|>\n",
        "",
        "",
        "");

static char const * const s_model_names_qwen3[] = {
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3 8B",
    "Qwen3 14B",
    "Qwen3-14B",
    "Qwen3-30B-A3B",
    "Qwen3-32B",
    "SmolLM3 3B", // Officially recommended: Temperature = 0.6, top-p = 0.95.
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
// This is what they call "ChatML".
//
// Source: https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
static const struct prompt_template s_qwen3 = // <- Add this to "the" array.
    MT_LLM_MODEL_CREATE(
        s_model_names_qwen3,
        "<|im_start|>",
        "system\n",
        "user\n",
        // NOTE that adding newline after the "assistant" role seems to be the
        // way to go here, as LLM on inference does not seem to begin with a
        // newline.
        "assistant" "\n",
        "<|im_end|>\n",
        "",
        "<think>",
        "</think>");

// TODO: "<think>" and "</think>" in prompt endings are not recognized as
//       sampled thinking mode delimiters, but this is OK, because they are not
//       sampled.
// 
//       Unfortunately, if thinking/reasoning is enabled, the actually
//       sampled reasoning-text is also NOT detected as sampled thinking tokens!
// 
// HARD-CODED: Reasoning/thinking is disabled.
//             Enable it by replacing "<think></think>\n" with "<think>\n".
static char const * const s_model_names_nemo3nano[] = {
    "Nemotron-3-Nano-30B-A3B",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
// This is what they call "ChatML", with additional "<think></think>\n" or
// "<think>\n" to disable or "jumpstart" assistant's thinking process.
// Source: https://unsloth.ai/docs/models/nemotron-3
static const struct prompt_template s_nemo3nano = // <- Add this to "the" array.
    MT_LLM_MODEL_CREATE(
        s_model_names_nemo3nano,
        "<|im_start|>",
        "system\n",
        "user\n",
        "assistant" "\n" "<think>" "</think>" "\n", // <- See comment above.
        "<|im_end|>\n",
        "",
        "<think>",
        "</think>");

static char const * const s_model_names_gemma[] = {
    "Gemma-3-270M-It",
    "Gemma 2 2b It",
    "Gemma 3 1b It",
    "Gemma-3-1B-It",
    "Gemma 3 4b It",
    "Gemma-3-4B-It",
    "Gemma 3 12b It",
    "Gemma-3-12B-It",
    "Gemma-3-27B-It",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
static const struct prompt_template s_gemma = { // <- Add this to "the" array.
    .model_names = s_model_names_gemma,

    // Source: https://ai.google.dev/gemma/docs/core/prompt-structure

    // For these models, the system prompt shall be added as prefix in the first
    // user prompt, so we are modelling the system prompt to do exactly that:

    .sys_prompt_beg_delim = "<start_of_turn>user\n",
    // System prompt will be put here.
    .sys_prompt_mid_delim = "\n\n",
    // User's first prompt will be put here.
    .sys_prompt_end_delim = "<end_of_turn>\n<start_of_turn>model" "\n",

    .prompt_beg_delim = "<start_of_turn>user\n",
    .prompt_end_delim = "<end_of_turn>\n<start_of_turn>model" "\n",

    // NOTE about newlines after "model" role: When letting the LLM during
    // inference take case of adding the newline, it adds two (at least
    // "Gemma-3-4B-It" did this). So adding them here manually seems the way to
    // go.

    .rev_prompt = "",

    .think_beg_delim = "",
    .think_end_delim = "",
 
    .n_sys_keep = -1 // Not supported (use create macro, if wanted).
};

static char const * const s_model_names_exaone3[] = {
    "EXAONE 3.0 7.8B Instruct",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
static const struct prompt_template s_exaone3 = { // <- Add this to "the" array.
    .model_names = s_model_names_exaone3,

    .sys_prompt_beg_delim = "[|system|]",
    .sys_prompt_mid_delim = "[|endofturn|]\n[|user|]",
    .sys_prompt_end_delim = "\n[|assistant|]",

    .prompt_beg_delim = "[|user|]",
    .prompt_end_delim = "\n[|assistant|]",

    .rev_prompt = "",

    .think_beg_delim = "",
    .think_end_delim = "",
 
    .n_sys_keep = -1 // Not supported (use create macro, if wanted).
};

static char const * const s_model_names_cohere4ai[] = {
    "CohereForAI.aya Expanse 8b",
    "Aya Expanse 8b",
    "C4Ai Command R7B 12 2024",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
// https://huggingface.co/CohereLabs/aya-expanse-8b/blob/main/tokenizer_config.json
static const struct prompt_template s_cohere4ai = // <- Add this to "the" array.
    MT_LLM_MODEL_CREATE(
        s_model_names_cohere4ai,
        "<|START_OF_TURN_TOKEN|>",
        "<|SYSTEM_TOKEN|>",
        "<|USER_TOKEN|>",
        "<|CHATBOT_TOKEN|>",
        "<|END_OF_TURN_TOKEN|>",
        "",
        "",
        "");

static char const * const s_model_names_mistral[] = { // V7-Tekken
    "Mistral Small 24B Instruct 2501",
    "Mistral-Small-3.2-24B-Instruct-2506",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
static const struct prompt_template s_mistral = { // <- Add this to "the" array.
    .model_names = s_model_names_mistral,

    .sys_prompt_beg_delim = /*"<s>" */"[SYSTEM_PROMPT]",
    // "You are a helpful AI assistant."
    .sys_prompt_mid_delim = "[/SYSTEM_PROMPT][INST]",
    // "Hello AI assistant, how are you?"
    .sys_prompt_end_delim = "[/INST]",
    // "Hi master, I am fine, thanks. How are you today?"
    .prompt_beg_delim = /*"</s>" */"[INST]",
    // "I am fine. Do you like to be an LLM?"
    .prompt_end_delim = "[/INST]",

    .rev_prompt = "",

    .think_beg_delim = "",
    .think_end_delim = "",
 
    .n_sys_keep = -1 // Not supported (use create macro, if wanted).
};

static char const * const s_model_names_mistral7b_v0_2[] = {
    "mistralai_mistral-7b-instruct-v0.2",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
static const struct prompt_template s_mistral7b_v0_2 = { // <- Add this to "the" array.
    .model_names = s_model_names_mistral7b_v0_2,

    // For these models, the system prompt shall be added as prefix in the first
    // user prompt, so we are modelling the system prompt to do exactly that:

    .sys_prompt_beg_delim = /*"<s>"*/"[INST]",
    // "You are a helpful AI assistant."
    .sys_prompt_mid_delim = "\n", // <- Incorrect, if no system prompt given..
    // "Hello AI assistant, how are you?"
    .sys_prompt_end_delim = " [/INST]",
    // "Hi master, I am fine, thanks. How are you today?"
    .prompt_beg_delim = /*"</s> "*/"[INST]",
    // "I am fine. Do you like to be an LLM?"
    .prompt_end_delim = " [/INST]",

    .rev_prompt = "",

    .think_beg_delim = "",
    .think_end_delim = "",
 
    .n_sys_keep = -1 // Not supported (use create macro, if wanted).
};

static char const * const s_model_names_olmo[] = {
    "OLMo 2 0425 1B Instruct",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
static const struct prompt_template s_olmo = { // <- Add this to "the" array.
    .model_names = s_model_names_olmo,

    .sys_prompt_beg_delim = "<|system|>\n",
    // "You are a helpful AI assistant."
    .sys_prompt_mid_delim = "\n<|user|>\n",
    // "Hello AI assistant, how are you?"
    .sys_prompt_end_delim = "\n<|assistant|>\n",
    // "Hi master, I am fine, thanks. How are you today?"
    .prompt_beg_delim = "\n<|user|>\n",
    // "I am fine. Do you like to be an LLM?"
    .prompt_end_delim = "\n<|assistant|>\n",

    .rev_prompt = "",

    .think_beg_delim = "",
    .think_end_delim = "",
 
    .n_sys_keep = -1 // Not supported (use create macro, if wanted).
};

static char const * const s_model_names_granite4[] = {
    "Granite-4.0-H-1B",
    "Granite-4.0-H-Micro",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
// https://huggingface.co/CohereLabs/aya-expanse-8b/blob/main/tokenizer_config.json
static const struct prompt_template s_granite4 = // <- Add this to "the" array.
    MT_LLM_MODEL_CREATE(
        s_model_names_granite4,
        "<|start_of_role|>",
        "system" "<|end_of_role|>",
        "user" "<|end_of_role|>",
        "assistant" "<|end_of_role|>",
        "<|end_of_text|>" "\n",
        "",
        "",
        "");

// TODO: Adds some white-space which may be wrong (e.g. after BOS <s>..).
static char const * const s_model_names_ernie45moe[] = {
    "Ernie-4.5-21B-A3B-Pt",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};
static const struct prompt_template s_ernie45moe = { // <- Add this to "the" array.
    .model_names = s_model_names_ernie45moe,

    .sys_prompt_beg_delim = "",
    .sys_prompt_mid_delim = "\nUser: ",
    .sys_prompt_end_delim = "\nAssistant: ",// Model adds EOS ("</s>") at end.

    // Ignoring the use of CLS and SEP tokens as it should be done, if you would
    // follow the Jinja template, also see:
    // https://github.com/ggml-org/llama.cpp/pull/14658#issuecomment-3082745420
    .prompt_beg_delim = "<s>User: ", // <- Hard-coded BOS token.
    .prompt_end_delim = "\nAssistant: ", // Model adds EOS ("</s>") at the end.

    .rev_prompt = "",

    .think_beg_delim = "",
    .think_end_delim = "",

    .n_sys_keep = -1 // Not supported (use create macro, if wanted).
};

// "[gMASK]<sop><|system|>You are a helpful assistant<|user|>Hello<|assistant|></think>Hi there<|user|>How are you?<|assistant|><think>"

static char const * const s_model_names_glm[] = {
    "Glm-4.7-Flash",
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};

static const struct prompt_template s_glm = { // <- Add this to "the" array.
    .model_names = s_model_names_glm,

    .sys_prompt_beg_delim = "[gMASK]<sop><|system|>",
    .sys_prompt_mid_delim = "<|user|>",
    .sys_prompt_end_delim = "<|assistant|></think>", // <- No thinking!

    .prompt_beg_delim = "<|user|>",
    .prompt_end_delim = "<|assistant|></think>", // <- No thinking!

    .rev_prompt = "",

    .think_beg_delim = "",
    .think_end_delim = "",

    .n_sys_keep = -1 // Not supported (use create macro, if wanted).
};

// *****************************************************************************

static struct prompt_template const * const s_prompt_templates[] = {
    &s_phi3,
    &s_phi4,
    &s_llama2,
    &s_llama3,
    &s_qwen,
    &s_qwen3,
    &s_nemo3nano,
    &s_gemma,
    &s_exaone3,
    &s_cohere4ai,
    &s_mistral,
    &s_mistral7b_v0_2,
    &s_olmo,
    &s_granite4,
    &s_ernie45moe,
    &s_glm,
    // Add more, when necessary.
    NULL // <- DON'T FORGET THIS TERMINATING NULL ENTRY!
};

// *****************************************************************************

// Only used by get_model_params() (and read-accessed externally):
static struct llama_model_tensor_buft_override s_tensor_buft_overrides[2] =
    {
        { .pattern = NULL, .buft = NULL },
        { .pattern = NULL, .buft = NULL }
    };

static bool is_space(char const c)
{
    return std::isspace(static_cast<int>(static_cast<unsigned char>(c)));
}

static bool is_char_plus_opt_whitespace(
    char const * const str, char const wanted)
{
    assert(str != nullptr);

    char const * ptr = str;

    while(is_space(*ptr)) // Moves past leading whitespace.
    {
        ++ptr;
    }

    if(*ptr == '\0')
    {
        return false;
    }
    if(*ptr != wanted)
    {
        return false;
    }

    ++ptr;

    while(is_space(*ptr)) // Skips trailing whitespace.
    {
        ++ptr;
    }

    return *ptr == '\0';
}

/** Returns, if the given strings are equal or not.
 *
 *  - Both strings must end with a '\0'!
 */
static bool are_strings_equal(char const * const a, char const * const b)
{
    int i = 0;

    while(true)
    {
        if(a[i] != b[i])
        {
            return false; // Not equal.
        }
        if(a[i] == '\0')
        {
            return true; // Equal
        }
        ++i;
    }
    assert(false); // Must never get here.
}

/** Does one of the strings in the given array equal the given string?
 *
 * - Given array of strings' last entry must be NULL, which signalizes the end
 *   of the array!
 *
 * - Returns -1, if not found or index in given array, if found.
 */
static int get_str_index(char const * const * const arr, char const * const str)
{
    int i = 0;

    while(true)
    {
        char const * const cur = arr[i];

        if(cur == NULL) // NULL in the array indicates the array's end.
        {
            return -1; // Not found.
        }
        if(are_strings_equal(cur, str))
        {
            return i; // Found
        }
        ++i;
    }
    assert(false); // Must never get here.
}

/**
 * - Given array's last entry must be NULL, which signalizes the end
 *   of the array!
 *
 * - Returns NULL, if not found or index in given array, if found.
 */
static struct prompt_template const * try_get_prompt_template(
    char const * const model_name)
{
    assert(model_name != NULL);

    int i = 0;

    while(true)
    {
        struct prompt_template const * const cur = s_prompt_templates[i];

        if(cur == NULL) // NULL in the array indicates the array's end.
        {
            return NULL; // Not found.
        }
        if(get_str_index(cur->model_names, model_name) != -1)
        {
            return cur;
        }
        ++i;
    }
    assert(false); // Must never get here.
}

static llama_model_params get_model_params(mt_llm_p const & mt_p)
{
    llama_model_params ret_val = llama_model_default_params();

    ret_val.n_gpu_layers = mt_p.n_gpu_layers;
    //ret_val.use_mlock // Default: false.
    //ret_val.use_mmap // Default: true.

    if(mt_p.cpu_moe != 0)
    {
        // Keep all Mixture of Experts (MoE) weights in the CPU.

        // Not sure, if this is the best way to do this:

        s_tensor_buft_overrides[0] = llm_ffn_exps_cpu_override();
    }
    else
    {
        // First entry will act as NULL-terminator:

        s_tensor_buft_overrides[0].pattern = NULL;
        s_tensor_buft_overrides[0].buft = NULL;
    }
    ret_val.tensor_buft_overrides = s_tensor_buft_overrides;

    return ret_val;
}

/**
 * - Caller takes ownership of return value.
 * - Returns null, if given key was not found.
 */
static char* get_meta_val_str(llama_model const & model, char const * const key)
{
    assert(key != nullptr);
    assert(key[0] != '\0');

    char* buf = static_cast<char*>(malloc(
        MT_LLM_MODEL_DEFAULT_META_VAL_STR_MAX_LEN * sizeof *buf));
    int32_t len = -1;

    assert(buf != nullptr);

    len = llama_model_meta_val_str(
            &model, key, buf, MT_LLM_MODEL_DEFAULT_META_VAL_STR_MAX_LEN);
    if(len < 0)
    {
        free(buf);
        return nullptr;
    }
    if(MT_LLM_MODEL_DEFAULT_META_VAL_STR_MAX_LEN - 1 < len)
    {
        int const meta_val_str_max_len = len + 1;

        free(buf);
        buf = static_cast<char*>(
                malloc(meta_val_str_max_len * sizeof *buf));

        assert(buf != nullptr);

        len = llama_model_meta_val_str(
                &model, key, buf, meta_val_str_max_len);
        
        assert(len == meta_val_str_max_len - 1);
    }
    return buf;
}

bool mt_llm_model_try_set_prompts(
    struct llama_model const & model, struct mt_llm_p & p)
{
    char * const model_name = get_meta_val_str(model, MT_LLM_MODEL_NAME_KEY);
    struct prompt_template const * pt = nullptr;

    if(model_name == nullptr)
    {
        MT_LOG("No name found in model data, using given strings.\n");
        return false;
    }

    pt = try_get_prompt_template(model_name);
    if(pt == nullptr)
    {
        MT_LOG(
            "No default prompt strings found for model with name \"%s\", using given strings!\n",
            model_name);
        free(model_name);
        return false;
    }

    MT_LOG(
        "Setting default prompt strings for model with name \"%s\"..\n",
        model_name);

    strcpy(p.sys_prompt_beg_delim, pt->sys_prompt_beg_delim);
    strcpy(p.sys_prompt_mid_delim, pt->sys_prompt_mid_delim);
    strcpy(p.sys_prompt_end_delim, pt->sys_prompt_end_delim);

    strcpy(p.prompt_beg_delim, pt->prompt_beg_delim);
    strcpy(p.prompt_end_delim, pt->prompt_end_delim);

    strcpy(p.rev_prompt, pt->rev_prompt);

    strcpy(p.think_beg_delim, pt->think_beg_delim);
    strcpy(p.think_end_delim, pt->think_end_delim);

    free(model_name);
    return true;
}

std::vector<std::vector<int>> mt_llm_model_get_digit_tokens(
    struct llama_model const & model)
{
    std::vector<std::vector<int>> ret_val;

    ret_val.resize(10); // For the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9.

    llama_vocab const * const vocab = llama_model_get_vocab(&model);
    int32_t const n_vocab = llama_vocab_n_tokens(vocab);

    // Not performance-optimized, better switch loops:
    //
    for(int digit = 0; digit < 10; ++digit)
    {
        std::vector<int> row;

        for(int token_id = 0; token_id < n_vocab; ++token_id)
        {
            char const * const text = vocab->token_get_text(token_id);

            if(is_char_plus_opt_whitespace(text, '0' + digit))
            {
                row.push_back(token_id);
                continue;
            }
        }
        ret_val[digit] = row;
    }
    return ret_val;
}

llama_model* mt_llm_model_create(mt_llm_p const & mt_p)
{
    return llama_model_load_from_file(
        mt_p.model_file_path, get_model_params(mt_p));
}
