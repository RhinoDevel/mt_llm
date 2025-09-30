
// Marcel Timm, RhinoDevel, 2025sep17

// An example about how to implement a
//
// Speech-To-Text -> Large-Language-Model -> Text-To-Speech
//
// pipeline in C, via mt_stt, mt_llm and mt_tts, which are C/C++ wrapper
// libraries for Whisper.cpp, Llama.cpp and Piper.

#include <stdio.h>
#include <stdlib.h>

#include "mt_tts.h"
#include "mt_stt.h"
#include "mt_llm.h"
#include "mt_llm_p.h"
#include "mt_llm_tok_type.h"

// The text-to-speech model & configuration for Piper:
static char const * const s_tts_onnx_path = "en_GB-jenny_dioco-medium.onnx";
static char const * const s_tts_json_path = "en_GB-jenny_dioco-medium.onnx.json";

// The large-language model for Llama.cpp:
static char const * const s_llm_model_path = "gemma-3-1b-it-Q5_K_M.gguf";
// The system prompt for the LLM:
static char const * const s_llm_sys_prompt = "You are a helpful AI assistant.";

// The speech-to-text model for Whisper.cpp:
static char const * const s_stt_model_path = "ggml-small-q5_1.bin";

// Where the LLM's response will be stored as wave file:
static char const * const s_response_wav_path = "llm_response.wav";

// This will be converted to audio to simulate initial audio input, converted to
// text and send to the LLM to query for an answer: 
static char const * const s_prompt = "Please tell me your name!";

// For a real application, do not restrict the LLM's answer length this way,
// done here to keep this example simple enough:
#define MT_BUF_LEN (2048 + 1) // The max. LLM response length (for simplicity).
static char s_buf[MT_BUF_LEN] = ""; // Filled by llm_callback().
static int s_buf_pos = 0;

/** TTS: Create raw audio data from a text given.
 *
 * - Caller takes ownership of return value, which must be deallocated via
 *   free().
 */
static float* create_audio(
    char const * const text, int * const out_sample_count)
{
    int16_t* tts_result = NULL;

    // Initialize TTS system with a model/voice for output:
    mt_tts_reinit(s_tts_onnx_path, s_tts_json_path);

    // Get the actual raw audio data:
    tts_result = mt_tts_to_raw(text, out_sample_count);

    // Convert the audio data into normalized floating-point representation:

    float * const ret_val = malloc(*out_sample_count * sizeof *ret_val);

    for(int i = 0; i < *out_sample_count; ++i)
    {
        ret_val[i] = (float)tts_result[i] / 16384.0f;
    }

    // Free memory and de-initialize TTS system:

    mt_tts_free_raw(tts_result);
    tts_result = NULL;

    mt_tts_deinit();

    return ret_val;
}

static bool llm_callback(
    int tok,
    char const * piece,
    int type,
    float const * dig_probs)
{
    if(type == MT_TOK_TYPE_SAMPLED_NON_EOG_NON_CONTROL)
    {
        // Append piece to the buffer holding the LLM's answer.
        // Stop inference early, if buffer is filled.

        char const * piece_ptr = piece;

        if(s_buf_pos == MT_BUF_LEN - 1)
        {
            s_buf[s_buf_pos] = '\0';
            return true; // Stop inference.
        }
        do
        {
            char const cur_char = *piece_ptr;

            s_buf[s_buf_pos] = cur_char;
            if(cur_char == '\0')
            {
                break; // Appended current piece completely.
            }
            ++s_buf_pos;
            ++piece_ptr;
        }while(true);
    }
    else
    {
        if(type == MT_TOK_TYPE_SAMPLED_EOG)
        {
            printf("\n\n");
        }
    }
    return false;
}

/** Example use of mt_stt, mt_llm and mt_tts together to demonstrate the
 *  possibility of a Speech-To-Text -> Large-Language Model -> Text-To-Speech
 *  pipeline in C.
 */
int main(void)
{
    int sample_count = -1;
    float* stt_input = NULL;
    char* llm_input = NULL;
    struct mt_llm_p p;

    // *************************************************************************
    // *** TTS: Create sample audio input via TTS:                           ***
    // *************************************************************************

    // Get the audio data (a placeholder for real audio, e.g. from your
    // microphone):
    stt_input = create_audio(s_prompt, &sample_count);
    // For a real application you'd want to load audio data from a WAV file or
    // read it directly from your microphone.

    // *************************************************************************
    // *** STT: Transcribe the audio to text:                                ***
    // *************************************************************************
    
    llm_input = mt_stt_transcribe_with_file(
        false,
        4,
        NULL,
        false,
        NULL,
        s_stt_model_path,
        stt_input,
        sample_count,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        0);

    free(stt_input);
    stt_input = NULL;

    printf("STT result: \"%s\"\n", llm_input);
    
    // *********************************
    // *** Setup the LLM parameters: ***
    // *********************************
    
    p.n_gpu_layers = 0;
    
    p.seed = -1;
    p.n_ctx = 2048;
    p.threads = 0;
    
    p.top_k = 40;
    p.top_p = 0.95;
    p.min_p = 0.05;
    p.temp = 0.8;
    p.grammar[0] = '\0';
    
    strncpy(p.model_file_path, s_llm_model_path, MT_LLM_P_LEN_MODEL_FILE_PATH);
    strncpy(p.sys_prompt, s_llm_sys_prompt, MT_LLM_P_LEN_SYS_PROMPT);
    
    // These will be automatically set by mt_llm (if model is supported..):
    p.prompt_beg_delim[0] = '\0';
    p.prompt_end_delim[0] = '\0';
    p.sys_prompt_beg_delim[0] = '\0';
    p.sys_prompt_mid_delim[0] = '\0';
    p.sys_prompt_end_delim[0] = '\0';
    p.rev_prompt[0] = '\0';
    p.think_beg_delim[0] = '\0';
    p.think_end_delim[0] = '\0';

    p.try_prompts_by_model = true;

    p.callback = llm_callback;

    // **************************
    // *** Initialize mt_llm: ***
    // **************************

    mt_llm_reinit(&p); // Ignoring return value, here..
    
    // **********************
    // *** Query the LLM: ***
    // **********************
    
    mt_llm_query(llm_input);
    
    // (inference is running here, and will call the callback for each token)

    free(llm_input);
    llm_input = NULL;

    // ****************************
    // *** Deinitialize mt_llm: ***
    // ****************************

    mt_llm_deinit();

    printf("LLM's answer: \"%s\"\n", s_buf);

    // *************************************************************************
    // *** TTS: Convert LLM's answer into speech, save as WAV file:          ***
    // *************************************************************************

    mt_tts_reinit(s_tts_onnx_path, s_tts_json_path);

    // Optionally use something like create_audio() and send the audio data
    // directly to speakers for output.
    mt_tts_to_wav_file(s_buf, s_response_wav_path);

    mt_tts_deinit();

    printf("Play the WAV file and listen to the LLM's answer! :-)\n");

    return 0;
}