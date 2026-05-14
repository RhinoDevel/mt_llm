
// Marcel Timm, RhinoDevel, 2026may14

#include <cstdint>
#include <string>
#include <cassert>

#include "mt_llm_import.h"
#include "mt_llm.h"
#include "mt_llm_log.h"

MT_EXPORT_LLM_API bool __stdcall mt_llm_import(
	char const * const * const messages,
    int const msg_cnt,
    int const slot_index)
{
    int i = -1;

    if(messages == nullptr)
    {
        MT_LOG_ERR("No messages given!\n");
        return false;
    }
    if(msg_cnt < 1)
    {
        MT_LOG_ERR("Invalid message count given (0 or lower)!\n");
        return false;
    }
    while(++i < msg_cnt)
    {
        if(messages[i] == nullptr)
        {
            MT_LOG_ERR("Message at index %d is \"unset\" (null)!\n", i);
            return false;
        }
        if(strnlen(messages[i], MT_LLM_STR_LIM) == MT_LLM_STR_LIM)
        {
            MT_LOG_ERR("Message at index %d is a too long C-string!\n", i);
            return false;
        }
    }

    // (not updating configured system prompt, here)
    mt_llm_reset(nullptr, slot_index);

    i = 0;
    if(!mt_llm_decode_sys_prompt(messages[i], slot_index))
    {
        return false; // (called function logged)
    }
    while(++i < msg_cnt)
    {
        if(i % 2 == 1)
        {
            // User prompt/query/request.
            if(!mt_llm_decode_request(messages[i], slot_index, i == 1))
            {
                return false; // (called function logged)
            }
            continue;
        }

        // LLM answer/response.
        if(!mt_llm_decode_response(messages[i], slot_index))
        {
            return false;
        }
    }
    return true;
}
