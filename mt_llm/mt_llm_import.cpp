
// Marcel Timm, RhinoDevel, 2026may14

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include "mt_llm_import.h"
#include "mt_llm.h"
#include "mt_llm_log.h"

MT_EXPORT_LLM_API bool __stdcall mt_llm_import(
	char const * const * const messages,
    int const msg_cnt,
    int const slot_index,
    bool const no_last_prompt_beg_delim)
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

    // (not updating configured system prompt or sampling params. here)
    mt_llm_reset(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, slot_index);

    if(msg_cnt % 2 == 1)
    {
        // Odd message count. <=> First message must be the system prompt.

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
            if(!mt_llm_decode_response(
                    messages[i],
                    slot_index,
                    no_last_prompt_beg_delim && i + 1 == msg_cnt))
            {
                return false;
            }
        }
        return true;
    }

    // Even message count. <=> User and assistant message pairs, only.

    for(i = 0; i < msg_cnt; ++i)
    {
        if(i % 2 == 0)
        {
            // User prompt/query/request.

            if(i == 0)
            {
                if(!mt_llm_decode_prompt_beg_delim(slot_index))
                {
                    return false; // (called function logged)
                }
            }

            if(!mt_llm_decode_request(messages[i], slot_index, false))
            {
                return false; // (called function logged)
            }
            continue;
        }

        // LLM answer/response.
        if(!mt_llm_decode_response(
                messages[i],
                slot_index,
                no_last_prompt_beg_delim && i + 1 == msg_cnt))
        {
            return false;
        }
    }

    return true;
}
