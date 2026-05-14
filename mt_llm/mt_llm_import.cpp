
// Marcel Timm, RhinoDevel, 2026may14

#include <cstdint>
#include <string>
#include <cassert>

#include "mt_llm_import.h"
#include "mt_llm_log.h"

MT_EXPORT_LLM_API bool __stdcall mt_llm_import(
	char const * const * const messages,
    int const messages_cnt,
    int const slot_index)
{
    if(messages == nullptr)
    {
        MT_LOG_ERR("No messages given!\n");
        return false;
    }
    if(messages_cnt < 1)
    {
        MT_LOG_ERR("Invalid message count given (0 or lower)!\n");
        return false;
    }
    for(int i = 0; i < messages_cnt; ++i)
    {
        if(messages[i] == nullptr)
        {
            MT_LOG_ERR("Message at index %d is \"unset\" (null)!\n", i);
            return false;
        }
        if(strnlen(messages[i], 65535) == 65535) // <- Hard-coded limit.
        {
            MT_LOG_ERR("Message at index %d is a too long C-string!\n", i);
            return false;
        }
    }

    if(slot_index != 0 && slot_index != 1)
    {
        MT_LOG_ERR("Unsupported slot index given, doing nothing.\n");
        return false;
    }

	return false; // TODO: Implement!
}
