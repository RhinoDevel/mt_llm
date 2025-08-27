
// Marcel Timm, RhinoDevel, 2025feb23

#include <cstdint>
#include <string>
#include <cassert>

#include "mt_llm_snapshot.h"
#include "mt_llm_state.h"
#include "mt_llm.h"
#include "mt_llm_log.h"

static mt_llm_state * s_snapshot = nullptr;

MT_EXPORT_LLM_API void __stdcall mt_llm_snapshot_clear()
{
	if(s_snapshot != nullptr)
	{
		free(s_snapshot->state);
		s_snapshot->state = nullptr;
		free(s_snapshot);
		s_snapshot = nullptr;
	}
}

MT_EXPORT_LLM_API bool mt_llm_snapshot_restore()
{
	if(s_snapshot == nullptr)
	{
		MT_LOG_ERR("No snapshot was taken!");
		return false;
	}

	assert(0 < s_snapshot->size);

	return mt_llm_state_restore(s_snapshot); // (logs on error)
}

MT_EXPORT_LLM_API bool mt_llm_snapshot_update()
{
	mt_llm_snapshot_clear();

	assert(s_snapshot == nullptr);

	mt_llm_state * const state = mt_llm_state_create(); // (logs on error)

	if(state == nullptr)
	{
		return false;
	}

	assert(0 < state->size);

	s_snapshot = state;
	return true;
}
