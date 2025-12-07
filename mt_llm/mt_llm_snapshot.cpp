
// Marcel Timm, RhinoDevel, 2025feb23

#include <cstdint>
#include <string>
#include <cassert>

#include "mt_llm_snapshot.h"
#include "mt_llm_state.h"
#include "mt_llm.h"
#include "mt_llm_log.h"

// Hard-coded for a maximum of two LLMs, see mt_llm!
static mt_llm_state * s_snapshots[] = { nullptr, nullptr };

static bool snapshot_clear(int const slot_index)
{
	if(slot_index != 0 && slot_index != 1)
	{
		MT_LOG_ERR("Invalid snapshot index given!\n");
		return false;
	}

	if(s_snapshots[slot_index] != nullptr)
	{
		free(s_snapshots[slot_index]->state);
		s_snapshots[slot_index]->state = nullptr;
		free(s_snapshots[slot_index]);
		s_snapshots[slot_index] = nullptr;
	}
	return true;
}

MT_EXPORT_LLM_API void __stdcall mt_llm_snapshot_clear(int const slot_index)
{
	if(!snapshot_clear(slot_index))
	{
		return;
	}
	MT_LOG("Cleared state at slot with index %d.\n", slot_index);
}

MT_EXPORT_LLM_API bool mt_llm_snapshot_restore(int const slot_index)
{
	if(slot_index != 0 && slot_index != 1)
	{
		MT_LOG_ERR("Invalid snapshot index given!\n");
		return false;
	}

	if(s_snapshots[slot_index] == nullptr)
	{
		MT_LOG_ERR("No snapshot was taken!");
		return false;
	}

	assert(0 < s_snapshots[slot_index]->size);

	return mt_llm_state_restore(
		s_snapshots[slot_index], slot_index); // (logs on error)
}

MT_EXPORT_LLM_API bool mt_llm_snapshot_update(int const slot_index)
{
	if(slot_index != 0 && slot_index != 1)
	{
		MT_LOG_ERR("Invalid snapshot index given!\n");
		return false;
	}

	mt_llm_snapshot_clear(slot_index);

	assert(s_snapshots[slot_index] == nullptr);

	mt_llm_state * const state =
		mt_llm_state_create(slot_index); // (logs on error)

	if(state == nullptr)
	{
		return false;
	}

	assert(0 < state->size);

	s_snapshots[slot_index] = state;
	return true;
}
