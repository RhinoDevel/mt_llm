
// Marcel Timm, RhinoDevel, 2025feb23

#include <cstdint>
#include <string>
#include <cassert>

#include "mt_llm_snapshot.h"
#include "mt_llm_state.h"
#include "mt_llm_slot.h"
#include "mt_llm.h"
#include "mt_llm_log.h"

static mt_llm_slot * s_snapshots = nullptr;

MT_EXPORT_LLM_API void __stdcall mt_llm_snapshot_clear(int const slot_index)
{
	s_snapshots = mt_llm_slot_remove(s_snapshots, slot_index, true);
	MT_LOG("State at slot with index %d is cleared.\n", slot_index);
}

MT_EXPORT_LLM_API bool mt_llm_snapshot_restore(int const slot_index)
{
	struct mt_llm_slot const * const slot = mt_llm_slot_find(
		s_snapshots, slot_index);

	if(slot == nullptr)
	{
		MT_LOG_ERR(
			"No snapshot was taken for slot with index %d!\n", slot_index);
		return false;
	}

	assert(0 < slot->state->size);

	return mt_llm_state_restore(slot->state, slot_index); // (logs on error)
}

MT_EXPORT_LLM_API bool mt_llm_snapshot_update(int const slot_index)
{
	mt_llm_snapshot_clear(slot_index);

	mt_llm_state * const state =
		mt_llm_state_create(slot_index); // (logs on error)

	if(state == nullptr)
	{
		return false;
	}

	assert(0 < state->size);

	s_snapshots = mt_llm_slot_update(s_snapshots, slot_index, state);
	return true;
}

MT_EXPORT_LLM_API bool mt_llm_snapshot_to_file(
	int const slot_index, char const * const abs_file_path)
{
	FILE* fp = nullptr;
	size_t bytes_to_write = 0;
	size_t items_written = 0;

	struct mt_llm_slot const * const slot = mt_llm_slot_find(
		s_snapshots, slot_index);

	if(slot == nullptr)
	{
		MT_LOG_ERR(
			"No snapshot was taken for slot with index %d!\n", slot_index);
		return false;
	}
	if(abs_file_path == nullptr)
	{
		MT_LOG_ERR("Invalid absolute file path given!\n");
		return false;
	}

	mt_llm_state * const state = slot->state;

	fp = fopen(abs_file_path, "wb");
	if(fp == nullptr)
	{
		MT_LOG_ERR(
			"Failed to open file at \"%s\" for binary write!\n", abs_file_path);
		return false;
	}

	bytes_to_write = sizeof state->last_tok_type;
	items_written = fwrite(&(state->last_tok_type), bytes_to_write, 1, fp);
	if(items_written != 1)
	{
		MT_LOG_ERR("Not all bytes were written (1)!\n");
		fclose(fp);
		fp = nullptr;
		return false;
	}

	bytes_to_write = sizeof state->tok_cnt;
	items_written = fwrite(&(state->tok_cnt), bytes_to_write, 1, fp);
	if(items_written != 1)
	{
		MT_LOG_ERR("Not all bytes were written (2)!\n");
		fclose(fp);
		fp = nullptr;
		return false;
	}

	bytes_to_write = sizeof state->size;
	items_written = fwrite(&(state->size), bytes_to_write, 1, fp);
	if(items_written != 1)
	{
		MT_LOG_ERR("Not all bytes were written (3)!\n");
		fclose(fp);
		fp = nullptr;
		return false;
	}

	assert(sizeof *(state->state) == 1);
	bytes_to_write = state->size;
	items_written = fwrite(state->state, 1, bytes_to_write, fp);
	if(items_written != bytes_to_write)
	{
		MT_LOG_ERR("Not all bytes were written (4)!\n");
		fclose(fp);
		fp = nullptr;
		return false;
	}

	fclose(fp);
	fp = nullptr;
	return true;
}

MT_EXPORT_LLM_API bool mt_llm_snapshot_from_file(
	char const * const abs_file_path, int const slot_index)
{
	FILE* fp = nullptr;
	size_t bytes_to_read = 0;
	size_t items_read = 0;
	mt_llm_state * state = nullptr;

	if(abs_file_path == nullptr)
	{
		MT_LOG_ERR("Invalid absolute file path given!\n");
		return false;
	}

	mt_llm_snapshot_clear(slot_index);

	fp = fopen(abs_file_path, "rb");
	if(fp == nullptr)
	{
		MT_LOG_ERR("Failed to open source file \"%s\"!\n", abs_file_path);
		return false;
	}

	state = static_cast<mt_llm_state*>(malloc(sizeof * state));
	if(state == nullptr)
	{
		MT_LOG_ERR("Failed to allocate state object!\n");
		fclose(fp);
		fp = nullptr;
		return false;
	}

	bytes_to_read = sizeof state->last_tok_type;
	items_read = fread(&state->last_tok_type, bytes_to_read, 1, fp);
	if(items_read != 1)
	{
		MT_LOG_ERR("Failed to read from file at \"%s\" (1)!\n", abs_file_path);
		fclose(fp);
		fp = nullptr;
		free(state);
		state = nullptr;
		return false;
	}

	bytes_to_read = sizeof state->tok_cnt;
	items_read = fread(&state->tok_cnt, bytes_to_read, 1, fp);
	if(items_read != 1)
	{
		MT_LOG_ERR("Failed to read from file at \"%s\" (2)!\n", abs_file_path);
		fclose(fp);
		fp = nullptr;
		free(state);
		state = nullptr;
		return false;
	}

	bytes_to_read = sizeof state->size;
	items_read = fread(&state->size, bytes_to_read, 1, fp);
	if(items_read != 1)
	{
		MT_LOG_ERR("Failed to read from file at \"%s\" (3)!\n", abs_file_path);
		fclose(fp);
		fp = nullptr;
		free(state);
		state = nullptr;
		return false;
	}
	if(state->size == 0)
	{
		MT_LOG_ERR("Read byte count of zero!\n");
		fclose(fp);
		fp = nullptr;
		free(state);
		state = nullptr;
		return false;
	}

	assert(sizeof *(state->state) == 1);
	state->state = static_cast<uint8_t*>(malloc(state->size));
	if(state->state == nullptr)
	{
		MT_LOG_ERR("Failed to allocate state's state object!\n");
		fclose(fp);
		fp = nullptr;
		free(state);
		state = nullptr;
		return false;
	}

	assert(sizeof *(state->state) == 1);
	bytes_to_read = state->size;
	items_read = fread(state->state, 1, bytes_to_read, fp);
	if(items_read != bytes_to_read)
	{
		MT_LOG_ERR("Failed to read from file at \"%s\" (4)!\n", abs_file_path);
		fclose(fp);
		fp = nullptr;
		free(state->state);
		state->state = nullptr;
		free(state);
		state = nullptr;
		return false;
	}

	fclose(fp);
	fp = nullptr;

	s_snapshots = mt_llm_slot_update(s_snapshots, slot_index, state);
	return true;
}