
// Marcel Timm, RhinoDevel, 2026jun06

#ifndef MT_LLM_SLOT
#define MT_LLM_SLOT

#include "mt_llm_state.h"

struct mt_llm_slot
{
	int index;
	struct mt_llm_state * state;

	struct mt_llm_slot * last;
	struct mt_llm_slot * next;
};

void mt_llm_slot_free(struct mt_llm_slot * const slot);

/**
 * - If existing, updates slot with given index (also frees existing state).
 * - If not existing, creates slot with given index and insert it into the
 *   linked list.
 * - Returns nullptr, if given index is invalid.
 * - Returns new first slot on success.
 */
struct mt_llm_slot * mt_llm_slot_update(
	struct mt_llm_slot * const first,
	int const index,
	struct mt_llm_state * const state);

struct mt_llm_slot * mt_llm_slot_find(
	struct mt_llm_slot * const first, int const index);

/**
 * - Removes slot from linked list.
 * - Returns new first slot in the linked list, after removal.
 * - Optionally frees memory of slot.
 * - Does nothing, if slot with given index was not found.
 */
struct mt_llm_slot * mt_llm_slot_remove(
	struct mt_llm_slot * const first, int const index, bool const free);

#endif //MT_LLM_SLOT
