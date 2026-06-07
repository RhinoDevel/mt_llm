
// Marcel Timm, RhinoDevel, 2026jun06

#include <cassert>
#include <cstdlib>

#include "mt_llm_log.h"
#include "mt_llm_slot.h"

static void free_state(struct mt_llm_slot * const slot)
{
	if(slot->state == nullptr)
	{
		return;
	}
	free(slot->state->state);
	free(slot->state);
	slot->state = nullptr;
}

static struct mt_llm_slot * get_last(struct mt_llm_slot * const first)
{
	struct mt_llm_slot * cur = first;

	while(cur != nullptr)
	{
		if(cur->next == nullptr)
		{
			return cur;
		}
		cur = cur->next;
	}
	assert(cur == nullptr);
	return nullptr;
}

void mt_llm_slot_free(struct mt_llm_slot * const slot)
{
	free_state(slot);
	free(slot);
}

struct mt_llm_slot * mt_llm_slot_find(
    struct mt_llm_slot * const first, int const index)
{
    struct mt_llm_slot * cur = first;

    while(cur != nullptr)
    {
        if(cur->index == index)
        {
            return cur;
        }
        cur = cur->next;
    }
    assert(cur == nullptr);
    return nullptr;
}

struct mt_llm_slot * mt_llm_slot_update(
	struct mt_llm_slot * const first,
	int const index,
	struct mt_llm_state * const state)
{
	struct mt_llm_slot * new_first = first;
	struct mt_llm_slot * slot = nullptr;

	if(index < 0)
	{
		MT_LOG_ERR("Invalid index %d given!\n", index);
		return nullptr;
	}

	slot = mt_llm_slot_find(first, index);

	if(slot == nullptr)
	{
		slot = (struct mt_llm_slot *)malloc(sizeof * slot);
		if(slot == nullptr)
		{
			assert(false); // Must never happen.
			return nullptr;
		}
		slot->last = nullptr;
		slot->next = nullptr;
		slot->state = nullptr;
		slot->index = index;
	}

	free_state(slot);
	slot->state = state;

	if(first == nullptr)
	{
		// Linked list is empty.
		new_first = slot;
	}
	else
	{
		struct mt_llm_slot * const last = get_last(first);

		assert(last != nullptr);
		assert(last->next == nullptr);

		slot->last = last;
		last->next = slot;
	}
	return new_first;
}

struct mt_llm_slot * mt_llm_slot_remove(
	struct mt_llm_slot * const first, int const index, bool const free)
{
	struct mt_llm_slot * new_first = first;
	struct mt_llm_slot * cur = mt_llm_slot_find(first, index);

	if(cur == nullptr)
	{
		MT_LOG("Slot index %d was not found, doing nothing..", index);
		return new_first;
	}

	if(cur->last == nullptr)
	{
		// Found slot is the first entry of linked list.
		new_first = cur->next;
	}
	else
	{
		assert(cur->last->next == cur);
		cur->last->next = cur->next;
	}

	if(cur->next != nullptr)
	{
		assert(cur->next->last == cur);
		cur->next->last = cur->last;
	}

	if(free)
	{
		mt_llm_slot_free(cur);
	}
	return new_first;
}