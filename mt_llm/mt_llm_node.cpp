
// Marcel Timm, RhinoDevel, 2026jun07

#include <cassert>
#include <cstdlib>

#include "mt_llm_log.h"
#include "mt_llm_node.h"

static mt_llm_node* get_last(mt_llm_node * const first)
{
	mt_llm_node* cur = first;

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

void mt_llm_node_free(mt_llm_node * const node)
{
	if(node == nullptr)
	{
		return;
	}

	assert(node->data == nullptr); // Ignoring node->data, here.

	free(node);
}

mt_llm_node* mt_llm_node_create(void)
{
	mt_llm_node * const node = static_cast<mt_llm_node*>(malloc(sizeof * node));

	if(node == nullptr)
	{
		assert(false); // Must never happen.
		return nullptr;
	}
	node->last = nullptr;
	node->next = nullptr;
	node->data = nullptr;
	node->index = -1;
	return node;
}

struct mt_llm_node * mt_llm_node_find(
	struct mt_llm_node * const first, int const index)
{
	mt_llm_node* cur = first;

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

mt_llm_node* mt_llm_node_add(
	mt_llm_node * const first, mt_llm_node * const to_add)
{
	if(first == nullptr)
	{
		// Linked list is empty.
		to_add->last = nullptr;
		to_add->next = nullptr;
		return to_add;
	}

	// Linked list is not empty.

	mt_llm_node * const last = get_last(first);

	assert(last != nullptr);
	assert(last->next == nullptr);

	to_add->last = last;
	last->next = to_add;

	return first;
}

mt_llm_node* mt_llm_node_remove(
	mt_llm_node * const first, mt_llm_node * const to_remove)
{
	mt_llm_node* new_first = first;
	mt_llm_node* cur = mt_llm_node_find(first, to_remove->index);

	if(cur == nullptr)
	{
		MT_LOG(
			"A node with index %d was not found, doing nothing..",
			to_remove->index);
		return new_first/*first*/;
	}

	assert(cur == to_remove); // Should be the same object.

	if(cur->last == nullptr)
	{
		// Found node is the first entry of linked list.
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

	to_remove->last = nullptr;
	to_remove->next = nullptr;
	return new_first;
}