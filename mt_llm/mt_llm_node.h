
// Marcel Timm, RhinoDevel, 2026jun07

#ifndef MT_LLM_NODE
#define MT_LLM_NODE

struct mt_llm_node
{
	int index;
	void* data;

	struct mt_llm_node * last;
	struct mt_llm_node * next;
};

/**
 * - Does nothing, if nullptr given.
 * - Does NOT free node's data.
 */
void mt_llm_node_free(mt_llm_node * const node);

/**
 * - Caller takes ownership.
 * - Needs to be freed via mt_llm_node_free(), later.
 */
mt_llm_node* mt_llm_node_create(void);

/**
 * - Just returns nullptr, if nullptr given.
 */
mt_llm_node* mt_llm_node_find(mt_llm_node * const first, int const index);

/**
 * - Also works, if nullptr as "first node" given.
 * - Returns new first node in the linked list, after addition.
 */
mt_llm_node* mt_llm_node_add(
	mt_llm_node * const first, mt_llm_node * const to_add);

/**
 * - Removes node from linked list.
 * - Returns new first node in the linked list, after removal.
 * - Does nothing, if given node is not included in linked list.
 * - Also works, if nullptr as "first node" given.
 */
mt_llm_node* mt_llm_node_remove(
	mt_llm_node * const first, mt_llm_node * const to_remove);

#endif //MT_LLM_NODE
