
// Marcel Timm, RhinoDevel, 2024aug28

#ifndef MT_LLM_TOK_TYPE
#define MT_LLM_TOK_TYPE

// Could be an enum:

// Tokens represent a user prompt.
#define MT_TOK_TYPE_PROMPT 1

// Tokens to-be-added on (user) interrupt request to stop inference.
#define MT_TOK_TYPE_IRQ 2

// Tokens to be send AFTER the reverse prompt was already send once [as another
// token type, see mt_llm.cpp/inference()], the callback/client code must handle
// this (reverse prompt being given twice).
#define MT_TOK_TYPE_REV_PROMPT 3

// The string representation of these tokens are meant to be seen by the user.
// Thinking mode tokens (and their begin and end tags/delimiters) have their own
// token type.
#define MT_TOK_TYPE_SAMPLED_NON_EOG_NON_CONTROL 4

// An EOG (end-of-generation) token, e.g. EOS or EOT.
#define MT_TOK_TYPE_SAMPLED_EOG 5

// For prompt delimiters, used by/for instruction-tuned models.
#define MT_TOK_TYPE_DELIM 6

// Tokens of the actual system prompt (text).
#define MT_TOK_TYPE_SYS_PROMPT 7

// Non-EOG control tokens.
#define MT_TOK_TYPE_SAMPLED_CONTROL_NON_EOG 8

// Sampled tokens that represent the thinking mode text or the begin or end
// delimiter.
#define MT_TOK_TYPE_SAMPLED_THINK 9

#endif //MT_LLM_TOK_TYPE