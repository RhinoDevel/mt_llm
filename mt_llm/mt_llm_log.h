
// Marcel Timm, RhinoDevel, 2024aug28

#ifndef MT_LLM_LOG
#define MT_LLM_LOG

#define MT_LOG_ERR(fmt, ...) \
    fprintf(stderr, "%s : ERROR: " fmt, __func__, ##__VA_ARGS__)

#define MT_LOG(fmt, ...) \
    printf("%s : " fmt, __func__, ##__VA_ARGS__)

#endif //MT_LLM_LOG
