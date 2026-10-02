#ifndef SCALA_OLLAMA_DIRECT_H
#define SCALA_OLLAMA_DIRECT_H

#include <stdint.h>

#if defined(_WIN32)
#define SOD_API __declspec(dllexport)
#else
#define SOD_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void * sod_handle;

SOD_API sod_handle sod_create(void);

SOD_API int32_t sod_load(
    sod_handle handle,
    const char * packed_model_paths,
    int32_t model_path_count,
    int32_t context_size,
    int32_t batch_size,
    int32_t gpu_layers,
    float temperature,
    float min_p,
    uint32_t seed);

SOD_API int32_t sod_generate(
    sod_handle handle,
    const char * prompt,
    int32_t max_tokens);

SOD_API int32_t sod_chat(
    sod_handle handle,
    const char * message,
    int32_t max_tokens);

SOD_API int32_t sod_set_system_prompt(
    sod_handle handle,
    const char * prompt);

SOD_API int32_t sod_reset(sod_handle handle);

SOD_API int64_t sod_response_size(sod_handle handle);

SOD_API int64_t sod_copy_response(
    sod_handle handle,
    uint8_t * destination,
    int64_t capacity);

SOD_API int64_t sod_error_size(sod_handle handle);

SOD_API int64_t sod_copy_error(
    sod_handle handle,
    uint8_t * destination,
    int64_t capacity);

SOD_API void sod_close(sod_handle handle);

#ifdef __cplusplus
}
#endif

#endif
