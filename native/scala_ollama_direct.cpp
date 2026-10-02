#include "scala_ollama_direct.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Message {
    std::string role;
    std::string content;
};

struct Engine {
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_context * context = nullptr;
    llama_sampler * sampler = nullptr;

    int32_t batch_size = 512;
    std::vector<Message> messages;
    int32_t previous_formatted_length = 0;

    std::string last_response;
    std::string last_error;
    std::mutex mutex;

    ~Engine() {
        if (sampler != nullptr) {
            llama_sampler_free(sampler);
        }
        if (context != nullptr) {
            llama_free(context);
        }
        if (model != nullptr) {
            llama_model_free(model);
        }
    }
};

std::once_flag backend_once;

void initialize_backend() {
    std::call_once(backend_once, []() {
        llama_log_set([](enum ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                std::fputs(text, stderr);
            }
        }, nullptr);
        llama_backend_init();
    });
}

Engine * engine_from(sod_handle handle) {
    if (handle == nullptr) {
        throw std::runtime_error("Null llama engine handle");
    }
    return static_cast<Engine *>(handle);
}

std::vector<const char *> unpack_paths(
    const char * packed,
    int32_t count) {

    if (packed == nullptr) {
        throw std::runtime_error("Model path buffer is null");
    }
    if (count <= 0) {
        throw std::runtime_error("At least one model path is required");
    }

    std::vector<const char *> paths;
    paths.reserve(static_cast<size_t>(count));

    const char * cursor = packed;
    for (int32_t i = 0; i < count; ++i) {
        if (*cursor == '\0') {
            throw std::runtime_error("Encountered an empty model path");
        }
        paths.push_back(cursor);
        cursor += std::strlen(cursor) + 1;
    }
    return paths;
}

int32_t used_context_tokens(Engine * engine) {
    const llama_pos max_pos = llama_memory_seq_pos_max(
        llama_get_memory(engine->context),
        0);
    return max_pos < 0 ? 0 : max_pos + 1;
}

std::vector<llama_token> tokenize(
    Engine * engine,
    const std::string & text,
    bool add_special) {

    int32_t required = llama_tokenize(
        engine->vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        add_special,
        true);

    if (required == 0) {
        return {};
    }
    if (required > 0) {
        throw std::runtime_error("Unexpected llama_tokenize sizing result");
    }

    std::vector<llama_token> tokens(static_cast<size_t>(-required));
    const int32_t actual = llama_tokenize(
        engine->vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        true);

    if (actual < 0) {
        throw std::runtime_error("llama_tokenize failed");
    }
    tokens.resize(static_cast<size_t>(actual));
    return tokens;
}

std::string token_to_piece(Engine * engine, llama_token token) {
    std::vector<char> buffer(256);
    int32_t length = llama_token_to_piece(
        engine->vocab,
        token,
        buffer.data(),
        static_cast<int32_t>(buffer.size()),
        0,
        true);

    if (length < 0) {
        buffer.resize(static_cast<size_t>(-length));
        length = llama_token_to_piece(
            engine->vocab,
            token,
            buffer.data(),
            static_cast<int32_t>(buffer.size()),
            0,
            true);
    }

    if (length < 0) {
        throw std::runtime_error("llama_token_to_piece failed");
    }

    return std::string(buffer.data(), static_cast<size_t>(length));
}

void decode_tokens(Engine * engine, std::vector<llama_token> & tokens) {
    size_t offset = 0;

    while (offset < tokens.size()) {
        const size_t remaining = tokens.size() - offset;
        const int32_t count = static_cast<int32_t>(std::min(
            remaining,
            static_cast<size_t>(engine->batch_size)));

        llama_batch batch = llama_batch_get_one(tokens.data() + offset, count);
        const int32_t result = llama_decode(engine->context, batch);

        if (result != 0) {
            throw std::runtime_error(
                "llama_decode failed while processing prompt, result = " +
                std::to_string(result));
        }

        offset += static_cast<size_t>(count);
    }
}

std::string generate_append(
    Engine * engine,
    const std::string & prompt,
    int32_t max_tokens) {

    const bool first = used_context_tokens(engine) == 0;
    std::vector<llama_token> prompt_tokens = tokenize(engine, prompt, first);

    if (prompt_tokens.empty()) {
        throw std::runtime_error("Prompt produced no tokens");
    }

    const int32_t context_size = static_cast<int32_t>(llama_n_ctx(engine->context));
    const int32_t used = used_context_tokens(engine);

    if (used + static_cast<int32_t>(prompt_tokens.size()) > context_size) {
        throw std::runtime_error("Prompt exceeds remaining context capacity");
    }

    decode_tokens(engine, prompt_tokens);

    std::string response;

    for (int32_t generated = 0; generated < max_tokens; ++generated) {
        llama_token token = llama_sampler_sample(
            engine->sampler,
            engine->context,
            -1);

        if (llama_vocab_is_eog(engine->vocab, token)) {
            break;
        }

        response += token_to_piece(engine, token);

        if (used_context_tokens(engine) >= context_size) {
            break;
        }

        llama_batch batch = llama_batch_get_one(&token, 1);
        const int32_t result = llama_decode(engine->context, batch);

        if (result != 0) {
            throw std::runtime_error(
                "llama_decode failed while generating, result = " +
                std::to_string(result));
        }
    }

    return response;
}

std::string apply_chat_template(Engine * engine, bool add_assistant) {
    std::vector<llama_chat_message> native_messages;
    native_messages.reserve(engine->messages.size());

    for (const Message & message : engine->messages) {
        native_messages.push_back({message.role.c_str(), message.content.c_str()});
    }

    const char * chat_template = llama_model_chat_template(engine->model, nullptr);

    int32_t required = llama_chat_apply_template(
        chat_template,
        native_messages.data(),
        native_messages.size(),
        add_assistant,
        nullptr,
        0);

    if (required < 0) {
        throw std::runtime_error("llama_chat_apply_template failed");
    }

    std::vector<char> buffer(static_cast<size_t>(required) + 1);
    const int32_t actual = llama_chat_apply_template(
        chat_template,
        native_messages.data(),
        native_messages.size(),
        add_assistant,
        buffer.data(),
        static_cast<int32_t>(buffer.size()));

    if (actual < 0) {
        throw std::runtime_error("llama_chat_apply_template failed");
    }

    return std::string(buffer.data(), static_cast<size_t>(actual));
}

void reset_engine(Engine * engine) {
    if (engine->context != nullptr) {
        llama_memory_clear(llama_get_memory(engine->context), true);
    }
    if (engine->sampler != nullptr) {
        llama_sampler_reset(engine->sampler);
    }
    engine->messages.clear();
    engine->previous_formatted_length = 0;
    engine->last_response.clear();
}

int64_t copy_string(
    const std::string & source,
    uint8_t * destination,
    int64_t capacity) {

    const int64_t required = static_cast<int64_t>(source.size());
    if (capacity < required) {
        return -required;
    }
    if (required > 0 && destination != nullptr) {
        std::memcpy(destination, source.data(), static_cast<size_t>(required));
    }
    return required;
}

} // namespace

extern "C" {

SOD_API sod_handle sod_create(void) {
    try {
        initialize_backend();
        return new Engine();
    } catch (...) {
        return nullptr;
    }
}

SOD_API int32_t sod_load(
    sod_handle handle,
    const char * packed_model_paths,
    int32_t model_path_count,
    int32_t context_size,
    int32_t batch_size,
    int32_t gpu_layers,
    float temperature,
    float min_p,
    uint32_t seed) {

    Engine * engine = nullptr;
    try {
        engine = engine_from(handle);
        std::lock_guard<std::mutex> guard(engine->mutex);
        engine->last_error.clear();

        if (engine->model != nullptr) {
            throw std::runtime_error("A model is already loaded in this engine");
        }

        auto paths = unpack_paths(packed_model_paths, model_path_count);

        engine->batch_size = batch_size;

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = gpu_layers;

        if (paths.size() == 1) {
            engine->model = llama_model_load_from_file(paths.front(), model_params);
        } else {
            engine->model = llama_model_load_from_splits(
                paths.data(),
                paths.size(),
                model_params);
        }

        if (engine->model == nullptr) {
            throw std::runtime_error("Unable to load the GGUF model layer(s)");
        }

        engine->vocab = llama_model_get_vocab(engine->model);
        if (engine->vocab == nullptr) {
            throw std::runtime_error("Unable to obtain model vocabulary");
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(context_size);
        context_params.n_batch = static_cast<uint32_t>(batch_size);
        context_params.n_ubatch = static_cast<uint32_t>(batch_size);

        engine->context = llama_init_from_model(engine->model, context_params);
        if (engine->context == nullptr) {
            throw std::runtime_error("Unable to create llama context");
        }

        engine->sampler = llama_sampler_chain_init(
            llama_sampler_chain_default_params());
        if (engine->sampler == nullptr) {
            throw std::runtime_error("Unable to create sampler chain");
        }

        llama_sampler_chain_add(
            engine->sampler,
            llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(
            engine->sampler,
            llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(
            engine->sampler,
            llama_sampler_init_dist(seed));

        return 0;
    } catch (const std::exception & exception) {
        if (engine != nullptr) {
            engine->last_error = exception.what();
        }
        return -1;
    }
}

SOD_API int32_t sod_generate(
    sod_handle handle,
    const char * prompt,
    int32_t max_tokens) {

    Engine * engine = nullptr;
    try {
        engine = engine_from(handle);
        std::lock_guard<std::mutex> guard(engine->mutex);
        engine->last_error.clear();
        reset_engine(engine);

        if (prompt == nullptr) {
            throw std::runtime_error("Prompt is null");
        }

        engine->last_response = generate_append(engine, prompt, max_tokens);
        return 0;
    } catch (const std::exception & exception) {
        if (engine != nullptr) {
            engine->last_error = exception.what();
        }
        return -1;
    }
}

SOD_API int32_t sod_chat(
    sod_handle handle,
    const char * message,
    int32_t max_tokens) {

    Engine * engine = nullptr;
    try {
        engine = engine_from(handle);
        std::lock_guard<std::mutex> guard(engine->mutex);
        engine->last_error.clear();

        if (message == nullptr) {
            throw std::runtime_error("Chat message is null");
        }

        engine->messages.push_back({"user", message});

        const std::string complete_prompt = apply_chat_template(engine, true);
        if (engine->previous_formatted_length >
            static_cast<int32_t>(complete_prompt.size())) {
            throw std::runtime_error("Invalid incremental chat-template state");
        }

        const std::string incremental_prompt = complete_prompt.substr(
            static_cast<size_t>(engine->previous_formatted_length));

        engine->last_response = generate_append(
            engine,
            incremental_prompt,
            max_tokens);

        engine->messages.push_back({"assistant", engine->last_response});

        const std::string completed_chat = apply_chat_template(engine, false);
        engine->previous_formatted_length =
            static_cast<int32_t>(completed_chat.size());

        return 0;
    } catch (const std::exception & exception) {
        if (engine != nullptr) {
            engine->last_error = exception.what();
        }
        return -1;
    }
}

SOD_API int32_t sod_set_system_prompt(
    sod_handle handle,
    const char * prompt) {

    Engine * engine = nullptr;
    try {
        engine = engine_from(handle);
        std::lock_guard<std::mutex> guard(engine->mutex);
        engine->last_error.clear();
        reset_engine(engine);

        if (prompt != nullptr && prompt[0] != '\0') {
            engine->messages.push_back({"system", prompt});
        }

        return 0;
    } catch (const std::exception & exception) {
        if (engine != nullptr) {
            engine->last_error = exception.what();
        }
        return -1;
    }
}

SOD_API int32_t sod_reset(sod_handle handle) {
    Engine * engine = nullptr;
    try {
        engine = engine_from(handle);
        std::lock_guard<std::mutex> guard(engine->mutex);
        engine->last_error.clear();
        reset_engine(engine);
        return 0;
    } catch (const std::exception & exception) {
        if (engine != nullptr) {
            engine->last_error = exception.what();
        }
        return -1;
    }
}

SOD_API int64_t sod_response_size(sod_handle handle) {
    Engine * engine = engine_from(handle);
    std::lock_guard<std::mutex> guard(engine->mutex);
    return static_cast<int64_t>(engine->last_response.size());
}

SOD_API int64_t sod_copy_response(
    sod_handle handle,
    uint8_t * destination,
    int64_t capacity) {

    Engine * engine = engine_from(handle);
    std::lock_guard<std::mutex> guard(engine->mutex);
    return copy_string(engine->last_response, destination, capacity);
}

SOD_API int64_t sod_error_size(sod_handle handle) {
    Engine * engine = engine_from(handle);
    std::lock_guard<std::mutex> guard(engine->mutex);
    return static_cast<int64_t>(engine->last_error.size());
}

SOD_API int64_t sod_copy_error(
    sod_handle handle,
    uint8_t * destination,
    int64_t capacity) {

    Engine * engine = engine_from(handle);
    std::lock_guard<std::mutex> guard(engine->mutex);
    return copy_string(engine->last_error, destination, capacity);
}

SOD_API void sod_close(sod_handle handle) {
    if (handle != nullptr) {
        delete static_cast<Engine *>(handle);
    }
}

} // extern "C"
