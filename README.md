# scala-ollama-direct-ffm 0.3.0-direct-gguf

This regenerated release has first-class `--gguf PATH` support. Verify it immediately after extraction with `./scripts/verify-project.sh`, `sbt 'run --version'`, and `sbt 'run --help'`. The help output must contain `--gguf PATH`.

The defaults are intentionally conservative for Apple Silicon after Metal out-of-memory testing: context 8192, batch 128, GPU layers 20, and maximum generation 1024 tokens. Use command-line options to increase them when the machine has sufficient memory.

# Scala 3 direct access to a locally installed Ollama model

This project selects a model already present in the local Ollama model store, resolves the underlying GGUF blob from the Ollama manifest, and loads that GGUF directly into llama.cpp. Scala calls a small native C ABI through the JDK 25 Foreign Function and Memory API.

The prompt path is

```text
Scala 3 CLI
    |
    | JDK 25 FFM
    v
libscala_ollama_direct
    |
    v
llama.cpp
    |
    v
Ollama GGUF blob already on disk
```

The Ollama daemon, REST API, `ollama run`, and port 11434 are not used.

## Requirements

Install the following software.

```text
JDK 25
Scala/sbt
CMake
C++17 compiler
Git
Ollama with at least one locally installed model
```

The project uses Scala 3.9.0 and sbt 1.13.0. JDK 25 is required because the Scala binding uses the standard `java.lang.foreign` API.

Verify Java before building.

```bash
java -version
```

On macOS, if several JDKs are installed, a typical selection is

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 25)
export PATH="$JAVA_HOME/bin:$PATH"
```

## 1. See the models that Ollama has already installed

Ollama normally stores model data under `~/.ollama/models` on macOS and under `/usr/share/ollama/.ollama/models` for the standard Linux installation. The `OLLAMA_MODELS` environment variable takes precedence when it is defined.

After the Scala project is compiled, list the manifests with

```bash
sbt 'run --list'
```

Typical output looks like

```text
Ollama model store: /Users/me/.ollama/models
MODEL        SIZE       SPLITS
qwen3:8b     5.0 GiB    1
llama3.2:3b  1.9 GiB    1
```

The listing is produced by reading the local Ollama manifest files. It does not call the Ollama API.

If the model store is somewhere else, use

```bash
sbt 'run --list --models-dir /path/to/models'
```

## 2. Build the native llama.cpp bridge

Run

```bash
./scripts/build-native.sh
```

If `external/llama.cpp` does not exist, the script clones the upstream llama.cpp repository and builds a shared library containing the small C facade plus llama.cpp.

If llama.cpp is already installed as source somewhere else, avoid the clone with

```bash
LLAMA_CPP_DIR=/path/to/llama.cpp ./scripts/build-native.sh
```

For NVIDIA CUDA builds, use

```bash
GGML_CUDA=1 ./scripts/build-native.sh
```

On Apple Silicon, the normal llama.cpp CMake configuration enables the supported Metal backend automatically. To force Metal off, use

```bash
GGML_METAL=0 ./scripts/build-native.sh
```

The resulting native library is normally written under

```text
native/build/ffi/
```

The Scala client automatically looks there. An alternative path can be supplied with `--native-lib` or the `SCALA_LLAMA_NATIVE` environment variable.

## 3. Start an interactive direct-model session

The simplest launch is

```bash
sbt run
```

If several Ollama models are installed, the program shows a numbered menu and asks you to select one. If only one model is installed, it is selected automatically.

For a reproducible or scripted launch, specify the model explicitly.

```bash
sbt 'run --model qwen3:8b'
```

Startup prints the resolved model blob.

```text
Model: qwen3:8b
Ollama store: /Users/me/.ollama/models
GGUF layer(s): 1
  /Users/me/.ollama/models/blobs/sha256-...
Native runtime: .../native/build/ffi/libscala_ollama_direct.dylib
Ollama daemon/API: not used
```

You can then type instructions.

```text
you> Explain the difference between Lamport clocks and vector clocks.
model> ...

you> Give a concrete example with three processes.
model> ...
```

The conversation remains in the llama.cpp context between turns.

Interactive commands are

```text
/help
/reset
/system TEXT
/exit
```

`/reset` clears the conversation while keeping the model loaded. `/system TEXT` installs a new system message and also starts a fresh conversation.

## 4. Send one instruction and exit

Use

```bash
sbt 'run --model qwen3:8b --prompt "Explain Chandy-Lamport snapshots in 5 paragraphs"'
```

This still uses the model chat template, but the process exits after the response.

A custom system message can be supplied with

```bash
sbt 'run --model qwen3:8b --system "You are a distributed-systems instructor." --prompt "Explain vector clocks"'
```

## 5. Raw model completion

Chat mode asks llama.cpp to use the chat template embedded in the GGUF metadata. If you want to bypass chat-role formatting and feed text directly into the tokenizer, use `--raw`.

```bash
sbt 'run --model qwen3:8b --raw --prompt "The Chandy-Lamport algorithm works by"'
```

In interactive raw mode, every line is an independent prompt because the native context is reset before each raw completion.

## 6. Inference options

Examples are shown below.

```bash
sbt 'run --model qwen3:8b --context 32768 --max-tokens 4096'
```

```bash
sbt 'run --model qwen3:8b --temperature 0.2 --min-p 0.05'
```

```bash
sbt 'run --model qwen3:8b --gpu-layers 0'
```

The last command requests CPU-only model layers. The default is `--gpu-layers 20`; `--gpu-layers -1`, which asks current llama.cpp to offload all supported model layers where a configured GPU backend is available.

## 7. How an Ollama model name becomes a GGUF path

For a model such as `qwen3:8b`, Ollama has a manifest under a hierarchy conceptually resembling

```text
models/
  manifests/
    registry.ollama.ai/
      library/
        qwen3/
          8b
```

The manifest contains a layer such as

```json
{
  "mediaType": "application/vnd.ollama.image.model",
  "digest": "sha256:012345...",
  "size": 123456789
}
```

The project converts the digest into the local blob filename

```text
sha256-012345...
```

and resolves it under

```text
models/blobs/
```

If a model consists of several GGUF model layers, the paths remain in manifest order and are passed to `llama_model_load_from_splits`.

## 8. What is and is not inherited from Ollama

The model weights are reused directly from the local Ollama blob store.

The program does not ask Ollama to generate the response. Consequently, Ollama runtime behavior is not automatically reproduced. In particular, an Ollama manifest can contain separate template, system, parameter, adapter, projector, or other layers. This project intentionally uses the GGUF model and the chat template available to llama.cpp from the model itself, plus the sampling values supplied on this program's command line.

Therefore a response can differ from `ollama run MODEL` even when the same model weights are used.

The current project is intended primarily for text-generation GGUF models. It does not yet wire projector layers for image input or LoRA adapter layers.

## 9. Why there is a small C facade

FFM could bind every llama.cpp structure directly, but llama.cpp exposes several C structures by value and those structures evolve relatively quickly. The project keeps those details inside `native/scala_ollama_direct.cpp` and exposes a stable C ABI made only of pointers, fixed-width integers, floats, and byte buffers.

The Scala side therefore deals with operations such as

```text
create engine
load model path(s)
chat
generate
reset
copy response
close
```

while the native side deals with llama.cpp model parameters, contexts, batches, samplers, tokenization, chat templates, and model memory.

## 10. Source layout

```text
build.sbt
project/build.properties

native/
  CMakeLists.txt
  scala_ollama_direct.h
  scala_ollama_direct.cpp

scripts/
  build-native.sh
  run.sh

src/main/scala/directollama/
  Cli.scala
  LlamaConfig.scala
  LlamaModel.scala
  LlamaNative.scala
  Main.scala
  OllamaStore.scala
```

## 11. Important implementation detail

This is direct access to the locally stored model, not direct access to an Ollama process. Ollama is only the source of the already downloaded manifest and GGUF blobs.

At generation time the execution path is

```text
command line
    |
    v
Scala
    |
    v
FFM MethodHandle
    |
    v
C ABI facade
    |
    v
llama.cpp tokenizer and model context
    |
    v
sampler
    |
    v
response bytes
    |
    v
Scala String
```

The Ollama service can be stopped and the inference path remains the same, provided the model files remain accessible.

## Load an arbitrary GGUF directly

The client can bypass Ollama model discovery completely and load a GGUF file by path.

```bash
sbt 'run --gguf /Users/drmark/models/qwen3-base/qwen3-8b-base-q4_k_m.gguf --raw'
```

For lower Metal memory pressure on an 8B model, a conservative invocation is

```bash
sbt 'run --gguf /Users/drmark/models/qwen3-base/qwen3-8b-base-q4_k_m.gguf --raw --context 8192 --batch 128 --gpu-layers 20 --max-tokens 1024'
```

`--model` and `--gguf` are mutually exclusive. `--gguf` does not inspect or require the Ollama model store.
