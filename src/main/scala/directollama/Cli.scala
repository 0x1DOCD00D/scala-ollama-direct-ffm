package directollama

import java.nio.file.Path

final case class CliOptions(
    model: Option[String] = None,
    gguf: Option[Path] = None,
    listModels: Boolean = false,
    modelsDir: Option[Path] = None,
    nativeLibrary: Option[Path] = None,
    prompt: Option[String] = None,
    raw: Boolean = false,
    systemPrompt: Option[String] = None,
    config: LlamaConfig = LlamaConfig(),
    help: Boolean = false,
    version: Boolean = false
)

object Cli:

  def parse(args: Array[String]): CliOptions =
    var options = CliOptions()
    var i = 0

    def value(flag: String): String =
      if i + 1 >= args.length then
        throw IllegalArgumentException(s"Missing value after $flag")
      i += 1
      args(i)

    while i < args.length do
      args(i) match
        case "--model" | "-m" =>
          options = options.copy(model = Some(value(args(i))))
        case "--gguf" =>
          options = options.copy(gguf = Some(Path.of(value(args(i)))))
        case "--list" | "-l" =>
          options = options.copy(listModels = true)
        case "--models-dir" =>
          options = options.copy(modelsDir = Some(Path.of(value(args(i)))))
        case "--native-lib" =>
          options = options.copy(nativeLibrary = Some(Path.of(value(args(i)))))
        case "--prompt" | "-p" =>
          options = options.copy(prompt = Some(value(args(i))))
        case "--raw" =>
          options = options.copy(raw = true)
        case "--system" =>
          options = options.copy(systemPrompt = Some(value(args(i))))
        case "--context" =>
          options = options.copy(config = options.config.copy(contextSize = value(args(i)).toInt))
        case "--batch" =>
          options = options.copy(config = options.config.copy(batchSize = value(args(i)).toInt))
        case "--gpu-layers" =>
          options = options.copy(config = options.config.copy(gpuLayers = value(args(i)).toInt))
        case "--temperature" =>
          options = options.copy(config = options.config.copy(temperature = value(args(i)).toFloat))
        case "--min-p" =>
          options = options.copy(config = options.config.copy(minP = value(args(i)).toFloat))
        case "--seed" =>
          options = options.copy(config = options.config.copy(seed = value(args(i)).toInt))
        case "--max-tokens" =>
          options = options.copy(config = options.config.copy(maxTokens = value(args(i)).toInt))
        case "--help" | "-h" =>
          options = options.copy(help = true)
        case "--version" =>
          options = options.copy(version = true)
        case unknown =>
          throw IllegalArgumentException(s"Unknown option: $unknown")
      i += 1

    if options.model.nonEmpty && options.gguf.nonEmpty then
      throw IllegalArgumentException("Use either --model or --gguf, not both")

    options

  val usage: String =
    """Scala Ollama Direct FFM
      |
      |Usage
      |  sbt 'run --list'
      |  sbt 'run --model qwen3:8b'
      |  sbt 'run --model qwen3:8b --prompt "Explain vector clocks"'
      |  sbt 'run --gguf /path/to/model.gguf --raw'
      |  sbt 'run --gguf /path/to/model.gguf --raw --prompt "Text prefix"'
      |
      |Model source
      |  -m, --model NAME          Locally installed Ollama model to load
      |      --gguf PATH           Load a GGUF file directly, without Ollama discovery
      |  -l, --list                List locally installed Ollama models and exit
      |      --models-dir PATH     Override the Ollama model-store directory
      |
      |Runtime
      |      --native-lib PATH     Override the FFM native-library path
      |  -p, --prompt TEXT         Run one instruction and exit
      |      --raw                 Use raw completion instead of chat templating
      |      --system TEXT         System message for chat mode
      |      --context N           Context size, default 8192
      |      --batch N             Decode batch size, default 128
      |      --gpu-layers N        GPU layers, default 20; use 0 for CPU only, -1 for all
      |      --temperature X       Temperature, default 0.8
      |      --min-p X             Min-p sampler threshold, default 0.05
      |      --seed N              Seed, default -1
      |      --max-tokens N        Maximum generated tokens, default 1024
      |      --version             Show project version
      |  -h, --help                Show this help
      |
      |Interactive commands
      |  /help                     Show commands
      |  /reset                    Start a fresh conversation
      |  /system TEXT              Set a new system message and reset conversation
      |  /exit                     Exit
      |""".stripMargin
