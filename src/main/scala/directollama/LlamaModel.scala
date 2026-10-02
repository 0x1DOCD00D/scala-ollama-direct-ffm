package directollama

import java.lang.foreign.MemorySegment
import java.nio.file.Path

final class LlamaModel(
    nativeLibrary: Path,
    modelPaths: Vector[Path],
    config: LlamaConfig = LlamaConfig()
) extends AutoCloseable:

  require(modelPaths.nonEmpty, "At least one GGUF model path is required")

  private val native = new LlamaNative(nativeLibrary)
  private var engine: MemorySegment = native.create()
  private var closed = false

  native.load(engine, modelPaths, config)

  private def ensureOpen(): Unit =
    if closed then throw IllegalStateException("The model has already been closed")

  def generate(prompt: String, maxTokens: Int = config.maxTokens): String = synchronized {
    ensureOpen()
    native.generate(engine, prompt, maxTokens)
  }

  def startChat(systemPrompt: Option[String] = None): Unit = synchronized {
    ensureOpen()
    systemPrompt match
      case Some(prompt) => native.setSystemPrompt(engine, prompt)
      case None => native.reset(engine)
  }

  def chat(message: String, maxTokens: Int = config.maxTokens): String = synchronized {
    ensureOpen()
    native.chat(engine, message, maxTokens)
  }

  def reset(): Unit = synchronized {
    ensureOpen()
    native.reset(engine)
  }

  override def close(): Unit = synchronized {
    if !closed then
      native.closeEngine(engine)
      engine = MemorySegment.NULL
      native.close()
      closed = true
  }
