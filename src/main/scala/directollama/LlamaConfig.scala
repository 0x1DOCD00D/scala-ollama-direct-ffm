package directollama

final case class LlamaConfig(
    contextSize: Int = 8192,
    batchSize: Int = 128,
    gpuLayers: Int = 20,
    temperature: Float = 0.8f,
    minP: Float = 0.05f,
    seed: Int = -1,
    maxTokens: Int = 1024
)
