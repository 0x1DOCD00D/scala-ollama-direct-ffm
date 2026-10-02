package directollama

import java.nio.file.{Files, Path}

private final case class ModelSelection(
    name: String,
    paths: Vector[Path],
    source: String
)

object Main:

  def main(args: Array[String]): Unit =
    try
      val options = Cli.parse(args)

      if options.help then
        println(Cli.usage)
        return

      if options.version then
        println(ProjectVersion.value)
        return

      if options.listModels then
        if options.gguf.nonEmpty then
          throw IllegalArgumentException("--list cannot be combined with --gguf")
        val store = OllamaStore.locate(options.modelsDir)
        printModels(store)
        return

      val selected = resolveModel(options)
      val nativeLibrary = resolveNativeLibrary(options.nativeLibrary)

      println(s"Client version: ${ProjectVersion.value}")
      println(s"Model: ${selected.name}")
      println(s"Model source: ${selected.source}")
      println(s"GGUF file(s): ${selected.paths.size}")
      selected.paths.foreach(path => println(s"  $path"))
      println(s"Native runtime: $nativeLibrary")
      println("Ollama daemon/API: not used")
      println()

      val model = new LlamaModel(nativeLibrary, selected.paths, options.config)
      try
        if options.raw then
          options.prompt match
            case Some(prompt) => println(model.generate(prompt))
            case None => rawRepl(model)
        else
          model.startChat(options.systemPrompt)
          options.prompt match
            case Some(prompt) => println(model.chat(prompt))
            case None => chatRepl(model, options.systemPrompt)
      finally model.close()

    catch
      case e: Throwable =>
        Console.err.println(s"ERROR: ${e.getMessage}")
        System.exit(1)


  private def resolveModel(options: CliOptions): ModelSelection =
    options.gguf match
      case Some(path) =>
        val normalized = path.toAbsolutePath.normalize()
        if !Files.isRegularFile(normalized) then
          throw IllegalArgumentException(s"GGUF file not found: $normalized")
        if !Files.isReadable(normalized) then
          throw IllegalArgumentException(s"GGUF file is not readable: $normalized")

        ModelSelection(
          name = Option(normalized.getFileName).map(_.toString).getOrElse(normalized.toString),
          paths = Vector(normalized),
          source = "direct GGUF file"
        )

      case None =>
        val store = OllamaStore.locate(options.modelsDir)
        val ollamaModel = options.model match
          case Some(modelName) => OllamaStore.resolve(store, modelName)
          case None => selectInstalledModel(store)

        ModelSelection(
          name = ollamaModel.name,
          paths = ollamaModel.modelPaths,
          source = s"Ollama model store $store"
        )


  private def selectInstalledModel(store: Path): OllamaModel =
    val models = OllamaStore.list(store)
    if models.isEmpty then
      throw IllegalStateException(s"No Ollama model manifests found under $store")
    else if models.size == 1 then
      println(s"Using the only installed model: ${models.head.name}")
      models.head
    else
      println("Select a locally installed Ollama model")
      models.zipWithIndex.foreach { case (model, index) =>
        println(f"  ${index + 1}%2d. ${model.name}%-30s ${humanBytes(model.modelBytes)}")
      }
      print("model> ")
      val input = Option(scala.io.StdIn.readLine()).map(_.trim).getOrElse("")
      input.toIntOption match
        case Some(number) if number >= 1 && number <= models.size => models(number - 1)
        case _ if input.nonEmpty => OllamaStore.resolve(store, input)
        case _ => throw IllegalArgumentException("No model selected")


  private def printModels(store: Path): Unit =
    val models = OllamaStore.list(store)
    println(s"Ollama model store: $store")
    if models.isEmpty then println("No model manifests found.")
    else
      val nameWidth = models.map(_.name.length).max.max("MODEL".length)
      val format = s"%-${nameWidth}s  %-10s  %s%n"
      printf(format, "MODEL", "SIZE", "SPLITS")
      models.foreach { model =>
        printf(format, model.name, humanBytes(model.modelBytes), model.splitCount.toString)
      }


  private def chatRepl(model: LlamaModel, initialSystem: Option[String]): Unit =
    var systemPrompt = initialSystem
    var running = true
    println("Interactive chat. Type /help for commands.")

    while running do
      print("you> ")
      val line = scala.io.StdIn.readLine()
      if line == null then running = false
      else
        val input = line.trim
        input match
          case "" => ()
          case "/exit" | "/quit" => running = false
          case "/help" =>
            println("/reset, /system TEXT, /exit")
          case "/reset" =>
            model.startChat(systemPrompt)
            println("conversation reset")
          case command if command.startsWith("/system ") =>
            val text = command.stripPrefix("/system ").trim
            systemPrompt = Option(text).filter(_.nonEmpty)
            model.startChat(systemPrompt)
            println("system message updated and conversation reset")
          case message =>
            val answer = model.chat(message)
            println(s"model> $answer")


  private def rawRepl(model: LlamaModel): Unit =
    var running = true
    println("Raw completion mode. Each line is an independent prompt. Type /exit to stop.")
    while running do
      print("prompt> ")
      val line = scala.io.StdIn.readLine()
      if line == null || line.trim == "/exit" then running = false
      else if line.trim.nonEmpty then println(s"model> ${model.generate(line)}")


  private def resolveNativeLibrary(explicit: Option[Path]): Path =
    val candidate = explicit
      .orElse(sys.env.get("SCALA_LLAMA_NATIVE").map(value => Path.of(value)))
      .getOrElse {
        Path.of(
          "native",
          "build",
          "ffi",
          System.mapLibraryName("scala_ollama_direct")
        )
      }
      .toAbsolutePath
      .normalize()

    if !Files.isRegularFile(candidate) then
      throw IllegalStateException(
        s"Native library not found at $candidate. Run scripts/build-native.sh first " +
          "or pass --native-lib."
      )
    candidate


  private def humanBytes(bytes: Long): String =
    val units = Vector("B", "KiB", "MiB", "GiB", "TiB")
    var value = bytes.toDouble
    var index = 0
    while value >= 1024.0 && index < units.size - 1 do
      value /= 1024.0
      index += 1
    if index == 0 then s"${bytes} B" else f"$value%.1f ${units(index)}"
