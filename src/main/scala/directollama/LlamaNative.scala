package directollama

import java.lang.foreign.{Arena, FunctionDescriptor, Linker, MemorySegment, SymbolLookup, ValueLayout}
import java.lang.invoke.MethodHandle
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path
import scala.jdk.CollectionConverters.*

final class LlamaNative(libraryPath: Path) extends AutoCloseable:

  private val libraryArena = Arena.ofShared()
  private val linker = Linker.nativeLinker()
  private val lookup = SymbolLookup.libraryLookup(
    libraryPath.toAbsolutePath.normalize(),
    libraryArena
  )

  private def symbol(name: String): MemorySegment = lookup.findOrThrow(name)

  private def downcall(name: String, descriptor: FunctionDescriptor): MethodHandle =
    linker.downcallHandle(symbol(name), descriptor)

  private def call(handle: MethodHandle, arguments: AnyRef*): AnyRef =
    handle.invokeWithArguments(arguments.toList.asJava)

  private val createHandle = downcall(
    "sod_create",
    FunctionDescriptor.of(ValueLayout.ADDRESS)
  )

  private val loadHandle = downcall(
    "sod_load",
    FunctionDescriptor.of(
      ValueLayout.JAVA_INT,
      ValueLayout.ADDRESS,
      ValueLayout.ADDRESS,
      ValueLayout.JAVA_INT,
      ValueLayout.JAVA_INT,
      ValueLayout.JAVA_INT,
      ValueLayout.JAVA_INT,
      ValueLayout.JAVA_FLOAT,
      ValueLayout.JAVA_FLOAT,
      ValueLayout.JAVA_INT
    )
  )

  private val generateHandle = downcall(
    "sod_generate",
    FunctionDescriptor.of(
      ValueLayout.JAVA_INT,
      ValueLayout.ADDRESS,
      ValueLayout.ADDRESS,
      ValueLayout.JAVA_INT
    )
  )

  private val chatHandle = downcall(
    "sod_chat",
    FunctionDescriptor.of(
      ValueLayout.JAVA_INT,
      ValueLayout.ADDRESS,
      ValueLayout.ADDRESS,
      ValueLayout.JAVA_INT
    )
  )

  private val setSystemHandle = downcall(
    "sod_set_system_prompt",
    FunctionDescriptor.of(
      ValueLayout.JAVA_INT,
      ValueLayout.ADDRESS,
      ValueLayout.ADDRESS
    )
  )

  private val resetHandle = downcall(
    "sod_reset",
    FunctionDescriptor.of(
      ValueLayout.JAVA_INT,
      ValueLayout.ADDRESS
    )
  )

  private val responseSizeHandle = downcall(
    "sod_response_size",
    FunctionDescriptor.of(
      ValueLayout.JAVA_LONG,
      ValueLayout.ADDRESS
    )
  )

  private val copyResponseHandle = downcall(
    "sod_copy_response",
    FunctionDescriptor.of(
      ValueLayout.JAVA_LONG,
      ValueLayout.ADDRESS,
      ValueLayout.ADDRESS,
      ValueLayout.JAVA_LONG
    )
  )

  private val errorSizeHandle = downcall(
    "sod_error_size",
    FunctionDescriptor.of(
      ValueLayout.JAVA_LONG,
      ValueLayout.ADDRESS
    )
  )

  private val copyErrorHandle = downcall(
    "sod_copy_error",
    FunctionDescriptor.of(
      ValueLayout.JAVA_LONG,
      ValueLayout.ADDRESS,
      ValueLayout.ADDRESS,
      ValueLayout.JAVA_LONG
    )
  )

  private val closeHandle = downcall(
    "sod_close",
    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
  )

  def create(): MemorySegment =
    val result = call(createHandle).asInstanceOf[MemorySegment]
    if result == MemorySegment.NULL || result.address() == 0L then
      throw RuntimeException("Unable to allocate native llama engine")
    result

  def load(engine: MemorySegment, modelPaths: Vector[Path], config: LlamaConfig): Unit =
    val arena = Arena.ofConfined()
    try
      val packedPaths = packPaths(arena, modelPaths)
      val status = call(
        loadHandle,
        engine,
        packedPaths,
        Int.box(modelPaths.size),
        Int.box(config.contextSize),
        Int.box(config.batchSize),
        Int.box(config.gpuLayers),
        Float.box(config.temperature),
        Float.box(config.minP),
        Int.box(config.seed)
      ).asInstanceOf[java.lang.Integer].intValue()
      check(engine, status)
    finally arena.close()

  def generate(engine: MemorySegment, prompt: String, maxTokens: Int): String =
    val arena = Arena.ofConfined()
    try
      val nativePrompt = arena.allocateFrom(prompt)
      val status = call(
        generateHandle,
        engine,
        nativePrompt,
        Int.box(maxTokens)
      ).asInstanceOf[java.lang.Integer].intValue()
      check(engine, status)
      readResponse(engine)
    finally arena.close()

  def chat(engine: MemorySegment, message: String, maxTokens: Int): String =
    val arena = Arena.ofConfined()
    try
      val nativeMessage = arena.allocateFrom(message)
      val status = call(
        chatHandle,
        engine,
        nativeMessage,
        Int.box(maxTokens)
      ).asInstanceOf[java.lang.Integer].intValue()
      check(engine, status)
      readResponse(engine)
    finally arena.close()

  def setSystemPrompt(engine: MemorySegment, prompt: String): Unit =
    val arena = Arena.ofConfined()
    try
      val nativePrompt = arena.allocateFrom(prompt)
      val status = call(
        setSystemHandle,
        engine,
        nativePrompt
      ).asInstanceOf[java.lang.Integer].intValue()
      check(engine, status)
    finally arena.close()

  def reset(engine: MemorySegment): Unit =
    val status = call(resetHandle, engine)
      .asInstanceOf[java.lang.Integer]
      .intValue()
    check(engine, status)

  def closeEngine(engine: MemorySegment): Unit =
    call(closeHandle, engine)
    ()

  private def packPaths(arena: Arena, paths: Vector[Path]): MemorySegment =
    require(paths.nonEmpty, "At least one model path is required")
    val encoded = paths.map(_.toAbsolutePath.normalize().toString.getBytes(UTF_8))
    val totalBytes = encoded.foldLeft(0L)((sum, bytes) => sum + bytes.length + 1L)
    val packed = arena.allocate(totalBytes, 1)
    var offset = 0L
    encoded.foreach { bytes =>
      val source = MemorySegment.ofArray(bytes)
      packed.asSlice(offset, bytes.length.toLong).copyFrom(source)
      offset += bytes.length
      packed.set(ValueLayout.JAVA_BYTE, offset, 0.toByte)
      offset += 1
    }
    packed

  private def readResponse(engine: MemorySegment): String =
    readNativeString(engine, responseSizeHandle, copyResponseHandle, "response")

  private def readError(engine: MemorySegment): String =
    val text = readNativeString(engine, errorSizeHandle, copyErrorHandle, "error")
    if text.isBlank then "Unknown native llama.cpp error" else text

  private def readNativeString(
      engine: MemorySegment,
      sizeHandle: MethodHandle,
      copyHandle: MethodHandle,
      label: String
  ): String =
    val size = call(sizeHandle, engine)
      .asInstanceOf[java.lang.Long]
      .longValue()

    if size <= 0 then ""
    else
      val arena = Arena.ofConfined()
      try
        val buffer = arena.allocate(size, 1)
        val copied = call(
          copyHandle,
          engine,
          buffer,
          Long.box(size)
        ).asInstanceOf[java.lang.Long].longValue()
        if copied != size then
          throw RuntimeException(s"Expected $size $label bytes but copied $copied")
        new String(buffer.toArray(ValueLayout.JAVA_BYTE), UTF_8)
      finally arena.close()

  private def check(engine: MemorySegment, status: Int): Unit =
    if status != 0 then throw RuntimeException(readError(engine))

  override def close(): Unit = libraryArena.close()
