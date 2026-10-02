package directollama

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*

final case class OllamaModel(
    name: String,
    manifest: Path,
    modelPaths: Vector[Path],
    modelBytes: Long
):
  def splitCount: Int = modelPaths.size

object OllamaStore:

  private val ModelMediaType = "application/vnd.ollama.image.model"

  def locate(explicit: Option[Path] = None): Path =
    val candidates =
      explicit.toVector ++
        sys.env.get("OLLAMA_MODELS").map(value => Path.of(value)).toVector ++
        Vector(
          Path.of(System.getProperty("user.home"), ".ollama", "models"),
          Path.of("/usr/share/ollama/.ollama/models")
        )

    candidates
      .map(_.toAbsolutePath.normalize())
      .find(path => Files.isDirectory(path.resolve("manifests")))
      .getOrElse {
        val searched = candidates.map(_.toString).mkString(", ")
        throw IllegalStateException(
          s"Cannot find an Ollama model store. Checked: $searched. " +
            "Set OLLAMA_MODELS or pass --models-dir."
        )
      }

  def list(root: Path): Vector[OllamaModel] =
    val manifestsRoot = root.resolve("manifests")
    if !Files.isDirectory(manifestsRoot) then Vector.empty
    else
      val stream = Files.walk(manifestsRoot)
      try
        stream.iterator().asScala
          .filter(path => Files.isRegularFile(path))
          .flatMap(path => parseManifest(root, manifestsRoot, path).toOption)
          .toVector
          .sortBy(_.name)
      finally stream.close()

  def resolve(root: Path, requested: String): OllamaModel =
    val models = list(root)
    if models.isEmpty then
      throw IllegalStateException(s"No Ollama manifests found under $root")

    val normalized = normalizeRequested(requested)
    val matches = models.filter { model =>
      val canonical = model.name.toLowerCase
      val withoutLatest = canonical.stripSuffix(":latest")
      canonical == normalized || withoutLatest == normalized
    }

    matches match
      case Vector(model) => model
      case Vector() =>
        val available = models.map(_.name).mkString(", ")
        throw IllegalArgumentException(
          s"Ollama model '$requested' was not found. Available models: $available"
        )
      case many =>
        throw IllegalArgumentException(
          s"Model name '$requested' is ambiguous: ${many.map(_.name).mkString(", ")}"
        )

  private def normalizeRequested(name: String): String =
    val n = name.trim.toLowerCase
    if n.contains(":") then n else s"$n:latest"

  private def parseManifest(
      root: Path,
      manifestsRoot: Path,
      manifestPath: Path
  ): Either[Throwable, OllamaModel] =
    try
      val relative = manifestsRoot.relativize(manifestPath)
      val parts = relative.iterator().asScala.map(_.toString).toVector
      if parts.size < 4 then
        Left(IllegalArgumentException(s"Unexpected manifest path: $manifestPath"))
      else
        val registry = parts.head
        val namespace = parts(1)
        val modelParts = parts.slice(2, parts.size - 1)
        val tag = parts.last
        val model = modelParts.mkString("/")

        val displayName =
          if registry == "registry.ollama.ai" && namespace == "library" then
            s"$model:$tag"
          else if registry == "registry.ollama.ai" then
            s"$namespace/$model:$tag"
          else
            s"$registry/$namespace/$model:$tag"

        val json = ujson.read(Files.readString(manifestPath))
        val modelLayers = json("layers").arr.toVector.filter { layer =>
          layer.obj.get("mediaType").exists(_.str == ModelMediaType)
        }

        if modelLayers.isEmpty then
          Left(IllegalArgumentException(s"No GGUF model layer in $manifestPath"))
        else
          val paths = modelLayers.map { layer =>
            val digest = layer("digest").str
            root.resolve("blobs").resolve(blobFileName(digest)).toAbsolutePath.normalize()
          }
          val missing = paths.filterNot(Files.isRegularFile(_))
          if missing.nonEmpty then
            Left(IllegalStateException(
              s"Manifest $manifestPath refers to missing blob(s): ${missing.mkString(", ")}"
            ))
          else
            val size = modelLayers.map(layer => layer.obj.get("size").map(_.num.toLong).getOrElse(0L)).sum
            Right(OllamaModel(displayName, manifestPath, paths, size))
    catch
      case t: Throwable => Left(t)

  private def blobFileName(digest: String): String =
    if digest.startsWith("sha256:") then digest.replaceFirst("sha256:", "sha256-")
    else digest
