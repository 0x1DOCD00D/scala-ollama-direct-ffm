ThisBuild / scalaVersion := "3.9.0"
ThisBuild / organization := "edu.example"
ThisBuild / version := "0.3.0"

lazy val root = (project in file("."))
  .settings(
    name := "scala-ollama-direct-ffm",
    libraryDependencies ++= Seq(
      "com.lihaoyi" %% "ujson" % "4.4.3"
    ),
    Compile / run / fork := true,
    Compile / run / connectInput := true,
    Compile / run / javaOptions += "--enable-native-access=ALL-UNNAMED",
    Test / fork := true,
    Test / javaOptions += "--enable-native-access=ALL-UNNAMED"
  )
