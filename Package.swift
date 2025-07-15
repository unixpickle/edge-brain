// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "EdgeBrain",
  platforms: [.macOS(.v13)],
  products: [
    .library(name: "EdgeBrain", targets: ["EdgeBrain"])
  ],
  dependencies: [],
  targets: [
    .target(
      name: "EdgeBrain",
      dependencies: []
    ),
    .testTarget(
      name: "EdgeBrainTests",
      dependencies: [
        "EdgeBrain"
      ],
      swiftSettings: [
        .enableExperimentalFeature("Testing")
      ]
    ),
  ]
)
