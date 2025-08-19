// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "EdgeBrain",
  platforms: [.macOS(.v13)],
  products: [
    .library(name: "EdgeBrain", targets: ["EdgeBrain"])
  ],
  dependencies: [
    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    .package(url: "https://github.com/unixpickle/honeycrisp-examples.git", from: "0.0.4"),
  ],
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
    .executableTarget(
      name: "ClassifyMNIST",
      dependencies: [
        "EdgeBrain",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "MNIST", package: "honeycrisp-examples"),
      ]
    ),
  ]
)
