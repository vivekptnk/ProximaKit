import XCTest
@testable import ProximaEmbeddings
import CoreGraphics
import ProximaKit

final class VisionEmbeddingProviderTests: XCTestCase {

    /// Creates a simple solid-color CGImage for testing.
    private func makeTestImage(width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else { return nil }

        context.setFillColor(CGColor(
            red: CGFloat(red) / 255,
            green: CGFloat(green) / 255,
            blue: CGFloat(blue) / 255,
            alpha: 1.0
        ))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        return context.makeImage()
    }

    func testEmbedImage() async throws {
        guard let image = makeTestImage(width: 224, height: 224, red: 128, green: 64, blue: 200) else {
            throw XCTSkip("Could not create test image")
        }

        let provider = VisionEmbeddingProvider()
        let vector = try await provider.embed(image)

        XCTAssertGreaterThan(vector.dimension, 0, "Feature print should have positive dimension")
        XCTAssertGreaterThan(vector.magnitude, 0, "Feature print should be non-zero")
    }

    func testDifferentImagesProduceDifferentVectors() async throws {
        guard let red = makeTestImage(width: 224, height: 224, red: 255, green: 0, blue: 0),
              let blue = makeTestImage(width: 224, height: 224, red: 0, green: 0, blue: 255) else {
            throw XCTSkip("Could not create test images")
        }

        let provider = VisionEmbeddingProvider()
        let redVec = try await provider.embed(red)
        let blueVec = try await provider.embed(blue)

        // Two solid-color images should produce different feature prints.
        // They won't be identical (different colors).
        XCTAssertNotEqual(redVec, blueVec, "Different images should produce different vectors")
    }

    func testConsistentDimension() async throws {
        guard let img1 = makeTestImage(width: 100, height: 100, red: 50, green: 50, blue: 50),
              let img2 = makeTestImage(width: 300, height: 300, red: 200, green: 200, blue: 200) else {
            throw XCTSkip("Could not create test images")
        }

        let provider = VisionEmbeddingProvider()
        let v1 = try await provider.embed(img1)
        let v2 = try await provider.embed(img2)

        XCTAssertEqual(v1.dimension, v2.dimension,
                       "Different image sizes should produce same dimension vectors")
    }
}
