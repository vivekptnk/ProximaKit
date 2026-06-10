import AppKit
import CoreText
import CoreGraphics
import Foundation

// Renders text as SVG path data using real CoreText shaping (SF Pro),
// so the wordmark renders identically on every platform.
func svgPath(text: String, font: CTFont, originX: CGFloat, originY: CGFloat) -> String {
    let attr = [kCTFontAttributeName as NSAttributedString.Key: font]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attr))
    let runs = CTLineGetGlyphRuns(line) as! [CTRun]
    var d = ""
    for run in runs {
        let count = CTRunGetGlyphCount(run)
        var glyphs = [CGGlyph](repeating: 0, count: count)
        var positions = [CGPoint](repeating: .zero, count: count)
        CTRunGetGlyphs(run, CFRange(location: 0, length: count), &glyphs)
        CTRunGetPositions(run, CFRange(location: 0, length: count), &positions)
        let runFont = (CTRunGetAttributes(run) as! [NSAttributedString.Key: Any])[kCTFontAttributeName as NSAttributedString.Key] as! CTFont
        for i in 0..<count {
            guard let path = CTFontCreatePathForGlyph(runFont, glyphs[i], nil) else { continue }
            // Flip Y (CoreText is y-up; SVG is y-down) and translate.
            var t = CGAffineTransform(translationX: originX + positions[i].x, y: originY + positions[i].y)
            t = t.scaledBy(x: 1, y: -1)
            let moved = path.copy(using: &t)!
            moved.applyWithBlock { elem in
                let e = elem.pointee
                func f(_ p: CGPoint) -> String { String(format: "%.2f %.2f", p.x, p.y) }
                switch e.type {
                case .moveToPoint: d += "M\(f(e.points[0]))"
                case .addLineToPoint: d += "L\(f(e.points[0]))"
                case .addQuadCurveToPoint: d += "Q\(f(e.points[0])) \(f(e.points[1]))"
                case .addCurveToPoint: d += "C\(f(e.points[0])) \(f(e.points[1])) \(f(e.points[2]))"
                case .closeSubpath: d += "Z"
                @unknown default: break
                }
            }
        }
    }
    return d
}

let args = CommandLine.arguments
// usage: outline <text> <size> <weight: semibold|regular> <x> <y>
let text = args[1]
let size = CGFloat(Double(args[2])!)
let weight = args[3]
let x = CGFloat(Double(args[4])!)
let y = CGFloat(Double(args[5])!)

let nsWeight: NSFont.Weight = (weight == "semibold") ? .semibold : .regular
let font = NSFont.systemFont(ofSize: size, weight: nsWeight) as CTFont
let attrW = [kCTFontAttributeName as NSAttributedString.Key: font]
let lineW = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attrW))
let width = CTLineGetTypographicBounds(lineW, nil, nil, nil)
FileHandle.standardError.write("FONT: \(CTFontCopyPostScriptName(font))  WIDTH: \(width)\n".data(using: .utf8)!)
print(svgPath(text: text, font: font, originX: x, originY: y))
