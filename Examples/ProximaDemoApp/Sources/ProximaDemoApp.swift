import SwiftUI
import ProximaKit
import ProximaEmbeddings

@main
struct ProximaDemoApp: App {
    @State private var engine = SearchEngine()

    var body: some Scene {
        // WindowGroup is the one scene type available on macOS, iOS,
        // iPadOS, and visionOS alike (`Window` is macOS-only).
        WindowGroup {
            MainView(engine: engine)
            #if os(macOS)
                .frame(minWidth: 750, minHeight: 550)
            #endif
                .task { await engine.buildIndex() }
        }
    }
}
