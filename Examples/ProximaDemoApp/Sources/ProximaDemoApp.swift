import SwiftUI
import ProximaKit
import ProximaEmbeddings

@main
struct ProximaDemoApp: App {
    @State private var engine = SearchEngine()

    var body: some Scene {
        Window("ProximaKit Demo", id: "main") {
            MainView(engine: engine)
                .frame(minWidth: 750, minHeight: 550)
                .task { await engine.buildIndex() }
        }
    }
}
