import SwiftUI
import UniformTypeIdentifiers
import ProximaKit

/// The feature screens the demo surfaces in the detail pane / tab bar.
/// Search is the original experience; the inspector/import/export screens are
/// the demo-app phase-2 additions alongside the persistence and benchmark labs.
enum DemoScreen: String, CaseIterable, Identifiable {
    case search
    case inspector
    case importDocuments = "import"
    case exportResults = "export"
    case persistence
    case benchmark

    var id: String { rawValue }
    static let detailScreens: [DemoScreen] = [.search, .inspector, .importDocuments, .exportResults, .persistence, .benchmark]

    init?(launchArgument: String) {
        switch launchArgument.lowercased() {
        case "index", "inspector":
            self = .inspector
        case "import":
            self = .importDocuments
        case "export":
            self = .exportResults
        default:
            self.init(rawValue: launchArgument.lowercased())
        }
    }

    var title: String {
        switch self {
        case .search: "Search"
        case .inspector: "Inspector"
        case .importDocuments: "Import"
        case .exportResults: "Export"
        case .persistence: "Persistence"
        case .benchmark: "Benchmark"
        }
    }
    var icon: String {
        switch self {
        case .search: "magnifyingglass"
        case .inspector: "point.3.connected.trianglepath.dotted"
        case .importDocuments: "tray.and.arrow.down"
        case .exportResults: "square.and.arrow.up"
        case .persistence: "internaldrive"
        case .benchmark: "chart.xyaxis.line"
        }
    }
}

private enum DemoCompactTab: Hashable {
    case search, inspector, importDocuments, exportResults, labs

    init(screen: DemoScreen) {
        switch screen {
        case .search:
            self = .search
        case .inspector:
            self = .inspector
        case .importDocuments:
            self = .importDocuments
        case .exportResults:
            self = .exportResults
        case .persistence, .benchmark:
            self = .labs
        }
    }

    var title: String {
        switch self {
        case .search: "Search"
        case .inspector: "Inspector"
        case .importDocuments: "Import"
        case .exportResults: "Export"
        case .labs: "Labs"
        }
    }

    var icon: String {
        switch self {
        case .search: DemoScreen.search.icon
        case .inspector: DemoScreen.inspector.icon
        case .importDocuments: DemoScreen.importDocuments.icon
        case .exportResults: DemoScreen.exportResults.icon
        case .labs: "testtube.2"
        }
    }
}

struct MainView: View {
    @Bindable var engine: SearchEngine
    @State private var searchText = ""
    @State private var newNote = ""
    @State private var showImagePicker = false
    @State private var showCorpusImporter = false
    // The two v1.6.0 lab controllers are self-contained (synthetic corpora, no
    // embedder) so they live and die with the view, never disturbing search.
    @State private var lab = PersistenceLab()
    @State private var bench = BenchmarkEngine()
    @State private var screen: DemoScreen = .search
    @State private var labsScreen: DemoScreen = .benchmark
    #if !os(macOS)
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    #endif

    var body: some View {
        layout
            .task {
                // Screenshot/UI-test hook: `simctl launch <sim> <bundle-id>
                // -demoScreen benchmark|persistence|search|inspector|import|export` selects the
                // initial screen/tab (launch args of the -key value form land in
                // UserDefaults). Pairs with -demoQuery below.
                if let s = UserDefaults.standard.string(forKey: "demoScreen"),
                   let mapped = DemoScreen(launchArgument: s) {
                    screen = mapped
                    if mapped == .benchmark || mapped == .persistence {
                        labsScreen = mapped
                    }
                }
                // Screenshot/UI-test hook: `simctl launch <sim> <bundle-id>
                // -demoQuery "..."` pre-fills the search field (launch
                // arguments of the -key value form land in UserDefaults).
                if let demo = UserDefaults.standard.string(forKey: "demoQuery"),
                   searchText.isEmpty {
                    // Wait for the index build to have started AND finished —
                    // checking isIndexing alone races the buildIndex() task
                    // and can fire the query against an empty index.
                    await waitForIndexReady()
                    searchText = demo
                }
                // Screenshot/UI-test hook: `-demoAutorun 1` kicks the selected
                // screen's headline action on appear (benchmark sweep, or build
                // the persistence lab + grow its WAL) so a populated screen can
                // be captured non-interactively.
                if UserDefaults.standard.bool(forKey: "demoAutorun") {
                    switch screen {
                    case .benchmark:
                        await bench.run()
                    case .persistence:
                        await lab.build()
                        // `-demoFlow memory` walks the happy path (checkpoint →
                        // measure); default grows the WAL and surfaces the
                        // paged-open error → checkpoint recovery flow.
                        if UserDefaults.standard.string(forKey: "demoFlow") == "memory" {
                            await lab.checkpoint()
                            await lab.measureMemory()
                        } else {
                            await lab.addOps(50)
                            try? await lab.openJournaled(mode: .paged)
                        }
                    case .importDocuments:
                        await waitForIndexReady()
                        await engine.importDemoCorpus()
                    case .exportResults:
                        await waitForIndexReady()
                        let query = UserDefaults.standard.string(forKey: "demoQuery") ?? "space exploration"
                        searchText = query
                        await engine.search(query)
                        _ = try? engine.writeExportFile(format: .csv)
                        _ = try? engine.writeExportFile(format: .json)
                    case .inspector:
                        await waitForIndexReady()
                    default:
                        break
                    }
                }
            }
            .onChange(of: screen) {
                if screen == .benchmark || screen == .persistence {
                    labsScreen = screen
                }
            }
            .onChange(of: searchText) {
                Task { await engine.search(searchText) }
            }
            .fileImporter(
                isPresented: $showImagePicker,
                allowedContentTypes: [.image],
                allowsMultipleSelection: true
            ) { result in
                guard let urls = try? result.get() else { return }
                Task { await engine.addImages(urls) }
            }
            .fileImporter(
                isPresented: $showCorpusImporter,
                allowedContentTypes: corpusImportTypes,
                allowsMultipleSelection: true
            ) { result in
                do {
                    let urls = try result.get()
                    Task { await engine.importCorpus(from: urls) }
                } catch {
                    engine.importErrorMessage = error.localizedDescription
                }
            }
    }

    // Compact widths (iPhone, narrow iPad splits) lead with search in a tab
    // bar; regular widths (macOS, iPad full-screen, visionOS) keep the
    // sidebar + detail split.
    @ViewBuilder
    private var layout: some View {
        #if os(macOS)
        splitLayout
        #else
        if horizontalSizeClass == .compact {
            TabView(selection: compactTabBinding) {
                NavigationStack {
                    searchPane.navigationTitle("Search")
                }
                .tabItem { Label(DemoCompactTab.search.title, systemImage: DemoCompactTab.search.icon) }
                .tag(DemoCompactTab.search)

                NavigationStack {
                    IndexInspectorView(engine: engine)
                }
                .tabItem { Label(DemoCompactTab.inspector.title, systemImage: DemoCompactTab.inspector.icon) }
                .tag(DemoCompactTab.inspector)

                NavigationStack {
                    CorpusImportView(engine: engine) { showCorpusImporter = true }
                }
                .tabItem { Label(DemoCompactTab.importDocuments.title, systemImage: DemoCompactTab.importDocuments.icon) }
                .tag(DemoCompactTab.importDocuments)

                NavigationStack {
                    ResultsExportView(engine: engine, queryText: $searchText)
                }
                .tabItem { Label(DemoCompactTab.exportResults.title, systemImage: DemoCompactTab.exportResults.icon) }
                .tag(DemoCompactTab.exportResults)

                NavigationStack {
                    labsPane
                }
                .tabItem { Label(DemoCompactTab.labs.title, systemImage: DemoCompactTab.labs.icon) }
                .tag(DemoCompactTab.labs)
            }
        } else {
            splitLayout
        }
        #endif
    }

    private var compactTabBinding: Binding<DemoCompactTab> {
        Binding {
            DemoCompactTab(screen: screen)
        } set: { tab in
            switch tab {
            case .search:
                screen = .search
            case .inspector:
                screen = .inspector
            case .importDocuments:
                screen = .importDocuments
            case .exportResults:
                screen = .exportResults
            case .labs:
                screen = labsScreen
            }
        }
    }

    private var corpusImportTypes: [UTType] {
        [.plainText, UTType(filenameExtension: "md") ?? .plainText, .folder]
    }

    private var splitLayout: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 220, ideal: 260)
        } detail: {
            detailPane
        }
    }

    // Regular-width detail: a segmented switch across the three feature screens,
    // keeping the familiar Search experience as the default while surfacing the
    // v1.6.0 Persistence and Benchmark panels alongside it.
    private var detailPane: some View {
        VStack(spacing: 0) {
            Picker("View", selection: $screen) {
                ForEach(DemoScreen.detailScreens) { screen in
                    Label(screen.title, systemImage: screen.icon).tag(screen)
                }
            }
            .pickerStyle(.segmented)
            .labelStyle(.titleAndIcon)
            .padding(.horizontal, 12)
            .padding(.top, 8)

            Divider().padding(.top, 8)

            switch screen {
            case .search: searchPane
            case .inspector: IndexInspectorView(engine: engine)
            case .importDocuments: CorpusImportView(engine: engine) { showCorpusImporter = true }
            case .exportResults: ResultsExportView(engine: engine, queryText: $searchText)
            case .persistence: PersistencePanel(lab: lab)
            case .benchmark: BenchmarkView(engine: bench)
            }
        }
    }

    private var labsPane: some View {
        VStack(spacing: 0) {
            Picker("Lab", selection: $labsScreen) {
                Label(DemoScreen.benchmark.title, systemImage: DemoScreen.benchmark.icon)
                    .tag(DemoScreen.benchmark)
                Label(DemoScreen.persistence.title, systemImage: DemoScreen.persistence.icon)
                    .tag(DemoScreen.persistence)
            }
            .pickerStyle(.segmented)
            .labelStyle(.titleAndIcon)
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .onChange(of: labsScreen) {
                screen = labsScreen
            }

            Divider().padding(.top, 8)

            switch labsScreen {
            case .persistence:
                PersistencePanel(lab: lab)
            default:
                BenchmarkView(engine: bench)
            }
        }
        .navigationTitle("Labs")
    }

    private func waitForIndexReady() async {
        var ticks = 0
        while (engine.isIndexing || engine.indexedCount == 0) && ticks < 300 {
            try? await Task.sleep(for: .milliseconds(100))
            ticks += 1
        }
    }

    private var searchPane: some View {
        VStack(spacing: 0) {
            // Search bar
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search by meaning...", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(.title3)
                if !searchText.isEmpty {
                    Button { searchText = ""; engine.results = [] } label: {
                        Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                    }.buttonStyle(.plain)
                }
            }
            .padding(12)

            Divider()

            #if !os(macOS)
            if horizontalSizeClass == .compact {
                compactSearchControls
                Divider()
            }
            #endif

            // Content
            if engine.isIndexing {
                Spacer()
                ProgressView("Indexing \(engine.indexedCount) / \(SampleData.sentences.count)...")
                Spacer()
            } else if engine.results.isEmpty && searchText.isEmpty {
                welcomeView
            } else if engine.results.isEmpty {
                Spacer()
                Text("No results for \"\(searchText)\"").foregroundStyle(.secondary)
                Spacer()
            } else {
                resultsList
            }

            Divider()
            statusBar
        }
    }

    // MARK: - Welcome

    private var welcomeView: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "magnifyingglass")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Semantic Search").font(.title2)
            Text("Search \(engine.indexedCount) items by meaning.\nAdd notes or import documents to grow the index.")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 300)
            Spacer()
        }
    }

    // MARK: - Results

    private var resultsList: some View {
        List(engine.results) { result in
            HStack(alignment: .top, spacing: 12) {
                Text(String(format: "%.2f", result.distance))
                    .font(.system(.caption, design: .monospaced))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(distanceColor(result.distance).opacity(0.15))
                    .foregroundStyle(distanceColor(result.distance))
                    .clipShape(RoundedRectangle(cornerRadius: 4))

                VStack(alignment: .leading, spacing: 4) {
                    Text(result.text)
                    Text(result.category)
                        .font(.caption)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(categoryColor(result.category).opacity(0.1))
                        .foregroundStyle(categoryColor(result.category))
                        .clipShape(RoundedRectangle(cornerRadius: 3))
                }
            }
            .padding(.vertical, 2)
        }
    }

    private var compactSearchControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("efSearch \(Int(engine.efSearch))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Slider(value: $engine.efSearch, in: 10...200, step: 10)
                    .onChange(of: engine.efSearch) {
                        if !searchText.isEmpty {
                            Task { await engine.search(searchText) }
                        }
                    }
            }

            HStack(spacing: 8) {
                TextField("Add a note...", text: $newNote)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit {
                        Task { await engine.addNote(newNote); newNote = "" }
                    }
                Button {
                    Task { await engine.addNote(newNote); newNote = "" }
                } label: {
                    Image(systemName: "plus.circle.fill")
                }
                .disabled(newNote.trimmingCharacters(in: .whitespaces).isEmpty)
            }

            HStack(spacing: 10) {
                Button {
                    showImagePicker = true
                } label: {
                    Label("Images", systemImage: "photo")
                }
                Button {
                    Task { await engine.rebuildIndex() }
                } label: {
                    Label("Rebuild", systemImage: "arrow.clockwise")
                }
            }
            .buttonStyle(.bordered)
            .font(.callout)
        }
        .padding(12)
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List {
            Section("Search Settings") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("efSearch: \(Int(engine.efSearch))")
                        .font(.caption).foregroundStyle(.secondary)
                    Slider(value: $engine.efSearch, in: 10...200, step: 10)
                        .onChange(of: engine.efSearch) {
                            // Re-run current search with new efSearch
                            if !searchText.isEmpty {
                                Task { await engine.search(searchText) }
                            }
                        }
                    Text("Higher = better results, slower").font(.caption2).foregroundStyle(.tertiary)
                }
                .padding(.vertical, 4)

                Button("Rebuild Index") {
                    Task { await engine.rebuildIndex() }
                }

                if engine.loadedFromDisk {
                    Text("Loaded from disk (instant)")
                        .font(.caption2)
                        .foregroundStyle(.green)
                }
            }

            Section("Your Notes") {
                HStack {
                    TextField("Add a note...", text: $newNote)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit {
                            Task { await engine.addNote(newNote); newNote = "" }
                        }
                    Button {
                        Task { await engine.addNote(newNote); newNote = "" }
                    } label: {
                        Image(systemName: "plus.circle.fill")
                    }
                    .disabled(newNote.trimmingCharacters(in: .whitespaces).isEmpty)
                }

                if engine.userNotes.isEmpty {
                    Text("Notes you add appear in search results")
                        .font(.caption).foregroundStyle(.tertiary)
                } else {
                    ForEach(engine.userNotes, id: \.self) { note in
                        Text(note).font(.caption).lineLimit(2)
                    }
                }
            }

            Section("Images") {
                Button("Add Images...") { showImagePicker = true }
                if engine.imageCount > 0 {
                    Text("\(engine.imageCount) images indexed")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("Add photos to search visually")
                        .font(.caption).foregroundStyle(.tertiary)
                }
            }

            Section("Index Stats") {
                LabeledContent("Vectors", value: "\(engine.indexedCount)")
                LabeledContent("efSearch", value: "\(Int(engine.efSearch))")
                if engine.lastQueryTimeMs > 0 {
                    LabeledContent("Last query", value: String(format: "%.1f ms", engine.lastQueryTimeMs))
                }
                LabeledContent("Notes", value: "\(engine.userNotes.count)")
                LabeledContent("Images", value: "\(engine.imageCount)")
                if !engine.embeddingSource.isEmpty {
                    Text(engine.embeddingSource)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .listStyle(.sidebar)
    }

    // MARK: - Status Bar

    private var statusBar: some View {
        HStack {
            Label("\(engine.indexedCount) vectors", systemImage: "square.stack.3d.up")
            Spacer()
            if engine.lastQueryTimeMs > 0 {
                Label(String(format: "%.1f ms", engine.lastQueryTimeMs), systemImage: "clock")
            }
            Label("ef=\(Int(engine.efSearch))", systemImage: "slider.horizontal.3")
        }
        .font(.caption).foregroundStyle(.secondary)
        .padding(.horizontal, 12).padding(.vertical, 6)
    }

    // MARK: - Colors

    // NLEmbedding distances are typically 0.5-0.8 range.
    // Adjusted thresholds for Apple's built-in model quality.
    private func distanceColor(_ d: Float) -> Color {
        d < 0.55 ? .green : d < 0.68 ? .orange : .red
    }

    private func categoryColor(_ c: String) -> Color {
        switch c {
        case "Animals": .brown; case "Food": .orange; case "Technology": .blue
        case "Nature": .green; case "Sports": .red; case "Science": .purple
        case "Travel": .cyan; case "Music": .pink; case "Arts": .mint
        case "Your Notes": .indigo; default: .gray
        }
    }
}
