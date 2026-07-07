import SwiftUI

struct IndexInspectorView: View {
    @Bindable var engine: SearchEngine
    @State private var graph: IndexInspectorGraph = .empty
    @State private var layout = ForceGraphLayout()
    @State private var selectedID: UUID?
    @State private var isPaused = false
    @State private var isLoading = false
    @State private var errorMessage: String?

    private let sampleLimit = 150

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                metrics
                graphPanel
                selectedNodePanel
                layerPanel

                if let errorMessage {
                    Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
            .padding(20)
            .frame(maxWidth: 860, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle("Inspector")
        .task(id: engine.indexedCount) {
            await reload()
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Index Inspector")
                    .font(.title2.weight(.semibold))
                Text("Live HNSW layer-0 adjacency, levels, and metadata from the read-only graph snapshot.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                Task { await reload() }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
            .disabled(isLoading)
        }
    }

    private var metrics: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 10)], spacing: 10) {
            metric("Live vectors", "\(graph.totalNodeCount)")
            metric("Showing", "\(graph.sampledNodeCount)")
            metric("Layer-0 links", "\(graph.edges.count)")
            metric("Avg degree", String(format: "%.1f", graph.averageLayer0Degree))
            metric("Max layer", "\(graph.maxLayer)")
        }
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.title3, design: .rounded).weight(.semibold))
                .monospacedDigit()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
    }

    private var graphPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label(
                    "Showing \(graph.sampledNodeCount) of \(graph.totalNodeCount)",
                    systemImage: graph.sampledNodeCount < graph.totalNodeCount ? "rectangle.stack.badge.person.crop" : "point.3.connected.trianglepath.dotted")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    isPaused.toggle()
                } label: {
                    Label(isPaused ? "Resume" : "Pause", systemImage: isPaused ? "play.fill" : "pause.fill")
                }
                .buttonStyle(.bordered)
                .disabled(graph.nodes.isEmpty)
            }

            ZStack {
                ForceGraphCanvas(
                    graph: graph,
                    layout: $layout,
                    selectedID: $selectedID,
                    isPaused: isPaused)
                if isLoading {
                    ProgressView("Loading graph...")
                        .padding(12)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                } else if graph.nodes.isEmpty {
                    Text("No vectors indexed yet.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(height: 420)
            .background(.background, in: RoundedRectangle(cornerRadius: 8))
            .overlay {
                RoundedRectangle(cornerRadius: 8)
                    .stroke(.quaternary, lineWidth: 1)
            }
        }
        .padding(14)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private var selectedNodePanel: some View {
        if let selected = selectedNode {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(selected.title)
                            .font(.headline)
                        Text(selected.subtitle)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("L\(selected.level)")
                        .font(.system(.caption, design: .monospaced).weight(.semibold))
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(.blue.opacity(0.14), in: Capsule())
                        .foregroundStyle(.blue)
                }

                Text(selected.text)
                    .font(.callout)
                    .lineLimit(6)
                    .textSelection(.enabled)

                Divider()

                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                    GridRow {
                        Text("UUID").foregroundStyle(.secondary)
                        Text(selected.id.uuidString).font(.system(.caption, design: .monospaced))
                    }
                    GridRow {
                        Text("Degree").foregroundStyle(.secondary)
                        Text("\(selected.degree)")
                    }
                    GridRow {
                        Text("Snapshot row").foregroundStyle(.secondary)
                        Text("\(selected.internalIndex)")
                    }
                }
                .font(.caption)
            }
            .padding(14)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var layerPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Layer height distribution")
                .font(.headline)
            HStack(alignment: .bottom, spacing: 8) {
                ForEach(graph.layerBuckets) { bucket in
                    VStack(spacing: 5) {
                        RoundedRectangle(cornerRadius: 3)
                            .fill(.blue.opacity(0.75))
                            .frame(height: layerBarHeight(bucket.count))
                        Text("L\(bucket.level)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Text("\(bucket.count)")
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: 46)
                }
            }
            .frame(height: 118, alignment: .bottom)
        }
        .padding(14)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
    }

    private var selectedNode: IndexInspectorNode? {
        guard let selectedID else { return graph.nodes.first }
        return graph.nodes.first { $0.id == selectedID } ?? graph.nodes.first
    }

    private func layerBarHeight(_ count: Int) -> CGFloat {
        let maxCount = max(graph.layerBuckets.map(\.count).max() ?? 1, 1)
        return max(8, CGFloat(count) / CGFloat(maxCount) * 72)
    }

    private func reload() async {
        isLoading = true
        errorMessage = nil
        do {
            let next = try await engine.makeInspectorGraph(sampleLimit: sampleLimit)
            graph = next
            if selectedID == nil || !next.nodes.contains(where: { $0.id == selectedID }) {
                selectedID = next.nodes.first?.id
            }
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}

private struct ForceGraphCanvas: View {
    let graph: IndexInspectorGraph
    @Binding var layout: ForceGraphLayout
    @Binding var selectedID: UUID?
    let isPaused: Bool

    var body: some View {
        GeometryReader { proxy in
            TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: isPaused)) { timeline in
                Canvas { context, size in
                    draw(in: context, size: size)
                }
                .onAppear {
                    layout.prepare(for: graph, in: proxy.size)
                }
                .onChange(of: graph.signature) {
                    layout.prepare(for: graph, in: proxy.size)
                }
                .onChange(of: timeline.date) {
                    guard !isPaused else { return }
                    layout.step(graph: graph, in: proxy.size)
                }
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onEnded { value in
                            selectedID = layout.nearestNode(to: value.location, graph: graph)
                        })
            }
        }
    }

    private func draw(in context: GraphicsContext, size: CGSize) {
        for edge in graph.edges {
            guard let a = layout.position(for: edge.source),
                  let b = layout.position(for: edge.target) else { continue }
            var path = Path()
            path.move(to: a)
            path.addLine(to: b)
            context.stroke(path, with: .color(.secondary.opacity(0.22)), lineWidth: 0.8)
        }

        for node in graph.nodes {
            guard let point = layout.position(for: node.id) else { continue }
            let radius = layout.radius(for: node)
            let rect = CGRect(
                x: point.x - radius,
                y: point.y - radius,
                width: radius * 2,
                height: radius * 2)
            let shape = Path(ellipseIn: rect)
            let isSelected = node.id == selectedID
            context.fill(shape, with: .color(.blue.opacity(isSelected ? 0.95 : 0.62)))
            context.stroke(
                shape,
                with: .color(isSelected ? .primary : .blue.opacity(node.level > 0 ? 0.95 : 0.35)),
                lineWidth: isSelected ? 2.4 : (node.level > 0 ? 1.5 : 0.8))
        }
    }
}

private struct ForceGraphLayout {
    private struct Body {
        var position: CGPoint
        var velocity: CGVector
    }

    private var signature = ""
    private var bodies: [UUID: Body] = [:]

    mutating func prepare(for graph: IndexInspectorGraph, in size: CGSize) {
        guard signature != graph.signature || bodies.count != graph.nodes.count else { return }
        signature = graph.signature
        bodies.removeAll(keepingCapacity: true)

        let usable = max(CGFloat(120), min(size.width, size.height))
        let center = CGPoint(x: max(size.width, 1) / 2, y: max(size.height, 1) / 2)
        var rng = DemoRNG(seed: 0xC0FF_EE)
        for (index, node) in graph.nodes.enumerated() {
            let jitter = Double(rng.next() % 10_000) / 10_000.0
            let angle = (Double(index) / Double(max(graph.nodes.count, 1))) * Double.pi * 2 + jitter * 0.55
            let radius = usable * CGFloat(0.20 + jitter * 0.22)
            bodies[node.id] = Body(
                position: CGPoint(
                    x: center.x + CGFloat(cos(angle)) * radius,
                    y: center.y + CGFloat(sin(angle)) * radius),
                velocity: .zero)
        }
    }

    mutating func step(graph: IndexInspectorGraph, in size: CGSize) {
        prepare(for: graph, in: size)
        guard graph.nodes.count > 1 else { return }

        var forces = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id, CGVector.zero) })
        let center = CGPoint(x: max(size.width, 1) / 2, y: max(size.height, 1) / 2)

        for i in graph.nodes.indices {
            for j in graph.nodes.indices where j > i {
                let aID = graph.nodes[i].id
                let bID = graph.nodes[j].id
                guard let a = bodies[aID]?.position, let b = bodies[bID]?.position else { continue }
                let dx = b.x - a.x
                let dy = b.y - a.y
                let distanceSquared = max(dx * dx + dy * dy, 64)
                let distance = sqrt(distanceSquared)
                let strength = CGFloat(2_700) / distanceSquared
                let fx = dx / distance * strength
                let fy = dy / distance * strength
                addForce(CGVector(dx: -fx, dy: -fy), to: aID, in: &forces)
                addForce(CGVector(dx: fx, dy: fy), to: bID, in: &forces)
            }
        }

        for edge in graph.edges {
            guard let a = bodies[edge.source]?.position,
                  let b = bodies[edge.target]?.position else { continue }
            let dx = b.x - a.x
            let dy = b.y - a.y
            let distance = max(sqrt(dx * dx + dy * dy), 1)
            let desired = CGFloat(58)
            let strength = (distance - desired) * 0.014
            let fx = dx / distance * strength
            let fy = dy / distance * strength
            addForce(CGVector(dx: fx, dy: fy), to: edge.source, in: &forces)
            addForce(CGVector(dx: -fx, dy: -fy), to: edge.target, in: &forces)
        }

        for node in graph.nodes {
            guard var body = bodies[node.id] else { continue }
            var force = forces[node.id] ?? .zero
            force.dx += (center.x - body.position.x) * 0.002
            force.dy += (center.y - body.position.y) * 0.002

            body.velocity.dx = (body.velocity.dx + force.dx) * 0.86
            body.velocity.dy = (body.velocity.dy + force.dy) * 0.86
            let speed = max(sqrt(body.velocity.dx * body.velocity.dx + body.velocity.dy * body.velocity.dy), 1)
            if speed > 12 {
                body.velocity.dx = body.velocity.dx / speed * 12
                body.velocity.dy = body.velocity.dy / speed * 12
            }

            body.position.x += body.velocity.dx
            body.position.y += body.velocity.dy
            let padding = CGFloat(24)
            body.position.x = min(max(body.position.x, padding), max(padding, size.width - padding))
            body.position.y = min(max(body.position.y, padding), max(padding, size.height - padding))
            bodies[node.id] = body
        }
    }

    func position(for id: UUID) -> CGPoint? {
        bodies[id]?.position
    }

    func radius(for node: IndexInspectorNode) -> CGFloat {
        CGFloat(5 + min(node.level, 4) * 2)
    }

    func nearestNode(to point: CGPoint, graph: IndexInspectorGraph) -> UUID? {
        var best: (id: UUID, distance: CGFloat)?
        for node in graph.nodes {
            guard let position = position(for: node.id) else { continue }
            let dx = position.x - point.x
            let dy = position.y - point.y
            let distance = sqrt(dx * dx + dy * dy)
            if best == nil || distance < best!.distance {
                best = (node.id, distance)
            }
        }
        guard let best, best.distance <= 28 else { return nil }
        return best.id
    }

    private func addForce(_ force: CGVector, to id: UUID, in forces: inout [UUID: CGVector]) {
        var existing = forces[id] ?? .zero
        existing.dx += force.dx
        existing.dy += force.dy
        forces[id] = existing
    }
}

private extension IndexInspectorGraph {
    var signature: String {
        "\(totalNodeCount)-\(sampledNodeCount)-\(edges.count)-" + nodes.map(\.id.uuidString).joined(separator: "|")
    }
}
