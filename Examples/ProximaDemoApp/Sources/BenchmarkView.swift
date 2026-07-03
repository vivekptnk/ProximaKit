// BenchmarkView.swift
// ProximaDemoApp
//
// v1.6.0 Benchmark tab UI: runs the efSearch sweep and charts recall@10
// against live median latency with SwiftUI Charts.

import SwiftUI
import Charts

struct BenchmarkView: View {
    @Bindable var engine: BenchmarkEngine

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header

                if engine.phase == .running {
                    runningCard
                } else {
                    runButton
                }

                if let error = engine.errorMessage {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                if !engine.results.isEmpty {
                    chartCard
                    tableCard
                    footnote
                }
            }
            .padding(20)
            .frame(maxWidth: 640, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle("Benchmark")
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("efSearch Sweep")
                .font(.title2.weight(.semibold))
            Text("Recall@\(engine.k) vs latency across ef = \(engine.efValues.map(String.init).joined(separator: " / ")). "
                 + "Recall is measured against an exact BruteForceIndex over the same corpus.")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text("Corpus: \(engine.corpusSize) synthetic vectors × \(engine.dimension)d · \(engine.queryCount) seeded queries · reproducible")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }

    // MARK: - Run controls

    private var runButton: some View {
        Button {
            Task { await engine.run() }
        } label: {
            Label(engine.phase == .done ? "Run Again" : "Run Benchmark",
                  systemImage: "play.fill")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
    }

    private var runningCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            ProgressView(value: engine.progress) {
                Text(engine.statusLabel.isEmpty ? "Running…" : engine.statusLabel)
                    .font(.callout)
            }
            Text("Building an exact + approximate index and sweeping — this does real work on \(engine.corpusSize) vectors.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(16)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Chart

    private var chartCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recall vs. median latency")
                .font(.headline)
            Chart(engine.results) { point in
                LineMark(
                    x: .value("Median latency (ms)", point.medianMs),
                    y: .value("Recall@10", point.recallAt10))
                .foregroundStyle(.blue)
                .interpolationMethod(.catmullRom)

                PointMark(
                    x: .value("Median latency (ms)", point.medianMs),
                    y: .value("Recall@10", point.recallAt10))
                .foregroundStyle(.blue)
                .annotation(position: .top, alignment: .center) {
                    Text("ef \(point.efSearch)")
                        .font(.system(.caption2, design: .rounded))
                        .foregroundStyle(.secondary)
                }
            }
            .chartYScale(domain: 0...1)
            .chartXAxisLabel("Median query latency (ms)")
            .chartYAxisLabel("Recall@10")
            .frame(height: 260)
        }
        .padding(16)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Table

    private var tableCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("efSearch").frame(width: 80, alignment: .leading)
                Text("Recall@10").frame(maxWidth: .infinity, alignment: .trailing)
                Text("Median").frame(maxWidth: .infinity, alignment: .trailing)
                Text("p90").frame(maxWidth: .infinity, alignment: .trailing)
            }
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
            .padding(.vertical, 6)

            Divider()

            ForEach(engine.results) { point in
                HStack {
                    Text("\(point.efSearch)")
                        .frame(width: 80, alignment: .leading)
                    Text(String(format: "%.3f", point.recallAt10))
                        .foregroundStyle(recallColor(point.recallAt10))
                        .frame(maxWidth: .infinity, alignment: .trailing)
                    Text(String(format: "%.3f ms", point.medianMs))
                        .frame(maxWidth: .infinity, alignment: .trailing)
                    Text(String(format: "%.3f ms", point.p90Ms))
                        .frame(maxWidth: .infinity, alignment: .trailing)
                }
                .font(.system(.callout, design: .monospaced))
                .padding(.vertical, 5)
                Divider()
            }
        }
        .padding(16)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 12))
    }

    private var footnote: some View {
        Text("Recall@10 is reproducible (seeded corpus + fixed graph seed). Latency is measured live and varies run to run; last run took \(String(format: "%.1f", engine.lastRunSeconds)) s.")
            .font(.caption2)
            .foregroundStyle(.tertiary)
    }

    // Green when recall is high, orange mid, red low — mirrors the search
    // pane's distance color language.
    private func recallColor(_ r: Double) -> Color {
        r >= 0.9 ? .green : r >= 0.7 ? .orange : .red
    }
}
