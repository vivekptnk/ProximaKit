// Heap.swift
// ProximaKit
//
// A binary heap (priority queue) used internally by HNSW search.
// Swift doesn't ship a built-in heap, so we implement a minimal one.
//
// Binary heap properties:
// - Insert: O(log n)
// - Pop min/max: O(log n)
// - Peek min/max: O(1)
// - Storage: a flat array (cache-friendly, no pointer chasing)

/// A binary heap that can be configured as min-heap or max-heap.
///
/// Used by NSW/HNSW search to maintain the candidate set (min-heap: always
/// explore the closest node) and result set (max-heap: evict the furthest
/// result when the set is full).
struct Heap<Element> {
    private var elements: [Element] = []
    private let comparator: (Element, Element) -> Bool

    /// The number of elements in the heap.
    var count: Int { elements.count }

    /// Whether the heap is empty.
    var isEmpty: Bool { elements.isEmpty }

    /// Creates a heap with a custom comparator.
    ///
    /// - Parameter comparator: Returns `true` if the first element should be
    ///   closer to the top. Use `<` for a min-heap, `>` for a max-heap.
    init(comparator: @escaping (Element, Element) -> Bool) {
        self.comparator = comparator
    }

    /// Returns the top element without removing it.
    func peek() -> Element? {
        elements.first
    }

    /// Adds an element to the heap.
    mutating func push(_ element: Element) {
        elements.append(element)
        siftUp(from: elements.count - 1)
    }

    /// Removes and returns the top element.
    @discardableResult
    mutating func pop() -> Element? {
        guard !elements.isEmpty else { return nil }
        if elements.count == 1 { return elements.removeLast() }
        let top = elements[0]
        elements[0] = elements.removeLast()
        siftDown(from: 0)
        return top
    }

    /// Returns all elements as an unsorted array.
    func toArray() -> [Element] {
        elements
    }

    // ── Heap Operations ──────────────────────────────────────────────

    // siftUp: After inserting at the end, bubble the element up
    // until the heap property is restored.
    //
    // Parent of index i is at (i - 1) / 2.
    private mutating func siftUp(from index: Int) {
        var child = index
        while child > 0 {
            let parent = (child - 1) / 2
            if comparator(elements[child], elements[parent]) {
                elements.swapAt(child, parent)
                child = parent
            } else {
                break
            }
        }
    }

    // siftDown: After removing the top and moving the last element there,
    // push it down until the heap property is restored.
    //
    // Children of index i are at 2*i+1 (left) and 2*i+2 (right).
    private mutating func siftDown(from index: Int) {
        var parent = index
        let count = elements.count

        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var candidate = parent

            if left < count && comparator(elements[left], elements[candidate]) {
                candidate = left
            }
            if right < count && comparator(elements[right], elements[candidate]) {
                candidate = right
            }
            if candidate == parent { break }

            elements.swapAt(parent, candidate)
            parent = candidate
        }
    }
}
