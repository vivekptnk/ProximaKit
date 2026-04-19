// Platform.swift
// Reports OS / arch / CPU model / RSS. Uses libc + mach APIs only.

import Darwin
import Foundation

enum PlatformProbe {
    static func current() -> Platform {
        Platform(
            os: uname("sysname").lowercased(),
            kernel: uname("release"),
            arch: uname("machine"),
            cpuModel: sysctlString("machdep.cpu.brand_string") ?? "unknown",
            swiftVersion: swiftVersion(),
            pythonVersion: nil
        )
    }

    /// Resident set size in bytes, via `mach_task_basic_info`.
    /// Returns 0 on failure rather than throwing — benchmarks should not abort
    /// just because a memory probe failed.
    static func residentMemoryBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
            }
        }
        return result == KERN_SUCCESS ? info.resident_size : 0
    }

    // MARK: - Private

    private static func uname(_ field: String) -> String {
        var buf = utsname()
        guard Darwin.uname(&buf) == 0 else { return "unknown" }
        switch field {
        case "sysname":
            return withUnsafePointer(to: &buf.sysname) { ptr in
                ptr.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) { String(cString: $0) }
            }
        case "release":
            return withUnsafePointer(to: &buf.release) { ptr in
                ptr.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) { String(cString: $0) }
            }
        case "machine":
            return withUnsafePointer(to: &buf.machine) { ptr in
                ptr.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) { String(cString: $0) }
            }
        default:
            return "unknown"
        }
    }

    private static func sysctlString(_ name: String) -> String? {
        var size: size_t = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
        return String(cString: buf)
    }

    private static func swiftVersion() -> String {
        #if swift(>=6.0)
        return "6.0"
        #elseif swift(>=5.10)
        return "5.10"
        #elseif swift(>=5.9)
        return "5.9"
        #else
        return "unknown"
        #endif
    }
}
