// chapel_stream.chpl — GPU STREAM benchmark
// Measures sustained memory bandwidth for Copy/Scale/Add/Triad.
// Arrays are allocated on the GPU locale so all kernels run on device.

use GPU;
use Time;

config const N      = 40_000_000;   // elements per array (~960 MB total for 3×real64)
config const NTIMES = 20;           // repetitions per operation (best of all taken)
config const scalar: real(64) = 3.0;

proc main() {
    if here.gpus.size == 0 { writeln("No GPU found"); return; }

    writeln("Chapel GPU STREAM  N=", N, "  NTIMES=", NTIMES);
    writeln("Array size: ", (N * 8 / 1024.0 / 1024.0):string, " MB each");
    writeln();

    on here.gpus[0] {
        var A: [0..#N] real(64);
        var B: [0..#N] real(64);
        var C: [0..#N] real(64);

        // Initialise
        foreach i in 0..#N { A[i] = 1.0; B[i] = 2.0; C[i] = 0.0; }

        // --- warmup ---
        foreach i in 0..#N { C[i] = A[i]; }

        // ---- Copy: C = A  (2 arrays touched) ----
        var bestCopy = 0.0;
        for _rep in 1..NTIMES {
            var t: stopwatch; t.start();
            foreach i in 0..#N { C[i] = A[i]; }
            t.stop();
            const bw = 2.0 * 8 * N / t.elapsed() / 1e9;
            if bw > bestCopy then bestCopy = bw;
        }
        writeln("Copy:  ", bestCopy, " GB/s");

        // ---- Scale: B = scalar * C  (2 arrays) ----
        var bestScale = 0.0;
        for _rep in 1..NTIMES {
            var t: stopwatch; t.start();
            foreach i in 0..#N { B[i] = scalar * C[i]; }
            t.stop();
            const bw = 2.0 * 8 * N / t.elapsed() / 1e9;
            if bw > bestScale then bestScale = bw;
        }
        writeln("Scale: ", bestScale, " GB/s");

        // ---- Add: C = A + B  (3 arrays) ----
        var bestAdd = 0.0;
        for _rep in 1..NTIMES {
            var t: stopwatch; t.start();
            foreach i in 0..#N { C[i] = A[i] + B[i]; }
            t.stop();
            const bw = 3.0 * 8 * N / t.elapsed() / 1e9;
            if bw > bestAdd then bestAdd = bw;
        }
        writeln("Add:   ", bestAdd, " GB/s");

        // ---- Triad: A = B + scalar*C  (3 arrays) ----
        var bestTriad = 0.0;
        for _rep in 1..NTIMES {
            var t: stopwatch; t.start();
            foreach i in 0..#N { A[i] = B[i] + scalar * C[i]; }
            t.stop();
            const bw = 3.0 * 8 * N / t.elapsed() / 1e9;
            if bw > bestTriad then bestTriad = bw;
        }
        writeln("Triad: ", bestTriad, " GB/s");
    }
}
