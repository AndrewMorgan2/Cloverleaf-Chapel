// gpu_sync_bench.chpl
//
// Tests whether Chapel's GPU overhead is per `on here.gpus` block crossing
// or per `foreach` kernel launch.
//
// All three configs do identical work: 2*REPS kernels over an N×N array.
// Arrays are pre-allocated once — no allocation inside the timing loops.
//
// Config B: 1 on-block, for-loop with 2 foreach inside.
//           Minimum possible block crossings (1 total).
//
// Config C: REPS on-blocks, each with 2 foreach inside.
//           Half the block crossings of Config A.
//
// Config A: 2*REPS on-blocks, each with 1 foreach.
//           Maximum block crossings — mirrors current CloverLeaf pattern.
//
// Interpretation:
//   If per-on-block overhead dominates:   B << C << A
//   If per-foreach overhead dominates:    B ≈ C ≈ A

use GPU;
use Time;

config const N    = 960;
config const REPS = 87;

// Hold GPU-resident arrays.  Allocated once on the GPU locale.
class GpuBuf {
    var X: [1..N, 1..N] real;
    var Y: [1..N, 1..N] real;
}

proc main() {
    if here.gpus.size == 0 { writeln("No GPU available"); return; }
    const gpu = here.gpus[0];

    // --- Allocate arrays on the GPU locale ---
    var buf: owned GpuBuf?;
    on gpu { buf = new owned GpuBuf(); }
    const D = {1..N, 1..N};

    // --- Warmup: a few passes to pages/JIT settled ---
    for _w in 1..5 {
        on gpu {
            ref X = buf!.X; ref Y = buf!.Y;
            foreach (i,j) in D { X[i,j] += Y[i,j]; }
            foreach (i,j) in D { Y[i,j] += X[i,j]; }
        }
    }

    // -------------------------------------------------------
    // Config B — 1 on-block, 2*REPS foreach inside
    // -------------------------------------------------------
    var tB: stopwatch;
    tB.start();
    on gpu {
        ref X = buf!.X; ref Y = buf!.Y;
        for _rep in 1..REPS {
            foreach (i,j) in D { X[i,j] += Y[i,j]; }
            foreach (i,j) in D { Y[i,j] += X[i,j]; }
        }
    }
    tB.stop();
    writeln("B  1 on-block,    ", 2*REPS, " foreach:  ", tB.elapsed():string, " s");

    // -------------------------------------------------------
    // Config C — REPS on-blocks, 2 foreach each
    // -------------------------------------------------------
    var tC: stopwatch;
    tC.start();
    for _rep in 1..REPS {
        on gpu {
            ref X = buf!.X; ref Y = buf!.Y;
            foreach (i,j) in D { X[i,j] += Y[i,j]; }
            foreach (i,j) in D { Y[i,j] += X[i,j]; }
        }
    }
    tC.stop();
    writeln("C  ", REPS, " on-blocks,   ", 2*REPS, " foreach:  ", tC.elapsed():string, " s");

    // -------------------------------------------------------
    // Config A — 2*REPS on-blocks, 1 foreach each
    // -------------------------------------------------------
    var tA: stopwatch;
    tA.start();
    for _rep in 1..REPS {
        on gpu { ref X = buf!.X; ref Y = buf!.Y; foreach (i,j) in D { X[i,j] += Y[i,j]; } }
        on gpu { ref X = buf!.X; ref Y = buf!.Y; foreach (i,j) in D { Y[i,j] += X[i,j]; } }
    }
    tA.stop();
    writeln("A  ", 2*REPS, " on-blocks,  ", 2*REPS, " foreach:  ", tA.elapsed():string, " s");

    writeln();
    writeln("B→C overhead (per on-block crossing, REPS blocks): ",
            ((tC.elapsed() - tB.elapsed()) / REPS * 1000):string, " ms/block");
    writeln("C→A overhead (extra REPS block crossings):          ",
            ((tA.elapsed() - tC.elapsed()) / REPS * 1000):string, " ms/block");
    writeln("B→A overhead (all ", 2*REPS, " extra crossings):       ",
            ((tA.elapsed() - tB.elapsed()) / (2*REPS) * 1000):string, " ms/block");
}
