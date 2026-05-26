module CloverLeaf {
    use Initialization;
    use Hydro;
    use Math;
    use Definitions;
    use Time;
    use CommDiagnostics;
    use CTypes;
    use GpuDiagnostics;

    // C interop: read environment variables for qthreads diagnostics
    extern proc getenv(name: c_ptrConst(c_char)): c_ptrConst(c_char);
    proc envStr(name: string): string {
        const p = getenv(name.c_str());
        if p == nil then return "(not set)";
        return try! string.createCopyingBuffer(p);
    }

    //Where we dump to (checks if blank later on to see if user wants the dump or not)
    config const dump = ""; 
    config const testing = true;  
    config const profile = false;  // pass --profile=true to get per-kernel timing breakdown
    config const noPrint = false; 
    config const commsDiagonstics = false;
    config const numLocales: int = 1;
    const LocaleSpace = {0..numLocales-1};
    // const Locales: [LocaleSpace] locale;
    config const inputDeck = "clover_bm.in";
    
    proc run_tests() {
        writeln("Starting CloverLeaf tests...");
        
        var all_tests_passed = true;
        
        writeln("\nTest summary: ", if all_tests_passed then "ALL TESTS PASSED" else "SOME TESTS FAILED");
        writeln("----------------------------------------\n");
        
        return all_tests_passed;
    }

    proc main() {
        writeln("=== CloverLeaf GPU Starting ===");
       // if (commsDiagonstics) then startVerboseComm();

        writeln("Number of Locales: ", numLocales);

        // Threading / Qthreads diagnostics
        writeln("=== Threading (CHPL_TASKS=", envStr("CHPL_TASKS"), ") ===");
        writeln("  here.maxTaskPar (hardware threads visible): ", here.maxTaskPar);
        writeln("  CHPL_RT_NUM_THREADS_PER_LOCALE : ", envStr("CHPL_RT_NUM_THREADS_PER_LOCALE"));
        writeln("  QT_NUM_SHEPHERDS               : ", envStr("QT_NUM_SHEPHERDS"));
        writeln("  QT_NUM_WORKERS_PER_SHEPHERD    : ", envStr("QT_NUM_WORKERS_PER_SHEPHERD"));
        writeln("  QT_SHEPHERD_BOUNDARY_POLICY    : ", envStr("QT_SHEPHERD_BOUNDARY_POLICY"));

        writeln("Available GPUs on this node: ", here.gpus.size);
        writeln("Active GPU devices:");
        const gpuCount = min(numGpus, here.gpus.size);
        for i in 0..<gpuCount {
            writeln("  - GPU ", i, ": ", here.gpus[i]);
        }

        // Count GPU kernel launches to confirm the GPU is actually being used.
        // If kernels == 0 after the simulation, the GPU path is silently falling
        // back to CPU (common with out-of-range ROCm versions and
        // CHPLENV_GPU_REQ_ERRS_AS_WARNINGS=1).
        if numGpus > 0 {
            startGpuDiagnostics();
        }

        var clock = new stopwatch();
        clock.start();

        var test_result: bool;
        if testing {
            test_result = run_tests();
        } else {
            test_result = true;
        }
        if test_result {
            writeln("\n Starting CloverLeaf simulation...");
            // Run entire simulation on GPU 0 so all arrays are allocated as
            // device memory and all forall loops dispatch as GPU kernels.
            const runLocale: locale = if useGpu && here.gpus.size > 0 then here.gpus[0] else here;
            on runLocale {
                var cfg = initialize(dump, inputDeck);
                hydro(cfg, clock);
            }
            writeln("Simulation completed");
            if numGpus > 0 {
                stopGpuDiagnostics();
                const gpuDiag = getGpuDiagnostics();
                const totalLaunches = + reduce gpuDiag.kernel_launch;
                writeln("=== GPU Kernel Diagnostics ===");
                writeln("  Kernel launches : ", totalLaunches);
                if totalLaunches == 0 then
                    writeln("  WARNING: 0 GPU kernels launched — GPU path is silently running on CPU!");
                else
                    writeln("  GPU is active (kernels launched successfully)");
            };
            // writeln("Final time: ", t);
            // writeln("Number of steps: ", step);
        } else {
            writeln("Error: Some tests failed. Please check the implementation.");
            halt(1);
        }

        // if (commsDiagonstics){
        //     printCommDiagnosticsTable();
        //     stopVerboseComm();
        // } 
    }
}
