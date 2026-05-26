module Definitions {
    use Math;
    use IO;
    use List;
    use GpuDiagnostics;
    use GPU;

    // GPU configuration
    config param gpuBlockSize = 256;
    config const useGpu = true;
    config const numGpus = 1;  // informational only

    // Set to true to print [diag] kernel-completion breadcrumbs
    config const diagPrints = false;

    //Profiling data
    class ProfilingTimes {
        var timestep, idealGas, viscosity, pdv, revert, accelerate, fluxes, cellAdvection, momAdvection, reset, halo: real = 0.0;
    }

    // Constants that are known at compile time
    param g_small = 1.0e-16;
    param g_big = 1.0e+21;

    param dtinit = 0.1;
    param dtmax = 0.04;
    param dtmin = 0.0000001;
    param dtrise = 1.5;
    param dtc_safe = 0.7;
    param dtu_safe = 0.5;
    param dtv_safe = 0.5;
    param dtdiv_safe = 0.7;

    config const x_size = 100;
    config const y_size = 100;

    /* Enums for geometry types */
    enum GeometryType {
        g_rect = 0,
        g_circ = 1,
        g_point = 2
    }

    param chunk_left = 0;
    param chunk_right = 1;
    param chunk_bottom = 2;
    param chunk_top = 3;
    param external_face = -1;

    param tile_left = 0;
    param tile_right = 1;
    param tile_bottom = 2;
    param tile_top = 3;
    param external_tile = -1;

    /* Grid configuration record */
    class GridConfig {
        var xmin, xmax, ymin, ymax, left, right, bottom, top, x_cells, y_cells, total_cells: int = 0;
    }

    /* State record to hold state properties */
    record State {
        var density, energy, xvel, yvel, xmin, xmax, ymin, ymax, radius: real = 0.0;
        var geometry: int = GeometryType.g_rect: int;
    }

    // For GPU, we use local distribution
    param distribution = "local";

    class Field {
        param distType = "local";

        var xsize, ysize: int;

        var gridDom = {0..xsize, 0..ysize};
        var vargridDom = {0..xsize-1, 0..ysize-1};
        var varGridDom = {0..xsize-1, 0..ysize-1};
        var varYGridDom = {0..xsize-1, 0..ysize};
        var varXGridDom = {0..xsize, 0..ysize-1};
        var xCellDom = {0..xsize-1};
        var yCellDom = {0..ysize-1};
        var xDom = {1..xsize};
        var yDom = {1..ysize};

        // Time-evolved variables (double buffered)
        var density0, density1, energy0, energy1, work_array1, work_array2, work_array3, work_array4, work_array5, work_array6, work_array7: [varGridDom] real;
        var xvel0, xvel1, yvel0, yvel1: [gridDom] real;

        // Diagnostic variables
        var pressure, viscosity, soundspeed: [varGridDom] real;

        // Flux variables
        var vol_flux_x, mass_flux_x: [varXGridDom] real;
        var vol_flux_y, mass_flux_y: [varYGridDom] real;

        // Grid geometry - 1D arrays
        var cellx, celldx: [xCellDom] real;
        var celly, celldy: [yCellDom] real;
        var vertexx, vertexdx: [xDom] real;
        var vertexy, vertexdy: [yDom] real;

        // Grid geometry - 2D arrays
        var volume: [varGridDom] real;
        var xarea: [varXGridDom] real;
        var yarea: [varYGridDom] real;

        //Strides
        var flux_x_stride, vels_wk_stride = xsize;
        var flux_y_stride = xsize - 1;
        var base_stride = xsize - 1;

        proc init(param distType: string, xsize: int, ysize: int) {
            this.distType = distType;
            this.xsize = xsize;
            this.ysize = ysize;
        }
    }

    class TileInfo{
        var t_xmax, t_ymax, t_xmin, t_ymin, t_bottom, t_top, t_right, t_left: int;
    }

    class Config {
        var xsize, ysize: int;
        var field = new Field(distribution, xsize, ysize);
        var dumpDir: string;
        var states: list(State);
        var grid = new GridConfig();
        var info = new TileInfo();
        var max_timestep, timestep_rise, time, initial_timestep: real;
        var jdt, kdt, test_problem, end_step, step, number_of_states: int;
        var dt = 0.04;
        var end_time = 1000.0;
        var dtold = 0.04;
        var dtinit =0.1;
        var dtmax = 0.04;
        var dtmin = 0.0000001;
        var dtrise = 1.5;
        var dtc_safe = 0.7;
        var dtu_safe = 0.5;
        var dtv_safe = 0.5;
        var dtdiv_safe = 0.7;

        var report_test_fail: bool;
        var advect_x: bool = true;

        //Profiling
        var profile_times = new ProfilingTimes();

        // Method to add a state
        proc addState(state: State) {
            number_of_states += 1;
            states.pushBack(state);
        }
    }

    //Answers, the values we check at the end to see if we are near
    record TestProblem {
        var id: int;
        var expected_ke: real;
    }

    const testProblems: [1..8] TestProblem = [
        new TestProblem(1, 1.82280367310258),
        new TestProblem(2, 1.19316898756307),   // bm * 87
        new TestProblem(3, 2.58984003503994),   // bm * 2955
        new TestProblem(4, 0.307475452287895),  // bm16 @ 87
        new TestProblem(5, 4.85350315783719),   // bm15 * 2955
        new TestProblem(487, 6.088288e-01),     // bm4 @ 87
        new TestProblem(287, 6.062609e-01),     // bm2 @ 87
        new TestProblem(168, 2.465082e-02)      // bm16 @ 8
    ];
}
