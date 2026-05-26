// src/initialization.chpl
module Initialization {
    use Debug;
    use BlockDist;
    use IO;
    use FileSystem;
    use List;
    use Definitions;
    use ReadingConfig;
    use IdealGas;
    use Update_halo;

    /* Function to print state variables from a list of states */
    proc printStates(states: list(State)) {
        writeln("State List Information:");
        writeln("----------------------");
        
        for i in 0..states.size-1 {
            writeln("State ", i, ":");
            writeln("  Density: ", states[i].density);
            writeln("  Energy: ", states[i].energy);
            writeln("  Velocity (x,y): (", states[i].xvel, ", ", states[i].yvel, ")");
            writeln("  X bounds: [", states[i].xmin, ", ", states[i].xmax, "]");
            writeln("  Y bounds: [", states[i].ymin, ", ", states[i].ymax, "]");
            writeln("  Radius: ", states[i].radius);
            if (states[i].geometry == 0) {
                writeln("  Geometry: rectangle");
            }
            if (states[i].geometry == 1){
                writeln("  Geometry: circle");
            }
            if (states[i].geometry == 2) {
                writeln("  Geometry: point");
            }
            writeln();
        }
    }

    proc initialise_chunk(tile: int, cfg: Config) {
        // Calculate grid spacing
        const dx = (cfg.info.t_xmax- cfg.info.t_xmin) / cfg.grid.x_cells: real;
        const dy = (cfg.info.t_ymax - cfg.info.t_ymin) / cfg.grid.y_cells: real;

        writeln("Grid configuration:");
        writeln("xmin: ", cfg.info.t_xmin);
        writeln("xmax: ", cfg.info.t_xmax);
        writeln("ymin: ", cfg.info.t_ymin); 
        writeln("ymax: ", cfg.info.t_xmax);
        writeln("x_cells: ", cfg.grid.x_cells);
        writeln("y_cells: ", cfg.grid.y_cells);
        writeln("Grid spacing - dx: ", dx, " dy: ", dy);        

        // Calculate minimum coordinates for this tile
        const xmin = cfg.info.t_xmin + dx * (cfg.info.t_left - 1): real;
        const ymin = cfg.info.t_ymin + dy * (cfg.info.t_bottom - 1): real;
        writeln("Minimum coordinates - xmin: ", xmin, " ymin: ", ymin);

        const x_min = cfg.grid.xmin;
        const x_max = cfg.grid.xmax;
        const y_min = cfg.grid.ymin;
        const y_max = cfg.grid.ymax;
        writeln("Tile bounds - x_min:", x_min,", x_max: ", x_max, ", y_min: " , y_min,", y_max:", y_max);
        
        // Calculate ranges including ghost cells
        const xrange = ((cfg.grid.xmax + 3) - (cfg.grid.xmin - 2) + 1):int;
        const yrange = ((cfg.grid.ymax + 3) - (cfg.grid.ymin - 2) + 1):int;
        writeln("Ranges with ghost cells - xrange: ", xrange, " yrange: ", yrange);

        // Reference to field for cleaner access
        ref field = cfg.field;
        
        // Create domains for parallel iteration (1-based to match array bounds)
        const vertexDomX = {1..xrange};
        const vertexDomY = {1..yrange};
        writeln("Vertex domains - vertexDomX: ", vertexDomX, " vertexDomY: ", vertexDomY);

        // Initialize vertex coordinates and spacing
        forall j in vertexDomX {
            field.vertexx[j] = xmin + dx * (j: real - 1 - x_min);
            field.vertexdx[j] = dx;
        }

        forall k in vertexDomY {
            field.vertexy[k] = ymin + dy * (k: real - 1 - y_min);
            field.vertexdy[k] = dy;
        }

        // Calculate ranges for cell-centered quantities
        const xrange1 = ((cfg.grid.xmax + 2) - (cfg.grid.xmin - 2) + 1):int;
        const yrange1 = ((cfg.grid.ymax + 2) - (cfg.grid.ymin - 2) + 1):int;
        writeln("Cell-centered ranges - xrange1: ", xrange1, " yrange1: ", yrange1);

        const cellDomX = {1..xrange1};
        const cellDomY = {1..yrange1};
        // writeln("Cell domains - cellDomX: ", cellDomX, " cellDomY: ", cellDomY);

        // Initialize cell-centered coordinates and spacing
        field.cellx[0] = field.vertexx[1] - 0.5 * dx;
        field.celldx[0] = dx;
        forall j in cellDomX {
            field.cellx[j] = 0.5 * (field.vertexx[j] + field.vertexx[j + 1]);
            field.celldx[j] = dx;
        }

        field.celly[0] = field.vertexy[1] - 0.5 * dy;
        field.celldy[0] = dy;
        forall k in cellDomY {
            field.celly[k] = 0.5 * (field.vertexy[k] + field.vertexy[k + 1]);
            field.celldy[k] = dy;
        }

        // Assuming field is a record/class with similar structure
        const base_stride = field.base_stride: int;
        const flux_x_stride = field.flux_x_stride: int;
        const flux_y_stride = field.flux_y_stride: int;

        writeln("Hello! base_stride: ", base_stride, 
                ", flux_x_stride: ", flux_x_stride,
                ", flux_y_stride: ", flux_y_stride);


        // Initialize volume and areas (1-based to match array bounds)
        const volDom2D = {1..xrange1, 1..yrange1};
        // writeln("Cell-centered ranges - xrange1 and yrange1 ", cellDom2D);
        var count: int = 0;
        forall (i,j) in volDom2D {
            field.volume[i,j] = dx * dy;
            field.xarea[i, j] = field.celldy[j];
            field.yarea[i,j] = field.celldx[i];
            //count += 1;
        }

        writeln("count ", count);                      
    }

    proc generate_chunk(tile: int, cfg: Config): (int, int){
        // Declare arrays to store state data
        var state_density, state_energy, state_xvel, state_yvel, state_xmax, state_xmin, state_ymax, state_ymin, state_radius, state_geometry: [0..#cfg.number_of_states] real;

        // Copy data from globals into arrays
        forall state in 0..#cfg.number_of_states {
            state_density[state] = cfg.states[state].density;
            state_energy[state] = cfg.states[state].energy;
            state_xvel[state] = cfg.states[state].xvel;
            state_yvel[state] = cfg.states[state].yvel;
            state_xmin[state] = cfg.states[state].xmin;
            state_xmax[state] = cfg.states[state].xmax;
            state_ymin[state] = cfg.states[state].ymin;
            state_ymax[state] = cfg.states[state].ymax;
            state_radius[state] = cfg.states[state].radius;
            state_geometry[state] = cfg.states[state].geometry;
        }
        
        writeln("xmin", state_xmin[1]);
        writeln("xmax",state_xmax[1]);
        writeln("ymin",state_ymin[1]);
        writeln("ymax",state_ymax[1]);

        const x_min = cfg.grid.xmin;
        const x_max = cfg.grid.xmax;
        const y_min = cfg.grid.ymin;
        const y_max = cfg.grid.ymax;

        const xrange = ((x_max + 2) - (x_min - 2) + 1):int;
        const yrange = ((y_max + 2) - (y_min - 2) + 1):int;

        ref field = cfg.field;

        // Create a domain for the 2D range (1-based to match array bounds)
        const xyDomain = {1..xrange, 1..yrange};
        // Set background state (state 0)
        forall (i,j) in xyDomain {
            field.energy0[i,j] = state_energy[0];
            field.density0[i,j] = state_density[0];
            field.xvel0[i,j] = state_xvel[0];
            field.yvel0[i,j] = state_yvel[0];
        }

        // Find min/max j indices (1-based)
        var min_j = xrange+1;
        var max_j = 1;
        for j in 1..xrange {
            if (field.vertexx[j] < state_xmax[1] && field.vertexx[j+1] >= state_xmin[1]) {
                min_j = min(min_j, j);
                max_j = max(max_j, j);
            }
        }

        // Find min/max k indices (1-based)
        var min_k = yrange+1;
        var max_k = 1;
        for k in 1..yrange {
            if (field.vertexy[k] < state_ymax[1] && field.vertexy[k+1] >= state_ymin[1]) {
                min_k = min(min_k, k);
                max_k = max(max_k, k);
            }
        }

        writeln("Min j: ", min_j, " Max j: ", max_j);
        writeln("Min k: ", min_k, " Max k: ", max_k);

        // Process other states
        for state in 1..#cfg.number_of_states-1 {
            // writeln(state_energy[state]);
            forall (j,k) in xyDomain{
                const x_cent = state_xmin[state];
                const y_cent = state_ymin[state];
                select state_geometry[state] {
                    when GeometryType.g_rect:int {
                        if (field.vertexx[j+1] >= state_xmin[state] && field.vertexx[j] < state_xmax[state]) {
                            if (field.vertexy[k+1] >= state_ymin[state] && field.vertexy[k] < state_ymax[state]) {        
                                if(j == 1 || k == 1){
                                    // writeln("Here's the problem");
                                }                    
                                field.energy0[j,k] = state_energy[state];
                                field.density0[j,k] = state_density[state];
                                //writeln(j,",",k);
                                for kt in k..k+1 {
                                    for jt in j..j+1 {
                                        field.xvel0[jt,kt] = state_xvel[state];
                                        field.yvel0[jt,kt] = state_yvel[state];
                                    }
                                }
                            }
                        }
                    }
                    when GeometryType.g_circ:int {
                        const radius = sqrt((field.cellx[j] - x_cent)**2 + (field.celly[k] - y_cent)**2);
                        if (radius <= state_radius[state]) {
                            field.energy0[j,k] = state_energy[state];
                            field.density0[j,k] = state_density[state];
                            
                            for kt in k..k+1 {
                                for jt in j..j+1 {
                                    field.xvel0[jt,kt] = state_xvel[state];
                                    field.yvel0[jt,kt] = state_yvel[state];
                                }
                            }
                        }
                    }
                    when GeometryType.g_point:int {
                        if (field.vertexx[j] == x_cent && field.vertexy[k] == y_cent) {
                            field.energy0[j,k] = state_energy[state];
                            field.density0[j,k] = state_density[state];
                            
                            for kt in k..k+1 {
                                for jt in j..j+1 {
                                    field.xvel0[jt,kt] = state_xvel[state];
                                    field.yvel0[jt,kt] = state_yvel[state];
                                }
                            }
                        }
                    }
                }
            }
        }
        return (max_j, max_k);
    }

    //This replaced readConfig and start in the other cloverleaf implementations
    proc initialize(dump_dir: string, inputDeck: string) : Config {
        //Reading in setup
        try {
            var cfg = readin(inputDeck);
            cfg.dumpDir = dump_dir;
            try {
                dumpState(cfg, "0_0_00_build_field.txt", 0.0, 0); //need to update this to show which rank/device + step etc
            } catch e {
                writeln("Failed to dump debug state: ", e.message());
            }
            writeln("Problem initialising");
            //why not working total grid cells is an int?
            cfg.grid.total_cells = 1;
            for i in 0..<cfg.grid.total_cells do 
                initialise_chunk(i, cfg);  

            //Proof indexing works for debug
            // cfg.field.volume[0,0] = 0.000108507;

            try {
                dumpState(cfg, "0_0_01_init_chunk.txt", 0.0, 0); //need to update this to show which rank/device + step etc
            } catch e {
                writeln("Failed to dump debug state: ", e.message());
            } 

            writeln("Problem initialised, problem generating");

            var max_j: int = 0;
            var max_k: int = 0;

            for i in 0..<cfg.grid.total_cells {
                const (curr_j, curr_k) = generate_chunk(i, cfg);
                max_j = max(max_j, curr_j);
                max_k = max(max_k, curr_k);
            }

            try {
                dumpState(cfg, "0_0_01_generate_chunk.txt", 0.0, 0); //need to update this to show which rank/device + step etc
            } catch e {
                writeln("Failed to dump debug state: ", e.message());
            } 

            writeln("Problem generated");


            for i in 0..<cfg.grid.total_cells do 
                ideal_gas(cfg, i,false);  

            try {
                dumpState(cfg, "0_0_02_ideal_gas.txt", 0.0, 0); //need to update this to show which rank/device + step etc
            } catch e {
                writeln("Failed to dump debug state: ", e.message());
            } 
            
            //Halo exchange time
            update_halo(cfg, UH_DEN0|UH_ENG0|UH_PRESS|UH_VISC|UH_DEN1|UH_ENG1|UH_XV0|UH_XV1|UH_YV0|UH_YV1, 2);

            try {
                dumpState(cfg, "0_0_03_update_halo.txt", 0.0, 0); //need to update this to show which rank/device + step etc
            } catch e {
                writeln("Failed to dump debug state: ", e.message());
            } 

            writeln("Problem initialised and generated");

            try {
                dumpState(cfg, "0_0_04_field_summary.txt", 0.0, 0); //need to update this to show which rank/device + step etc
            } catch e {
                writeln("Failed to dump debug state: ", e.message());
            }

            return cfg;
        }
        catch e: Error {
            writeln("Error reading configuration: ", e.message());
        }

        return new Config(0, 0);
    }
}
