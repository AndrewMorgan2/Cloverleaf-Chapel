module ReadingConfig{
    use Initialization;
    use Debug;
    use BlockDist;
    use IO;
    use FileSystem;
    use List;
    use Definitions;
    
    proc readConfig(filename: string) throws {
        //First we work out how big to make the buffers 
        var x_size, y_size: int;
         
        // Open the file with explicit error handling
        var file = try! open(filename, ioMode.r);
        var channel = try! file.reader();

        // Read line by line
        var line: string;       
        while (try! channel.readLine(line)) {
            // Skip empty lines
            if line.size == 0 then continue;
            
            // Remove whitespace
            line = line.strip();
            
            // Check for start/end markers
            if line == "" then continue;
            if line == "*clover" then continue;
            if line == "*endclover" then break;
            
            // Split the line into tokens
            var tokens = line.split();
            // Only process lines with '=' in the first token
            if tokens[0].find('=') != -1 {
                try {
                    var parts = tokens[0].split('=');
                    var labelStr = parts[0].strip();
                    var valueStr = parts[1].strip();

                    select labelStr {
                        when "x_cells" do x_size = valueStr:int;
                        when "y_cells" do y_size = valueStr:int;
                    }
                } catch e: Error {
                    writeln("Error processing configuration value for ", tokens[0]);
                    throw e;
                }
            }
        }

        var cfg = new Config(x_size+5, y_size+5); //MAGIC NUMBER, due to this being the value in the config's relationship to the grid size
        writeln("so we there");
        var currentState = 0;
        cfg.number_of_states = 0;
        
        // Open the file with explicit error handling
        file = try! open(filename, ioMode.r);
        channel = try! file.reader();
        while (try! channel.readLine(line)) {
            // Skip empty lines
            if line.size == 0 then continue;
            
            // Remove whitespace
            line = line.strip();
            
            // Check for start/end markers
            if line == "" then continue;
            if line == "*clover" then continue;
            if line == "*endclover" then break;
            
            // Split the line into tokens
            var tokens = line.split();
            // Process based on first token
            if tokens[0] == "state" {
                try {
                    var stateNum = tokens[1]:int;
                    //currentState = stateNum - 1;  // Convert to 0-based index
                    //cfg.number_of_states = max(cfg.number_of_states, stateNum);
                    var state = new State();
                    // Process state parameters
                    for i in 2..tokens.size-1 {
                        var keyValue = tokens[i].split("=");
                        if keyValue.size >= 2 {
                            var key = keyValue[0];
                            var value = keyValue[1].strip();
                            
                            try {
                                select key {
                                    when "density" do state.density = value:real;
                                    when "energy" do state.energy = value:real;
                                    when "geometry" {
                                        select value {
                                            when "rectangle" do state.geometry = GeometryType.g_rect:int;
                                            when "circle" do state.geometry = GeometryType.g_circ:int;
                                            when "point" do state.geometry = GeometryType.g_point:int;
                                        }
                                    }
                                    when "xmin" do state.xmin = value:real;
                                    when "xmax" do state.xmax = value:real;
                                    when "ymin" do state.ymin = value:real;
                                    when "ymax" do state.ymax = value:real;
                                    when "radius" do state.radius = value:real;
                                }
                            } catch e: Error {
                                writeln("Error parsing value for key ", key, ": ", value);
                                throw e;
                            }
                        }
                    }
                    cfg.addState(state);
                } catch e: Error {
                    writeln("Error processing state definition");
                    throw e;
                }
            }
            else if tokens[0] == "test_problem"{
                cfg.test_problem = tokens[1]:int;
            }
            // Grid configuration - only process lines with '='
            else if tokens[0].find('=') != -1 {
                try {
                    var parts = tokens[0].split('=');
                    var labelStr = parts[0].strip();
                    var valueStr = parts[1].strip();

                    select labelStr {
                            when "x_cells" do {
                                cfg.grid.x_cells = valueStr:int;
                                cfg.grid.xmax = valueStr:int;
                                cfg.info.t_xmax = valueStr:int;
                                cfg.grid.right = valueStr:int;
                                cfg.grid.xmin = 1;
                                cfg.grid.left = 1;
                            }
                            when "y_cells" do {
                                cfg.grid.y_cells = valueStr:int;
                                cfg.grid.ymax = valueStr:int;
                                cfg.grid.top = valueStr:int;
                                cfg.grid.ymin = 1;
                                cfg.grid.bottom = 1;
                            }
                            when "xmin" do {
                                cfg.info.t_xmin = trunc(valueStr:real):int;
                                cfg.info.t_left = 1;
                            }
                            when "xmax" do {
                                cfg.info.t_xmax = trunc(valueStr:real):int;
                                cfg.info.t_right = trunc(valueStr:real):int;
                            }
                            when "ymin" do {
                                cfg.info.t_ymin = trunc(valueStr:real):int;
                                cfg.info.t_bottom = 1;
                            }
                            when "ymax" do {
                                cfg.info.t_ymax = trunc(valueStr:real):int;
                                cfg.info.t_top = trunc(valueStr:real):int;
                            }
                            when "initial_timestep" do cfg.initial_timestep = valueStr:real;
                            when "timestep_rise" do cfg.timestep_rise = valueStr:real;
                            when "max_timestep" do cfg.max_timestep = valueStr:real;
                            when "end_time" do cfg.end_time = valueStr:real;
                            when "end_step" do cfg.end_step = valueStr:int;
                    }
                } catch e: Error {
                    writeln("Error processing configuration value for ", tokens[0]);
                    throw e;
                }
            }
        }

        if (cfg.states[0].energy == 0.0){
            writeln("Common issue! Make sure energy has a = to the init energy state for the first state");
        }

        //Need to post process states x and y mins and maxs
        var dx = (cfg.info.t_xmax - cfg.info.t_xmin) / cfg.grid.x_cells: real;
        var dy = (cfg.info.t_ymax - cfg.info.t_ymin) / cfg.grid.y_cells: real;

        writeln("dx = ", dx);
        writeln("dy = ", dy);
        writeln("cfg.grid.xmax = ", cfg.grid.xmax);
        writeln("cfg.grid.xmin = ", cfg.grid.xmin);
        writeln("cfg.grid.ymax = ", cfg.grid.ymax); 
        writeln("cfg.grid.ymin = ", cfg.grid.ymin);
        writeln("cfg.grid.x_cells = ", cfg.grid.x_cells);
        writeln("cfg.grid.y_cells = ", cfg.grid.y_cells);

        for n in 1..cfg.number_of_states-1 {
            cfg.states[n].xmin += dx/100.0;
            cfg.states[n].ymin += dy/100.0;
            cfg.states[n].xmax -= dx/100.0;
            cfg.states[n].ymax -= dy/100.0;
        }

        return cfg;
    }

    // Modified main procedure to use the config file
    proc readin(inputDeck: string) throws {
        // Create and read configuration
        

        try {
            var cfg = readConfig("./config/" + inputDeck);
            cfg.grid.total_cells = cfg.grid.x_cells * cfg.grid.y_cells;
            writeln("Configuration loaded:");
            writeln("Number of states: ", cfg.number_of_states);
            printStates(cfg.states);
            writeln("Number of cells: ", cfg.grid.total_cells);
            writeln("Grid: ", cfg.grid.x_cells, "x", cfg.grid.y_cells);
            writeln("Domain: (", cfg.grid.xmin, ",", cfg.grid.ymin, ") to (",
                    cfg.grid.xmax, ",", cfg.grid.ymax, ") From " + inputDeck);

            return cfg;
        } catch e: Error {
            writeln("Error reading configuration: ", e.message());
            throw e;
        }
    }
}
