// src/debug.chpl
module Debug {
    use FileSystem;
    use IO;
    use Path;
    use Initialization;
    use Definitions;

    // Format and write 2D array content to stream
    proc showBuffer(writer: ?fileWriter, name: string, buffer: [] real) throws {
        const dims = buffer.domain.dims();
        // if name != "energy0" && name != "density0" && name !="yarea" {return;}
        try {
            if dims.size == 1{
                // Handle 1D array
                writer.writeln("\n", name, "(1) [", buffer.domain.size, "]");
                writer.write("\t");
                for i in 0..#buffer.domain.size {
                    writer.write(buffer[i], ", ");
                }
            }
            if dims.size == 2 {
                writer.writeln(" \n", name, "(2) [", dims(0).size, "x", dims(1).size, "]");
                writer.write("\t");
                
                // Check if buffer is all zeros
                var allZeros = true;
                for value in buffer {
                    if value != 0.0 {
                        allZeros = false;
                        break;
                    }
                }
                

                for i in 0..#dims(0).size {
                    writer.write("\t");
                    for j in 0..#dims(1).size {
                        // writer.write(i, ",", j, ":");
                        writer.write(buffer[i,j]);
                        writer.write(", ");
                    }
                    writer.writeln();
                }
            }
        } catch e {
            writeln("Error writing buffer ", name, ": ", e.message());
            throw e;
        }
    }

    // Main dump function for debugging
    proc dumpState(configData: ?Config, filename: string, currentTime: real, step: int) throws {
        if configData.dumpDir == "" then return;

        writeln("Dumping state to ", filename);

        //meant to speed up for debug
        // if filename != "0_1_4_PdV.txt" then return;

        const dirPath = configData.dumpDir + "/";
        
        // Create directory if it doesn't exist
        if !exists(dirPath) {
            try {
                mkdir(dirPath);
                writeln("Created ", dirPath, " for field dump");
            } catch e {
                writeln("Cannot create ", dirPath, ": ", e.message(), ", skipping field dump");
                return;
            }
        }

        const filepath = dirPath + filename;
        var writer = try! open(filepath, ioMode.cw).writer();

        try {
            // writer.writeln("Dump(tileCount = ", tiles.size, ")");
            // writer.writeln("error_condition = ", errorCondition);
            writer.writeln("step = ", configData.step);
            writer.writeln("advect_x = ", configData.advect_x);
            writer.writeln("time = ", configData.time);
            writer.writeln("dt = ", configData.dt);
            writer.writeln("dtold = ", configData.dtold);
            writer.writeln("jdt = ", configData.jdt);
            writer.writeln("kdt = ", configData.kdt);
            // writer.writeln("dtold = ", dtold);
            // writer.writeln("complete = ", isComplete);

            // Tile boundary information
            writer.writeln("t_xmin = ", configData.grid.xmin);
            writer.writeln("t_xmax = ", configData.grid.xmax);
            writer.writeln("t_ymin = ", configData.grid.ymin);
            writer.writeln("t_ymax = ", configData.grid.ymax);
            writer.writeln("t_left = ", configData.grid.left);
            writer.writeln("t_right = ", configData.grid.right);
            writer.writeln("t_bottom = ", configData.grid.bottom);
            writer.writeln("t_top = ", configData.grid.top);

            // Dump field data
            showBuffer(writer, "density0", configData.field.density0);
            showBuffer(writer, "density1", configData.field.density1);
            showBuffer(writer, "energy0", configData.field.energy0);
            showBuffer(writer, "energy1", configData.field.energy1);
            showBuffer(writer, "pressure", configData.field.pressure);
            showBuffer(writer, "viscosity", configData.field.viscosity);
            showBuffer(writer, "soundspeed", configData.field.soundspeed);
                
            // // Velocities
            showBuffer(writer, "xvel0", configData.field.xvel0);
            showBuffer(writer, "xvel1", configData.field.xvel1);
            showBuffer(writer, "yvel0", configData.field.yvel0);
            showBuffer(writer, "yvel1", configData.field.yvel1);

            // // Fluxes
            showBuffer(writer, "vol_flux_x",  configData.field.vol_flux_x);
            showBuffer(writer, "vol_flux_y",  configData.field.vol_flux_y);
            showBuffer(writer, "mass_flux_x", configData.field.mass_flux_x);
            showBuffer(writer, "mass_flux_y", configData.field.mass_flux_y);

            // // Work arrays
            showBuffer(writer, "work_array1", configData.field.work_array1);
            showBuffer(writer, "work_array2", configData.field.work_array2);
            showBuffer(writer, "work_array3", configData.field.work_array3);
            showBuffer(writer, "work_array4", configData.field.work_array4);
            showBuffer(writer, "work_array5", configData.field.work_array5);
            showBuffer(writer, "work_array6", configData.field.work_array6);
            showBuffer(writer, "work_array7", configData.field.work_array7);

            // // Grid information
            showBuffer(writer, "cellx", configData.field.cellx);
            showBuffer(writer, "celldx", configData.field.celldx);
            showBuffer(writer, "celly", configData.field.celly);
            showBuffer(writer, "celldy", configData.field.celldy);
            showBuffer(writer, "vertexx", configData.field.vertexx);
            showBuffer(writer, "vertexdx", configData.field.vertexdx);
            showBuffer(writer, "vertexy", configData.field.vertexy);
            showBuffer(writer, "vertexdy", configData.field.vertexdy);

            // // Volume and areas
            showBuffer(writer, "volume", configData.field.volume);
            showBuffer(writer, "xarea", configData.field.xarea);
            showBuffer(writer, "yarea", configData.field.yarea);

        } catch e {
            writeln("Error writing to file: ", e.message());
            throw e;
        }
        writer.close();
    }
}