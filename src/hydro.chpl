// src/main.chpl
module Hydro {
    use Timestep;
    use Advection;
    use Accelerate;
    use PdV;
    use Flux_calc;
    use Reset_field;
    use Definitions;
    use Debug;
    use Initialization;
    use Report;
    use Time;

    // Print grind-time line every N steps (0 = never, except on the last step)
    config const reportFreq = 10;

    proc hydro(ref cfg: Config, ref clock: stopwatch) {

        if cfg.dumpDir != "" {
            try {
                dumpState(cfg, "0_0_05_hydro.txt", 0.0, 0);
            } catch e {
                writeln("Failed to dump debug state: ", e.message());
            }
        }

        writeln("Hydro Running");

        while(true){

            var clock_step = new stopwatch();
            clock_step.start();

            cfg.step = cfg.step + 1;

            if cfg.dumpDir != "" {
                try {
                    dumpState(cfg, "0_"+ cfg.step:string +"_1_checky.txt", 0.0, 0);
                } catch e {
                    writeln("Failed to dump debug state: ", e.message());
                }
            }

            //timestep
            calculate_timestep(cfg);

            if cfg.dumpDir != "" {
                try {
                    dumpState(cfg, "0_"+ cfg.step:string +"_1_timestep.txt", 0.0, 0);
                } catch e {
                    writeln("Failed to dump debug state: ", e.message());
                }
            }

            //PdV predict
            PdV(cfg, true);

            //accelerate
            accelerate(cfg);

            //PdV not predict
            PdV(cfg, false);

            //Flux calc
            flux_calculation(cfg);

            //advection
            advection(cfg);

            if cfg.dumpDir != "" {
                try {
                    dumpState(cfg, "0_"+ cfg.step:string +"_6_advection.txt", 0.0, 0);
                } catch e {
                    writeln("Failed to dump debug state: ", e.message());
                }
            }

            //reset fields
            reset_field(cfg);

            if cfg.dumpDir != "" {
                try {
                    dumpState(cfg, "0_"+ cfg.step:string +"_7_reset_field.txt", 0.0, 0);
                } catch e {
                    writeln("Failed to dump debug state: ", e.message());
                }
            }

            cfg.advect_x = !cfg.advect_x;
            cfg.time = cfg.time + cfg.dt;

            if isNan(cfg.field.density0[2,2]) || isNan(cfg.field.density1[2,2]) {
                writeln("NaN FIRST SEEN at step ", cfg.step,
                        " density0[2,2]=", cfg.field.density0[2,2],
                        " density1[2,2]=", cfg.field.density1[2,2]);
            }

            var wall_clock = clock.elapsed();
            clock_step.stop();
            var step_clock = clock_step.elapsed();

            const isLast = (cfg.time + g_small > cfg.end_time || cfg.step >= cfg.end_step);
            if isLast || (reportFreq > 0 && cfg.step % reportFreq == 0) {
                var cells = cfg.grid.x_cells * cfg.grid.y_cells;
                var rstep = cfg.step: real;
                var grind_time = wall_clock / (rstep * cells);
                var step_grind = step_clock / cells;
                writeln(" Wall clock ", wall_clock);
                writeln(" Average time per cell ", grind_time);
                writeln("  Step time per cell    ", step_grind);
            }

            if isLast {
                //Tells us how far we are from the summated answers
                report(cfg);
                clock.stop();
                var wall_clock2 = clock.elapsed();
                var kernel_total = cfg.profile_times.timestep + cfg.profile_times.idealGas + cfg.profile_times.viscosity +
                                cfg.profile_times.pdv + cfg.profile_times.revert + cfg.profile_times.accelerate +
                                cfg.profile_times.fluxes + cfg.profile_times.cellAdvection + cfg.profile_times.momAdvection +
                                cfg.profile_times.reset + cfg.profile_times.halo;
                writeln();
                writeln(" Profiler Output        Time     Percentage");
                writeln(" Timestep              : ", cfg.profile_times.timestep,       " ", 100.0 * (cfg.profile_times.timestep       / wall_clock2));
                writeln(" Ideal Gas             : ", cfg.profile_times.idealGas,       " ", 100.0 * (cfg.profile_times.idealGas       / wall_clock2));
                writeln(" Viscosity             : ", cfg.profile_times.viscosity,      " ", 100.0 * (cfg.profile_times.viscosity      / wall_clock2));
                writeln(" PdV                   : ", cfg.profile_times.pdv,            " ", 100.0 * (cfg.profile_times.pdv            / wall_clock2));
                writeln(" Revert                : ", cfg.profile_times.revert,         " ", 100.0 * (cfg.profile_times.revert         / wall_clock2));
                writeln(" Acceleration          : ", cfg.profile_times.accelerate,     " ", 100.0 * (cfg.profile_times.accelerate     / wall_clock2));
                writeln(" Fluxes                : ", cfg.profile_times.fluxes,         " ", 100.0 * (cfg.profile_times.fluxes         / wall_clock2));
                writeln(" Cell Advection        : ", cfg.profile_times.cellAdvection,  " ", 100.0 * (cfg.profile_times.cellAdvection  / wall_clock2));
                writeln(" Momentum Advection    : ", cfg.profile_times.momAdvection,   " ", 100.0 * (cfg.profile_times.momAdvection   / wall_clock2));
                writeln(" Reset                 : ", cfg.profile_times.reset,          " ", 100.0 * (cfg.profile_times.reset          / wall_clock2));
                writeln(" Halo Exchange         : ", cfg.profile_times.halo,           " ", 100.0 * (cfg.profile_times.halo           / wall_clock2));
                writeln(" Total                 : ", kernel_total,                      " ", 100.0 * (kernel_total                     / wall_clock2));
                writeln(" The Rest              : ", wall_clock2 - kernel_total,        " ", 100.0 * (wall_clock2 - kernel_total)      / wall_clock2);
                writeln();
                writeln();
                writeln("Calculation complete at step: ", cfg.step, " and time:", cfg.time);
                writeln("Clover is finishing");
                break;
            }
        }
    }

}
