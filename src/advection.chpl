module Advection {
    use Definitions;
    use Advec_cell;
    use Advec_mom;
    use Update_halo;
    use Debug;
    use CloverLeaf;
    use Time;
    
    const g_xdir = 1;
    const g_ydir = 2;

    proc advection(ref cfg: Config) {
        var sweep_number = 1;
        var direction: int;
        
        // Set initial direction
        if cfg.advect_x then direction = g_xdir;
        else direction = g_ydir;
        
        const xvel = g_xdir;
        const yvel = g_ydir;
        
        var clock_halo = new stopwatch();
        if(profile) then clock_halo.start();

        //Halo exchange time
        update_halo(cfg, UH_ENG1|UH_DEN1|UH_VFX|UH_VFY, 2);

        if(profile) {clock_halo.stop(); cfg.profile_times.halo = cfg.profile_times.halo + clock_halo.elapsed();}
    
        // try {
        //     dumpState(cfg, "0_1_6_unexpected.txt", 0.0, 0); //need to update this to show which rank/device + step etc
        // } catch e {
        //     writeln("Failed to dump debug state: ", e.message());
        // } 

        var clock_advec_cell = new stopwatch();
        if(profile) then clock_advec_cell.start();

        // Perform first sweep cell advection
        advec_cell_driver(cfg, sweep_number, direction);

        if(profile) {clock_advec_cell.stop(); cfg.profile_times.cellAdvection = cfg.profile_times.cellAdvection + clock_advec_cell.elapsed();}

        var clock_halo2 = new stopwatch();
        if(profile) then clock_halo2.start();

        update_halo(cfg, UH_ENG1|UH_DEN1|UH_XV1|UH_YV1|UH_MFX|UH_MFY, 2);

        if(profile) {clock_halo2.stop(); cfg.profile_times.halo = cfg.profile_times.halo + clock_halo2.elapsed();}

        var clock_advec_mom = new stopwatch();
        if(profile) then clock_advec_mom.start();

        // Perform momentum advection in x and y
        advec_mom_driver(cfg, xvel, direction, sweep_number);
        advec_mom_driver(cfg, yvel, direction, sweep_number);

        if(profile) {clock_advec_mom.stop(); cfg.profile_times.momAdvection = cfg.profile_times.momAdvection + clock_advec_mom.elapsed();}

        if cfg.dumpDir != "" {
            try {
                dumpState(cfg, "0_"+ cfg.step:string +"_6_mom_drive.txt", 0.0, 0);
            } catch e {
                writeln("Failed to dump debug state: ", e.message());
            }
        }
        
        // Second sweep
        sweep_number = 2;
        
        // Set second sweep direction
        if cfg.advect_x then direction = g_ydir;
        else direction = g_xdir;
        
        var clock_advec_cell2 = new stopwatch();
        if(profile) then clock_advec_cell2.start();

        // Perform second sweep cell advection
        advec_cell_driver(cfg, sweep_number, direction);

        if(profile) {clock_advec_cell2.stop(); cfg.profile_times.cellAdvection = cfg.profile_times.cellAdvection + clock_advec_cell2.elapsed();}

        var clock_halo3 = new stopwatch();
        if(profile) then clock_halo3.start();

        update_halo(cfg, UH_ENG1|UH_DEN1|UH_XV1|UH_YV1|UH_MFX|UH_MFY, 2);

        if(profile) {clock_halo3.stop(); cfg.profile_times.halo = cfg.profile_times.halo + clock_halo3.elapsed();}

        var clock_advec_mom2 = new stopwatch();
        if(profile) then clock_advec_mom2.start();

        // Perform final momentum advection in x and y
        advec_mom_driver(cfg, xvel, direction, sweep_number);
        advec_mom_driver(cfg, yvel, direction, sweep_number);

        if(profile) {clock_advec_mom2.stop(); cfg.profile_times.momAdvection = cfg.profile_times.momAdvection + clock_advec_mom2.elapsed();}

        if cfg.dumpDir != "" {
            try {
                dumpState(cfg, "0_"+ cfg.step:string +"_6_mom_drive_2.txt", 0.0, 0);
            } catch e {
                writeln("Failed to dump debug state: ", e.message());
            }
        }

    }
}