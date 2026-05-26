module Flux_calc {
    use Definitions;
    use Time;
    use CloverLeaf;
    use GPU;

    proc flux_calc_kernel(ref cfg: Config) {
        const x_min = cfg.grid.xmin;
        const x_max = cfg.grid.xmax;
        const y_min = cfg.grid.ymin;
        const y_max = cfg.grid.ymax;
        const dt = cfg.dt;

        ref vol_flux_x = cfg.field.vol_flux_x;
        ref vol_flux_y = cfg.field.vol_flux_y;
        ref xarea      = cfg.field.xarea;
        ref yarea      = cfg.field.yarea;
        ref xvel0      = cfg.field.xvel0;
        ref xvel1      = cfg.field.xvel1;
        ref yvel0      = cfg.field.yvel0;
        ref yvel1      = cfg.field.yvel1;

        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max+2), (y_min+1)..(y_max+2)} {
            vol_flux_x[i,j] = 0.25 * dt * xarea[i,j] *
                (xvel0[i,j] + xvel0[i,j+1] + xvel1[i,j] + xvel1[i,j+1]);
            vol_flux_y[i,j] = 0.25 * dt * yarea[i,j] *
                (yvel0[i,j] + yvel0[i+1,j] + yvel1[i,j] + yvel1[i+1,j]);
        }
    }

    proc flux_calculation(ref cfg: Config) {
        var clock_flux_calculation = new stopwatch();
        if(profile) then clock_flux_calculation.start();

        flux_calc_kernel(cfg);

        if(profile) {clock_flux_calculation.stop(); cfg.profile_times.fluxes = cfg.profile_times.fluxes + clock_flux_calculation.elapsed();}
    }
}
