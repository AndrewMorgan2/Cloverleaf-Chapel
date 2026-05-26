module Accelerate {
    use Definitions;
    use Time;
    use CloverLeaf;
    use GPU;

    proc accelerate_kernel(ref cfg: Config) {
        const x_min = cfg.grid.xmin;
        const x_max = cfg.grid.xmax;
        const y_min = cfg.grid.ymin;
        const y_max = cfg.grid.ymax;
        const dt    = cfg.dt;
        const halfdt = 0.5 * dt;

        ref xarea    = cfg.field.xarea;
        ref yarea    = cfg.field.yarea;
        ref volume   = cfg.field.volume;
        ref density0 = cfg.field.density0;
        ref pressure = cfg.field.pressure;
        ref viscosity = cfg.field.viscosity;
        ref xvel0    = cfg.field.xvel0;
        ref yvel0    = cfg.field.yvel0;
        ref xvel1    = cfg.field.xvel1;
        ref yvel1    = cfg.field.yvel1;

        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max+2), (y_min+1)..(y_max+2)} {
            const stepbymass_s = halfdt / (
                (density0[i-1,j-1] * volume[i-1,j-1] +
                 density0[i-1,j]   * volume[i-1,j]   +
                 density0[i,j]     * volume[i,j]     +
                 density0[i,j-1]   * volume[i,j-1]) * 0.25
            );

            xvel1[i,j] = xvel0[i,j] - stepbymass_s * (
                xarea[i,j]   * (pressure[i,j]   - pressure[i-1,j]) +
                xarea[i,j-1] * (pressure[i,j-1] - pressure[i-1,j-1])
            );
            yvel1[i,j] = yvel0[i,j] - stepbymass_s * (
                yarea[i,j]   * (pressure[i,j]   - pressure[i,j-1]) +
                yarea[i-1,j] * (pressure[i-1,j] - pressure[i-1,j-1])
            );
            xvel1[i,j] = xvel1[i,j] - stepbymass_s * (
                xarea[i,j]   * (viscosity[i,j]   - viscosity[i-1,j]) +
                xarea[i,j-1] * (viscosity[i,j-1] - viscosity[i-1,j-1])
            );
            yvel1[i,j] = yvel1[i,j] - stepbymass_s * (
                yarea[i,j]   * (viscosity[i,j]   - viscosity[i,j-1]) +
                yarea[i-1,j] * (viscosity[i-1,j] - viscosity[i-1,j-1])
            );
        }
    }

    proc accelerate(ref cfg: Config) {
        var clock_accelerate = new stopwatch();
        if(profile) then clock_accelerate.start();

        accelerate_kernel(cfg);

        if(profile) {clock_accelerate.stop(); cfg.profile_times.accelerate = cfg.profile_times.accelerate + clock_accelerate.elapsed();}
    }
}
