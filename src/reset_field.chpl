module Reset_field {
    use Definitions;
    use CloverLeaf;
    use Time;
    use GPU;

    proc reset_field(ref cfg: Config) {

        var reset = new stopwatch();
        if(profile) then reset.start();

        const x_min = cfg.grid.xmin;
        const x_max = cfg.grid.xmax;
        const y_min = cfg.grid.ymin;
        const y_max = cfg.grid.ymax;

        ref density0 = cfg.field.density0;
        ref density1 = cfg.field.density1;
        ref energy0  = cfg.field.energy0;
        ref energy1  = cfg.field.energy1;
        ref xvel0    = cfg.field.xvel0;
        ref xvel1    = cfg.field.xvel1;
        ref yvel0    = cfg.field.yvel0;
        ref yvel1    = cfg.field.yvel1;

        // Two clean foralls — no per-element branches, SIMD-friendly.
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max+1), (y_min+1)..(y_max+1)} {
            density0[i,j] = density1[i,j];
            energy0[i,j]  = energy1[i,j];
        }
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max+2), (y_min+1)..(y_max+2)} {
            xvel0[i,j] = xvel1[i,j];
            yvel0[i,j] = yvel1[i,j];
        }

        if(profile) {reset.stop(); cfg.profile_times.reset = cfg.profile_times.reset + reset.elapsed();}
    }
}
