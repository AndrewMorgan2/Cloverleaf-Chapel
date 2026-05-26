module IdealGas {
    use Definitions;
    use Math;
    use Initialization;
    use GPU;

    proc ideal_gas_kernel(ref cfg: Config, predict: bool, compSS: bool = true) {
        const x_min = cfg.grid.xmin;
        const x_max = cfg.grid.xmax;
        const y_min = cfg.grid.ymin;
        const y_max = cfg.grid.ymax;

        ref density    = if predict then cfg.field.density1 else cfg.field.density0;
        ref energy     = if predict then cfg.field.energy1  else cfg.field.energy0;
        ref pressure   = cfg.field.pressure;
        ref soundspeed = cfg.field.soundspeed;

        if compSS {
            @gpu.blockSize(gpuBlockSize)
            forall (i,j) in {(x_min+1)..(x_max+1), (y_min+1)..(y_max+1)} {
                var v = 1.0 / density[i,j];
                pressure[i,j] = (1.4 - 1.0) * density[i,j] * energy[i,j];
                var pressurebyenergy = (1.4 - 1.0) * density[i,j];
                var pressurebyvolume = -density[i,j] * pressure[i,j];
                var sound_speed_squared = v * v * (pressure[i,j] * pressurebyenergy - pressurebyvolume);
                soundspeed[i,j] = sqrt(max(0.0, sound_speed_squared));
            }
        } else {
            @gpu.blockSize(gpuBlockSize)
            forall (i,j) in {(x_min+1)..(x_max+1), (y_min+1)..(y_max+1)} {
                pressure[i,j] = (1.4 - 1.0) * density[i,j] * energy[i,j];
            }
        }
    }

    proc ideal_gas(ref cfg: Config, tile: int, predict: bool, compSS: bool = true) {
        ideal_gas_kernel(cfg, predict, compSS);
    }
}
