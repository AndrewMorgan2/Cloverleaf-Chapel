module Revert {
   use Definitions;
   use GPU;

   proc revert_PdV(ref cfg: Config) {
        const x_min = cfg.grid.xmin;
        const x_max = cfg.grid.xmax;
        const y_min = cfg.grid.ymin;
        const y_max = cfg.grid.ymax;

        ref density0 = cfg.field.density0;
        ref density1 = cfg.field.density1;
        ref energy0  = cfg.field.energy0;
        ref energy1  = cfg.field.energy1;

        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max+1), (y_min+1)..(y_max+1)} {
            density1[i,j] = density0[i,j];
            energy1[i,j]  = energy0[i,j];
        }
   }
}
