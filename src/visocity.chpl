module Viscosity {
   use Definitions;
   use Math;
   use GPU;

   proc viscosity(ref cfg: Config) {
        const x_min = cfg.grid.xmin;
        const x_max = cfg.grid.xmax;
        const y_min = cfg.grid.ymin;
        const y_max = cfg.grid.ymax;

        ref celldx   = cfg.field.celldx;
        ref celldy   = cfg.field.celldy;
        ref density0 = cfg.field.density0;
        ref pressure = cfg.field.pressure;
        ref visc     = cfg.field.viscosity;
        ref xvel0    = cfg.field.xvel0;
        ref yvel0    = cfg.field.yvel0;

        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max+1), (y_min+1)..(y_max+1)} {
            var ugrad = (xvel0[i+1,j] + xvel0[i+1,j+1]) -
                        (xvel0[i,j]   + xvel0[i,j+1]);
            var vgrad = (yvel0[i,j+1]   + yvel0[i+1,j+1]) -
                        (yvel0[i,j]     + yvel0[i+1,j]);
            var div = celldx[i] * ugrad + celldy[j] * vgrad;

            var strain2 = 0.5 * (xvel0[i,j+1]   + xvel0[i+1,j+1] -
                                 xvel0[i,j]     - xvel0[i+1,j]) / celldy[j] +
                          0.5 * (yvel0[i+1,j]   + yvel0[i+1,j+1] -
                                 yvel0[i,j]     - yvel0[i,j+1]) / celldx[i];

            var pgradx = (pressure[i+1,j] - pressure[i-1,j]) /
                         (celldx[i] + celldx[i+1]);
            var pgrady = (pressure[i,j+1] - pressure[i,j-1]) /
                         (celldy[j] + celldy[j+2]);

            var pgradx2 = pgradx * pgradx;
            var pgrady2 = pgrady * pgrady;

            var limiter = ((0.5 * ugrad / celldx[i]) * pgradx2 +
                            (0.5 * vgrad / celldy[j]) * pgrady2 +
                            strain2 * pgradx * pgrady) /
                            max(pgradx2 + pgrady2, g_small);

            if (limiter > 0.0 || div >= 0.0) {
                visc[i,j] = 0.0;
            } else {
                var dirx = if pgradx < 0.0 then -1.0 else 1.0;
                pgradx = dirx * max(g_small, abs(pgradx));

                var diry = if pgrady < 0.0 then -1.0 else 1.0;
                pgrady = diry * max(g_small, abs(pgrady));

                var pgrad = sqrt(pgradx * pgradx + pgrady * pgrady);
                var xgrad = abs(celldx[i] * pgrad / pgradx);
                var ygrad = abs(celldy[j] * pgrad / pgrady);
                var grad  = min(xgrad, ygrad);
                var grad2 = grad * grad;

                visc[i,j] = 2.0 * density0[i,j] * grad2 * limiter * limiter;
            }
        }
   }
}
