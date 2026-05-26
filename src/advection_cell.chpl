module Advec_cell {
    use Definitions;
    use GPU;
    use Update_halo;

    const g_xdir = 1;
    const g_ydir = 2;
    const one_by_six = 1.0 / 6.0;

    // ── Kernel 1: pre/post volume ────────────────────────────────────────────

    inline proc k1_xdir_sweep1(ref pre_vol, ref post_vol,
                                ref volume, ref vol_flux_x, ref vol_flux_y,
                                iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            pre_vol[i,j]  = volume[i,j] + (vol_flux_x[i+1,j] - vol_flux_x[i,j]
                                          + vol_flux_y[i,j+1] - vol_flux_y[i,j]);
            post_vol[i,j] = pre_vol[i,j] - (vol_flux_x[i+1,j] - vol_flux_x[i,j]);
        }
    }

    inline proc k1_xdir_sweep2(ref pre_vol, ref post_vol,
                                ref volume, ref vol_flux_x,
                                iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            pre_vol[i,j]  = volume[i,j] + vol_flux_x[i+1,j] - vol_flux_x[i,j];
            post_vol[i,j] = volume[i,j];
        }
    }

    inline proc k1_ydir_sweep1(ref pre_vol, ref post_vol,
                                ref volume, ref vol_flux_x, ref vol_flux_y,
                                iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            pre_vol[i,j]  = volume[i,j] + (vol_flux_y[i,j+1] - vol_flux_y[i,j]
                                          + vol_flux_x[i+1,j] - vol_flux_x[i,j]);
            post_vol[i,j] = pre_vol[i,j] - (vol_flux_y[i,j+1] - vol_flux_y[i,j]);
        }
    }

    inline proc k1_ydir_sweep2(ref pre_vol, ref post_vol,
                                ref volume, ref vol_flux_y,
                                iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            pre_vol[i,j]  = volume[i,j] + vol_flux_y[i,j+1] - vol_flux_y[i,j];
            post_vol[i,j] = volume[i,j];
        }
    }

    // ── Kernel 2: mass and energy flux (x direction) ─────────────────────────
    // Branchless: upwind direction and limiter computed with select expressions
    // so LLVM emits cmov/select instructions — no warp divergence on GPU.
    inline proc k2_xdir(ref pre_vol, ref vol_flux_x, ref vertexdx,
                         ref density1, ref energy1,
                         ref mass_flux_x, ref ener_flux,
                         x_max: int,
                         iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            // Upwind direction — branchless index selection
            const fwd      = vol_flux_x[i,j] > 0.0;
            const upwind   = if fwd then i-2           else min(i+1, x_max+2);
            const donor    = if fwd then i-1           else i;
            const downwind = if fwd then i             else i-1;
            const dif      = if fwd then i-1           else min(i+1, x_max+2);

            const sigmat = abs(vol_flux_x[i,j]) / pre_vol[donor,j];
            const sigma3 = (1.0 + sigmat) * (vertexdx[i] / vertexdx[dif]);
            const sigma4 = 2.0 - sigmat;
            const sigmav = sigmat;

            // Density limiter — compute unconditionally, select at end
            const diffuw_d = density1[donor,j]    - density1[upwind,j];
            const diffdw_d = density1[downwind,j] - density1[donor,j];
            const wind_d   = if diffdw_d <= 0.0 then -1.0 else 1.0;
            const lim_d    = (1.0 - sigmav) * wind_d *
                             min(min(abs(diffuw_d), abs(diffdw_d)),
                                 one_by_six * (sigma3 * abs(diffuw_d) + sigma4 * abs(diffdw_d)));
            const limiter_d = if diffuw_d * diffdw_d > 0.0 then lim_d else 0.0;

            mass_flux_x[i,j] = vol_flux_x[i,j] * (density1[donor,j] + limiter_d);

            const sigmam   = abs(mass_flux_x[i,j]) / (density1[donor,j] * pre_vol[donor,j]);

            // Energy limiter — compute unconditionally, select at end
            const diffuw_e = energy1[donor,j]    - energy1[upwind,j];
            const diffdw_e = energy1[downwind,j] - energy1[donor,j];
            const wind_e   = if diffdw_e <= 0.0 then -1.0 else 1.0;
            const lim_e    = (1.0 - sigmam) * wind_e *
                             min(min(abs(diffuw_e), abs(diffdw_e)),
                                 one_by_six * (sigma3 * abs(diffuw_e) + sigma4 * abs(diffdw_e)));
            const limiter_e = if diffuw_e * diffdw_e > 0.0 then lim_e else 0.0;

            ener_flux[i,j] = mass_flux_x[i,j] * (energy1[donor,j] + limiter_e);
        }
    }

    // ── Kernel 2: mass and energy flux (y direction) ─────────────────────────
    // Branchless: same treatment as k2_xdir.
    inline proc k2_ydir(ref pre_vol, ref vol_flux_y, ref vertexdy,
                         ref density1, ref energy1,
                         ref mass_flux_y, ref ener_flux,
                         y_max: int,
                         iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            const fwd      = vol_flux_y[i,j] > 0.0;
            const upwind   = if fwd then j-2           else min(j+1, y_max+2);
            const donor    = if fwd then j-1           else j;
            const downwind = if fwd then j             else j-1;
            const dif      = if fwd then j-1           else min(j+1, y_max+2);

            const sigmat = abs(vol_flux_y[i,j]) / pre_vol[i,donor];
            const sigma3 = (1.0 + sigmat) * (vertexdy[j] / vertexdy[dif]);
            const sigma4 = 2.0 - sigmat;
            const sigmav = sigmat;

            const diffuw_d = density1[i,donor]    - density1[i,upwind];
            const diffdw_d = density1[i,downwind] - density1[i,donor];
            const wind_d   = if diffdw_d <= 0.0 then -1.0 else 1.0;
            const lim_d    = (1.0 - sigmav) * wind_d *
                             min(min(abs(diffuw_d), abs(diffdw_d)),
                                 one_by_six * (sigma3 * abs(diffuw_d) + sigma4 * abs(diffdw_d)));
            const limiter_d = if diffuw_d * diffdw_d > 0.0 then lim_d else 0.0;

            mass_flux_y[i,j] = vol_flux_y[i,j] * (density1[i,donor] + limiter_d);

            const sigmam   = abs(mass_flux_y[i,j]) / (density1[i,donor] * pre_vol[i,donor]);

            const diffuw_e = energy1[i,donor]    - energy1[i,upwind];
            const diffdw_e = energy1[i,downwind] - energy1[i,donor];
            const wind_e   = if diffdw_e <= 0.0 then -1.0 else 1.0;
            const lim_e    = (1.0 - sigmam) * wind_e *
                             min(min(abs(diffuw_e), abs(diffdw_e)),
                                 one_by_six * (sigma3 * abs(diffuw_e) + sigma4 * abs(diffdw_e)));
            const limiter_e = if diffuw_e * diffdw_e > 0.0 then lim_e else 0.0;

            ener_flux[i,j] = mass_flux_y[i,j] * (energy1[i,donor] + limiter_e);
        }
    }

    // ── Kernel 3: density and energy update (x direction) ────────────────────
    inline proc k3_xdir(ref density1, ref energy1,
                         ref pre_vol, ref vol_flux_x,
                         ref mass_flux_x, ref ener_flux,
                         iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            var pre_mass_s  = density1[i,j] * pre_vol[i,j];
            var post_mass_s = pre_mass_s + mass_flux_x[i,j] - mass_flux_x[i+1,j];
            var post_ener_s = (energy1[i,j] * pre_mass_s + ener_flux[i,j] - ener_flux[i+1,j]) / post_mass_s;
            density1[i,j]   = post_mass_s / (pre_vol[i,j] + vol_flux_x[i,j] - vol_flux_x[i+1,j]);
            energy1[i,j]    = post_ener_s;
        }
    }

    // ── Kernel 3: density and energy update (y direction) ────────────────────
    inline proc k3_ydir(ref density1, ref energy1,
                         ref pre_vol, ref vol_flux_y,
                         ref mass_flux_y, ref ener_flux,
                         iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            var pre_mass_s  = density1[i,j] * pre_vol[i,j];
            var post_mass_s = pre_mass_s + mass_flux_y[i,j] - mass_flux_y[i,j+1];
            var post_ener_s = (energy1[i,j] * pre_mass_s + ener_flux[i,j] - ener_flux[i,j+1]) / post_mass_s;
            density1[i,j]   = post_mass_s / (pre_vol[i,j] + vol_flux_y[i,j] - vol_flux_y[i,j+1]);
            energy1[i,j]    = post_ener_s;
        }
    }

    // ── Driver ───────────────────────────────────────────────────────────────
    proc advec_cell_driver(ref cfg: Config, sweep_number: int, dir: int) {
        const x_min = cfg.grid.xmin;
        const x_max = cfg.grid.xmax;
        const y_min = cfg.grid.ymin;
        const y_max = cfg.grid.ymax;

        ref pre_vol     = cfg.field.work_array1;
        ref post_vol    = cfg.field.work_array2;
        ref volume      = cfg.field.volume;
        ref vol_flux_x  = cfg.field.vol_flux_x;
        ref vol_flux_y  = cfg.field.vol_flux_y;
        ref vertexdx    = cfg.field.vertexdx;
        ref vertexdy    = cfg.field.vertexdy;
        ref density1    = cfg.field.density1;
        ref mass_flux_x = cfg.field.mass_flux_x;
        ref mass_flux_y = cfg.field.mass_flux_y;
        ref energy1     = cfg.field.energy1;
        ref ener_flux   = cfg.field.work_array7;

        const k1_iLo = max(1, x_min - 1);
        const k1_iHi = x_max + 3;
        const k1_jLo = max(1, y_min - 1);
        const k1_jHi = y_max + 3;

        const k23_iLo = x_min + 1;
        const k2_iHi  = x_max + 3;
        const k3_iHi  = x_max + 1;
        const k23_jLo = y_min + 1;
        const k23_jHi = y_max + 1;

        // Kernel 1 — branch hoisted
        if dir == g_xdir {
            if sweep_number == 1 then
                k1_xdir_sweep1(pre_vol, post_vol, volume, vol_flux_x, vol_flux_y,
                               k1_iLo, k1_iHi, k1_jLo, k1_jHi);
            else
                k1_xdir_sweep2(pre_vol, post_vol, volume, vol_flux_x,
                               k1_iLo, k1_iHi, k1_jLo, k1_jHi);
        } else {
            if sweep_number == 1 then
                k1_ydir_sweep1(pre_vol, post_vol, volume, vol_flux_x, vol_flux_y,
                               k1_iLo, k1_iHi, k1_jLo, k1_jHi);
            else
                k1_ydir_sweep2(pre_vol, post_vol, volume, vol_flux_y,
                               k1_iLo, k1_iHi, k1_jLo, k1_jHi);
        }

        // Kernel 2 — branch hoisted
        if dir == g_xdir then
            k2_xdir(pre_vol, vol_flux_x, vertexdx, density1, energy1,
                    mass_flux_x, ener_flux, x_max,
                    k23_iLo, k2_iHi, k23_jLo, k23_jHi);
        else
            k2_ydir(pre_vol, vol_flux_y, vertexdy, density1, energy1,
                    mass_flux_y, ener_flux, y_max,
                    k23_iLo, k2_iHi, k23_jLo, k23_jHi);

        // Kernel 3 — branch hoisted
        if dir == g_xdir then
            k3_xdir(density1, energy1, pre_vol, vol_flux_x, mass_flux_x, ener_flux,
                    k23_iLo, k3_iHi, k23_jLo, k23_jHi);
        else
            k3_ydir(density1, energy1, pre_vol, vol_flux_y, mass_flux_y, ener_flux,
                    k23_iLo, k3_iHi, k23_jLo, k23_jHi);
    }
}
