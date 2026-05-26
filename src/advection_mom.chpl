module Advec_mom {
    use Definitions;
    use GPU;
    use Update_halo;

    // ── Kernel 1: pre/post volume (4 specialised variants for mom_sweep) ──────

    inline proc k1_sweep1(ref post_vol, ref pre_vol, ref volume,
                           ref vol_flux_x, ref vol_flux_y,
                           iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            post_vol[i,j] = volume[i,j] + vol_flux_y[i,j+1] - vol_flux_y[i,j];
            pre_vol[i,j]  = post_vol[i,j] + vol_flux_x[i+1,j] - vol_flux_x[i,j];
        }
    }

    inline proc k1_sweep2(ref post_vol, ref pre_vol, ref volume,
                           ref vol_flux_x, ref vol_flux_y,
                           iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            post_vol[i,j] = volume[i,j] + vol_flux_x[i+1,j] - vol_flux_x[i,j];
            pre_vol[i,j]  = post_vol[i,j] + vol_flux_y[i,j+1] - vol_flux_y[i,j];
        }
    }

    inline proc k1_sweep3(ref post_vol, ref pre_vol, ref volume,
                           ref vol_flux_y,
                           iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            post_vol[i,j] = volume[i,j];
            pre_vol[i,j]  = post_vol[i,j] + vol_flux_y[i,j+1] - vol_flux_y[i,j];
        }
    }

    inline proc k1_sweep4(ref post_vol, ref pre_vol, ref volume,
                           ref vol_flux_x,
                           iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            post_vol[i,j] = volume[i,j];
            pre_vol[i,j]  = post_vol[i,j] + vol_flux_x[i+1,j] - vol_flux_x[i,j];
        }
    }

    // ── Kernel 2: node flux ───────────────────────────────────────────────────

    inline proc k2_dir1_xvel(ref node_flux, ref mass_flux_x,
                              x_min: int, x_max: int, y_min: int, y_max: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max), (y_min+1)..(y_max+1)} {
            node_flux[i,j] = 0.25 * (mass_flux_x[i,j-1] + mass_flux_x[i,j]
                                    + mass_flux_x[i+1,j-1] + mass_flux_x[i+1,j]);
        }
    }

    inline proc k2_dir2_xvel(ref node_flux, ref mass_flux_y,
                              x_min: int, x_max: int, y_min: int, y_max: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max+1), (y_min+1)..(y_max)} {
            node_flux[i,j] = 0.25 * (mass_flux_y[i-1,j] + mass_flux_y[i,j]
                                    + mass_flux_y[i-1,j+1] + mass_flux_y[i,j+1]);
        }
    }

    // ── Kernel 3: node mass ───────────────────────────────────────────────────

    inline proc k3_dir1(ref node_mass_post, ref node_mass_pre,
                         ref density1, ref post_vol, ref node_flux,
                         x_min: int, x_max: int, y_min: int, y_max: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max+2), (y_min+1)..(y_max+2)} {
            node_mass_post[i,j] = 0.25 * (
                density1[i,j-1]   * post_vol[i,j-1]   +
                density1[i,j]     * post_vol[i,j]     +
                density1[i-1,j-1] * post_vol[i-1,j-1] +
                density1[i-1,j]   * post_vol[i-1,j]
            );
            node_mass_pre[i,j] = node_mass_post[i,j] - node_flux[i-1,j] + node_flux[i,j];
        }
    }

    inline proc k3_dir2(ref node_mass_post, ref node_mass_pre,
                         ref density1, ref post_vol, ref node_flux,
                         x_min: int, x_max: int, y_min: int, y_max: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max+2), (y_min+1)..(y_max+2)} {
            node_mass_post[i,j] = 0.25 * (
                density1[i,j-1]   * post_vol[i,j-1]   +
                density1[i,j]     * post_vol[i,j]     +
                density1[i-1,j-1] * post_vol[i-1,j-1] +
                density1[i-1,j]   * post_vol[i-1,j]
            );
            node_mass_pre[i,j] = node_mass_post[i,j] - node_flux[i,j-1] + node_flux[i,j];
        }
    }

    // ── Kernel 4: momentum flux ───────────────────────────────────────────────
    // Branchless: all data-dependent branches replaced with select expressions.
    // Loop bounds tightened to the exact active domain — no inner guard.
    //
    // dir1 active domain: i in [x_min+1 .. x_max],   j in [y_min+1 .. y_max+2]
    // dir2 active domain: i in [x_min+1 .. x_max+2], j in [y_min+1 .. y_max]

    inline proc k4_dir1(ref mom_flux, ref node_flux, ref node_mass_pre,
                         ref vel1, ref celldx,
                         x_min: int, x_max: int, y_min: int, y_max: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..x_max, (y_min+1)..(y_max+2)} {
            const neg      = node_flux[i,j] < 0.0;
            const upwind   = if neg then i+2 else i-1;
            const donor    = if neg then i+1 else i;
            const downwind = if neg then i   else i+1;
            const dif      = if neg then i+1 else i-1;   // donor when neg, upwind when pos

            const sigma    = abs(node_flux[i,j]) / node_mass_pre[donor,j];
            const width    = celldx[i];

            const vdiffuw  = vel1[donor,j]    - vel1[upwind,j];
            const vdiffdw  = vel1[downwind,j] - vel1[donor,j];
            const auw      = abs(vdiffuw);
            const adw      = abs(vdiffdw);
            const wind     = if vdiffdw <= 0.0 then -1.0 else 1.0;
            const lim_val  = wind * min(min(
                width * ((2.0-sigma)*adw/width + (1.0+sigma)*auw/celldx[dif]) / 6.0,
                auw), adw);
            const limiter  = if vdiffuw * vdiffdw > 0.0 then lim_val else 0.0;

            mom_flux[i,j]  = (vel1[donor,j] + (1.0 - sigma) * limiter) * node_flux[i,j];
        }
    }

    inline proc k4_dir2(ref mom_flux, ref node_flux, ref node_mass_pre,
                         ref vel1, ref celldy,
                         x_min: int, x_max: int, y_min: int, y_max: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max+2), (y_min+1)..y_max} {
            const neg      = node_flux[i,j] < 0.0;
            const upwind   = if neg then j+2 else j-1;
            const donor    = if neg then j+1 else j;
            const downwind = if neg then j   else j+1;
            const dif      = if neg then j+1 else j-1;

            const sigma    = abs(node_flux[i,j]) / node_mass_pre[i,donor];
            const width    = celldy[j];

            const vdiffuw  = vel1[i,donor]    - vel1[i,upwind];
            const vdiffdw  = vel1[i,downwind] - vel1[i,donor];
            const auw      = abs(vdiffuw);
            const adw      = abs(vdiffdw);
            const wind     = if vdiffdw <= 0.0 then -1.0 else 1.0;
            const lim_val  = wind * min(min(
                width * ((2.0-sigma)*adw/width + (1.0+sigma)*auw/celldy[dif]) / 6.0,
                auw), adw);
            const limiter  = if vdiffuw * vdiffdw > 0.0 then lim_val else 0.0;

            mom_flux[i,j]  = (vel1[i,donor] + (1.0 - sigma) * limiter) * node_flux[i,j];
        }
    }

    // ── Kernel 5: velocity update ─────────────────────────────────────────────

    inline proc k5_dir1(ref vel1, ref node_mass_pre, ref node_mass_post, ref mom_flux,
                         iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            vel1[i,j] = (vel1[i,j] * node_mass_pre[i,j]
                         + mom_flux[i-1,j] - mom_flux[i,j]) / node_mass_post[i,j];
        }
    }

    inline proc k5_dir2(ref vel1, ref node_mass_pre, ref node_mass_post, ref mom_flux,
                         iLo: int, iHi: int, jLo: int, jHi: int) {
        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {iLo..iHi, jLo..jHi} {
            vel1[i,j] = (vel1[i,j] * node_mass_pre[i,j]
                         + mom_flux[i,j-1] - mom_flux[i,j]) / node_mass_post[i,j];
        }
    }

    // ── Driver ───────────────────────────────────────────────────────────────
    proc advec_mom_driver(ref cfg: Config, which_vel: int,
                          direction: int, sweep_number: int) {

        const x_min = cfg.grid.xmin;
        const x_max = cfg.grid.xmax;
        const y_min = cfg.grid.ymin;
        const y_max = cfg.grid.ymax;

        const mom_sweep = direction + 2 * (sweep_number - 1);

        ref volume      = cfg.field.volume;
        ref vol_flux_y  = cfg.field.vol_flux_y;
        ref vol_flux_x  = cfg.field.vol_flux_x;
        ref mass_flux_x = cfg.field.mass_flux_x;
        ref mass_flux_y = cfg.field.mass_flux_y;
        ref density1    = cfg.field.density1;
        ref celldx      = cfg.field.celldx;
        ref celldy      = cfg.field.celldy;
        ref vel1;
        if which_vel == 1 { vel1 = cfg.field.xvel1; } else { vel1 = cfg.field.yvel1; }
        ref node_flux       = cfg.field.work_array1;
        ref node_mass_post  = cfg.field.work_array2;
        ref node_mass_pre   = cfg.field.work_array3;
        ref mom_flux        = cfg.field.work_array4;
        ref pre_vol         = cfg.field.work_array5;
        ref post_vol        = cfg.field.work_array6;

        const k1_iLo = max(1, x_min - 1);
        const k1_iHi = x_max + 3;
        const k1_jLo = max(1, y_min - 1);
        const k1_jHi = y_max + 3;

        const k5_iLo = x_min + 1;
        const k5_iHi = x_max + 2;
        const k5_jLo = y_min + 1;
        const k5_jHi = y_max + 2;

        // Kernels 1-3: only run for xvel call.
        // k1 writes pre_vol/post_vol; k2/k3 write node_flux/node_mass_pre/post.
        // All three produce results that are identical for both vel components
        // (same mom_sweep, same mass fluxes), so the yvel call reuses them.
        if which_vel == 1 {
            select mom_sweep {
                when 1 do k1_sweep1(post_vol, pre_vol, volume, vol_flux_x, vol_flux_y,
                                    k1_iLo, k1_iHi, k1_jLo, k1_jHi);
                when 2 do k1_sweep2(post_vol, pre_vol, volume, vol_flux_x, vol_flux_y,
                                    k1_iLo, k1_iHi, k1_jLo, k1_jHi);
                when 3 do k1_sweep3(post_vol, pre_vol, volume, vol_flux_y,
                                    k1_iLo, k1_iHi, k1_jLo, k1_jHi);
                when 4 do k1_sweep4(post_vol, pre_vol, volume, vol_flux_x,
                                    k1_iLo, k1_iHi, k1_jLo, k1_jHi);
            }

            // Zero node_flux: k3/k4 read one border cell outside k2's active write region.
            node_flux = 0.0;

            if direction == 1 then
                k2_dir1_xvel(node_flux, mass_flux_x, x_min, x_max, y_min, y_max);
            else
                k2_dir2_xvel(node_flux, mass_flux_y, x_min, x_max, y_min, y_max);

            if direction == 1 then
                k3_dir1(node_mass_post, node_mass_pre, density1, post_vol, node_flux,
                        x_min, x_max, y_min, y_max);
            else
                k3_dir2(node_mass_post, node_mass_pre, density1, post_vol, node_flux,
                        x_min, x_max, y_min, y_max);
        }

        // Kernels 4-5: run for BOTH xvel and yvel.
        // k4 uses vel1 (xvel1 or yvel1) and the shared node_flux/node_mass_pre from k2/k3.
        // k5 applies the momentum update to vel1.
        if direction == 1 then
            k4_dir1(mom_flux, node_flux, node_mass_pre, vel1, celldx,
                    x_min, x_max, y_min, y_max);
        else
            k4_dir2(mom_flux, node_flux, node_mass_pre, vel1, celldy,
                    x_min, x_max, y_min, y_max);

        if direction == 1 then
            k5_dir1(vel1, node_mass_pre, node_mass_post, mom_flux,
                    k5_iLo, k5_iHi, k5_jLo, k5_jHi);
        else
            k5_dir2(vel1, node_mass_pre, node_mass_post, mom_flux,
                    k5_iLo, k5_iHi, k5_jLo, k5_jHi);
    }
}
