module Update_halo {
    use Definitions;
    use GPU;

    // Field bitmask constants — param values allow compile-time specialisation:
    // each unique `fields` combination generates a separate kernel with dead
    // branches eliminated by LLVM, and zero string D2H copies at runtime.
    param UH_DEN0  =     1;   // density0
    param UH_DEN1  =     2;   // density1
    param UH_ENG0  =     4;   // energy0
    param UH_ENG1  =     8;   // energy1
    param UH_PRESS =    16;   // pressure
    param UH_VISC  =    32;   // viscosity
    param UH_SOUND =    64;   // soundspeed
    param UH_XV0   =   128;   // xvel0
    param UH_XV1   =   256;   // xvel1
    param UH_YV0   =   512;   // yvel0
    param UH_YV1   =  1024;   // yvel1
    param UH_VFX   =  2048;   // vol_flux_x
    param UH_MFX   =  4096;   // mass_flux_x
    param UH_VFY   =  8192;   // vol_flux_y
    param UH_MFY   = 16384;   // mass_flux_y

    // Two fused foralls (halves kernel-launch count vs the old 4-forall version):
    //
    //   Forall LR  (j_idx × k) — Left + Right halos, both non-vertex and vertex:
    //     Non-vertex (doF1): density0/1, energy0/1, pressure, viscosity,
    //                        soundspeed, vol_flux_x, mass_flux_x
    //                        → active k range  nv_k_lo..nv_k_hi
    //     Vertex    (doF2): xvel0/1, yvel0/1, vol_flux_y, mass_flux_y
    //                        → active k range  nv_k_lo..v_k_hi  (one extra row)
    //     Loop iterates the wider vertex range; F1 fields guarded by k <= nv_k_hi.
    //
    //   Forall BT  (j × k_idx) — Bottom + Top halos, both non-vertex and vertex:
    //     Non-vertex (doF3): density0/1, energy0/1, pressure, viscosity,
    //                        soundspeed, vol_flux_y, mass_flux_y
    //                        → active j range  nv_j_lo..nv_j_hi
    //     Vertex    (doF4): xvel0/1, yvel0/1, vol_flux_x, mass_flux_x
    //                        → active j range  nv_j_lo..v_j_hi  (one extra column)
    //     Loop iterates the wider vertex range; F3 fields guarded by j <= nv_j_hi.

    proc update_halo_kernel(x_min: int, x_max: int, y_min: int, y_max: int,
                            chunk_neighbours: [0..3] int, tile_neighbours: [0..3] int,
                            field: Field, param fields: int, depth: int) {

        ref density0    = field.density0;
        ref density1    = field.density1;
        ref energy0     = field.energy0;
        ref energy1     = field.energy1;
        ref pressure    = field.pressure;
        ref viscosity   = field.viscosity;
        ref soundspeed  = field.soundspeed;
        ref xvel0       = field.xvel0;
        ref xvel1       = field.xvel1;
        ref yvel0       = field.yvel0;
        ref yvel1       = field.yvel1;
        ref vol_flux_x  = field.vol_flux_x;
        ref vol_flux_y  = field.vol_flux_y;
        ref mass_flux_x = field.mass_flux_x;
        ref mass_flux_y = field.mass_flux_y;

        // ── Decode fields bitmask into compile-time param booleans ───────────
        // Each flag is a compile-time constant; dead branches are eliminated
        // entirely by LLVM — no runtime comparison, no D2H copy.
        param doDen0  = (fields & UH_DEN0)  != 0;
        param doDen1  = (fields & UH_DEN1)  != 0;
        param doEng0  = (fields & UH_ENG0)  != 0;
        param doEng1  = (fields & UH_ENG1)  != 0;
        param doPress = (fields & UH_PRESS) != 0;
        param doVisc  = (fields & UH_VISC)  != 0;
        param doSound = (fields & UH_SOUND) != 0;
        param doXv0   = (fields & UH_XV0)   != 0;
        param doXv1   = (fields & UH_XV1)   != 0;
        param doYv0   = (fields & UH_YV0)   != 0;
        param doYv1   = (fields & UH_YV1)   != 0;
        param doVFX   = (fields & UH_VFX)   != 0;
        param doMFX   = (fields & UH_MFX)   != 0;
        param doVFY   = (fields & UH_VFY)   != 0;
        param doMFY   = (fields & UH_MFY)   != 0;

        // Group flags: skip entire foralls when no field in that group is needed
        param doF1 = doDen0||doDen1||doEng0||doEng1||doPress||doVisc||doSound||doVFX||doMFX;
        param doF2 = doXv0||doXv1||doYv0||doYv1||doVFY||doMFY;
        param doF3 = doDen0||doDen1||doEng0||doEng1||doPress||doVisc||doSound||doVFY||doMFY;
        param doF4 = doXv0||doXv1||doYv0||doYv1||doVFX||doMFX;

        const nv_k_lo = max(1, y_min - depth + 1);
        const nv_k_hi = y_max + depth + 1;
        const  v_k_hi = y_max + 1 + depth + 1;

        const nv_j_lo = max(1, x_min - depth + 1);
        const nv_j_hi = x_max + depth + 1;
        const  v_j_hi = x_max + 1 + depth + 1;

        // ── Forall LR: Left + Right, all fields (F1 + F2 fused) ──────────────
        if doF1 || doF2 {
            @gpu.blockSize(gpuBlockSize)
            forall (j_idx, k) in {0..2*depth-1, nv_k_lo..v_k_hi} {
                if j_idx < depth {
                    const j_off = j_idx;
                    const jg    = 1 - j_off;
                    const jm    = 2 + j_off;
                    // Non-vertex fields stop one row before vertex fields
                    if doF1 && k <= nv_k_hi {
                        if doDen0  { density0[jg,k]               = density0[jm,k]; }
                        if doDen1  { density1[jg,k]               = density1[jm,k]; }
                        if doEng0  { energy0[jg,k]                = energy0[jm,k]; }
                        if doEng1  { energy1[jg,k]                = energy1[jm,k]; }
                        if doPress { pressure[jg,k]               = pressure[jm,k]; }
                        if doVisc  { viscosity[jg,k]              = viscosity[jm,k]; }
                        if doSound { soundspeed[jg,k]             = soundspeed[jm,k]; }
                        if doVFX   { vol_flux_x[jg,k]             = -vol_flux_x[jm+1,k]; }
                        if doMFX   { mass_flux_x[jg,k]            = -mass_flux_x[jm+1,k]; }
                    }
                    // Vertex fields cover full v_k_hi range
                    if doF2 {
                        if doXv0 { xvel0[jg,k]                    = -xvel0[jm+1,k]; }
                        if doXv1 { xvel1[jg,k]                    = -xvel1[jm+1,k]; }
                        if doYv0 { yvel0[jg,k]                    =  yvel0[jm+1,k]; }
                        if doYv1 { yvel1[jg,k]                    =  yvel1[jm+1,k]; }
                        if doVFY { vol_flux_y[jg,k]               =  vol_flux_y[jm,k]; }
                        if doMFY { mass_flux_y[jg,k]              =  mass_flux_y[jm,k]; }
                    }
                } else {
                    const j_off = j_idx - depth;
                    const jg    = x_max + 2 + j_off;
                    const jm    = x_max + 1 - j_off;
                    if doF1 && k <= nv_k_hi {
                        if doDen0  { density0[jg,k]               = density0[jm,k]; }
                        if doDen1  { density1[jg,k]               = density1[jm,k]; }
                        if doEng0  { energy0[jg,k]                = energy0[jm,k]; }
                        if doEng1  { energy1[jg,k]                = energy1[jm,k]; }
                        if doPress { pressure[jg,k]               = pressure[jm,k]; }
                        if doVisc  { viscosity[jg,k]              = viscosity[jm,k]; }
                        if doSound { soundspeed[jg,k]             = soundspeed[jm,k]; }
                        if doVFX   { vol_flux_x[x_max+3+j_off,k]  = -vol_flux_x[jm,k]; }
                        if doMFX   { mass_flux_x[x_max+3+j_off,k] = -mass_flux_x[jm,k]; }
                    }
                    if doF2 {
                        if doXv0 { xvel0[x_max+3+j_off,k]         = -xvel0[jm,k]; }
                        if doXv1 { xvel1[x_max+3+j_off,k]         = -xvel1[jm,k]; }
                        if doYv0 { yvel0[x_max+3+j_off,k]         =  yvel0[jm,k]; }
                        if doYv1 { yvel1[x_max+3+j_off,k]         =  yvel1[jm,k]; }
                        if doVFY { vol_flux_y[x_max+2+j_off,k]    =  vol_flux_y[jm,k]; }
                        if doMFY { mass_flux_y[x_max+2+j_off,k]   =  mass_flux_y[jm,k]; }
                    }
                }
            }
        }

        // ── Forall BT: Bottom + Top, all fields (F3 + F4 fused) ──────────────
        if doF3 || doF4 {
            @gpu.blockSize(gpuBlockSize)
            forall (j, k_idx) in {nv_j_lo..v_j_hi, 0..2*depth-1} {
                if k_idx < depth {
                    const k_off = k_idx;
                    const kg    = 1 - k_off;
                    const km    = 2 + k_off;
                    // Non-vertex fields stop one column before vertex fields
                    if doF3 && j <= nv_j_hi {
                        if doDen0  { density0[j,kg]                = density0[j,km]; }
                        if doDen1  { density1[j,kg]                = density1[j,km]; }
                        if doEng0  { energy0[j,kg]                 = energy0[j,km]; }
                        if doEng1  { energy1[j,kg]                 = energy1[j,km]; }
                        if doPress { pressure[j,kg]                = pressure[j,km]; }
                        if doVisc  { viscosity[j,kg]               = viscosity[j,km]; }
                        if doSound { soundspeed[j,kg]              = soundspeed[j,km]; }
                        if doVFY   { vol_flux_y[j,kg]              = -vol_flux_y[j,km+1]; }
                        if doMFY   { mass_flux_y[j,kg]             = -mass_flux_y[j,km+1]; }
                    }
                    // Vertex fields cover full v_j_hi range
                    if doF4 {
                        if doXv0 { xvel0[j,kg]                     =  xvel0[j,km+1]; }
                        if doXv1 { xvel1[j,kg]                     =  xvel1[j,km+1]; }
                        if doYv0 { yvel0[j,kg]                     = -yvel0[j,km+1]; }
                        if doYv1 { yvel1[j,kg]                     = -yvel1[j,km+1]; }
                        if doVFX { vol_flux_x[j,kg]                =  vol_flux_x[j,km]; }
                        if doMFX { mass_flux_x[j,kg]               =  mass_flux_x[j,km]; }
                    }
                } else {
                    const k_off = k_idx - depth;
                    const kg    = y_max + 2 + k_off;
                    const km    = y_max + 1 - k_off;
                    if doF3 && j <= nv_j_hi {
                        if doDen0  { density0[j,kg]                = density0[j,km]; }
                        if doDen1  { density1[j,kg]                = density1[j,km]; }
                        if doEng0  { energy0[j,kg]                 = energy0[j,km]; }
                        if doEng1  { energy1[j,kg]                 = energy1[j,km]; }
                        if doPress { pressure[j,kg]                = pressure[j,km]; }
                        if doVisc  { viscosity[j,kg]               = viscosity[j,km]; }
                        if doSound { soundspeed[j,kg]              = soundspeed[j,km]; }
                        if doVFY   { vol_flux_y[j,y_max+3+k_off]   = -vol_flux_y[j,km]; }
                        if doMFY   { mass_flux_y[j,y_max+3+k_off]  = -mass_flux_y[j,km]; }
                    }
                    if doF4 {
                        if doXv0 { xvel0[j,y_max+3+k_off]          =  xvel0[j,km]; }
                        if doXv1 { xvel1[j,y_max+3+k_off]          =  xvel1[j,km]; }
                        if doYv0 { yvel0[j,y_max+3+k_off]          = -yvel0[j,km]; }
                        if doYv1 { yvel1[j,y_max+3+k_off]          = -yvel1[j,km]; }
                        if doVFX { vol_flux_x[j,y_max+2+k_off]     =  vol_flux_x[j,km]; }
                        if doMFX { mass_flux_x[j,y_max+2+k_off]    =  mass_flux_x[j,km]; }
                    }
                }
            }
        }
    }

    proc update_halo(cfg: Config, param fields: int, depth: int) {
        var chunk_neighbours: [0..3] int = [-1, -1, -1, -1];
        var tile_neighbours:  [0..3] int = [-1, -1, -1, -1];
        update_halo_kernel(cfg.grid.xmin, cfg.grid.xmax, cfg.grid.ymin, cfg.grid.ymax,
                           chunk_neighbours, tile_neighbours, cfg.field, fields, depth);
    }
}
