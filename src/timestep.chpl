module Timestep {
    use CloverLeaf;
    use IdealGas;
    use Definitions;
    use Math;
    use Viscosity;
    use Update_halo;
    use Time;
    use GPU;

   proc calc_dt_kernel(ref cfg: Config, dtmin: real, dtc_safe: real, dtu_safe: real, dtv_safe: real, dtdiv_safe: real,
                   ref field: Field, ref dt_min_val: real, ref dtl_control: int,
                   ref xl_pos: real, ref yl_pos: real, ref jldt: int, ref kldt: int, ref small: int) {

        small = 0;
        dt_min_val = g_big;
        var jk_control: real = 1.1;
        var dt_min_val0 = g_big;

        ref x_min = cfg.grid.xmin;
        ref x_max = cfg.grid.xmax;
        ref y_min = cfg.grid.ymin;
        ref y_max = cfg.grid.ymax;

        ref cellx      = field.cellx;
        ref celly      = field.celly;
        ref density0_f = field.density0;
        ref energy0_f  = field.energy0;
        ref pressure_f = field.pressure;
        ref sound_f    = field.soundspeed;
        ref xvel0_f    = field.xvel0;
        ref yvel0_f    = field.yvel0;

        ref celldx     = field.celldx;
        ref celldy     = field.celldy;
        ref soundspeed = field.soundspeed;
        ref viscosity  = field.viscosity;
        ref xvel0      = field.xvel0;
        ref yvel0      = field.yvel0;
        ref xarea      = field.xarea;
        ref yarea      = field.yarea;
        ref volume     = field.volume;
        ref density0   = field.density0;

        @gpu.blockSize(gpuBlockSize)
        forall (i,j) in {(x_min+1)..(x_max+1), (y_min+1)..(y_max+1)}
            with (min reduce dt_min_val0) {
            const dsx = celldx[i];
            const dsy = celldy[j];
            const ss = soundspeed[i,j];
            var cc = ss * ss;
            cc += 2.0 * viscosity[i,j] / density0[i,j];
            cc = max(sqrt(cc), g_small);
            const dtct = dtc_safe * min(dsx, dsy) / cc;
            var div = 0.0;
            var dv1 = (xvel0[i,j] + xvel0[i,j+1]) * xarea[i,j];
            var dv2 = (xvel0[i+1,j] + xvel0[i+1,j+1]) * xarea[i+1,j];
            div += dv2 - dv1;
            const dtut = dtu_safe * 2.0 * volume[i,j] /
                        max(max(abs(dv1), abs(dv2)), g_small * volume[i,j]);
            dv1 = (yvel0[i,j] + yvel0[i+1,j]) * yarea[i,j];
            dv2 = (yvel0[i,j+1] + yvel0[i+1,j+1]) * yarea[i,j+1];
            div += dv2 - dv1;
            const dtvt = dtv_safe * 2.0 * volume[i,j] /
                        max(max(abs(dv1), abs(dv2)), g_small * volume[i,j]);
            div /= (2.0 * volume[i,j]);
            var dtdivt: real;
            if div < -g_small then
                dtdivt = dtdiv_safe * (-1.0 / div);
            else
                dtdivt = g_big;
            dt_min_val0 reduce= min(dtct, min(dtut, min(dtvt, min(dtdivt, g_big))));
        }

        dt_min_val = dt_min_val0;
        dtl_control = (10.01 * (jk_control - jk_control: int)): int;
        jk_control -= (jk_control - jk_control: int);
        jldt = (jk_control: int) % x_max;
        kldt = (1.0 + (jk_control / x_max)): int;

        if dt_min_val < dtmin then
            small = 1;

        if small != 0 && !noPrint{
            writeln("Timestep information:");
            writeln("j, k                 : ", jldt, " ", kldt);
            writeln("x, y                 : ", cellx[jldt], " ", celly[kldt]);
            writeln("timestep : ", dt_min_val);
            writeln("Cell velocities;");
            writeln(xvel0_f[jldt,kldt], " ", yvel0_f[jldt,kldt]);
            writeln(xvel0_f[jldt+1,kldt], " ", yvel0_f[jldt+1,kldt]);
            writeln(xvel0_f[jldt+1,kldt+1], " ", yvel0_f[jldt+1,kldt+1]);
            writeln(xvel0_f[jldt,kldt+1], " ", yvel0_f[jldt,kldt+1]);
            writeln("density, energy, pressure, soundspeed ");
            writeln(density0_f[jldt,kldt], " ", energy0_f[jldt,kldt], " ",
                    pressure_f[jldt,kldt], " ", sound_f[jldt,kldt]);
        }
    }

    proc calculate_timestep(ref cfg: Config) {

        cfg.dt = g_big;

        var clock = new stopwatch();
        if(profile) then clock.start();

        ideal_gas(cfg, 0, false);

        if(profile) { clock.stop(); cfg.profile_times.idealGas = cfg.profile_times.idealGas + clock.elapsed();}

        var clock_halo = new stopwatch();
        if(profile) then clock_halo.start();

        update_halo(cfg, UH_DEN0|UH_ENG0|UH_PRESS|UH_XV0|UH_YV0, 1);

        if(profile) { clock_halo.stop(); cfg.profile_times.halo = cfg.profile_times.halo + clock_halo.elapsed();}

        var clock_viscosity = new stopwatch();
        if(profile) then clock_viscosity.start();

        viscosity(cfg);

        if(profile) {clock_viscosity.stop(); cfg.profile_times.viscosity = cfg.profile_times.viscosity + clock_viscosity.elapsed();}

        var clock_halo2 = new stopwatch();
        if(profile) then clock_halo2.start();

        update_halo(cfg, UH_VISC, 1);

        if(profile) {clock_halo2.stop(); cfg.profile_times.halo = cfg.profile_times.halo + clock_halo2.elapsed();}

        var clock_timestep = new stopwatch();
        if(profile) then clock_timestep.start();

        var jldt, kldt: int;
        var x_pos, y_pos, xl_pos, yl_pos: real;
        var dt_control, dtl_control: string;
        var small: int = 0;
        var l_control: int;
        var dtlp: real;

        calc_dt_kernel(cfg, dtmin, dtc_safe, dtu_safe, dtv_safe, dtdiv_safe, cfg.field, dtlp,
            l_control, xl_pos, yl_pos, jldt, kldt, small);

        if dtlp <= cfg.dt {
            cfg.dt = dtlp;
            dt_control = dtl_control;
            x_pos = xl_pos;
            y_pos = yl_pos;
            cfg.jdt = jldt;
            cfg.kdt = kldt;
        }

        cfg.dt = min(min(cfg.dt, cfg.dtold * dtrise), dtmax);
        cfg.dtold = cfg.dt;

        if(profile) {clock_timestep.stop(); cfg.profile_times.timestep = cfg.profile_times.timestep + clock_timestep.elapsed();}

        if !noPrint then
            writeln("  Step ", cfg.step, " time ", cfg.time, " control ", dt_control,
                    " timestep ", cfg.dt, " ", cfg.jdt, ",", cfg.kdt,
                    " x ", x_pos, " y ", y_pos);
    }
}
