module PdV {
  use CloverLeaf;
  use Time;
  use Math;
  use Definitions;
  use IdealGas;
  use Update_halo;
  use Revert;
  use GPU;

  proc PdV_kernel(predict: bool, ref cfg: Config) {
    const x_min = cfg.grid.xmin;
    const x_max = cfg.grid.xmax;
    const y_min = cfg.grid.ymin;
    const y_max = cfg.grid.ymax;
    const dt = cfg.dt;

    ref xarea    = cfg.field.xarea;
    ref xvel0    = cfg.field.xvel0;
    ref xvel1    = cfg.field.xvel1;
    ref yarea    = cfg.field.yarea;
    ref yvel0    = cfg.field.yvel0;
    ref yvel1    = cfg.field.yvel1;
    ref volume   = cfg.field.volume;
    ref pressure = cfg.field.pressure;
    ref density0 = cfg.field.density0;
    ref density1 = cfg.field.density1;
    ref visc     = cfg.field.viscosity;
    ref energy0  = cfg.field.energy0;
    ref energy1  = cfg.field.energy1;

    if predict {
      // Predict branch: both vel0 and vel1 are vel0 (half-step estimate uses vel0 twice).
      // Original had: (vel0 + vel1 + vel0 + vel1) * 0.25 * dt * 0.5
      //             = 2*(vel0 + vel1) * 0.25 * dt * 0.5
      // With vel1==vel0 in predict: = 2*(vel0 + vel0) * 0.25 * dt * 0.5 = vel0 * 0.5 * dt
      // Rewritten: (vel0[a] + vel0[b]) * 0.25 * dt  (2 reads instead of 4 per flux)
      @gpu.blockSize(gpuBlockSize)
      forall (i,j) in {(x_min+1)..(x_max+1), (y_min+1)..(y_max+1)} {
        var left_flux   = xarea[i,j]   * (xvel0[i,j]   + xvel0[i,j+1])   * 0.25 * dt;
        var right_flux  = xarea[i+1,j] * (xvel0[i+1,j] + xvel0[i+1,j+1]) * 0.25 * dt;
        var bottom_flux = yarea[i,j]   * (yvel0[i,j]   + yvel0[i+1,j])   * 0.25 * dt;
        var top_flux    = yarea[i,j+1] * (yvel0[i,j+1] + yvel0[i+1,j+1]) * 0.25 * dt;
        var total_flux  = right_flux - left_flux + top_flux - bottom_flux;
        const vol = volume[i,j];
        var volume_change_s = vol / (vol + total_flux);
        var energy_change   = (pressure[i,j] + visc[i,j]) / density0[i,j] * total_flux / vol;
        energy1[i,j]  = energy0[i,j] - energy_change;
        density1[i,j] = density0[i,j] * volume_change_s;
      }
    } else {
      @gpu.blockSize(gpuBlockSize)
      forall (i,j) in {(x_min+1)..(x_max+1), (y_min+1)..(y_max+1)} {
        var left_flux   = (xarea[i,j]   * (xvel0[i,j]   + xvel0[i,j+1]   + xvel1[i,j]   + xvel1[i,j+1]))   * 0.25 * dt;
        var right_flux  = (xarea[i+1,j] * (xvel0[i+1,j] + xvel0[i+1,j+1] + xvel1[i+1,j] + xvel1[i+1,j+1])) * 0.25 * dt;
        var bottom_flux = (yarea[i,j]   * (yvel0[i,j]   + yvel0[i+1,j]   + yvel1[i,j]   + yvel1[i+1,j]))   * 0.25 * dt;
        var top_flux    = (yarea[i,j+1] * (yvel0[i,j+1] + yvel0[i+1,j+1] + yvel1[i,j+1] + yvel1[i+1,j+1])) * 0.25 * dt;
        var total_flux  = right_flux - left_flux + top_flux - bottom_flux;
        const vol = volume[i,j];
        var volume_change_s = vol / (vol + total_flux);
        var energy_change   = (pressure[i,j] + visc[i,j]) / density0[i,j] * total_flux / vol;
        energy1[i,j]  = energy0[i,j] - energy_change;
        density1[i,j] = density0[i,j] * volume_change_s;
      }
    }
  }

  proc PdV(ref cfg: Config, predict: bool) {
    var clock_pdv = new stopwatch();
    if(profile) then clock_pdv.start();

    PdV_kernel(predict, cfg);

    if(profile) {clock_pdv.stop(); cfg.profile_times.pdv = cfg.profile_times.pdv + clock_pdv.elapsed();}

    if (predict) {
      if cfg.dumpDir != "" {
        try {
          dumpState(cfg, "0_"+ cfg.step:string +"_2_PdV_ideal-pre.txt", 0.0, 0);
        } catch e {
          writeln("Failed to dump debug state: ", e.message());
        }
      }

      var clock_idealGas = new stopwatch();
      if(profile) then clock_idealGas.start();

      ideal_gas(cfg, 0, true, compSS=false);  // predict: pressure only, soundspeed not needed

      if(profile) {clock_idealGas.stop(); cfg.profile_times.idealGas = cfg.profile_times.idealGas + clock_idealGas.elapsed();}

      if cfg.dumpDir != "" {
        try {
          dumpState(cfg, "0_"+ cfg.step:string +"_2_PdV_ideal.txt", 0.0, 0);
        } catch e {
          writeln("Failed to dump debug state: ", e.message());
        }
      }

      var clock_halo = new stopwatch();
      if(profile) then clock_halo.start();

      update_halo(cfg, UH_PRESS, 1);

      if(profile) {clock_halo.stop(); cfg.profile_times.halo = cfg.profile_times.halo + clock_halo.elapsed();}

      var clock_revert = new stopwatch();
      if(profile) then clock_revert.start();

      revert_PdV(cfg);

      if(profile) {clock_revert.stop(); cfg.profile_times.revert = cfg.profile_times.revert + clock_revert.elapsed();}
    }
  }
}
