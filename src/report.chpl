module Report {
    use Definitions;
    use IdealGas;
    use Math;
    use Initialization;

    proc checking_summated_values(ref cfg: Config, vol: real, mass: real, ie:real, ke:real , press:real){
        writeln("checking problem: ", cfg.test_problem);

        if cfg.test_problem >= 1 {
            var qa_diff: real;
            var valid_problem = false;

            for test in testProblems {
                if test.id == cfg.test_problem {
                    qa_diff = abs((100.0 * (ke / test.expected_ke)) - 100.0);
                    valid_problem = true;
                    break;
                }
            }

            if !valid_problem {
                qa_diff = 100.0;
                writeln(" WARNING: Unknown test problem ", cfg.test_problem, ", validation will fail");
            }

            writeln(" Test problem ", cfg.test_problem, " is within ", qa_diff, "% of the expected solution");

            if !isNan(qa_diff) && qa_diff < 0.001 {
                writeln(" This test is considered PASSED");
                cfg.report_test_fail = false;
            } else {
                writeln(" This test is considered NOT PASSED");
                cfg.report_test_fail = true;
            }
        }
    }

    proc report(ref cfg: Config){
        for i in 0..<cfg.grid.total_cells do
            ideal_gas(cfg, i,false);

        ref field = cfg.field;
        const x_min = cfg.grid.xmin;
        const x_max = cfg.grid.xmax;
        const y_min = cfg.grid.ymin;
        const y_max = cfg.grid.ymax;

        const cellDom = {(x_min+1)..(x_max+1), (y_min+1)..(y_max+1)};

        var vol: real = 0.0;
        forall (j,k) in cellDom with (+ reduce vol) {
            vol += field.volume[j,k];
        }

        var mass: real = 0.0;
        forall (j,k) in cellDom with (+ reduce mass) {
            mass += field.volume[j,k] * field.density0[j,k];
        }

        var ie: real = 0.0;
        forall (j,k) in cellDom with (+ reduce ie) {
            ie += field.volume[j,k] * field.density0[j,k] * field.energy0[j,k];
        }

        var ke: real = 0.0;
        forall (j,k) in cellDom with (+ reduce ke) {
            const cell_mass = field.volume[j,k] * field.density0[j,k];
            var vsqrd = 0.25 * (
                field.xvel0[j,k]   * field.xvel0[j,k]   + field.yvel0[j,k]   * field.yvel0[j,k]   +
                field.xvel0[j+1,k] * field.xvel0[j+1,k] + field.yvel0[j+1,k] * field.yvel0[j+1,k] +
                field.xvel0[j,k+1] * field.xvel0[j,k+1] + field.yvel0[j,k+1] * field.yvel0[j,k+1] +
                field.xvel0[j+1,k+1] * field.xvel0[j+1,k+1] + field.yvel0[j+1,k+1] * field.yvel0[j+1,k+1]);
            ke += cell_mass * 0.5 * vsqrd;
        }

        var press: real = 0.0;
        forall (j,k) in cellDom with (+ reduce press) {
            press += field.volume[j,k] * field.pressure[j,k];
        }

        checking_summated_values(cfg, vol, mass, ie, ke, press);
    }
}
