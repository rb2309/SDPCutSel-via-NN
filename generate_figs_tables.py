import utilities as utils
from cut_select_qp import CutSolver
from cut_select_qcqp import CutSolverQCQP
import numpy as np
import os


def main():
    # Lighter test configuration (omitting jumbo/large instances that take very long to test on for specific settings)
    test_cfg = True

    # All figures and tables
    run_everything(folder_tables="data_tables", folder_figures="data_figures", test_cfg=test_cfg)

    # All figures
    # run_all_figures(folder_name="data_figures", test_cfg=test_cfg)

    # An individual figure
    # figure_nb = 4
    # run_for_figure(figure_nb, folder_name="data_figures", test_cfg=test_cfg)

    # All tables
    # run_for_all_tables(table=0, folder_name="data_tables", test_cfg=test_cfg)

    # An individual table
    # table_nb = 4
    # run_for_all_tables(table=table_nb, folder_name="data_tables", test_cfg=test_cfg)


def run_everything(folder_tables="data_tables", folder_figures="data_figures", test_cfg=True):
    run_all_figures(folder_name=folder_figures, test_cfg=test_cfg)
    run_for_all_tables(table=0, folder_name=folder_tables, test_cfg=test_cfg)


def run_all_figures(folder_name="data_figures", test_cfg=True):
    for figure_nb in [1, 3, 4, 5, 7, 8, 9, 11, 15]:
        run_for_figure(figure_nb, folder_name=folder_name, test_cfg=test_cfg)


def run_for_figure(figure, folder_name="data_figures", test_cfg=True, write_flag='w'):
    """Run on all BoxQP instances Figures 1, 3-11, 15 in the manuscript
    :param figure: for which figure to run
    :param folder_name: what folder to save to
    :param test_cfg: run all figures fully (False) or run scaled down (True) for Figure 1 by limiting instance size
    and for Figure 9 by ommiting dense instances
    :param write_flag: overwrite to .csv data file or append
    :return: saved .csv data files in folder_name
    """
    dirname = os.path.join(os.path.curdir, folder_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    print("Run for Figure " + str(figure) + " ... ")
    # Create instances of cutting plane solvers (QP and QCQP)
    cs = CutSolver()
    cs_qcp = CutSolverQCQP()
    dirname = os.path.join(os.path.curdir, folder_name)
    # Disctionary of cut selection strategies for printing
    dict_sel_print = {1: "feasibility", 2: "optimality estimated", 3: "optimality exact",
                      4: "combined", 5: "random", 0: "dense"}
    # Random seed used in all figures
    seed_nb = 7
    ###########################################
    # Figure 1 analysing M+S_3 bounds on random instances of different size
    ###########################################
    # takes a very long time (~10h) to run to size of instances 60, consider stopping at 30 for testing
    if figure == 1:
        save_file = os.path.join(dirname, "fig1_bounds_3D.csv")
        reset_file(save_file, write_flag)
        if test_cfg:
            utils.get_average_bounds_3d(save_file, start=5, stop=30, step=5, nb_instances=30, rand_seed=seed_nb)
        else:
            utils.get_average_bounds_3d(save_file, start=5, stop=60, step=5, nb_instances=30, rand_seed=seed_nb)
    ###########################################
    # Figure 3 looking at the SDP relaxation surface for a 2D example
    ###########################################
    # very quick to run (~10s)
    elif figure == 3:
        save_file = os.path.join(dirname, "fig3_SDP_2D_points.csv")
        reset_file(save_file, write_flag)
        utils.gen_sdp_surface_2d_fig3(save_file)
    ###########################################
    # Figure 4 looking at distribution of data sampled for neural nets
    ###########################################
    # Can also be run to generate training/test sets for neural nets, so using Figure 7 data
    # ok to run (10-20 mins) for 10k samples each
    # Will match the existing file "GenDataTest3D.csv" with 500k entries in "neural_nets" (same seed)
    elif figure == 4:
        save_file = os.path.join(os.path.curdir, "neural_nets", "GenDataTest3D_fig7.csv")
        reset_file(save_file, write_flag)
        utils.gen_data_ndim(10000, 3, save_file, rand_seed=seed_nb)
    ###########################################
    # Figure 5 for uniform Q's distribution of eigenvalues
    ###########################################
    # ok to run (~30s)
    elif figure == 5:
        save_file = os.path.join(dirname, "fig5_data_randomQs.csv")
        reset_file(save_file, write_flag)
        utils.gen_data_3d_q(500000, save_file, rand_seed=seed_nb)
    ###########################################
    # Figure 7 plot for other neural net test data
    ###########################################
    # ok to run (10-20 mins) for 10k samples each
    # Will match the existing files with 500k entries in "neural_nets" (same seed)
    elif figure == 7:
        for nn_size in [2, 3, 4, 5]:
            save_file = os.path.join(os.path.curdir, "neural_nets", "GenDataTest" + str(nn_size) + "D_fig7.csv")
            reset_file(save_file, write_flag)
            utils.gen_data_ndim(10000, nn_size, save_file, rand_seed=seed_nb)
    ###########################################
    # Figure 8 plot comparing optimality selection via estimated vs. exact measures
    ###########################################
    # quick to run (~1m)
    elif figure == 8:
        save_file = os.path.join(dirname, "fig8_data.csv")
        cut_rounds = 4
        # The M+S^E_3 theoretical bound can be obtained (approximately)
        # by running feasibility selection with many rounds
        sol = cs.cut_select_algo("spar020-100-1", 3, 100, strat=1, nb_rounds_cuts=40)[0][-1]
        (objPercs, roundsStats, roundStdDevs, roundsAllCuts) = \
            cs.cut_select_algo("spar020-100-1", 3, 100, strat=-1, nb_rounds_cuts=cut_rounds, sol=sol, plots=True)
        with open(save_file, "w") as f:
            f.write("cuts_round,gap_closed,percent_same_sel,std_dev_exact_selection\n")
            for cutRound in range(1, cut_rounds + 1):
                f.write(str(cutRound) + "," + str(objPercs[cutRound]) + "," + str(roundsStats[cutRound - 1]) +
                        "," + str(roundStdDevs[cutRound - 1]) + "\n")
            f.write("cuts_round,cut_number,sel_estim,sel_exact,estim_measure,exact_measure\n")
            for cut in roundsAllCuts:
                f.write(",".join(str(x) for x in cut) + "\n")
    ###########################################
    # Figure 9 and 10 plots on optimality (estimator vs exact) vs feasibility vs combined vs random vs dense cuts
    ###########################################
    # reasonable run-time for "spar040-030-1", "spar040-100-1" (~15mins) but slow for "spar100-025-1", "spar100-075-1"
    #   - consider removing it or wait a very long time (~5h) esp. for "spar100-075-1"
    elif figure == 9:
        save_file = os.path.join(dirname, "fig9_10_data.csv")
        reset_file(save_file, write_flag)
        # M+S solutions from Table 11 in "Globally solving nonconvex quadratic programming problems
        # with box constraints via integer programming methods", P. Bonami, O. Gunluk, J. Linderoth
        sol_ms = [839.50, 2476.38, 4066.38, 7514.48]
        instances = list(zip(sol_ms, ["spar040-030-1", "spar040-100-1", "spar100-025-1", "spar100-075-1"]))
        if test_cfg:
            instances = instances[0:1]
        dim = 3  # dimension 3
        sel_size = 0.05  # sel_size 5% of sub-problems/cuts selected
        cut_rounds = 20
        rand_runs = 10
        # record gaps and cumulative times - the time spend on solving McCormick counts as 0
        for sol_ms, filename in instances:
            # The M+S_3 theoretical bound can be obtained (approximately)
            # by running feasibility selection with many rounds
            print("- " + filename + " finding M+S_3 bound ... ")
            bound_run = cs.cut_select_algo(filename, dim, sel_size, strat=1, plots=True, sol=sol_ms, nb_rounds_cuts=40)
            bound = bound_run[0:40 + 1][-1]
            max_cuts = np.floor(bound_run[-1] * sel_size)
            sols = []
            times_cumul = []    # cumulative times in round 1, rounds 1-2, rounds 1-3,... ,rounds 1-20
            # run cut selection with strategies:
            # optimality estimated, feasibility, combined, random, dense, optimality exact
            for strat in [2]: #[1, 2, 4, 5, 3, 0]:
                runs = 1
                print("-" + filename + " " + dict_sel_print[strat] + " ... ")
                if strat == 5:   # random
                    np.random.seed(seed_nb)
                    runs = rand_runs
                for i in range(runs):
                    sols_timed = cs.cut_select_algo(filename, dim, sel_size, strat=strat, plots=True, sol=sol_ms,
                                                    nb_rounds_cuts=cut_rounds)
                    sols.append(sols_timed[0:cut_rounds + 1])
                    times = [0, *sols_timed[cut_rounds + 2:2 * cut_rounds + 2]]
                    times_cumul.append([sum(times[0:idx + 1]) for idx, _ in enumerate(times)])          
            sols.append([bound] * (cut_rounds + 1))
            sols = np.array(sols).T
            times_cumul.append([0] * (cut_rounds + 1))
            times_cumul = np.array(times_cumul).T
            with open(save_file, "a") as f:
                f.write(filename + "\n")
                f.write("size,density,max_cuts_sel\n")
                size_inst = filename.split('-')
                dens_inst = int(size_inst[1])
                size_inst = int(size_inst[0].split('r')[1])
                f.write(str(size_inst) + "," + str(dens_inst) + "," + str(max_cuts) + "\n")
                title_head = "opt_nn,feas,comb," + ",".join(str(x) for x in ["rand " + str(i) for i in range(rand_runs)])\
                             + ",dense,opt_exact,M+S_3 \n"
                f.write("Bounds:\n")
                f.write(title_head)
                for line in sols:
                    f.write(",".join(str(x) for x in line) + "\n")
                f.write("Times (cumulative):\n")
                f.write(title_head)
                for line in times_cumul:
                    f.write(",".join(str(x) for x in line) + "\n")
    ###########################################
    # Figure 11, 12 - plots on selecting strong vs all violated cuts
    ###########################################
    # reasonable run-time (~15mins)
    elif figure == 11:
        save_file = os.path.join(dirname, "fig11_12_data.csv")
        reset_file(save_file, write_flag)
        filename = "spar100-025-1"
        # M+S solution for "spar100-025-1" from Table 11 in "Globally solving nonconvex quadratic programming problems
        # with box constraints via integer programming methods", P. Bonami, O. Gunluk, J. Linderoth
        sol_ms = 4066.38
        dim = 3
        sel_size = 0.05
        cut_rounds = 20
        rand_runs = 10
        sols = []
        # run cut selection with strategies:
        # optimality combined, optimality estimated, feasibility, random, optimality exact
        for strat in [4, 2, 1, 5, 3]:
            runs = 1
            print("-" + filename + " " + dict_sel_print[strat] + " ... ")
            if strat == 5:  # random
                np.random.seed(seed_nb)
                runs = rand_runs
            # optimality-only strategies add only violated cuts within the selection size
            # associated with positive optimality measure (that predicts their violation).
            strong_only = True if strat in [2, 3] else False
            for i in range(runs):
                sols.append(cs.cut_select_algo(filename, dim, sel_size, strat=strat, plots=True, sol=sol_ms,
                                               strong_only=strong_only, nb_rounds_cuts=cut_rounds))
        sols = np.array(sols).T
        nb_columns = sols.shape[1]
        bounds = sols[0:cut_rounds + 1, :]
        cuts = sols[2 * cut_rounds + 3:3 * cut_rounds + 3, :]
        bounds_per_cut = np.zeros((cut_rounds + 1, sols.shape[1]))
        for col in range(nb_columns):
            for cutRound in range(1, cut_rounds + 1):
                bounds_per_cut[cutRound, col] = bounds[cutRound, col] / sum(cuts[0:cutRound, col])
        max_cuts = np.floor(sols[-1, 0] * sel_size)
        valid_cuts = np.divide(cuts, max_cuts)
        with open(save_file, "a") as f:
            title_head = "comb,opt_nn,feas," + ",".join(str(x) for x in ["rand " + str(i) for i in range(rand_runs)])\
                    + ",opt_exact\n"
            f.write(filename + "\n")
            f.write("gap_closed_overall\n")
            f.write(title_head)
            for line in bounds:
                f.write(",".join(str(x) for x in line) + "\n")
            f.write("gap_closed_per_nb_cuts_used\n")
            f.write(title_head)
            for line in bounds_per_cut:
                f.write(",".join(str(x) for x in line) + "\n")
            f.write("percent_valid_cuts_found\n")
            f.write(title_head)
            for line in valid_cuts:
                f.write(",".join(str(x) for x in line) + "\n")
    ###########################################
    # Figure 15 - plots on QCQP instances
    ###########################################
    # reasonable run-time (~15mins)
    elif figure == 15:
        save_file = os.path.join(dirname, "fig15_data.csv")
        filenames = ["q_50_10_25_1", "q_50_10_100_1", "q_50_50_25_1", "q_50_50_100_1"]
        strats_names = ["feas", "opt", "opt exact", "combined", "random"]
        # CONOPT solutions obtained via GAMS
        with open(os.path.join(os.path.curdir, 'qcqp_instances', 'qcqp_sols.txt'), "r") as f:
            sols = [(inst.split(',')[0], float(inst.split(',')[1])) for inst in f.read().split('\n')]
        nb_of_rand = 5
        for filename in filenames: #['q_20_0_50_1']:#
            size_inst = filename.split('_')
            cons_inst = int(size_inst[2])
            dens_inst = int(size_inst[3])
            size_inst = int(size_inst[1])
            sol = [sol for (file, sol) in sols if file == filename][0]
            for sel_size, nb_rounds in [(0.05, 10)]:
                gaps = []
                np.random.seed(seed_nb)
                for strat in [1, 4, *([5] * nb_of_rand)]:
                    print("- " + filename + ", " + str(sel_size) + "cuts, " + strats_names[strat - 1] + " sel ... ")
                    objectives, max_cut_sel, nbs_sdp_cuts, nbs_cuts_opt = \
                        cs_qcp.cut_select_algo(filename, 3, sel_size=sel_size, strat=strat, nb_rounds_cuts=nb_rounds)
                    gap_closed_percent = [0] * len(objectives)
                    for idx in range(1, len(objectives)):
                        gap_closed_percent[idx] = (objectives[idx] - objectives[0]) / (sol - objectives[0])
                    gaps.append(gap_closed_percent)
                    # print(gap_closed_percent)
                gaps = np.array(gaps).T
                with open(save_file, 'a') as f:
                    f.write(filename + "\n")
                    f.write("size,cons,density,sel_size\n")
                    f.write(str(size_inst) + "," + str(cons_inst) + "," + str(dens_inst) + "," + str(max_cut_sel) + "\n")
                    f.write("feas,comb2," + ",".join(
                        str(x) for x in ["rand " + str(i) for i in range(nb_of_rand)]) + "\n")
                    for line in gaps:
                        f.write(",".join(str(x) for x in line) + "\n")
    print("Run for Figure " + str(figure) + " - Done")


def run_for_all_tables(table=0, folder_name="data_tables", test_cfg=True):
    """Run on all BoxQP instances (or except the very large ones for test_cfg=True) to construct the tables 4-7 and
    figures 13-14 and 16 in the manuscript
    :param table: for which table to run (5-9, 13-14) or 0 for all tables and figs 13-14 and 16
    :param folder_name: what folder to save to
    :param test_cfg: run all BoxQP instances (takes extremely long esp. for tables 7-8), or a reduced set
    :return: saved .csv data files in folder_name
    """
    assert (table in [0, 2, 5, 6, 7, 8, 9, 13, 14, 16]), \
        "Please select all tables (0) or a valid table (5-9) or figure (13-14) to run data for"
    boxqp_files = 'filenames_test2.txt' if test_cfg else 'filenames.txt'
    text_file = open(os.path.join(os.path.curdir, 'boxqp_instances', boxqp_files), "r")
    dirname = os.path.join(os.path.curdir, folder_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    write_flag = "w"
    problems = text_file.read().split('\n')
    text_file.close()
    dict_sel = {2: "opt", 1: "feas", 4: "comb", 5: "random", 0: "dense"}
    dict_sel2 = {2: "Optimality selection (neural nets)", 1: "Feasibility selection"}
    info_inst = []
    # Get filename, size, density and solution of each BoxQP instance for calculations
    for prob in problems:
        filename, sol = prob.split('  ')
        size_inst = filename.split('-')
        density_inst = int(size_inst[1])
        size_inst = int(size_inst[0].split('r')[1])
        info_inst.append((filename, size_inst, density_inst, float(sol)))
    # Create instances of cutting plane solvers (QP and QCQP)
    cs = CutSolver()
    cs_qcp = CutSolverQCQP()

    ###########################################
    # Data for subproblem timing comparisons (Table 2)
    ###########################################
    if table in [0, 2]:
        save_file = os.path.join(dirname, "table2.csv")
        nb_evals = 1000
        n_values = [2, 3, 4, 5]
        utils.compare_nn_solver(nb_evals, n_values, save_file, rand_seed=7)
    ###########################################
    # Data for M+S^E_3 (Table 6)
    ###########################################
    if table in [0, 6]:
        sel_size = 0.1
        dim = 3
        r_runs = 5  # random runs
        cut_rounds = 4
        save_file = os.path.join(dirname, "data_all_boxqp_"+str(cut_rounds)+"rounds" + ".csv")
        with open(save_file, write_flag) as f:
            f.write(
                "M+S^E_3 with selection size " + str(sel_size) + " comparison "+\
                "in terms of gap closed - total times at cut rounds - separation times at cut rounds and "+\
                "nb of sdp cuts added at cut rounds between "
                "opt - feas - comb - dense - rand\n")
            white_space = ",".join("_" for _ in range(cut_rounds*5))
            f.write(",,,,gap_closed_at_round" + white_space + \
                    ",time_cut_at_round" + white_space + \
                    ",sep_time_cut_at_round" + white_space + \
                    ",nb_sdp_cuts_at_round" + white_space + "\n")
            f.write("filename,size,density,nb_subproblems," +
                    ",".join(
                        ",".join(
                            ",".join("r"+str(r+1)+"_"+ sel for sel in ["opt","feas","comb","dense","rand"])
                            for r in range(cut_rounds))
                        for _ in range(4)) + "\n")
        for inst in range(0, len(problems)):
            filename, size_inst, dens_inst, sol = info_inst[inst]
            nb_subproblems = cs.cut_select_algo(filename, dim, sel_size, strat=5, all_comp=True, nb_rounds_cuts=0)[-1]
            row_string = filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(nb_subproblems)
            all_gaps, all_times, all_sep_times, all_nb_cuts = [], [], [], []
            for selectType in [2, 1, 4, 0, 5]:
                if selectType != 5:  # all non-random cut selections strategies
                    (time_overall, nb_subproblems, obj_values, round_times, sep_times, nbs_sdp_cuts) = \
                        cs.cut_select_algo(filename, dim, sel_size,
                                           strat=selectType, triangle_on=False, nb_rounds_cuts=cut_rounds,
                                           all_comp=True)
                else:  # random cut selection
                    np.random.seed(7)
                    time_overall, obj_values, round_times, sep_times, nbs_sdp_cuts = \
                        [0] * r_runs, [0] * r_runs, [0] * r_runs, [0] * r_runs, [0] * r_runs
                    for rr in range(r_runs):
                        (time_overall[rr], _, obj_values[rr], round_times[rr], sep_times[rr], nbs_sdp_cuts[rr]) = \
                            cs.cut_select_algo(filename, dim, sel_size,
                                               strat=selectType, triangle_on=False, nb_rounds_cuts=cut_rounds,
                                               all_comp=True)
                    time_overall = sum(time_overall) / r_runs
                    obj_values = [sum(i) / r_runs for i in zip(*obj_values)]
                    round_times = [sum(i) / r_runs for i in zip(*round_times)]
                    sep_times = [sum(i) / r_runs for i in zip(*sep_times)]
                    nbs_sdp_cuts = [sum(i) / r_runs for i in zip(*nbs_sdp_cuts)]
                gap_closed_percents = [0] * len(obj_values)
                for idx in range(1, len(obj_values)):
                    gap_closed_percents[idx] = (obj_values[idx] - obj_values[0]) / (sol - obj_values[0])
                all_gaps.append(gap_closed_percents)
                all_times.append(round_times)
                all_sep_times.append(sep_times)
                all_nb_cuts.append(nbs_sdp_cuts)
                if selectType != 0:
                    print(filename + " - M+S^E_3, " + str(sel_size) + " ," + dict_sel[selectType] + ", time:" + str(
                        time_overall))
                else:
                    print(filename + " - M+S, " + str(sel_size) + " , dense cuts, time:" + str(
                        time_overall))
            with open(save_file, "a") as f:
                for arr in [all_gaps, all_times, all_sep_times, all_nb_cuts]:
                    for r in range(1, cut_rounds + 1):
                        for sel in range(0, 5):
                            row_string += "," + str(arr[sel][r])
                f.write(row_string + "\n")
    ###########################################
    # Data for M+tri (Tables 5, 9 and Figures 13, 14)
    ###########################################
    if table in [0, 5, 9, 13, 14]:
        sel_size = 0.1
        # Do not add any SDP cuts, separate only triangle
        save_file = os.path.join(dirname, "data_(M+tri)_" + str(sel_size) + "_20.csv")
        with open(save_file, write_flag) as f:
            f.write("M+tri separated with selection size " + str(sel_size) + " and up to 20 cut rounds\n")
            f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts,nb_tri_cuts,"
                    "nb_total_cuts, time_total\n")
        for inst in range(len(problems)):
            filename, size_inst, dens_inst, sol = info_inst[inst]
            (obj_values, time_overall, _, _, _, nb_tri_cuts, nb_subproblems) = \
                cs.solve_mccormick_and_tri(filename, sel_size, term_on=True)
            percent_gap_closed = (obj_values[-1] - obj_values[0]) / (sol - obj_values[0])
            with open(save_file, "a") as f:
                f.write(filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + "," +
                        str(nb_subproblems) + "," + str(0) + "," + str(sum(nb_tri_cuts)) +
                        "," + str(sum(nb_tri_cuts)) + "," + str(time_overall) + "\n")
            print("- M+tri, " + str(sel_size) + ", only triangle cuts: " + filename + ", time:" + str(
                time_overall))
    ###########################################
    #  Data for M+S^E_3 with 5%, 10% selection size (Table 5, 8)
    ###########################################
    if table in [0, 5, 8]:
        dim = 3
        sel_sizes = [0.1] if table == 8 else [0.05, 0.1]
        # selection sizes 5%, 10%
        for sel_size in sel_sizes:
            # for optimality selection via neural nets (2) and feasibility selection (1)
            for selectType in [2, 1]:
                save_file = os.path.join(dirname,
                            "data_(M+S^E_" + str(dim) + ")_" + dict_sel[selectType] + "_" + str(sel_size) + "_20_t.csv")
                with open(save_file, write_flag) as f:
                    f.write(dict_sel2[selectType] + " for M+S^E_3 with selection size " + str(sel_size) + " and up 20 cut rounds \n")
                    f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts\n")
                for inst in range(len(problems)):
                    filename, size_inst, dens_inst, sol = info_inst[inst]
                    (obj_values, time_overall, _, _, nb_sdp_cuts, _, nb_subproblems) = \
                        cs.cut_select_algo(filename, dim, sel_size, strat=selectType, triangle_on=False, term_on=True)
                    percent_gap_closed = (obj_values[-1] - obj_values[0]) / (sol - obj_values[0])
                    with open(save_file, "a") as f:
                        f.write(
                            filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + "," +
                            str(nb_subproblems) + "," + str(sum(nb_sdp_cuts)) + "\n")
                    print("- M+S^E_3, " + str(sel_size) + " ," + dict_sel[selectType] + ": " + filename + ", time:" + str(
                        time_overall))
    ###########################################
    #  Data for (naive) M+triangle+S^E_3 with 10% selection size (Table 5, 9; Figure 13, 14)
    ###########################################
    if table in [0, 5, 9, 13, 14]:
        dim = 3
        sel_size = 0.1
        select_types = [2] if table in [9, 13, 14] else [2, 1]
        # for optimality selection via neural nets (2) and feasibility selection (1)
        for selectType in select_types:
            save_file = os.path.join(dirname,
                                "data_(M+tri+S^E_" + str(dim) + ")_" + dict_sel[selectType] + "_" + str(sel_size) + "_20.csv")
            with open(save_file, write_flag) as f:
                f.write(dict_sel2[selectType] + " for M+triangle+S^E_3 with selection size " + str(sel_size) + "\n")
                f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts,nb_tri_cuts,"
                        "nb_total_cuts, time_total\n")
            for inst in range(len(problems)):
                filename, size_inst, dens_inst, sol = info_inst[inst]
                (obj_values, time_overall, _, _, nb_sdp_cuts, nb_tri_cuts, nb_subproblems) = \
                    cs.cut_select_algo(filename, dim, sel_size, strat=selectType, triangle_on=True, term_on=True)
                percent_gap_closed = (obj_values[-1] - obj_values[0]) / (sol - obj_values[0])
                with open(save_file, "a") as f:
                    f.write(filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + "," +
                            str(nb_subproblems) + "," + str(sum(nb_sdp_cuts)) + "," + str(sum(nb_tri_cuts)) +
                            "," + str(sum(nb_sdp_cuts) + sum(nb_tri_cuts)) + "," + str(time_overall) + "\n")
                print("- M+tri+S^E_3, " + str(sel_size) + " ," + dict_sel[selectType] + ": " + filename + ", time:" + str(
                    time_overall))
    ###########################################
    #  Data for chordal extensions M+P^(bar(E))_3, M+bar(S(P_3*)) (Table 8)
    ###########################################
    if table in [0, 8]:
        sel_size = 0.1
        dim = 3
        # chordal extensions M+P^(bar(E))_3, M+S(bar(P*_3))
        dict_ch = {1: "S^(bar(E))_3", 2: "barSP3star"}
        for chordalExt in [1, 2]:
            # for optimality selection via neural nets (2) and feasibility selection (1)
            selections = [1, 2] if chordalExt == 1 else [2]
            for selectType in selections:
                save_file = os.path.join(dirname, "data_(M+" + dict_ch[chordalExt] + ")_" + dict_sel[selectType] + \
                                         "_" + str(sel_size) + "_" + str(20) + ".csv")
                with open(save_file, write_flag) as f:
                    f.write(dict_sel2[selectType] + " for M+" + dict_ch[chordalExt]
                            + " with selection size " + str(sel_size) + " and up to 20 cut rounds" + "\n")
                    if chordalExt == 1:
                        f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts\n")
                    else:
                        f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts,size_P3Eplus\n")
                    
                for inst in range(len(problems)):
                    filename, size_inst, dens_inst, sol = info_inst[inst]
                    (obj_values, time_overall, _, _, nb_sdp_cuts, _, nb_subproblems) = \
                        cs.cut_select_algo(filename, dim, sel_size, strat=selectType, triangle_on=False, term_on=True,
                                           ch_ext=chordalExt)
                    percent_gap_closed = (obj_values[-1] - obj_values[0]) / (sol - obj_values[0])
                    with open(save_file, "a") as f:
                        if chordalExt == 1:
                            f.write(filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + ","
                                    + str(nb_subproblems) + "," + str(sum(nb_sdp_cuts)) + "\n")
                        else:
                            size_P3Eplus = size_inst*(size_inst-1)*(size_inst-2) // 6
                            f.write(filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + ","
                                    + str(nb_subproblems) + "," + str(sum(nb_sdp_cuts)) + "," + str(size_P3Eplus) +"\n")
                    print("- M+" + dict_ch[chordalExt] + ", " + str(sel_size) + " ," + dict_sel[selectType] +
                          ": " + filename + ", time:" + str(time_overall))
    ###########################################
    #  Data for (heuristic) M+triangle+S^E_3-5 with 10% selection size (Table 9; Figure 13, 14)
    #   Apply heuristic:
    #   - M+triangle+S^E_5 for low and medium density      (combined selection)
    #   - M+triangle+S^E_4 for high density, <jumbo size   (optimality selection)
    #   - M+triangle+S^E_3 for high density, jumbo size    (optimality selection)
    ###########################################
    if table in [0, 9, 13, 14]:
        sel_size = 0.1
        save_file = os.path.join(dirname, "data_heur_(M+tri+S^E_3-5)_" + str(sel_size) + "_" + str(20) + ".csv")
        with open(save_file, write_flag) as f:
            f.write("Cut selection heuristic for M+triangle+S^E_3-5 with selection size " + str(sel_size) + \
                    " and up to 20 cut rounds" + "\n")
            f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts,"
                    "nb_tri_cuts,nb_total_cuts,time_total\n")
        for inst in range(len(problems)):
            filename, size_inst, dens_inst, sol = info_inst[inst]
            if dens_inst <= 60:  # low and medium dense, combined selection
                sol_info = cs.cut_select_algo(filename, 5, sel_size, strat=4, term_on=True, triangle_on=True)
            elif size_inst < 100:  # until jumbo size for dense, optimality selection
                sol_info = cs.cut_select_algo(filename, 4, sel_size, strat=2, term_on=True, triangle_on=True)
            else:  # jumbo dense, optimality selection
                sol_info = cs.cut_select_algo(filename, 3, sel_size, strat=2, term_on=True, triangle_on=True)
            (obj_values, time_overall, _, _, nb_sdp_cuts, nb_tri_cuts, nb_subproblems) = sol_info
            percent_gap_closed = (obj_values[-1] - obj_values[0]) / (sol - obj_values[0])
            with open(save_file, "a") as f:
                f.write(filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + "," + str(
                    nb_subproblems) +
                        "," + str(sum(nb_sdp_cuts)) + "," + str(sum(nb_tri_cuts)) +
                        "," + str(sum(nb_sdp_cuts) + sum(nb_tri_cuts)) + "," + str(time_overall) + "\n")
            print("- heur M+S^E_3-5, " + str(sel_size) + ": " + filename + ", time:" + str(
                time_overall))
    ###########################################
    # Data for M+S^E_3, M+S^E_4, M+S^E_5 with 10% selection size and 40 cut rounds (Table 7)
    ##########################################
    if table in [0, 7]:
        sel_size = 0.1
        # Number of cut rounds for table 7
        cuts_rounds = 40
        # decompositions M+S^E_3, M+S^E_4, M+S^E_5
        for dim in [3, 4, 5]:
            # for optimality selection via neural nets (2) and feasibility selection (1)
            for selectType in [2, 1]:
                save_file = os.path.join(dirname, "data_(M+S^E_" + str(dim) + ")_" + dict_sel[selectType] +
                                         "_" + str(sel_size) + "_" + str(cuts_rounds) + ".csv")
                with open(save_file, write_flag) as f:
                    f.write(
                        dict_sel2[selectType] + " for M+S^E_" + str(dim) + " with selection size " + str(sel_size) +
                        " and up to " + str(cuts_rounds) + " cut rounds \n")
                    f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts\n")
                for inst in range(len(problems)):
                    filename, size_inst, dens_inst, sol = info_inst[inst]
                    # Number of cut rounds for table 5 (40) unless jumbo high density category for M+S^E_5 where
                    # there are instances we run out of memory for, so don't do cuts rounds for that category)
                    cut_rounds = 0 if (dim == 5 and size_inst >= 100 and dens_inst >= 75) else cuts_rounds
                    (obj_values, time_overall, _, _, nb_sdp_cuts, _, nb_subproblems) = \
                        cs.cut_select_algo(filename, dim, sel_size, strat=selectType, triangle_on=False,
                                           term_on=True, nb_rounds_cuts=cut_rounds)
                    percent_gap_closed = (obj_values[-1] - obj_values[0]) / (sol - obj_values[0])
                    with open(save_file, "a") as f:
                        f.write(filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(
                            percent_gap_closed) + ","
                                + str(nb_subproblems) + "," + str(sum(nb_sdp_cuts)) + "\n")
                    print("- M+S^E_" + str(dim) + ", " + str(sel_size) + " ," + dict_sel[
                        selectType] + ": " + filename + ", time:" + str(
                        time_overall))
    ###########################################
    #  Data for all QCQP instances comparing combined to feasibility selections
    ###########################################
    if table in [0, 16]:
        sel_size = 0.05
        cut_rounds = 4
        save_file = os.path.join(dirname, "data_all_qcqp_"+str(cut_rounds)+"rounds" + ".csv")
        strats_names = ["feas", "opt", "opt exact", "combined", "random"]
        with open(save_file, write_flag) as f:
            f.write(
                "M+S^E_3 with selection size " + str(sel_size) + " comparison "+\
                "in terms of gap closed - nb of sdp cuts added at cut rounds - "+\
                "nb of sdp cuts added based on the optimality measure between selection strategies"+\
                "comb - feas\n")
            white_space = ",".join("_" for _ in range(cut_rounds*2))
            f.write(",,,,gap_closed_at_round" + white_space + \
                    ",nb_sdp_cuts_at_round" + white_space + \
                    ",nb_opt_cuts_at_round" + white_space + "\n")
            f.write("filename,size,density,nb_constraints," +
                    ",".join(
                        ",".join(
                            ",".join("r"+str(r+1)+"_"+ sel for sel in ["feas","comb"])
                            for r in range(cut_rounds))
                        for _ in range(3)) + "\n")
        # CONOPT solutions obtained via GAMS
        with open(os.path.join(os.path.curdir, 'qcqp_instances', 'qcqp_sols.txt'), "r") as f:
            sols = [(inst.split(',')[0], float(inst.split(',')[1])) for inst in f.read().split('\n')]
        for (filename, sol) in sols:
            all_gaps, all_nb_sdp_cuts, all_nb_opt_cuts = [], [], []
            size_inst = filename.split('_')
            cons_inst = int(size_inst[2])
            dens_inst = int(size_inst[3])
            size_inst = int(size_inst[1])
            sol = [sol for (file, sol) in sols if file == filename][0]
            row_string = filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(cons_inst)
            for strat in [1, 4]:
                print("- " + filename + ", " + str(sel_size) + "cuts, " + strats_names[strat - 1] + " sel ... ")
                objectives, max_cut_sel, nbs_sdp_cuts, nbs_opt_cuts = \
                    cs_qcp.cut_select_algo(filename, 3, sel_size=sel_size, strat=strat, nb_rounds_cuts=cut_rounds)
                gap_closed_percents = [0] * len(objectives)
                for idx in range(1, len(objectives)):
                    if abs(sol - objectives[0]) < 0.01:
                        gap_closed_percents[idx] = 1
                    else:
                        gap_closed_percents[idx] = (objectives[idx] - objectives[0]) / (sol - objectives[0])
                all_gaps.append(gap_closed_percents)
                all_nb_sdp_cuts.append(nbs_sdp_cuts)
                all_nb_opt_cuts.append(nbs_opt_cuts)
            with open(save_file, "a") as f:
                for arr in [all_gaps, all_nb_sdp_cuts, all_nb_opt_cuts]:
                    for r in range(1, cut_rounds + 1):
                        for sel in range(0, 2):
                            row_string += "," + str(arr[sel][r])
                f.write(row_string + "\n")

    # Table = 0 means aggregate for all tables
    if table == 0:
        aggregate_table(6, folder=folder_name, nb_header_lines=3)
        for table_nb in [5, 7, 8, 9, 13, 14]:
                aggregate_table(table_nb, folder=folder_name, nb_header_lines=2)
    elif table in [6]:
        aggregate_table(table, folder=folder_name, nb_header_lines=3)
    elif table not in [2, 16]:
        aggregate_table(table, folder=folder_name, nb_header_lines=2)


def aggregate_table(table, folder="data_tables", fig_folder="data_figures", nb_header_lines=1):
    """ Aggregates BoxQP data for a table in Table 3 categories
    :param table: table to aggregate
    :param folder: where to find raw .csv data and save table .csv data file
    :param fig_folder:  where to find raw .csv data and save figure .csv data file (Figures 13-14)
    :return: .csv data file
    """
    assert (table in [5, 6, 7, 8, 9, 13, 14]), "Please select a valid table (5-9) or figure to aggregate (13-14)"
    pr_categories = [
        "Small,Low,",   ",Medium,", ",High,",
        "Medium,Low,",  ",Medium,", ",High,",
        "Large,Low,",   ",Medium,", ",High,",
        "Jumbo,Low,",   ",Medium,", ",High,"]
    write_flag = "w"
    save_file = os.path.join(os.path.curdir, folder, "table" + str(table) + ".csv")
    hl = nb_header_lines
    if table in [13, 14]:
        save_file = os.path.join(os.path.curdir, fig_folder, "fig" + str(table) + "_data.csv")

    if table == 5:
        table_data = np.column_stack((
            aggregate_column(folder, "data_(M+S^E_3)_opt_0.05_20", 1, hl),
            aggregate_column(folder, "data_(M+S^E_3)_feas_0.05_20", 1, hl),
            aggregate_column(folder, "data_(M+S^E_3)_feas_0.05_20", 1, hl) - aggregate_column(folder, "data_(M+S^E_3)_opt_0.05_20", 1, hl),
            aggregate_column(folder, "data_(M+S^E_3)_opt_0.1_20", 1, hl),
            aggregate_column(folder, "data_(M+S^E_3)_feas_0.1_20", 1, hl),
            aggregate_column(folder, "data_(M+S^E_3)_feas_0.1_20", 1, hl) - aggregate_column(folder, "data_(M+S^E_3)_opt_0.1_20", 1, hl),
            aggregate_column(folder, "data_(M+tri)_0.1_20", 1, hl),
            aggregate_column(folder, "data_(M+tri+S^E_3)_opt_0.1_20", 1, hl),
            aggregate_column(folder, "data_(M+tri+S^E_3)_feas_0.1_20", 1, hl),
            aggregate_column(folder, "data_(M+tri+S^E_3)_feas_0.1_20", 1, hl) - aggregate_column(folder, "data_(M+tri+S^E_3)_opt_0.1_20", 1, hl)))
    elif table == 6:
        nb_cut_rounds = 4
        nb_strats = 5
        nb_cols = nb_cut_rounds*nb_strats
        table_data = np.column_stack((
                [[" &" + str(round(el, 1)) for el in aggregate_column(folder, "data_all_boxqp_4rounds", col, hl)] for
                 col in range(2, nb_cols + 2)] + \
                [[" &" + str(round(el[0], 2)) + " (" + str(int(round(el[1] / el[0] * 100, 0))) + "\%)" for el in
                  zip(aggregate_column(folder, "data_all_boxqp_4rounds", col, hl, nb_types=3),
                      aggregate_column(folder, "data_all_boxqp_4rounds", col + nb_cols, hl, nb_types=3))] for col in
                 range(nb_cols + 2, nb_cols*2 + 2)]
        ))
    elif table == 7:
        table_data = np.column_stack((
            aggregate_column(folder, "data_(M+S^E_3)_opt_0.1_40", 1, hl),
            aggregate_column(folder, "data_(M+S^E_3)_feas_0.1_40", 1, hl),
            aggregate_column(folder, "data_(M+S^E_4)_opt_0.1_40", 1, hl),
            aggregate_column(folder, "data_(M+S^E_4)_feas_0.1_40", 1, hl),
            aggregate_column(folder, "data_(M+S^E_5)_opt_0.1_40", 1, hl),
            aggregate_column(folder, "data_(M+S^E_5)_feas_0.1_40", 1, hl),
            # columns with number of sub-problems
            aggregate_column(folder, "data_(M+S^E_3)_opt_0.1_40", 2, hl, nb_types=1),
            aggregate_column(folder, "data_(M+S^E_4)_opt_0.1_40", 2, hl, nb_types=1),
            aggregate_column(folder, "data_(M+S^E_5)_opt_0.1_40", 2, hl, nb_types=1)))
    elif table == 8:
        table_data = np.column_stack((
            aggregate_column(folder, "data_(M+S^E_3)_opt_0.1_20", 1, hl),
            aggregate_column(folder, "data_(M+S^E_3)_feas_0.1_20", 1, hl),
            aggregate_column(folder, "data_(M+S^(bar(E))_3)_opt_0.1_20", 1, hl),
            aggregate_column(folder, "data_(M+S^(bar(E))_3)_feas_0.1_20", 1, hl),
            aggregate_column(folder, "data_(M+barSP3star)_opt_0.1_20", 1, hl),
            # columns with number of sub-problems
            aggregate_column(folder, "data_(M+S^E_3)_opt_0.1_20", 2, hl, nb_types=1),
            aggregate_column(folder, "data_(M+S^(bar(E))_3)_opt_0.1_20", 2, hl, nb_types=1),
            aggregate_column(folder, "data_(M+barSP3star)_opt_0.1_20", 2, hl, nb_types=1),
            aggregate_column(folder, "data_(M+barSP3star)_opt_0.1_20", 4, hl, nb_types=1)
        ))
    elif table == 9:
        data_known = np.array([
            # S,    S>,    M+S,   BGL
            [80.65, 99.11, 99.29, 99.51],
            [89.79, 99.40, 99.46, 99.29],
            [94.15, 99.76, 99.80, 99.13],
            [85.85, 99.33, 99.55, 99.90],
            [93.00, 98.77, 98.86, 98.01],
            [95.68, 99.24, 99.31, 93.52],
            [88.61, 98.20, 98.65, 98.28],
            [94.96, 99.05, 99.25, 97.48],
            [96.34, 99.14, 99.29, 90.60],
            [92.90, 98.35, 98.84, 96.28],
            [95.25, 98.60, 98.82, 91.42],
            [96.67, 98.96, 99.16, 85.68]])
        all_gap_closed = np.array([100]*len(data_known[:, 3]))		
        table_data = np.column_stack((
            data_known,
            # M+tri linear base of inequalities
            aggregate_column(folder, "data_(M+tri)_0.1_20", 1, hl),
            # naive
            aggregate_column(folder, "data_(M+tri+S^E_3)_opt_0.1_20", 1, hl),
            np.round((aggregate_column(folder, "data_(M+tri+S^E_3)_opt_0.1_20", 1, hl) - data_known[:, 3])/ 
			(all_gap_closed - data_known[:, 3])*100,2),
            np.round((aggregate_column(folder, "data_(M+tri+S^E_3)_opt_0.1_20", 1, hl) - aggregate_column(folder, "data_(M+tri)_0.1_20", 1, hl))/ 
			(all_gap_closed - aggregate_column(folder, "data_(M+tri)_0.1_20", 1, hl))*100,2),
            # heuristic
            aggregate_column(folder, "data_heur_(M+tri+S^E_3-5)_0.1_20", 1, hl),
			np.round((aggregate_column(folder, "data_heur_(M+tri+S^E_3-5)_0.1_20", 1, hl) - data_known[:, 3])/ 
			(all_gap_closed - data_known[:, 3])*100,2),
            np.round((aggregate_column(folder,"data_heur_(M+tri+S^E_3-5)_0.1_20", 1, hl) - aggregate_column(folder, "data_(M+tri)_0.1_20", 1, hl))/ 
			(all_gap_closed - aggregate_column(folder, "data_(M+tri)_0.1_20", 1, hl))*100,2)
		))
    elif table == 13:
        # Number of cuts per category of BoxQP problems for BGL
        # ("Globally solving nonconvex quadratic programming problems with box constraints via integer
        # programming methods", P. Bonami, O. Gunluk, J. Linderoth)
        bgl_nb_cuts = np.array([790.37, 2368.78, 4115.55, 2454.53, 15012.26, 71558.93, 11807.31,
                                54733.33, 144118.66, 37858.37, 165480.88, 354370.41])
        table_data = np.column_stack((
            aggregate_column(folder, "data_(M+tri)_0.1_20", 5, hl, nb_types=2),              # M+tri linear base of ineq.
            aggregate_column(folder, "data_(M+tri+S^E_3)_opt_0.1_20", 5, hl, nb_types=2),       # naive
            aggregate_column(folder, "data_heur_(M+tri+S^E_3-5)_0.1_20", 5, hl, nb_types=2),    # heuristic
            bgl_nb_cuts.T))
        # default value for missing entries
        table_data[table_data == 1] = 100
    elif table == 14:
        table_data = np.column_stack((
            aggregate_column(folder, "data_(M+tri)_0.1_20", 6, hl, nb_types=2),              # M+tri linear base of ineq.
            aggregate_column(folder, "data_(M+tri+S^E_3)_opt_0.1_20", 6, hl, nb_types=2),       # naive
            aggregate_column(folder, "data_heur_(M+tri+S^E_3-5)_0.1_20", 6, hl, nb_types=2)    # heuristic
        ))
        # default value for missing entries
        table_data[table_data == 1] = 0.1

    header_dict = {
        5: "opt_5_M+S^E_3,feas_5_M+S^E_3,diff_5_M+S^E_3,opt_10_M+S^E_3,feas_10_M+S^E_3,diff_10_M+S^E_3"
           + ",M+tri,opt_M+S^E_3+tri,feas_M+S^E_3+tri,diff_feas_opt\n",
        6: ",".join(",".join(",".join("r"+str(r+1)+"_"+ sel for sel in ["opt","feas","comb","dense","rand"]) for r in range(4)) for _ in range(2))+"\n",
        7: "opt_M+S^E_3,feas_M+S^E_3,opt_M+S^E_4,feas_M+S^E_4,opt_M+S^E_5,feas_M+S^E_5,|P^E_3|,|P^E_4|,|P^E_5|\n",
        8: "opt_M+S^E_3,feas_M+S^E_3,opt_M+S^bar(E)_3,feas_M+S^bar(E)_3,opt_M+S(P*_3)_3,|P^E_3|,|P^bar(E)_3|,"
           + "|bar(P^*_3)|,|P^E+_3|\n",
        9: "S,S>,M+S,BGL,M+tri,M+tri+S^E_3,diff_BGL,diff_M+tri,M+tri+S^E_3-5,diff_BGL,diff_M+tri\n",
        13: "BGL_nb_cuts,M+tri_nb_cuts,M+tri+S^E_3_nb_cuts,M+tri+S^E_3-5_nb_cuts\n",
        14: "BGL_time(s),M+tri_time(s),M+tri+S^E_3_time(s),M+tri+S^E_3-5_time(s)\n"
    }
    with open(save_file, write_flag) as f:
        f.write("size,density,"+header_dict[table])
        for line in range(len(pr_categories)):
            f.write(pr_categories[line] + ",".join(str(x) for x in table_data[line, :]) + "\n")


def aggregate_column(folder, filename, column_to_agg, nb_header_lines, nb_types=0):
    """Aggregates BoxQP data for a table column in Table 3 categories
    :param folder: where to find raw .csv data
    :param filename: name of .csv file with raw data
    :param column_to_agg: column number to aggregate from raw data .csv file
    :param nb_types: type of aggregated numbers needed (for formatting)
    :return: aggregated column of formatted values
    """
    dirname = os.path.join(os.path.curdir, folder, filename + ".csv")
    with open(dirname) as f:
        content = f.readlines()
        content = content[nb_header_lines:]
    # Get size, density and value from .csv data file for each instance
    for idx, line in enumerate(content):
        line = line.strip('\n').split(',')[1:]
        try:
            value_to_agg = int(line[1 + column_to_agg])
        except ValueError:
            value_to_agg = float(line[1 + column_to_agg])
        content[idx] = [int(line[0]), int(line[1]), value_to_agg]
    agg_arr = [[0, 0] for _ in range(12)]
    # Find size and density categories according to Table 3 in the manuscript
    for pr in content:
        if pr[0] <= 40:
            pr_size = 0
        elif pr[0] <= 70:
            pr_size = 1
        elif pr[0] <= 90:
            pr_size = 2
        else:
            pr_size = 3
        if pr[1] <= 40:
            pr_dens = 0
        elif pr[1] <= 60:
            pr_dens = 1
        else:
            pr_dens = 2
        pr_category = pr_size * 3 + pr_dens
        if nb_types == 3:
            agg_arr[pr_category][0] += pr[2]    # summing for arithmetic avg for times
        else:
            agg_arr[pr_category][0] += np.log(pr[2])    # summing logs for geometric average for gaps
        agg_arr[pr_category][1] += 1
    # Format value according to type of value being aggregated
    for idx, elem in enumerate(agg_arr):
        if nb_types == 3:
            nb_agg = elem[0] / elem[1] if elem[1] != 0 else 100
        else:
            nb_agg = np.exp(elem[0] / max(elem[1], 1))  # exp to get geometric average
        if nb_types == 0:  # if percentage of gap closed <1
            agg_arr[idx] = np.round(nb_agg * 100, 2)
        elif nb_types == 1:  # if number of cuts added
            agg_arr[idx] = np.round(nb_agg, 0)
        elif nb_types == 2:  # if figures 13-14
            agg_arr[idx] = np.round(nb_agg, 2)
        elif nb_types == 3:
            agg_arr[idx] = nb_agg    # no rounding for times
    return np.array(agg_arr)


def reset_file(save_file, write_flag):
    """Reset savefile if write_flag set to write
    """
    if write_flag == 'w':
        with open(save_file, write_flag):
            pass


if __name__ == '__main__':
    main()

