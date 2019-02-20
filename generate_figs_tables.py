import utilities as utils
from cut_select_qp import CutSolver
from cut_select_powerflow import CutSolverQCQP
import numpy as np
import os


def main():
    # Lighter test configuration (omitting jumbo/large instances that take very long to test on for specific settings)
    test_cfg = True

    # All figures and tables
    run_everything(folder_tables="data_tables_v1", folder_figures="data_figures", test_cfg=test_cfg)

    # All figures
    # run_all_figures(folder_name="data_figures", test_cfg=test_cfg)

    # An individual figure
    # figure_nb = 4
    # run_for_figure(figure_nb, folder_name="data_figures_1", test_cfg=test_cfg)

    # All tables
    # run_for_all_tables(table=0, folder_name="data_tables", test_cfg=test_cfg)

    # An individual table
    # table_nb = 4
    # run_for_all_tables(table=table_nb, folder_name="data_tables", test_cfg=test_cfg)


def run_everything(folder_tables="data_tables", folder_figures="data_figures", test_cfg=True):
    run_all_figures(folder_name=folder_figures, test_cfg=test_cfg)
    run_for_all_tables(table=0, folder_name=folder_tables, test_cfg=test_cfg)


def run_all_figures(folder_name="data_figures", test_cfg=True):
    for figure_nb in [1, 4, 6, 7, 8, 9, 10, 14]:
        run_for_figure(figure_nb, folder_name=folder_name, test_cfg=test_cfg)


def run_for_figure(figure, folder_name="data_figures", test_cfg=True, write_flag='w'):
    """Run on all BoxQP instances (or except the very large ones for test_cfg=True) to construct the tables 4-7 and
    figures 12-13 in the manuscript
    :param figure: for which figure to run (1, 4-10, 14)
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
    cs_qcqp = CutSolverQCQP()
    dirname = os.path.join(os.path.curdir, folder_name)
    # Random seed used in all figures
    seed_nb = 7
    ###########################################
    # Figure 1 analysing M+S_3 bounds on random instances of different size
    ###########################################
    # takes a very long time (~10h) to run to size of instances 65, consider stop at 30 for testing
    if figure == 1:
        save_file = os.path.join(dirname, "fig1_bounds_3D.csv")
        reset_file(save_file, write_flag)
        if test_cfg:
            utils.get_average_bounds_3d(save_file, start=5, stop=30, step=5, nb_instances=30, rand_seed=seed_nb)
        else:
            utils.get_average_bounds_3d(save_file, start=5, stop=65, step=5, nb_instances=30, rand_seed=seed_nb)
    ###########################################
    # Figure 4 looking at the SDP relaxation surface for a 2D example
    ###########################################
    # very quick to run (~10s)
    elif figure == 4:
        save_file = os.path.join(dirname, "fig4_SDP_2D_points.csv")
        reset_file(save_file, write_flag)
        utils.gen_sdp_surface_2d_fig4(save_file)
    ###########################################
    # Figure 5 looking at distribution of data sampled for neural nets
    ###########################################
    # Can also be run to generate training/test sets for neural nets, so using Figure 7 data
    # ok to run (10-20 mins) for 10k samples each
    # Will match the existing file "GenDataTest3D.csv" with 500k entries in "neural_nets" (same seed)
    elif figure == 5:
        save_file = os.path.join(os.path.curdir, "neural_nets", "GenDataTest3D_fig7.csv")
        reset_file(save_file, write_flag)
        utils.gen_data_ndim(10000, 3, save_file, rand_seed=seed_nb)
    ###########################################
    # Figure 6 for uniform Q's distribution of eigenvalues
    ###########################################
    # ok to run (~30s)
    elif figure == 6:
        save_file = os.path.join(dirname, "fig6_data_randomQs.csv")
        reset_file(save_file, write_flag)
        utils.gen_data_3d_q(500000, save_file, rand_seed=seed_nb)
    ###########################################
    # Figure 7 plot for other neural net test data
    ###########################################
    # ok to run (10-20 mins) for 10k samples each
    # Will match the existing files with 500k entries in "neural_nets" (same seed)
    elif figure == 7:
        for nn_size in [4]:
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
        # The M+S_3 theoretical bound can be obtained (approximately)
        # by running feasibility selection with a large % of cuts and many rounds
        sol = cs.cut_select_algo("spar020-100-1", 3, 100, strat=1, nb_rounds_cuts=40)[0][-1]
        (*objPercs, roundsStats, roundsAllCuts) = \
            cs.cut_select_algo("spar020-100-1", 3, 100, strat=-1, nb_rounds_cuts=cut_rounds, sol=sol, plots=True)
        with open(save_file, "w") as f:
            f.write("cuts_round,gap_closed,percent_same_sel\n")
            for cutRound in range(1, cut_rounds + 1):
                f.write(str(cutRound) + "," + str(objPercs[cutRound]) + "," + str(roundsStats[cutRound - 1]) + "\n")
            f.write("cuts_round,cut_number,sel_estim,sel_exact,estim_measure,exact_measure\n")
            for cut in roundsAllCuts:
                f.write(",".join(str(x) for x in cut) + "\n")
    ###########################################
    # Figure 9 plots on optimality vs feasibility vs random
    ###########################################
    # reasonable run-time for "spar040-030-1", "spar100-025-1" (~15mins) but slow for "spar040-100-1", "spar100-075-1"
    #   - consider removing it or wait a very long time (~5h) esp. for "spar100-075-1"
    elif figure == 9:
        save_file = os.path.join(dirname, "fig9_data.csv")
        reset_file(save_file, write_flag)
        # M+S solutions from Table 11 in "Globally solving nonconvex quadratic programming problems
        # with box constraints via integer programming methods", P. Bonami, O. Gunluk, J. Linderoth
        sols_ms = [839.50, 2476.38, 4066.38, 7514.48]
        instances = list(zip(sols_ms, ["spar040-030-1", "spar040-100-1", "spar100-025-1", "spar100-075-1"]))
        if test_cfg:
            instances = [instances[0], instances[2]]
        dim = 3  # dimension 3
        top = 0.05  # top 5% of sub-problems/cuts selected
        cut_rounds = 20
        for sol_ms, filename in instances:
            sols = []
            # run optimality selection with neural net estimator
            print("- " + filename + " optimality NN sel ... ")
            sols.append(
                cs.cut_select_algo(filename, dim, top, strat=2, plots=True, sol=sol_ms, nb_rounds_cuts=20)[
                0:cut_rounds + 1])
            # run feasibility selection
            print("- " + filename + " feasibility sel ... ")
            sols.append(
                cs.cut_select_algo(filename, dim, top, strat=1, plots=True, sol=sol_ms, nb_rounds_cuts=20)[
                0:cut_rounds + 1])
            # run random selection
            print("- " + filename + " random sel ... ")
            np.random.seed(seed_nb)
            for i in range(10):
                sols.append(
                    cs.cut_select_algo(filename, dim, top, strat=5, plots=True, sol=sol_ms, nb_rounds_cuts=20)[
                    0:cut_rounds + 1])
            # run optimality selection with exact SDP solutions (Mosek)
            print("- " + filename + " optimality exact sel ... ")
            sols.append(
                cs.cut_select_algo(filename, dim, top, strat=3, plots=True, sol=sol_ms, nb_rounds_cuts=20)[
                0:cut_rounds + 1])
            # The M+S_3 theoretical bound can be obtained (approximately)
            # by running feasibility selection with a large % of cuts and many rounds
            print("- " + filename + " finding M+S_3 bound ... ")
            bound_run = cs.cut_select_algo(filename, dim, top, strat=1, plots=True, sol=sol_ms, nb_rounds_cuts=40)
            bound = bound_run[0:40 + 1][-1]
            max_cuts = np.floor(bound_run[-1] * top)
            sols.append([bound] * (cut_rounds + 1))
            sols = np.array(sols).T
            with open(save_file, "a") as f:
                f.write(filename + "\n")
                f.write("size,density,max_cuts_sel\n")
                size_inst = filename.split('-')
                dens_inst = int(size_inst[1])
                size_inst = int(size_inst[0].split('r')[1])
                f.write(str(size_inst) + "," + str(dens_inst) + "," + str(max_cuts) + "\n")
                f.write("opt_nn,feas," + ",".join(str(x) for x in ["rand " + str(i) for i in range(10)])
                        + ",opt_exact,M+S_3 \n")
                for line in sols:
                    f.write(",".join(str(x) for x in line) + "\n")
    ###########################################
    # Figure 10, 11 - plots on strong vs valid cuts and combined strategy
    ###########################################
    # reasonable run-time (~15mins)
    elif figure == 10:
        save_file = os.path.join(dirname, "fig10-11_data.csv")
        reset_file(save_file, write_flag)
        filename = "spar100-025-1"
        # M+S solution for "spar100-025-1" from Table 11 in "Globally solving nonconvex quadratic programming problems
        # with box constraints via integer programming methods", P. Bonami, O. Gunluk, J. Linderoth
        sols_ms = 4066.38
        strong_only = True  # select strong-only (diff>0) valid cuts
        dim = 3
        top = 0.05
        cuts_rounds = 20
        sols = []
        # run combined selection (neural net opt + feas)
        print("- " + filename + " combined sel ...")
        sols.append(cs.cut_select_algo(filename, dim, top, strat=4, plots=True, sol=sols_ms, strong_only=strong_only))
        # run optimality selection with neural net estimator
        print("- " + filename + " optimality NN sel ... ")
        sols.append(cs.cut_select_algo(filename, dim, top, strat=2, plots=True, sol=sols_ms, strong_only=strong_only))
        # run feasibility selection
        print("- " + filename + " feasibility sel ... ")
        sols.append(cs.cut_select_algo(filename, dim, top, strat=1, plots=True, sol=sols_ms))
        # run random selection
        print("- " + filename + " random sel ... ")
        np.random.seed(seed_nb)
        for i in range(10):
            sols.append(cs.cut_select_algo(filename, dim, top, strat=5, plots=True, sol=sols_ms))
        # run optimality selection with exact SDP solutions (Mosek)
        print("- " + filename + " optimality exact sel ... ")
        sols.append(cs.cut_select_algo(filename, dim, top, strat=3, plots=True, sol=sols_ms, strong_only=strong_only))
        sols = np.array(sols).T
        nb_columns = sols.shape[1]
        bounds = sols[0:cuts_rounds + 1, :]
        cuts = sols[cuts_rounds + 1:2 * cuts_rounds + 1, :]
        bounds_per_cut = np.zeros((cuts_rounds + 1, sols.shape[1]))
        for col in range(nb_columns):
            for cutRound in range(1, cuts_rounds + 1):
                bounds_per_cut[cutRound, col] = bounds[cutRound, col] / sum(cuts[0:cutRound, col])
        max_cuts = np.floor(sols[2 * cuts_rounds + 1, 0] * top)
        valid_cuts = np.divide(cuts, max_cuts)
        with open(save_file, "a") as f:
            f.write(filename + "\n")
            f.write("gap_closed_overall\n")
            f.write("comb,opt_nn,feas," + ",".join(str(x) for x in ["rand " + str(i) for i in range(10)])
                    + ",opt_exact\n")
            for line in bounds:
                f.write(",".join(str(x) for x in line) + "\n")
            f.write("gap_closed_per_nb_cuts_used\n")
            f.write("comb,opt_nn,feas," + ",".join(str(x) for x in ["rand " + str(i) for i in range(10)])
                    + ",opt_exact\n")
            for line in bounds_per_cut:
                f.write(",".join(str(x) for x in line) + "\n")
            f.write("percent_valid_cuts_found\n")
            f.write("comb,opt_nn,feas," + ",".join(str(x) for x in ["rand " + str(i) for i in range(10)])
                    + ",opt_exact\n")
            for line in valid_cuts:
                f.write(",".join(str(x) for x in line) + "\n")
    ###########################################
    # Figure 14 - plots on QCQP instance powerflow009r
    ###########################################
    # reasonable run-time (~15mins)
    elif figure == 14:
        save_file = os.path.join(dirname, "fig14_data")
        filename = "powerflow0009r_preprocessed"
        strats_names = ["feasibility", "combined selection 2, one constraint at a time",
                        "combined selection 1, all constraints aggregated", "random"]
        # Primal bound for powerflow0009r instance from http://www.minlplib.org/powerflow0009r.html
        sol_0009r = 5296.68620400
        nb_of_rand = 5
        for sel_size, nb_rounds in [(12, 20), (5, 40)]:
            gaps, nb_cuts = [], []
            np.random.seed(seed_nb)
            for strat in [1, 2, *([4] * nb_of_rand)]:
                print("- " + filename + ", " + str(sel_size) + "cuts, " + strats_names[strat-1] + " sel ... ")
                objectives, nb_cuts_opt = cs_qcqp.cut_select_algo(filename, 3, sel_size=sel_size, strat=strat,
                                                                  nb_rounds_cuts=nb_rounds)
                gap_closed_percent = [0] * len(objectives)
                for idx in range(1, len(objectives)):
                    gap_closed_percent[idx] = (objectives[idx] - objectives[0]) / (sol_0009r - objectives[0])
                gaps.append(gap_closed_percent)
                nb_cuts.append(nb_cuts_opt)
            gaps = np.array(gaps).T
            nb_cuts = np.array(nb_cuts).T
            with open(save_file + "_" + str(sel_size) + "cuts.csv", "w") as f:
                f.write(filename + "\n")
            with open(save_file + "_" + str(sel_size) + "cuts.csv", "a") as f:
                f.write("Percent of gap to optimality from M2 closed\n")
                f.write("feas,comb2," + ",".join(
                    str(x) for x in ["rand " + str(i) for i in range(nb_of_rand)]) + "\n")
                for line in gaps:
                    f.write(",".join(str(x) for x in line) + "\n")
                f.write("Number of cuts selected by optimality at each cut round\n")
                f.write("feas,comb2," + ",".join(
                    str(x) for x in ["rand " + str(i) for i in range(nb_of_rand)]) + "\n")
                for line in nb_cuts:
                    f.write(",".join(str(x) for x in line) + "\n")
    print("Run for Figure " + str(figure) + " - Done")


def run_for_all_tables(table=0, folder_name="data_tables", test_cfg=True):
    """Run on all BoxQP instances (or except the very large ones for test_cfg=True) to construct the tables 4-7 and
    figures 12-13 in the manuscript
    :param table: for which table to run (4-7, 12-13) or 0 for all tables and fig 12-13
    :param folder_name: what folder to save to
    :param test_cfg: run all BoxQP instances (takes extremely long esp. for tables 5-6), or a reduced set
    :return: saved .csv data files in folder_name
    """
    assert (table in [0, 4, 5, 6, 7, 12, 13]), \
        "Please select all tables (0) or a valid table (4-7) or figure (12-13) to run data for"
    boxqp_files = 'filenames_test.txt' if test_cfg else 'filenames.txt'
    text_file = open(os.path.join(os.path.curdir, 'boxqp_instances', boxqp_files), "r")
    dirname = os.path.join(os.path.curdir, folder_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    write_flag = "w"
    problems = text_file.read().split('\n')
    text_file.close()
    dict_sel = {2: "opt", 1: "feas"}
    dict_sel2 = {2: "Optimality selection (neural nets)", 1: "Feasibility selection"}
    info_inst = []
    # Get filename, size, density and solution of each BoxQP instance for calculations
    for prob in problems:
        filename, sol = prob.split('  ')
        size_inst = filename.split('-')
        density_inst = int(size_inst[1])
        size_inst = int(size_inst[0].split('r')[1])
        info_inst.append((filename, size_inst, density_inst, float(sol)))
    # Create QP cut solver
    cs = CutSolver()

    ###########################################
    # Data for M+tri (Tables 4, 7 and Figures 12, 13)
    ###########################################
    if table in [0, 4, 7, 12, 13]:
        sel_size = 0.1
        # Do not add any SDP cuts, separate only triangle
        save_file = os.path.join(dirname, "data_M_tri_" + str(sel_size) + ".csv")
        with open(save_file, write_flag) as f:
            f.write("M+tri with selection size " + str(sel_size) + "\n")
            f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts,nb_tri_cuts,"
                    "nb_total_cuts, time_total\n")
        for inst in range(len(problems)):
            filename, size_inst, dens_inst, sol = info_inst[inst]
            (curObjValues, time_overall, _, _, _, nb_tri_cuts, nb_subproblems) = \
                cs.solve_mccormick_and_tri(filename, sel_size, term_on=True)
            percent_gap_closed = (curObjValues[-1] - curObjValues[0]) / (sol - curObjValues[0])
            with open(save_file, "a") as f:
                f.write(filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + "," +
                        str(nb_subproblems) + "," + str(0) + "," + str(sum(nb_tri_cuts)) +
                        "," + str(sum(nb_tri_cuts)) + "," + str(time_overall) + "\n")
            print("- M+tri, " + str(sel_size) + ", only triangle cuts: " + filename + ", time:" + str(
                time_overall))
    ###########################################
    #  Data for M+S3 with 5%, 10% selection size (Table 4, 5, 6)
    ###########################################
    if table in [0, 4, 6]:
        dim = 3
        sel_sizes = [0.1] if table == 6 else [0.05, 0.1]
        # selection sizes 5%, 10%
        for sel_size in sel_sizes:
            # for optimality selection via neural nets (2) and feasibility selection (1)
            for selectType in [2, 1]:
                save_file = os.path.join(dirname,
                                "data_M_S" + str(dim) + "_" + dict_sel[selectType] + "_" + str(sel_size) + ".csv")
                with open(save_file, write_flag) as f:
                    f.write(dict_sel2[selectType] + " for M+S3 with selection size " + str(sel_size) + "\n")
                    f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts\n")
                for inst in range(len(problems)):
                    filename, size_inst, dens_inst, sol = info_inst[inst]
                    (curObjValues, time_overall, _, _, nb_sdp_cuts, _, nb_subproblems) = \
                        cs.cut_select_algo(filename, dim, sel_size, strat=selectType, triangle_on=False, term_on=True)
                    percent_gap_closed = (curObjValues[-1] - curObjValues[0]) / (sol - curObjValues[0])
                    with open(save_file, "a") as f:
                        f.write(
                            filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + "," +
                            str(nb_subproblems) + "," + str(sum(nb_sdp_cuts)) + "\n")
                    print("- M+S3, " + str(sel_size) + " ," + dict_sel[selectType] + ": " + filename + ", time:" + str(
                        time_overall))
    ###########################################
    #  Data for (naive) M+triangle+S3 with 10% selection size (Table 4, 7; Figure 12, 13)
    ###########################################
    if table in [0, 4, 7, 12, 13]:
        dim = 3
        sel_size = 0.1
        select_types = [2] if table in [7, 12, 13] else [2, 1]
        # for optimality selection via neural nets (2) and feasibility selection (1)
        for selectType in select_types:
            save_file = os.path.join(dirname,
                                "data_M_tri_S" + str(dim) + "_" + dict_sel[selectType] + "_" + str(sel_size) + ".csv")
            with open(save_file, write_flag) as f:
                f.write(dict_sel2[selectType] + " for M+triangle+S3 with selection size " + str(sel_size) + "\n")
                f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts,nb_tri_cuts,"
                        "nb_total_cuts, time_total\n")
            for inst in range(len(problems)):
                filename, size_inst, dens_inst, sol = info_inst[inst]
                (curObjValues, time_overall, _, _, nb_sdp_cuts, nb_tri_cuts, nb_subproblems) = \
                    cs.cut_select_algo(filename, dim, sel_size, strat=selectType, triangle_on=True, term_on=True)
                percent_gap_closed = (curObjValues[-1] - curObjValues[0]) / (sol - curObjValues[0])
                with open(save_file, "a") as f:
                    f.write(filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + "," +
                            str(nb_subproblems) + "," + str(sum(nb_sdp_cuts)) + "," + str(sum(nb_tri_cuts)) +
                            "," + str(sum(nb_sdp_cuts) + sum(nb_tri_cuts)) + "," + str(time_overall) + "\n")
                print("- M+tri+S3, " + str(sel_size) + " ," + dict_sel[selectType] + ": " + filename + ", time:" + str(
                    time_overall))
    ###########################################
    # Data for M+S3, M+S4, M+S5 with 10% selection size and 40 cut rounds (Table 5)
    ##########################################
    if table in [0, 5]:
        sel_size = 0.1
        # Number of cut rounds for table 5
        table5_rounds = 40
        # decompositions M+S4, M+S5
        for dim in [3, 4, 5]:
            # for optimality selection via neural nets (2) and feasibility selection (1)
            for selectType in [2, 1]:
                save_file = os.path.join(dirname, "data_M_S" + str(dim) + "_" + dict_sel[selectType] +
                                         "_" + str(sel_size) + "_" + str(table5_rounds) + ".csv")
                with open(save_file, write_flag) as f:
                    f.write(dict_sel2[selectType] + " for M+S" + str(dim) + " with selection size " + str(sel_size) +
                            " and " + str(table5_rounds) + "cut rounds \n")
                    f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts\n")
                for inst in range(len(problems)):
                    filename, size_inst, dens_inst, sol = info_inst[inst]
                    # Number of cut rounds for table 5 (40) unless jumbo high density category for M+S5 where
                    # there are instances we run out of memory for, so don't do cuts rounds for that category)
                    cut_rounds = 0 if (dim == 5 and size_inst >= 100 and dens_inst >= 75) else table5_rounds
                    (curObjValues, time_overall, _, _, nb_sdp_cuts, _, nb_subproblems) = \
                        cs.cut_select_algo(filename, dim, sel_size, strat=selectType, triangle_on=False,
                                           term_on=True, nb_rounds_cuts=cut_rounds)
                    percent_gap_closed = (curObjValues[-1] - curObjValues[0]) / (sol - curObjValues[0])
                    with open(save_file, "a") as f:
                        f.write(filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + ","
                                + str(nb_subproblems) + "," + str(sum(nb_sdp_cuts)) + "\n")
                    print("- M+S" + str(dim) + ", " + str(sel_size) + " ," + dict_sel[
                        selectType] + ": " + filename + ", time:" + str(
                        time_overall))
    ###########################################
    #  Data for chordal extensions M+bar(S(P_3)), M+bar(S(P_3*)) (Table 6)
    ###########################################
    if table in [0, 6]:
        sel_size = 0.1
        dim = 3
        # chordal extensions M+bar(S(P3)), M+bar(S(P3*))
        dict_ch = {1: "barSP3", 2: "barSP3star"}
        for chordalExt in [1, 2]:
            # for optimality selection via neural nets (2) and feasibility selection (1)
            selections = [1, 2] if chordalExt == 1 else [2]
            for selectType in selections:
                save_file = os.path.join(dirname,
                            "data_M_" + dict_ch[chordalExt] + "_" + dict_sel[selectType] + "_" + str(sel_size) + ".csv")
                with open(save_file, write_flag) as f:
                    f.write(dict_sel2[selectType] + " for M+" + dict_ch[chordalExt]
                            + " with selection size " + str(sel_size) + "\n")
                    f.write("filename,size,density,percent_gap_closed,nb_subproblems,nb_sdp_cuts\n")
                for inst in range(len(problems)):
                    filename, size_inst, dens_inst, sol = info_inst[inst]
                    (curObjValues, time_overall, _, _, nb_sdp_cuts, _, nb_subproblems) = \
                        cs.cut_select_algo(filename, dim, sel_size, strat=selectType, triangle_on=False, term_on=True,
                                           ch_ext=chordalExt)
                    percent_gap_closed = (curObjValues[-1] - curObjValues[0]) / (sol - curObjValues[0])
                    with open(save_file, "a") as f:
                        f.write(filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + ","
                                + str(nb_subproblems) + "," + str(sum(nb_sdp_cuts)) + "\n")
                    print("- M+" + dict_ch[chordalExt] + ", " + str(sel_size) + " ," + dict_sel[selectType] +
                          ": " + filename + ", time:" + str(time_overall))
    ###########################################
    #  Data for (heuristic) M+triangle+S3_5 with 10% selection size (Table 7; Figure 12, 13)
    #   Apply heuristic:
    #   - M+triangle+S5 for low and medium density      (combined selection)
    #   - M+triangle+S4 for high density, <jumbo size   (optimality selection)
    #   - M+triangle+S3 for high density, jumbo size    (optimality selection)
    ###########################################
    if table in [0, 7, 12, 13]:
        sel_size = 0.1
        save_file = os.path.join(dirname, "data_heur_M_tri_S3_5_" + str(sel_size) + ".csv")
        with open(save_file, write_flag) as f:
            f.write("Cut selection heuristic for M+triangle+S3_5 with selection size " + str(sel_size) + "\n")
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
            (curObjValues, time_overall, _, _, nb_sdp_cuts, nb_tri_cuts, nb_subproblems) = sol_info
            percent_gap_closed = (curObjValues[-1] - curObjValues[0]) / (sol - curObjValues[0])
            with open(save_file, "a") as f:
                f.write(filename + "," + str(size_inst) + "," + str(dens_inst) + "," + str(percent_gap_closed) + "," + str(
                    nb_subproblems) +
                        "," + str(sum(nb_sdp_cuts)) + "," + str(sum(nb_tri_cuts)) +
                        "," + str(sum(nb_sdp_cuts) + sum(nb_tri_cuts)) + "," + str(time_overall) + "\n")
            print("- heur M+S3_5, " + str(sel_size) + ": " + filename + ", time:" + str(
                time_overall))
    # Table = 0 means aggregate for all tables
    if table == 0:
        for table_nb in [4, 5, 6, 7, 12, 13]:
                aggregate_table(table_nb, folder_name)
    elif table in [12, 13]:
        aggregate_table(12, folder_name)
        aggregate_table(13, folder_name)
    else:
        aggregate_table(table, folder_name)


def aggregate_table(table, folder="data_tables", fig_folder="data_figures"):
    """ Aggregates BoxQP data for a table in Table 3 categories
    :param table: table to aggregate
    :param folder: where to find raw .csv data and save table .csv data file
    :param fig_folder:  where to find raw .csv data and save figure .csv data file (Figures 12-13)
    :return: .csv data file
    """
    assert (table in [4, 5, 6, 7, 12, 13]), "Please select a valid table (4-7) or figure to aggregate (12-13)"
    pr_categories = [
        "Small,Low,", ",Medium,", ",High,",
        "Medium,Low,", ",Medium,", ",High,",
        "Large,Low,", ",Medium,", ",High,",
        "Jumbo,Low,", ",Medium,", ",High,"]
    write_flag = "w"
    save_file = os.path.join(os.path.curdir, folder, "table" + str(table) + ".csv")
    if table in [12, 13]:
        save_file = os.path.join(os.path.curdir, fig_folder, "fig" + str(table) + "_data.csv")

    if table == 4:
        table_data = np.column_stack((
            aggregate_column(folder, "data_M_S3_opt_0.05", 1),
            aggregate_column(folder, "data_M_S3_feas_0.05", 1),
            aggregate_column(folder, "data_M_S3_feas_0.05", 1) - aggregate_column(folder, "data_M_S3_opt_0.05", 1),
            aggregate_column(folder, "data_M_S3_opt_0.1", 1),
            aggregate_column(folder, "data_M_S3_feas_0.1", 1),
            aggregate_column(folder, "data_M_S3_feas_0.1", 1) - aggregate_column(folder, "data_M_S3_opt_0.1", 1),
            aggregate_column(folder, "data_M_tri_0.1", 1),
            aggregate_column(folder, "data_M_tri_S3_opt_0.1", 1),
            aggregate_column(folder, "data_M_tri_S3_feas_0.1", 1),
            aggregate_column(folder, "data_M_tri_S3_feas_0.1", 1) - aggregate_column(folder, "data_M_tri_S3_opt_0.1", 1)))
    elif table == 5:
        table_data = np.column_stack((
            aggregate_column(folder, "data_M_S3_opt_0.1_40", 1),
            aggregate_column(folder, "data_M_S3_feas_0.1_40", 1),
            aggregate_column(folder, "data_M_S4_opt_0.1_40", 1),
            aggregate_column(folder, "data_M_S4_feas_0.1_40", 1),
            aggregate_column(folder, "data_M_S5_opt_0.1_40", 1),
            aggregate_column(folder, "data_M_S5_feas_0.1_40", 1),
            # columns with number of sub-problems
            aggregate_column(folder, "data_M_S3_opt_0.1_40", 2, nb_types=1),
            aggregate_column(folder, "data_M_S4_opt_0.1_40", 2, nb_types=1),
            aggregate_column(folder, "data_M_S5_opt_0.1_40", 2, nb_types=1)))
    elif table == 6:
        table_data = np.column_stack((
            aggregate_column(folder, "data_M_S3_opt_0.1", 1),
            aggregate_column(folder, "data_M_S3_feas_0.1", 1),
            aggregate_column(folder, "data_M_barSP3_opt_0.1", 1),
            aggregate_column(folder, "data_M_barSP3_feas_0.1", 1),
            aggregate_column(folder, "data_M_barSP3star_opt_0.1", 1),
            # columns with number of sub-problems
            aggregate_column(folder, "data_M_S3_opt_0.1", 2, nb_types=1),
            aggregate_column(folder, "data_M_barSP3_opt_0.1", 2, nb_types=1),
            aggregate_column(folder, "data_M_barSP3star_opt_0.1", 2, nb_types=1)))
    elif table == 7:
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
        table_data = np.column_stack((
            data_known,
            # M+tri linear base of inequalities
            aggregate_column(folder, "data_M_tri_0.1", 1),
            # naive
            aggregate_column(folder, "data_M_tri_S3_opt_0.1", 1),
            aggregate_column(folder, "data_M_tri_S3_opt_0.1", 1) - data_known[:, 3],
            aggregate_column(folder, "data_M_tri_S3_opt_0.1", 1) - aggregate_column(folder, "data_M_tri_0.1", 1),
            # heuristic
            aggregate_column(folder, "data_heur_M_tri_S3_5_0.1", 1),
            aggregate_column(folder, "data_heur_M_tri_S3_5_0.1", 1) - data_known[:, 3],
            aggregate_column(folder, "data_heur_M_tri_S3_5_0.1", 1) - aggregate_column(folder, "data_M_tri_0.1", 1)))
    elif table == 12:
        # Number of cuts per category of BoxQP problems for BGL
        # ("Globally solving nonconvex quadratic programming problems with box constraints via integer
        # programming methods", P. Bonami, O. Gunluk, J. Linderoth)
        bgl_nb_cuts = np.array([790.37, 2368.78, 4115.55, 2454.53, 15012.26, 71558.93, 11807.31,
                                54733.33, 144118.66, 37858.37, 165480.88, 354370.41])
        table_data = np.column_stack((
            aggregate_column(folder, "data_M_tri_0.1", 5, nb_types=2),              # M+tri linear base of inequalities
            aggregate_column(folder, "data_M_tri_S3_opt_0.1", 5, nb_types=2),       # naive
            aggregate_column(folder, "data_heur_M_tri_S3_5_0.1", 5, nb_types=2),    # heuristic
            bgl_nb_cuts.T))
        table_data[table_data == 1] = 100
    elif table == 13:
        table_data = np.column_stack((
            aggregate_column(folder, "data_M_tri_0.1", 6, nb_types=2),              # M+tri linear base of inequalities
            aggregate_column(folder, "data_M_tri_S3_opt_0.1", 6, nb_types=2),       # naive
            aggregate_column(folder, "data_heur_M_tri_S3_5_0.1", 6, nb_types=2),    # heuristic
        ))
        table_data[table_data == 1] = 0.1

    with open(save_file, write_flag) as f:
        for line in range(len(pr_categories)):
            f.write(pr_categories[line] + ",".join(str(x) for x in table_data[line, :]) + "\n")


def aggregate_column(folder, filename, column_to_agg, nb_types=0):
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
        content = content[2:]
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
        agg_arr[pr_category][0] += np.log(pr[2])  # summing logs for geometric average
        agg_arr[pr_category][1] += 1
    # Format value according to type of value being aggregated
    for idx, elem in enumerate(agg_arr):
        nb_agg = np.exp(elem[0] / max(elem[1], 1))  # exp to get geometric average
        if nb_types == 0:  # if percentage of gap closed <1
            agg_arr[idx] = np.round(nb_agg * 100, 2)
        elif nb_types == 1:  # if number of cuts added
            agg_arr[idx] = np.round(nb_agg, 0)
        elif nb_types == 2:  # if figures 12-13
            agg_arr[idx] = np.round(nb_agg, 2)
    return np.array(agg_arr)


def reset_file(save_file, write_flag):
    """Reset savefile if write_flag set to write
    """
    if write_flag == 'w':
        with open(save_file, write_flag):
            pass


if __name__ == '__main__':
    main()

