from timeit import default_timer as timer
from operator import itemgetter, mul
from copy import deepcopy
import itertools
import os.path
import platform
import numpy as np
import ctypes
import cplex
from mosek.fusion import Model, Domain, ObjectiveSense, Expr
from cvxopt import spmatrix, amd
import chompack as cp
import warnings
warnings.filterwarnings("error")


class CutSolver(object):
    """QP cutting plane solver object (applied on BoxQP)
    """
    # Class algorithmic parameters
    # Threshold of minimum optimality measure to select cut in combined selection
    _THRES_MIN_OPT = 0
    # Epsilon margin to consider eigenvalue negative (and generate cuts based on)
    _THRES_NEG_EIGVAL = -10 ** (-15)
    # Big M constant value for combined strategy
    _BIG_M = 1000
    # Termination criteria on an improvement between two consecutive cut rounds of less than (CONVERGENCE_TOL) of
    # the gap closed overall so far
    _CONVERGENCE_TOL = 10**(-3)
    # Consider only triangle inequalities with at least 2/3 non-zero coefficients (edges in Q_adj) - see Sect. 2
    _THRES_TRI_DENSE = 2
    # Violation threshold for separation of triangle inequalities
    _THRES_TRI_VIOL = 10**(-7)
    # Max number of subproblems in SDP vertex cover to not exceed available RAM memory
    _THRES_MAX_SUBS = 4*(10**6)
    # Max number of SDP cuts to be added to relaxation at each cut round
    _SDP_CUTS_PER_ROUND_MAX = 5000
    # Min number of triangle cuts to be added to relaxation at each cut round
    _TRI_CUTS_PER_ROUND_MIN = 5000
    # Min number of triangle cuts to be added to relaxation at each cut round
    _TRI_CUTS_PER_ROUND_MAX = 10000

    def __init__(self):
        # Dimension of SDP vertex cover
        self._dim = 0
        # Number of unlifted x variables of the BoxQP instance considered
        self._nb_vars = 0
        # Number of lifted X variables of the BoxQP instance considered
        self._nb_lifted = 0
        # Matrix of objective coefficients
        self._Q = []
        # Adjacency matrix of Q
        self._Q_adj = []
        # Upper triangular part of Q in flat array (row major) for faster slice random access
        self._Q_arr = []
        # CPLEX problem instance
        self._my_prob = None
        # List of aggregate info needed for each subproblem in an SDP vertex cover
        self._agg_list = []
        # Oredered list of subproblems according to cut selection strategy
        # Neural networks list in terms of their functions and data types
        self._nns = None
        # Global variables to speed up eigen-decomposition by preforming matrix to be decomposed
        self._Mat = [np.zeros((i, i)) for i in [3, 4, 5, 6]]
        for i in range(4):
            self._Mat[i][0, 0] = 1
        self._inds = [(np.array([1 + x for x in np.triu_indices(dim, 0, dim)[0]]),
                       np.array([1 + x for x in np.triu_indices(dim, 0, dim)[1]])) for dim in [2, 3, 4, 5]]
        # Lists storing pre-processed info about triangle inequalities
        self._rank_list_tri = []
        self._idx_list_tri = []

    def cut_select_algo(self, filename, dim, sel_size, strat=2, nb_rounds_cuts=20, term_on=False,
                        triangle_on=False, ch_ext=False, strong_only=False, plots=False, sol=0, all_comp=False):
        """Implements cut selection strategies as in Algorithm 1 in the manuscript.
        :param filename: file for BoxQP problem instance
        :param dim: dimension of SDP vertex cover and eigcuts
        :param sel_size: selection size (% or number) of eigcuts and/or triangle cuts
        :param strat: selection strategy that ranks/orders subproblems (and associated eigcuts) to select
        (1-feasibility, 2-optimality via estimator, 3-optimality via exact sdp, 4-combined (opt+feas), 5-random)
        :param nb_rounds_cuts: number of cut rounds
        :param term_on: terminate on small improv. between succesive cut rounds (True) or after all cut rounds (False)
        :param triangle_on: flag for using triangle cuts or not
        :param ch_ext: flag for chordal extension in SDP decomp (0-P^E_3, 1-P^bar(E)_3, 2-bar(P*_3))
        :param strong_only: select only strong violated cuts
        :param plots: flag for returning info needed to plot bounds or not
        :param sol: if plotting bounds, solution to calculate percent gap closed w.r.t to
        :param all_comp: flag to register all information (for table 8)
        :return: solutions across cut rounds at termination (containing bounds and cut statistics)
        """
        assert (strat in [-1, 0, 1, 2, 3, 4, 5]), "Pick a violated cut selection strategy (-1, 1-5)!"
        assert (0 < sel_size), "The selection size must be a % or number (of cuts) >0!"
        assert (dim <= 5), "Keep SDP vertex cover low-dimensional (<=5)!"
        assert (ch_ext in [0, 1, 2]), "Chordal extension flags: 0-P^E_3, 1-P^bar(E)_3, 2-bar(P*_3)!"

        # Start time
        time_begin = timer()
        # Numbes of cuts, objective values and execution times
        nbs_sdp_cuts, nbs_tri_cuts, curr_obj_vals, round_times, sep_times = [0], [], [], [], []
        self._dim = dim

        # Load trained neural network functions with their input/output types (if optimality selection with neural nets)
        if strat in [2, 4, -1]:
            self._load_neural_nets()

        # Parse BoxQP instances and create CPLEX problem instance with lifted variables,
        self.__parse_boxqp_into_cplex(filename)
        my_prob = self._my_prob
        # Use dual simplex rather than automatic cplex choice (incl. barrier) when no triangle ineq are present
        if not triangle_on:
            my_prob.parameters.lpmethod.set(my_prob.parameters.lpmethod.values.dual)

        # SDP vertex cover in a list of subproblems with their details (agg_list) and adjacency matrix (Q_adj).
        nb_subprobs = self._get_sdp_vertex_cover(dim, ch_ext=ch_ext)
        # Guard when (agg_list) is possibly too large to store in RAM memory
        # Also return just number of subproblems if no cut rounds to go through
        if nb_subprobs >= CutSolver._THRES_MAX_SUBS or nb_rounds_cuts == 0:
            # for such instances can alternatively trade time penalty for memory by computing elements of (agg_list)
            # at every round one at a time, but will run into CPLEX impractically long solve times
            return [0, 0], timer() - time_begin, 0, 0, [0], 0, nb_subprobs
        agg_list = self._agg_list
        # Interpret selection size as % or absolute number and threshold the maximum number of SDP cuts per round
        sel_size_frac = sel_size
        sel_size = min(int(np.floor(sel_size * nb_subprobs)) if sel_size < 1 else min(sel_size, nb_subprobs),
                       CutSolver._SDP_CUTS_PER_ROUND_MAX)

        time_pre_solve = timer()
        # Add McCormick cuts to the instance just created (if chordal extension used in SDP decomp take it in account)
        self._add_mccormick_to_instance()
        sep_times.append(timer() - time_pre_solve)

        # Cut Round 0 - Solve McCormick relaxation M
        time_pre_solve = timer()
        my_prob.solve()
        round_times.append(timer() - time_pre_solve + sep_times[0])
        curr_obj_vals.append(my_prob.solution.get_objective_value())
        vars_values = np.array(my_prob.solution.get_values())

        # Info needed for separation of triangle inequalities (if they are used)
        if triangle_on:
            self.__preprocess_triangle_ineq()
        # Store info for figure 8 if needed
        if strat == -1:
            rounds_stats, round_std_devs, rounds_all_cuts = ([], [], [])
        strat_change = strat

        # ROUNDS OF CUTS on linear relaxation till termination
        ######################################################
        for cut_round in range(1, nb_rounds_cuts + 1):
            # Termination criteria - terminate on an improvement between two consecutive cut rounds
            # of < 0.1% of the gap closed overall so far from the M bound
            if term_on and len(curr_obj_vals) >= 3 and curr_obj_vals[-1] != curr_obj_vals[0] and\
                    (curr_obj_vals[-1] - curr_obj_vals[-2]) / (curr_obj_vals[-1] - curr_obj_vals[0]) < \
                    CutSolver._CONVERGENCE_TOL:
                break
            # If using fully dense cuts, if after 4th round and time > 1000s, stop.
            if strat == 0 and term_on and cut_round > 4 and sum(round_times) > 1000:
                break

            ###################
            #  Start separation (selection + generation) of cuts (eigcuts from sdp and triangle ineq.)
            time_pre_sep = timer()
            # Order subproblems for cut selection based on (strat) measure
            # e.g. optimality-based estimated (via neural networks) objective improvements
            if strat == 0:
                nb_sdp_cuts = self.__gen_dense_eigcuts(vars_values=vars_values)
            else:
                if strat != -1 and strat != 4:
                    rank_list = self._sel_eigcut_by_ordering_on_measure(strat, vars_values, cut_round)
                elif strat == 4:    # combined selection
                    (strat_change, rank_list) = self._sel_eigcut_by_ordering_on_measure(strat, vars_values, cut_round, sel_size=sel_size)
                else:   # for figure 8
                    rank_list, round_stats, round_std_dev, round_all_cuts = \
                        self._sel_eigcut_by_ordering_on_measure(strat, vars_values, cut_round, sel_size=sel_size)
                    # keep cut selections (both by estimated and exact measure)
                    rounds_all_cuts.extend(round_all_cuts)
                    # keep percent (of selection size) of cuts selected by both estimated and exact measures (overlap)
                    rounds_stats.append(round_stats)
                    round_std_devs.append(round_std_dev)
                # Generate eigen-cuts from selected subproblems
                nb_sdp_cuts = self._gen_eigcuts_selected(strat, sel_size, rank_list,
                                                          strong_only=strong_only, vars_values=vars_values)
            nbs_sdp_cuts.append(nb_sdp_cuts)
            # Separate and add triangle inequalities (if activated through (triangle_on) flag)
            nb_tri_cuts = self.__separate_and_add_triangle(sel_size_frac, vars_values) if triangle_on else 0
            nbs_tri_cuts.append(nb_tri_cuts)
            sep_times.append(timer() - time_pre_sep)
            strat = strat_change
            # End of cuts separation for this round
            ###################

            # Solve relaxation augmented with cuts at current round
            time_pre_solve = timer()
            my_prob.solve()
            round_times.append(timer() - time_pre_solve + sep_times[-1])
            # Store new value of objective
            curr_obj_vals.append(my_prob.solution.get_objective_value())
            # print((-curr_obj_vals[-1] + curr_obj_vals[0]) / (sol + curr_obj_vals[0]))
            # Store new values of all variables (unlifted x and lifted X)
            vars_values = np.array(my_prob.solution.get_values()).astype(float)
        # End of cut rounds - return info in different formats
        ######################################################

        # If doing the comparison for Table 6
        if all_comp:
            return (timer() - time_begin, len(agg_list), [-obj for obj in curr_obj_vals],
                    round_times, sep_times, nbs_sdp_cuts)
        # If plotting figures
        if plots:
            gap_closed_percent = [0] * len(curr_obj_vals)
            for idx in range(1, len(curr_obj_vals)):
                gap_closed_percent[idx] = (-curr_obj_vals[idx] + curr_obj_vals[0]) / (sol + curr_obj_vals[0])
            # Figure 8:
            if strat == -1:
                return gap_closed_percent, rounds_stats, round_std_devs, rounds_all_cuts
            # Other figures
            else:
                return (*gap_closed_percent, *round_times, *nbs_sdp_cuts, len(agg_list))  # other figures
        # Default behaviour (for constructing tables)
        return ([-obj for obj in curr_obj_vals], timer() - time_begin, round_times,
                sep_times, nbs_sdp_cuts, nbs_tri_cuts, len(agg_list))

    def solve_mccormick_and_tri(self, filename, sel_size, nb_rounds_cuts=20, term_on=False):
        """Cut separation for only triangle inequalities for M+tri (no overhead of separating SDP eigcuts)
        :param filename: file for BoxQP problem instance
        :param sel_size: selection size (% or number) of triangle cuts
        :param nb_rounds_cuts: number of cut rounds
        :param term_on: terminate on small improv. between succesive cut rounds (True) or after all cut rounds (False)
        :return: solutions across cut rounds at termination (containing bounds and cut statistics)
        """
        assert (0 < sel_size), "The selection size must be a % or number (of cuts) >0!"

        # Start time
        time_begin = timer()
        # Numbes of cuts, objective values, execution and separation times
        nbs_tri_cuts, curr_obj_vals, cplex_times, sep_times = [], [], [], 0

        # Parse BoxQP instances and create CPLEX problem instance with lifted variables,
        self.__parse_boxqp_into_cplex(filename)
        my_prob = self._my_prob

        # Add McCormick cuts to the instance just created
        self._add_mccormick_to_instance()

        # Cut Round 0 - Solve McCormick relaxation M
        time_pre_solve = timer()
        my_prob.solve()
        cplex_times.append(timer() - time_pre_solve)
        curr_obj_vals.append(my_prob.solution.get_objective_value())
        vars_values = np.array(my_prob.solution.get_values())
        #print(my_prob.solution.get_objective_value())
        # Info needed for separation of triangle inequalities
        self.__preprocess_triangle_ineq()

        # ROUNDS OF CUTS on linear relaxation till termination
        ######################################################
        for cut_round in range(1, nb_rounds_cuts + 1):
            # Termination criteria - terminate on an improvement between two consecutive cut rounds
            # of < 0.01% of the gap closed overall so far from the M bound
            if term_on and len(curr_obj_vals) >= 3 and \
                    (curr_obj_vals[-1] - curr_obj_vals[-2]) / (curr_obj_vals[-1] - curr_obj_vals[0]) < \
                    CutSolver._CONVERGENCE_TOL:
                break
            #  Separate triangle ineq. only
            time_pre_sep = timer()
            nb_tri_cuts = self.__separate_and_add_triangle(sel_size, vars_values)
            nbs_tri_cuts.append(nb_tri_cuts)
            sep_times += timer() - time_pre_sep     # End of cuts separation for this round
            # Solve relaxation augmented with cuts at current round
            time_pre_solve = timer()
            my_prob.solve()
            cplex_times.append(timer() - time_pre_solve)
            # Store new value of objective
            curr_obj_vals.append(my_prob.solution.get_objective_value())
            # Store new values of all variables (unlifted x and lifted X)
            vars_values = np.array(my_prob.solution.get_values()).astype(float)
            #print(my_prob.solution.get_objective_value())
        # End of cut rounds - return info in different formats
        ######################################################
        # Default behaviour (for constructing tables)
        return ([-obj for obj in curr_obj_vals], timer() - time_begin, cplex_times,
                sep_times, [], nbs_tri_cuts, 0)

    def _load_neural_nets(self):
        """Load trained neural networks (from /neural_nets/NNs.dll) up to the subproblem dimension needed for an SDP
        vertex cover. These neural networks estimate the expected objective improvement for a particular subproblem at the
        current solution point.
        """
        self._nns = []
        if platform.uname()[0] == "Windows":
            nn_library = os.path.join('neural_nets', 'NNs.dll')
        elif platform.uname()[0] == "Linux":
            nn_library = os.path.join('neural_nets', 'NNs.so')
        else:   # Mac OSX
            raise ValueError('The neural net library is compiled only for Windows/Linux! (OSX needs compiling)')
            #   nn_library = 'neural_nets/NNs.dylib' - Not compiled for OSX, will throw error
        nn_library = ctypes.cdll.LoadLibrary(nn_library)
        for d in range(2, self._dim + 1):  # (d=|rho|) - each subproblem rho has a neural net depending on its size
            func_dim = getattr(nn_library, "neural_net_%dD" % d)  # load each neural net
            func_dim.restype = ctypes.c_double  # return type from each neural net is a c_double
            # c_double array input: x_rho (the current point) and Q_rho (upper triangular part since symmetric)
            input_arr = (ctypes.c_double * (d * (d + 3) // 2))()
            self._nns.append((func_dim, input_arr))

    def __parse_boxqp_into_cplex(self, filename):
        """Parse BoxQP instance from file, store quadratic coefficient matrix in several formats and form CPLEX
        instance with the identified variables and objective
        """
        with open(os.path.join(os.path.dirname(__file__), 'boxqp_instances', filename + '.in')) as f:
            content = f.readlines()
            nb_vars = [int(n) for n in content[0].split()][0]
            # Minus in front because we minimize
            c = [-int(n) for n in content[1].split()]
            Q = np.zeros((nb_vars, nb_vars))
            for idx, elem in enumerate(content[2:]):
                Q[idx, :] = np.array([-float(n) for n in elem.split()])
        # Store upper triangular part of matrix in array
        Q_arr = deepcopy(Q)
        for i in range(0, nb_vars):     # divide only diagonal elements by 2 for BoxQP formulation
            Q_arr[i, i] /= 2.0
        Q_arr = Q_arr[np.triu_indices(nb_vars, k=0)]
        # Store matrix (divide Q by 2 for BoxQP formulation)
        Q = np.divide(np.array(Q), 2)
        # Store the sparsity pattern (0s and 1s) in Q_adj
        rows_cols = np.nonzero(Q)
        Q_adj = spmatrix(1.0, rows_cols[0], rows_cols[1], (nb_vars, nb_vars))
        self._Q, self._Q_adj, self._Q_arr = Q, Q_adj, Q_arr
        self._nb_vars, self._nb_lifted = nb_vars, nb_vars * (nb_vars + 1) // 2

        # Create CPLEX problem instance with lifted X variables corresponding
        # only to uper triangular coefficients in Q (due to Q symmetric and no explicit SDP constraint).
        # For sparse instances, variables X_ij with Q_ij=0 could be ommited, but needs sparse indexing logic.
        X_names = []
        for i in range(nb_vars):
            for j in range(i, nb_vars):
                X_names.append("X" + str(i) + str(j))
        my_prob = cplex.Cplex()
        my_prob.objective.set_sense(my_prob.objective.sense.minimize)
        obj_coeffs = list(Q_arr)
        obj_coeffs.extend(c)
        my_prob.variables.add(
            obj=obj_coeffs,
            lb=[0] * (nb_vars + self._nb_lifted),
            ub=[1] * (nb_vars + self._nb_lifted),
            names=[*X_names, *["x" + str(i) for i in range(nb_vars)]])
        # no CPLEX printing
        my_prob.set_results_stream(None)
        # my_prob.set_log_stream(sys.stdout)
        # my_prob.set_results_stream(sys.stdout)
        self._my_prob = my_prob

    def _add_mccormick_to_instance(self):
        """Get RLT/McCormick constraints constraints for CPLEX instance with variables in [0,1]^N domain
        """
        Q_adj, nb_vars, nb_lifted = self._Q_adj, self._nb_vars, self._nb_lifted
        coeffs_mck, rhs_mck, senses_mck = [], [], []
        for i in range(nb_vars):
            # Indices of Xii and xi in CPLEX variable array
            iXii, ixi = nb_vars * i - i * (i - 1) // 2, nb_lifted + i
            # Add diagonal RLT constraints
            coeffs_mck.extend([cplex.SparsePair(ind=[iXii, ixi], val=[1, -1]),
                               cplex.SparsePair(ind=[iXii, ixi], val=[-1, 2])])
            rhs_mck.extend([0, 1])
            senses_mck.extend(["L", "L"])
            # Add off-diagonal RLT constraints (only on off-diagonal terms with coefficient non-zero)
            for j in range(i + 1, nb_vars):
                iXij, ixj = iXii + j - i, ixi + j - i
                if Q_adj[i, j]:
                    coeffs_mck.extend([cplex.SparsePair(ind=[iXij, ixi, ixj], val=[-1, 1, 1]),
                                       cplex.SparsePair(ind=[iXij, ixi], val=[1, -1]),
                                       cplex.SparsePair(ind=[iXij, ixj], val=[1, -1])])
                    rhs_mck.extend([1, 0, 0])
                    senses_mck.extend(["L", "L", "L"])
        # Append all constraints at once to CPLEX instance
        self._my_prob.linear_constraints.add(lin_expr=coeffs_mck, rhs=rhs_mck, senses=senses_mck)

    def _get_sdp_vertex_cover(self, dim, ch_ext=0):
        """Implements a semidefinite vertex cover, finding the relevant index set, and its relevant sliced coefficients:
        - For chosen n-dimensionality (dim), using the sparsity pattern of Q (or its chordal extension for dim=3),
        build the index set P^E_dim (or P^bar(E)_3, bar(P*_3)) described in the manuscript.
        - Then for each element of the index set, aggregate sliced coefficients needed for further computations.
        """
        Q_arr, Q_adj, nb_vars = self._Q_arr, self._Q_adj, self._nb_vars
        self._agg_list = None   # if an old value from a previous instance is stored
        # If we consider chordal extensions in the sparsity pattern
        if ch_ext in [1, 2]:
            # Save backup of the Q sparsity pattern (without chordal extension)
            Q_adjc = deepcopy(Q_adj)
            # Sparsity pattern must include diagonal elements for chompack
            for i in range(nb_vars):
                Q_adj[i, i] = 1
            # Compute symbolic factorization using AMD ordering
            symb = cp.symbolic(Q_adj, p=amd.order)
            # Update Q_adj sparsity pattern with the chordal extension
            Q_adj = symb.sparsity_pattern(reordered=False, symmetric=True)
            self._Q_adj = Q_adj

        # Hold the list of indices for each subproblem in the vertex cover
        idx_list = []
        # Build list of indices (idx_list) according to dimensionality (dim)
        if dim == 3:
            # idx_list = P^E_3 or P^bar(E)_3
            # if chordal extension or no, select all unique cliques <=3 (that aren't subcliques) in Q_adj
            # with ordered elements e.g. (i1, i2, i3)
            if ch_ext in [0, 1]:
                for i1 in range(nb_vars):
                    for i2 in range(i1 + 1, nb_vars):
                        if Q_adj[i1, i2]:
                            triple_flag = False
                            # look forward (i3>i2) for triple clique to add
                            for i3 in range(i2 + 1, nb_vars):
                                if Q_adj[i1, i3] and Q_adj[i2, i3]:
                                    idx_list.append(([i1, i2, i3], 3))
                                    triple_flag = True
                            # look backward  (i3<i2) for triple clique to account for
                            # (already added in a previous iteration)
                            if not triple_flag:
                                for i3 in (set(range(i2)) - {i1}):
                                    if Q_adj[i1, i3] and Q_adj[i2, i3]:
                                        triple_flag = True
                                        break
                                # if no triple clique found add double clique
                                if not triple_flag:
                                    idx_list.append(([i1, i2], 2))
            # idx_list =  bar(P*_3)
            # Add only subsets of cliques in Q_adj (chordal ext) that are connected graphs in Q_adjc (no chordal ext.)
            # In practice: for triple (dim=3) cliques add clique in Q_adj if it has at least 2 edges in Q_adjc
            # otherwise add connected subset (double)
            elif ch_ext == 2:
                # logic is the same as for ch_ext=0 (P^E_3), but on the the chordal ext. Q_adj and with the
                # extra check for nb_edges>=2 (on the original Q_adjc)
                for i1 in range(nb_vars):
                    for i2 in range(i1 + 1, nb_vars):
                        if Q_adj[i1, i2]:
                            triple_flag = False
                            for i3 in range(i2 + 1, nb_vars):
                                if Q_adj[i1, i3] and Q_adj[i2, i3]:
                                    nb_edges = int(Q_adjc[i1, i2] + Q_adjc[i1, i3] + Q_adjc[i2, i3])
                                    if nb_edges >= 2:
                                        idx_list.append(([i1, i2, i3], 3))
                                        triple_flag = True
                            if not triple_flag:
                                for i3 in (set(range(i2)) - {i1}):
                                    nb_edges = int(Q_adjc[i1, i2] + Q_adjc[i1, i3] + Q_adjc[i2, i3])
                                    if Q_adj[i1, i3] and Q_adj[i2, i3] and nb_edges >= 2:
                                        triple_flag = True
                                        break
                                if not triple_flag and Q_adjc[i1, i2]:
                                    idx_list.append(([i1, i2], 2))
            # idx_list =  P^E+_3
            elif ch_ext == -1:
                for i1 in range(nb_vars):
                    for i2 in range(i1 + 1, nb_vars):
                        for i3 in range(i2 + 1, nb_vars):
                            idx_list.append(([i1, i2, i3], 3))
        # idx_list = P^E_4, extends logic for P^E_3 (see dim=3 case)
        elif dim == 4:
            for i1 in range(nb_vars):
                for i2 in range(i1 + 1, nb_vars):
                    if Q_adj[i1, i2]:
                        triple_flag = False
                        for i3 in range(i2 + 1, nb_vars):
                            if Q_adj[i1, i3] and Q_adj[i2, i3]:
                                quad_flag = False
                                triple_flag = True
                                for i4 in range(i3 + 1, nb_vars):
                                    if Q_adj[i1, i4] and Q_adj[i2, i4] and Q_adj[i3, i4]:
                                        idx_list.append(([i1, i2, i3, i4], 4))
                                        quad_flag = True
                                if not quad_flag:
                                    for i4 in (set(range(i3)) - {i1, i2}):
                                        if Q_adj[i1, i4] and Q_adj[i2, i4] and Q_adj[i3, i4]:
                                            quad_flag = True
                                            break
                                    if not quad_flag:
                                        idx_list.append(([i1, i2, i3], 3))
                        if not triple_flag:
                            for i3 in (set(range(i2)) - {i1}):
                                if Q_adj[i1, i3] and Q_adj[i2, i3]:
                                    triple_flag = True
                                    break
                            if not triple_flag:
                                idx_list.append(([i1, i2], 2))
        # idx_list = P^E_5, extends logic for P^E_3 (see dim=3 case)
        elif dim == 5:
            for i1 in range(nb_vars):
                for i2 in range(i1 + 1, nb_vars):
                    if Q_adj[i1, i2]:
                        triple_flag = False
                        for i3 in range(i2 + 1, nb_vars):
                            if Q_adj[i1, i3] and Q_adj[i2, i3]:
                                quad_flag = False
                                triple_flag = True
                                for i4 in range(i3 + 1, nb_vars):
                                    if Q_adj[i1, i4] and Q_adj[i2, i4] and Q_adj[i3, i4]:
                                        cinq_flag = False
                                        quad_flag = True
                                        for i5 in range(i4 + 1, nb_vars):
                                            if Q_adj[i1, i5] and Q_adj[i2, i5] and Q_adj[i3, i5] and Q_adj[i4, i5]:
                                                idx_list.append(([i1, i2, i3, i4, i5], 5))
                                                cinq_flag = True
                                        if not cinq_flag:
                                            for i5 in (set(range(i4)) - {i1, i2, i3}):
                                                if Q_adj[i1, i5] and Q_adj[i2, i5] and Q_adj[i3, i5] and Q_adj[i4, i5]:
                                                    cinq_flag = True
                                                    break
                                            if not cinq_flag:
                                                idx_list.append(([i1, i2, i3, i4], 4))
                                if not quad_flag:
                                    for i4 in (set(range(i3)) - {i1, i2}):
                                        if Q_adj[i1, i4] and Q_adj[i2, i4] and Q_adj[i3, i4]:
                                            quad_flag = True
                                            break
                                    if not quad_flag:
                                        idx_list.append(([i1, i2, i3], 3))
                        if not triple_flag:
                            for i3 in (set(range(i2)) - {i1}):
                                if Q_adj[i1, i3] and Q_adj[i2, i3]:
                                    triple_flag = True
                                    break
                            if not triple_flag:
                                idx_list.append(([i1, i2], 2))

        # Pool all needed information for each element in the vertex cover
        agg_list = [0] * len(idx_list)  # stores all info needed for the semidefinite vertex cover
        # Don't compute (agg_list) when it is possibly too large to store in RAM memory
        if len(agg_list) >= CutSolver._THRES_MAX_SUBS:
            return len(agg_list)
        for idx, (setInds, lenInds) in enumerate(idx_list):
            set_idxs = list(itertools.combinations_with_replacement(setInds, 2))
            Xarr_inds = [nb_vars * si[0] - si[0] * (si[0] + 1) // 2 + si[1] for si in set_idxs]
            # Slice symmetric matrix Q on indices set_idxs (only upper triangular part)
            Q_slice = itemgetter(*Xarr_inds)(Q_arr)
            # Bound input eigenvalues in [-1,1] via Lemma 4.1.2
            # max_elem can be 0 when using P^E+_3 subproblems, so set it to 1 then
            max_elem = lenInds * abs(max(Q_slice, key=abs))
            max_elem += 1 if not max_elem else 0
            Q_slice = np.divide(Q_slice, max_elem)
            agg_list[idx] = (setInds, Xarr_inds, tuple(Q_slice), max_elem)
        self._agg_list = agg_list
        return len(agg_list)

    def _sel_eigcut_by_ordering_on_measure(self, strat, vars_values, cut_round, sel_size=0):
        """Apply selection strategy to rank subproblems (feasibility/ optimality/ combined/ exact/ random/ figure 8)
        """
        nb_lifted, agg_list, Q, get_eigendecomp = self._nb_lifted, self._agg_list, self._Q, self._get_eigendecomp
        X_vals, x_vals = list(vars_values[0:nb_lifted]), list(vars_values[nb_lifted:])
        rank_list = [0] * len(agg_list)
        # Guard for selection size
        sel_size = min(sel_size, len(agg_list))
        feas_sel, opt_sel, exact_sel, comb_sel, rand_sel, figure_8 = \
            (strat == 1), (strat == 2), (strat == 3), (strat == 4), (strat == 5), (strat == -1)
        models = []
        # Build Mosek model templates for future use to speedup its execution for all subproblems
        if exact_sel or figure_8:
            for dim_act in [2, 3, 4, 5]:
                M = Model()
                M.setSolverParam("numThreads", 1)
                Z = M.variable("Z", dim_act + 1, Domain.inPSDCone())
                X = Z.slice([0, 0], [dim_act, dim_act])
                M.constraint(Z.index(dim_act, dim_act), Domain.equalsTo(1.))
                x_val = [0] * dim_act
                con1 = M.constraint(X.diag(), Domain.lessThan(x_val))
                con2 = M.constraint(Z.slice([0, dim_act], [dim_act, dim_act + 1]), Domain.equalsTo(x_val))
                Q_sub = np.zeros((dim_act, dim_act))
                Q_inds = np.triu_indices(dim_act, 0, dim_act)
                models.append((M, con1, con2, X, Q_sub, Q_inds))
        # If optimality selection involved
        if opt_sel or comb_sel or exact_sel:
            nns = self._nns
            for agg_idx, (set_inds, Xarr_inds, Q_slice, max_elem) in enumerate(agg_list):
                dim_act = len(set_inds)
                curr_pt = itemgetter(*set_inds)(x_vals)
                X_slice = itemgetter(*Xarr_inds)(X_vals)
                obj_improve = - sum(map(mul, Q_slice, X_slice)) * max_elem
                # Optimality selection via neural networks (alone or combined with feasibility)
                if opt_sel or comb_sel:
                    # Estimate objective improvement using neural network (after casting input to right ctype)
                    input_arr = nns[dim_act - 2][1]
                    input_arr[:dim_act] = curr_pt
                    input_arr[dim_act:] = Q_slice
                    obj_improve += nns[dim_act - 2][0](input_arr) * max_elem
                # Optimality selection via exact SDP solution (Mosek)
                elif exact_sel:
                    # Update Mosek model template for each subproblem
                    (M, con1, con2, X, Q_sub, Q_inds) = models[dim_act-2]
                    x_neg = np.negative(curr_pt)
                    x_pos = list(curr_pt)
                    # Update x
                    con1.add(x_neg)
                    con2.add(x_neg)
                    Q_sub[Q_inds] = Q_slice
                    M.objective(ObjectiveSense.Minimize, Expr.sum(Expr.mulElm(Q_sub, X)))
                    M.solve()
                    obj_improve += M.primalObjValue() * max_elem
                    # Roll back update of x
                    con1.add(x_pos)
                    con2.add(x_pos)
                rank_list[agg_idx] = (agg_idx, obj_improve, curr_pt, X_slice)
            # Order by expected objective improvement
            rank_list.sort(key=itemgetter(1), reverse=True)
            # Combined selection
            if comb_sel:
                strong_violated_cuts = 0
                violated_cuts = 0
                for ix, (agg_idx, obj_improve, curr_pt, X_slice) in enumerate(rank_list):
                    if obj_improve > CutSolver._THRES_MIN_OPT and strong_violated_cuts < sel_size:      # strong cut
                        # Check smallest eigenvalue for violated cuts
                        eigval = get_eigendecomp(len(curr_pt), curr_pt, X_slice, False)[0]
                        if eigval < CutSolver._THRES_NEG_EIGVAL:    # violated strong cut
                            rank_list[ix] = (agg_idx, obj_improve + CutSolver._BIG_M, curr_pt, X_slice)
                            strong_violated_cuts += 1
                            violated_cuts += 1
                        else:                                       # not violated cut
                            rank_list[ix] = (agg_idx, obj_improve - CutSolver._BIG_M, curr_pt, X_slice)
                    # if not enough strong violated cuts can be found, employ also selection by feasibility
                    elif strong_violated_cuts < sel_size:
                        eigval = get_eigendecomp(len(curr_pt), curr_pt, X_slice, False)[0]
                        if eigval < CutSolver._THRES_NEG_EIGVAL:    # violated cut
                            rank_list[ix] = (agg_idx, -eigval, curr_pt, X_slice)
                            violated_cuts += 1
                    else:
                        break
                # Sort again
                rank_list.sort(key=itemgetter(1), reverse=True)
                # if optimality cut selection finds no more violated cuts then a random selection,
                # then switch to feasibility selection alone for future rounds
                try:
                    return (1, rank_list) if strong_violated_cuts/sel_size < violated_cuts/len(rank_list)\
                        else (strat, rank_list)
                except ZeroDivisionError:
                    a=1
        # Random selection
        elif rand_sel:
            # random ordering shuffle with seed
            np.random.shuffle(agg_list)
            rank_list = agg_list
        # Feasibility (-only) selection
        elif feas_sel:
            get_eigendecomp = self._get_eigendecomp
            nb_violated = 0
            # Rank by eigenvalues (when negative)
            for agg_idx, (set_inds, Xarr_inds, _, _) in enumerate(agg_list):
                dim_act = len(set_inds)
                curr_pt = itemgetter(*set_inds)(x_vals)
                X_slice = itemgetter(*Xarr_inds)(X_vals)
                eigval = get_eigendecomp(dim_act, curr_pt, X_slice, False)[0]
                if eigval < CutSolver._THRES_NEG_EIGVAL:
                    rank_list[agg_idx] = (set_inds, -eigval, Xarr_inds, dim_act)
                    nb_violated += 1
                else:
                    rank_list[agg_idx] = (0, 0)
            rank_list.sort(key=itemgetter(1), reverse=True)
            rank_list = rank_list[0:nb_violated]

        ####################
        # Special separate case for Figure 8
        # Comparison of exact vs estimated (neural net) measures for optimality cut selection
        ####################
        elif figure_8:
            exact_list, this_round_cuts, nb_cuts_sel_by_both = [], [], 0
            nns = self._nns
            for agg_idx, (set_inds, Xarr_inds, Q_slice, max_elem) in enumerate(agg_list):
                dim_act = len(set_inds)
                curr_pt = itemgetter(*set_inds)(x_vals)
                X_slice = itemgetter(*Xarr_inds)(X_vals)
                curr_obj = sum(map(mul, Q_slice, X_slice)) * max_elem
                # optimality selection via estimated (neural network) ordering
                input_arr = nns[dim_act - 2][1]
                input_arr[:dim_act] = curr_pt
                input_arr[dim_act:] = Q_slice
                obj_improve = nns[dim_act - 2][0](input_arr) * max_elem - curr_obj
                rank_list[agg_idx] = (agg_idx, obj_improve, curr_pt, X_slice)
                # optimality based exact ordering (for comparison)
                (M, con1, con2, X, Q_sub, Q_inds) = models[dim_act - 2]
                x_neg = np.negative(curr_pt)
                x_pos = list(curr_pt)
                con1.add(x_neg)
                con2.add(x_neg)
                Q_sub[Q_inds] = Q_slice
                M.objective(ObjectiveSense.Minimize, Expr.sum(Expr.mulElm(Q_sub, X)))
                M.solve()
                obj_improve_exact = M.primalObjValue() * max_elem - curr_obj
                con1.add(x_pos)
                con2.add(x_pos)
                exact_list.append((agg_idx, obj_improve_exact))
            # Sort by estimated optimality measure
            rank_list.sort(key=itemgetter(1), reverse=True)
            # Sort by exact optimality measure
            exact_list.sort(key=itemgetter(1), reverse=True)
            std_dev_exact = np.std(np.array([obj_improve_exact for
                                      (_,obj_improve_exact) in exact_list[0:sel_size]]))
            for estim_idx, cut in enumerate(rank_list):
                cut_idx = cut[0]
                exact_idx = [elem[0] for elem in exact_list].index(cut_idx)
                sel_by_estim = 1 if estim_idx < sel_size else 0  # if cut is selected by estimated measure
                sel_by_exact = 1 if exact_idx < sel_size else 0  # if cut is selected by exact measure
                this_round_cuts.append([cut_round, cut_idx, sel_by_estim, sel_by_exact,
                                        cut[1], exact_list[exact_idx][1]])
                if sel_by_estim and sel_by_exact:
                    nb_cuts_sel_by_both += 1
            return rank_list, nb_cuts_sel_by_both / sel_size, std_dev_exact, this_round_cuts
        return rank_list

    def _gen_eigcuts_selected(self, strat, sel_size, rank_list, strong_only=False, vars_values=None):
        """Adds eigenvalue cuts for the subproblems selected by a particular strategy. This involves fetching the
        eigen-decomposition info needed for each subproblem (pre-calculated or not for speed depending on strategy)
        and then using this info to generate the actual cuts (the same way irrespective of selection strategy).
        """
        nb_subprobs, my_prob, nb_lifted = len(self._agg_list), self._my_prob, self._nb_lifted
        nb_sdp_cuts, coeffs_sdp, rhs_sdp, senses_sdp = 0, [], [], []
        opt_sel, feas_sel, rand_sel = (strat in [2, 3, 4, -1]), (strat == 1), (strat == 5)
        sel_size = min(sel_size, len(rank_list))  # extra condition for feasibility selection
        if not opt_sel:
            X_vals, x_vals = list(vars_values[0:nb_lifted]), list(vars_values[nb_lifted:])
        get_eigendecomp = self._get_eigendecomp
        ix = 0
        while nb_sdp_cuts < sel_size and ix < sel_size:
            # Get eigen-decomposition info needed, pre-calculated or not depending on the selection strategy used
            if not feas_sel:
                if opt_sel:
                    (idx, diff, curr_pt, X_slice) = rank_list[ix]
                    # for Figs. 11-12, optimality-only strategies add only violated cuts within the selection size
                    # associated with positive optimality measure (that predicts their violation).
                    if strong_only and diff <= 0:
                        break
                    (set_inds, Xarr_inds) = self._agg_list[idx][0:2]
                else:  # random selection
                    (set_inds, Xarr_inds, Q_slice, max_elem) = rank_list[ix]
                    curr_pt = itemgetter(*set_inds)(x_vals)
                    X_slice = itemgetter(*Xarr_inds)(X_vals)
                dim_act = len(set_inds)
            else:
                (set_inds, _, Xarr_inds, dim_act) = rank_list[ix]
                curr_pt = itemgetter(*set_inds)(x_vals)
                X_slice = itemgetter(*Xarr_inds)(X_vals)
            eigvals, evecs = get_eigendecomp(dim_act, curr_pt, X_slice, True)
            evect = evecs.T[0]
            #################
            # Form eigcuts based on the eigendecomposition of the subproblems selected
            # Note: The eigcuts are formed the same way irrespective of the selection strategy!
            #################
            if eigvals[0] < CutSolver._THRES_NEG_EIGVAL:
                evect = np.where(abs(evect) <= -CutSolver._THRES_NEG_EIGVAL, 0, evect)
                evect_arr = [evect[idx1] * evect[idx2] * 2 if idx1 != idx2 else evect[idx1] * evect[idx2]
                             for idx1 in range(dim_act + 1) for idx2 in range(max(idx1, 1), dim_act + 1)]
                coeffs_sdp.append(cplex.SparsePair(
                    ind=[x + nb_lifted for x in set_inds] + Xarr_inds, val=evect_arr))
                rhs_sdp.append(-evect[0] * evect[0])
                senses_sdp.append("G")
                nb_sdp_cuts += 1
            ix += 1
        # Add all SDP eigen-cuts at once to problem instance
        my_prob.linear_constraints.add(lin_expr=coeffs_sdp, rhs=rhs_sdp, senses=senses_sdp)
        return nb_sdp_cuts

    def __gen_dense_eigcuts(self, vars_values=None):
        """Adds dense eigenvalue cuts
        """
        my_prob, nb_lifted = self._my_prob, self._nb_lifted
        nb_sdp_cuts, coeffs_sdp, rhs_sdp, senses_sdp = 0, [], [], []
        X_vals, x_vals = list(vars_values[0:nb_lifted]), list(vars_values[nb_lifted:])
        dim_act = len(x_vals)
        mat = np.zeros((dim_act+1, dim_act+1))
        mat[0, 0] = 1
        mat[0, 1:] = x_vals
        inds = (np.array([1 + x for x in np.triu_indices(dim_act, 0, dim_act)[0]]),
                       np.array([1 + x for x in np.triu_indices(dim_act, 0, dim_act)[1]]))
        mat[inds] = X_vals
        # np.linalg.eigh(/eigvalsh) return eigenvalues in ascending order
        eigvals, evecs = np.linalg.eigh(mat, "U")
        ix = 0
        while ix < dim_act:
            if eigvals[ix] < CutSolver._THRES_NEG_EIGVAL:
                evect = evecs.T[ix]
                evect_arr = [evect[idx1] * evect[idx2] * 2 if idx1 != idx2 else evect[idx1] * evect[idx2]
                             for idx1 in range(dim_act + 1) for idx2 in range(max(idx1, 1), dim_act + 1)]
                coeffs_sdp.append(cplex.SparsePair(
                    ind=[x + nb_lifted for x in range(0, dim_act)] + list(range(0, nb_lifted)), val=evect_arr))
                rhs_sdp.append(-evect[0] * evect[0])
                senses_sdp.append("G")
                nb_sdp_cuts += 1
            ix += 1
        # Add all SDP eigen-cuts at once to problem instance
        my_prob.linear_constraints.add(lin_expr=coeffs_sdp, rhs=rhs_sdp, senses=senses_sdp)
        return nb_sdp_cuts

    def _get_eigendecomp(self, dim_subpr, curr_pt, X_slice, ev_yes):
        """Get eigen-decomposition of a matrix of type [1, x^T; x, X] where x=(curr_pt), X=(X_slice),
        with/(out) eigenvectors (ev_yes)
        """
        mat = self._Mat
        mat[dim_subpr - 2][0, 1:] = curr_pt
        mat[dim_subpr - 2][self._inds[dim_subpr - 2]] = X_slice
        # np.linalg.eigh(/eigvalsh) return eigenvalues in ascending order
        return np.linalg.eigh(mat[dim_subpr - 2], "U") if ev_yes \
            else np.linalg.eigvalsh(mat[dim_subpr - 2], "U")

    def __preprocess_triangle_ineq(self):
        """Find which variable aggregations are to be considered as triangle inequalities at each cut round
        based on the sparsity pattern of the instance. This pre-processing is done before cut rounds start.
        """
        nb_vars, Q_adj = self._nb_vars, self._Q_adj
        self._rank_list_tri, self._idx_list_tri = (None, None)
        rank_list_tri, idx_list_tri = [], []
        ix_agg = 0
        for i1 in range(nb_vars):
            for i2 in range(i1 + 1, nb_vars):
                for i3 in range(i2 + 1, nb_vars):
                    density = Q_adj[i1, i2] + Q_adj[i1, i3] + Q_adj[i2, i3]
                    if density >= CutSolver._THRES_TRI_DENSE:
                        set_inds = [i1, i2, i3]
                        set_idxs = list(itertools.combinations_with_replacement(set_inds, 2))
                        Xarr_inds = [nb_vars * si[0] - si[0] * (si[0] + 1) // 2 + si[1] for si in set_idxs]
                        # rank_list_tri holds for each variable aggregation: the idx of the aggregation
                        # the triangle cut type (0-3), the density of the aggregation, the violation (to be computed)
                        rank_list_tri.extend([[ix_agg, tri_cut_type, density, 0] for tri_cut_type in range(4)])
                        # idx_list_tri holds the index lists for each variable aggregation
                        idx_list_tri.append((Xarr_inds, set_inds))
                        ix_agg += 1
        self._rank_list_tri, self._idx_list_tri = rank_list_tri, idx_list_tri

    def __separate_and_add_triangle(self, sel_size, vars_values):
        """Separate and add a given selection size of triangle cuts given current solution as described in manuscript
        """
        rank_list_tri, idx_list_tri, my_prob, nb_lifted = \
            self._rank_list_tri, self._idx_list_tri, self._my_prob, self._nb_lifted
        X_vals, x_vals = list(vars_values[0:nb_lifted]), list(vars_values[nb_lifted:])
        # Separate triangle inequalities
        for idx in range(len(idx_list_tri)):
            (Xarr_inds, set_inds) = idx_list_tri[idx]
            X_slice = itemgetter(*Xarr_inds)(X_vals)
            curr_pt = itemgetter(*set_inds)(x_vals)
            X1, X2, X4 = X_slice[1], X_slice[2], X_slice[4]
            rank_list_tri[idx * 4][3] = X1 + X2 - X4 - curr_pt[0]
            rank_list_tri[idx * 4 + 1][3] = X1 - X2 + X4 - curr_pt[1]
            rank_list_tri[idx * 4 + 2][3] = -X1 + X2 + X4 - curr_pt[2]
            rank_list_tri[idx * 4 + 3][3] = -X1 - X2 - X4 + sum(curr_pt) - 1
        # Remove non-violated constraints and sort by density first and then violation second as in manuscript
        rank_list_tri_viol = [el for el in rank_list_tri if el[3] >= CutSolver._THRES_TRI_VIOL]
        rank_list_tri_viol.sort(key=itemgetter(2, 3), reverse=True)
        # Determine thresholded (upper & lower) number of triangle cuts to add (proportional to selection size of SDP)
        nb_tri_cuts = max(min(CutSolver._TRI_CUTS_PER_ROUND_MIN, int(np.floor(sel_size * len(rank_list_tri_viol)))),
                          min(CutSolver._TRI_CUTS_PER_ROUND_MAX, len(rank_list_tri_viol)))
        coeffs_tri, rhs_tri, senses_tri = [0]*nb_tri_cuts, [0]*nb_tri_cuts, ["G"]*nb_tri_cuts
        dict_coeffs_tri = {0: [-1, -1, 1, 1], 1: [-1, 1, -1, 1], 2: [1, -1, -1, 1], 3: [1, 1, 1, -1, -1, -1]}
        for ix in range(nb_tri_cuts):
            (ix2, ineq_type, _, _) = rank_list_tri_viol[ix]
            (Xarr_inds, set_inds) = idx_list_tri[ix2]
            # Generate constraints for the 4 different triangle inequality types
            if ineq_type == 3:
                coeffs_tri[ix] = cplex.SparsePair(
                    ind=[Xarr_inds[1], Xarr_inds[2], Xarr_inds[4], set_inds[0] + nb_lifted,
                         set_inds[1] + nb_lifted, set_inds[2] + nb_lifted], val=dict_coeffs_tri[ineq_type])
                rhs_tri[ix] = -1
            else:
                coeffs_tri[ix] = cplex.SparsePair(
                    ind=[Xarr_inds[1], Xarr_inds[2], Xarr_inds[4], set_inds[ineq_type] + nb_lifted],
                    val=dict_coeffs_tri[ineq_type])
                rhs_tri[ix] = 0
        # Add all triangle cuts to relaxation
        my_prob.linear_constraints.add(lin_expr=coeffs_tri, rhs=rhs_tri, senses=senses_tri)
        return nb_tri_cuts

