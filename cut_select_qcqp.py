from cut_select_qp import CutSolver
from operator import itemgetter
import numpy as np
import lxml.etree as parser
import os.path
import cplex
import warnings
warnings.filterwarnings("ignore")


class CutSolverQCQP(CutSolver):
    def __init__(self):
        super().__init__()
        self._Q_adj_cons = None

    def cut_select_algo(self, filename, dim, sel_size=0.1, strat=2, nb_rounds_cuts=20):
        """Implements cut selection strategies, adapting Algorithm 1 to QCQP, as explained in the manuscript.
        :param filename: file for powerflow instance (after Antigone pre-processing for bounds/eliminating variables)
        :param dim: dimension of SDP decomposition and eigcuts
        :param sel_size: selection size (percent) of eigcuts
        :param strat: selection strategy that ranks/orders sub-problems (and associated eigcuts) to select
        (1-feasibility, 2-combined sel. one constraint at a time, 3-combined sel. all constraints aggregated, 4-random)
        :param nb_rounds_cuts: number of cut rounds
        :return: solutions across cut rounds at termination (containing bounds and cut statistics)
        """
        assert (strat in [1, 2, 3, 4, 5]), "Pick a violated cut selection strategy (1-5)!"
        assert (0 < sel_size), "The selection size must be a % or number (of cuts) >0!"
        assert (dim <= 5), "Keep SDP vertex cover low-dimensional (<=5)!"
        feas_sel, opt_sel, exact_sel, comb_sel, rand_sel = \
            (strat == 1), (strat == 2), (strat == 3), (strat == 4), (strat == 5)

        nbs_opt_cuts = [0] * (nb_rounds_cuts+1)
        nbs_sdp_cuts = [0]
        self._dim = dim
        # Parse QCQP .osil instances and create CPLEX problem instance with lifted variables
        self.__parse_qcqp_osil_into_cplex(filename)
        my_prob = self._my_prob
        # Add McCorcmick M constraints
        super()._add_mccormick_to_instance()
        if not rand_sel or feas_sel:
            super()._load_neural_nets()

        obj_values_rounds = []
        # Cut Round 0 - Solve McCormick relaxation M
        my_prob.solve()
        obj_values_rounds.append(my_prob.solution.get_objective_value())
        vars_values = np.array(my_prob.solution.get_values())

        # Get the info needed from an SDP vertex cover (from obj + constraints)
        agg_list_cons = self.__get_vertex_cover(dim)
        agg_list = self._agg_list[:]
        nb_subprobs = len(agg_list)
        # Interpret selection size ar % or absolute number and threshold the maximum number of SDP cuts per round
        sel_size = min(int(np.floor(sel_size * nb_subprobs)) if sel_size < 1 else min(sel_size, nb_subprobs),
                       CutSolver._SDP_CUTS_PER_ROUND_MAX)
        # Minimum of 1 cut selected each round 
        if sel_size == 0:
            sel_size = 1
        strat_old = strat
        ##############################
        # Start rounds of cuts (with selection and generation of linear SDP cuts)
        ##############################
        for cut_round in range(1, nb_rounds_cuts + 1):
            if rand_sel:    # random selection
                rank_list = \
                    super()._sel_eigcut_by_ordering_on_measure(strat, vars_values, cut_round)
            else:
                if comb_sel:    # combined selection
                    strat, rank_list_comb_obj = \
                        super()._sel_eigcut_by_ordering_on_measure(strat, vars_values, cut_round, sel_size=sel_size)
                else:       # feasibility or optimality-only selection
                    rank_list_comb_obj = \
                        super()._sel_eigcut_by_ordering_on_measure(strat, vars_values, cut_round)
                # Swap the objective vertex cover with the constraints one
                self._agg_list = agg_list_cons
                rank_list_feas_cons = super()._sel_eigcut_by_ordering_on_measure(1, vars_values, cut_round)
                # Swap back
                self._agg_list = agg_list
                rank_list = (rank_list_comb_obj + rank_list_feas_cons)[0:sel_size]
            if feas_sel or rand_sel:
                # Generate eigen-cuts from selected subproblems
                nb_sdp_cuts = self._gen_eigcuts_selected(strat, sel_size, rank_list, vars_values=vars_values)
            else:
                # Count number of cuts selected via optimality measure
                nb_opt_cuts = 0
                for elem in rank_list_comb_obj:
                    nb_opt_cuts += elem[1] > super()._BIG_M
                nbs_opt_cuts[cut_round] = nb_opt_cuts
                # Count number of cuts selected via combined measure
                nb_cuts_combined = 0
                for elem in rank_list:
                    nb_cuts_combined += isinstance(elem[0], int)
                # separate cut generation into that for the combined measure and that for the feasibility measure
                nb_cuts_comb = self._gen_eigcuts_selected(1, sel_size-nb_cuts_combined, rank_list_feas_cons[0:(sel_size-nb_cuts_combined)],
                                                          vars_values=vars_values)
                nb_cuts_feas = self._gen_eigcuts_selected(strat_old, nb_cuts_combined, rank_list_comb_obj[0:nb_cuts_combined],
                                                          vars_values=vars_values)
                nb_sdp_cuts = nb_cuts_comb + nb_cuts_feas
            # Update strategy if it changed from combined to feasibility only
            feas_sel = (strat == 1)
            comb_sel = (strat == 4)
            strat_old = strat
            nbs_sdp_cuts.append(nb_sdp_cuts)
            # End of cuts separation for this round
            ###################

            # Solve relaxation augmented with cuts at current round
            my_prob.solve()
            # Store new value of objective
            obj_values_rounds.append(my_prob.solution.get_objective_value())
            # Store new values of all variables (unlifted x and lifted X)
            vars_values = np.array(my_prob.solution.get_values()).astype(float)
        return obj_values_rounds, sel_size, nbs_sdp_cuts, nbs_opt_cuts

    def __parse_qcqp_osil_into_cplex(self, filename):
        """Parsing osil file for QCQP instances
        """
        dirname = os.path.join(os.path.dirname(__file__), "qcqp_instances", filename + ".osil")
        # Parse xml in a tree structure
        expr_tree = parser.iterparse(dirname)
        for _, el in expr_tree:
            el.tag = el.tag.split('}', 1)[1]  # strip all namespaces
        tree = expr_tree.root
        # Get variables xml element
        vars_xml = tree.find('instanceData/variables')
        # Number of linear variables
        nb_vars = int(vars_xml.attrib['numberOfVariables'])
        self._nb_vars, self._nb_lifted = nb_vars, nb_vars * (nb_vars + 1) // 2
        nb_lifted = self._nb_lifted

        ######################
        #  Parse OSIL objective
        ######################
        # Get objective xml element
        obj_xml = tree.find('instanceData/objectives/obj')
        # Linear part of objective
        c = [0]*nb_vars
        obj_const = 0
        if 'constant' in obj_xml.attrib:
            obj_const += float(obj_xml.attrib['constant'])
        obj_lin_coeffs = np.array([float(coeff.text) for coeff in obj_xml.findall('coef')])
        if obj_xml.attrib['maxOrMin'] == 'min':
            # for all indexed (by x_idx) variables x that participate in linear objective terms)
            for idx, x_idx in enumerate(np.array([int(coeff.attrib['idx']) for coeff in obj_xml.findall('coef')])):
                c[x_idx] = obj_lin_coeffs[idx]
        ######################
        #  Parse OSIL constraints
        ######################
        # Get constraints xml element
        con_xml = tree.find('instanceData/constraints')
        nb_cons = 0
        if con_xml is not None:
            # Number of constraints (linear and quadratic)
            nb_cons = int(con_xml.attrib['numberOfConstraints'])
            # Type and right hand side constant of constraints
            cons_sgn_rhs = [0] * nb_cons
            # ub (upper-bound) and lb (lower-bound) are assumed equal when both present
            # Values for both ub and lb, but sometimes only for one (to signal inequality type)
            for idx, con in enumerate(con_xml.findall('con')):
                lb, ub = (None, None)
                try:
                    lb = con.attrib['lb']
                except KeyError:
                    pass
                try:
                    ub = con.attrib['ub']
                except KeyError:
                    pass
                if lb and ub:
                    cons_sgn_rhs[idx] = (0, float(lb))  # equality
                elif lb:
                    cons_sgn_rhs[idx] = (1, float(lb))  # >= ineq
                else:
                    cons_sgn_rhs[idx] = (2, float(ub))  # <= ineq

            # LINEAR CONSTRAINTS
            con_lin_xml = tree.find('instanceData/linearConstraintCoefficients')
            nb_lin_entries = int(con_lin_xml.attrib['numberOfValues'])
            # Create array of all indices ordered column-major
            lin_col_idx = np.zeros(nb_lin_entries)
            elem_idx = 0
            for idx, el in enumerate(con_lin_xml.find('colIdx').findall('el')):
                mult = 1
                incr = 1
                try:
                    mult = int(el.attrib['mult'])
                except KeyError:
                    pass
                try:
                    incr = int(el.attrib['incr'])
                except KeyError:
                    pass
                start_idx = int(el.text)
                if mult == 1:
                    lin_col_idx[elem_idx] = start_idx
                else:
                    lin_col_idx[elem_idx:elem_idx + mult] = np.arange(start_idx, start_idx + mult * incr, incr)
                elem_idx += mult
            # Create array of all values ordered column-major
            lin_col_val = np.zeros(nb_lin_entries)
            elem_idx = 0
            for idx, el in enumerate(con_lin_xml.find('value').findall('el')):
                mult = 1
                try:
                    mult = int(el.attrib['mult'])
                except KeyError:
                    pass
                lin_col_val[elem_idx:elem_idx + mult] = np.ones(mult) * float(el.text)
                elem_idx += mult
            # Create array with start index for each linear constraint
            con_start_list = [0] * (nb_cons + 1)
            elem_idx = 0
            for idx, el in enumerate(con_lin_xml.find('start').findall('el')):
                con_start_list[elem_idx] = int(el.text)
                try:
                    mult = int(el.attrib['mult'])
                except KeyError:
                    elem_idx += 1
                    continue
                try:
                    incr = int(el.attrib['incr'])
                except KeyError:
                    incr = 0
                for row_idx in range(1, mult):
                    con_start_list[elem_idx + row_idx] = con_start_list[elem_idx + row_idx - 1] + incr
                elem_idx += mult

        # QUADRATIC TERMS
        con_quad_xml = tree.find('instanceData/quadraticCoefficients')
        nb_quad_terms = int(con_quad_xml.attrib['numberOfQuadraticTerms'])
        # Find quadratic terms and their associated x variables, coefficients, and constraints to which they belong
        quad_term_list = [0] * nb_quad_terms
        quad_term_to_cons = np.zeros(nb_quad_terms)
        for idx, el in enumerate(con_quad_xml.findall('qTerm')):
            x_idxs = [int(el.attrib['idxOne']), int(el.attrib['idxTwo'])]
            quad_term_list[idx] = [int(el.attrib['idx']), *x_idxs, float(el.attrib['coef'])]
            # store constraint idx that quad term belongs to
            quad_term_to_cons[idx] = int(el.attrib['idx'])
        quad_term_to_cons = list(np.unique(np.sort(quad_term_to_cons)))

        # Form adjacency matrices for only objective and for all constraints + objective
        Q_adj = np.zeros((nb_vars, nb_vars))
        Q_adj_cons = np.zeros((nb_vars, nb_vars))
        # Form coeffs matrix for only objective
        Q = np.zeros((nb_vars, nb_vars))

        for q_idx, quad_term in enumerate(quad_term_list):
            if quad_term[0] == -1:  # only if objective adjacency
                Q_adj[quad_term[1], quad_term[2]] = 1
                Q_adj[quad_term[2], quad_term[1]] = 1
                Q[quad_term[1], quad_term[2]] = quad_term[3]
                Q[quad_term[2], quad_term[1]] = quad_term[3]
            Q_adj_cons[quad_term[1], quad_term[2]] = 1
            Q_adj_cons[quad_term[2], quad_term[1]] = 1
        self._Q = Q
        Q_arr = Q[np.triu_indices(nb_vars, k=0)]
        self._Q_adj = Q_adj
        self._Q_adj_cons = Q_adj_cons
        self._Q_arr = Q_arr
        self._nb_cons = nb_cons

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

        ######################
        # Add all constraints (linear or linearized quadratic via lifting)
        ######################
        dict_con_sign = {0: "E", 1: "G", 2: "L"}
        cons_expr = []
        consts_vals = []
        senses_arr = []
        # Go through each constraint
        for con_idx in range(0, nb_cons):
            con_inds = []
            con_coeffs = []
            ineq_sign, const_val = cons_sgn_rhs[con_idx]  # type of constraint and rhs constant
            # Quadratic part for any lifted quadratic constraint
            if con_idx in quad_term_to_cons:
                for q_idx, quad_term in enumerate(quad_term_list):
                    # If quad term is in constraint
                    if quad_term[0] == con_idx:
                        si, coeff = quad_term[1:3], quad_term[3]  # quad term 1st and 2nd x index and coefficient
                        con_inds.append(nb_vars * si[0] - si[0] * (si[0] + 1) // 2 + si[1])
                        con_coeffs.append(coeff)
            # Linear part for any constraint, with re-scaling of x_idx variable
            for term_idx in range(con_start_list[con_idx], con_start_list[con_idx + 1]):
                x_idx = int(lin_col_idx[term_idx])
                con_inds.append(x_idx + nb_lifted)
                con_coeffs.append(lin_col_val[term_idx])
            cons_expr.append(cplex.SparsePair(ind=con_inds, val=con_coeffs))
            consts_vals.append(const_val)
            senses_arr.append(dict_con_sign[ineq_sign])
        my_prob.linear_constraints.add(lin_expr=cons_expr, rhs=consts_vals, senses=senses_arr)

    def __get_vertex_cover(self, dim):
        """Get semidefinite vertex cover of given dimensionality for the objective-only adjacency matrix ( P_n^(E_0) )
        and for the obj+constraints adjacency matrix ( P_n^(E_0) )
        """
        # Get vertex cover for objective-only with the needed coeeficients
        super()._get_sdp_vertex_cover(dim)
        # Backup Q_adj, agg_list
        Q_adj = self._Q_adj[:]
        agg_list = self._agg_list[:]
        # Swap the adjacency matrix with the obj+constraints one
        self._Q_adj = self._Q_adj_cons
        # Get vertex cover for all constraints (for feasibility Q coefficients don't matter)
        super()._get_sdp_vertex_cover(dim)
        agg_list_cons = self._agg_list[:]
        # Swap back
        self._Q_adj = Q_adj
        # Set intersect P_n^(E_m) with P_n^(E_0)
        self._agg_list = [el for el in agg_list_cons if el in agg_list]
        # Set difference P_n^(E_m)- (intersection P_n^(E_0),P_n^(E_m))
        agg_list_cons = [el for el in agg_list_cons if el not in self._agg_list]
        return agg_list_cons
