from cut_select_qp import CutSolver
import itertools
from operator import itemgetter, mul
import numpy as np
import cvxpy as cvx
import lxml.etree as parser
import os.path
import warnings
warnings.filterwarnings('ignore')


class CutSolverQCQP(CutSolver):
    def __init__(self):
        super().__init__()
        self._nb_x_unlifted = 0
        self._dim_lifted = 0
        # Utility lists for parsing OSIL and making sense of constraints
        self._quad_term_list = []
        self._cons_quad_list = []
        # CVX constraints and objective
        self._constraints = []
        self._objective = []
        # CVX unlifted and lifted variables
        self._x = None
        self._X = None

    def cut_select_algo(self, filename, dim, sel_size=12, strat=2, nb_rounds_cuts=20):
        """Implements cut selection strategies, adapting Algorithm 1 to QCQP, as explained in the manuscript.
        :param filename: file for powerflow instance (after Antigone pre-processing for bounds/eliminating variables)
        :param dim: dimension of SDP decomposition and eigcuts
        :param sel_size: selection size (number) of eigcuts
        :param strat: selection strategy that ranks/orders sub-problems (and associated eigcuts) to select
        (1-feasibility, 2-combined sel. one constraint at a time, 3-combined sel. all constraints aggregated, 4-random)
        :param nb_rounds_cuts: number of cut rounds
        :return: solutions across cut rounds at termination (containing bounds and cut statistics)
        """
        assert (strat in [1, 2, 3, 4]), "Please pick a valid cut selection strategy!"
        feas_sel, comb_one_cons, comb_all_cons, rand_sel = \
            (strat == 1), (strat == 2), (strat == 3), (strat == 4)
        obj_values_rounds = [0] * (nb_rounds_cuts + 1)
        nb_cuts_opt = [0] * nb_rounds_cuts
        self._dim = dim

        # Load neural networks for optimality selection
        super()._load_neural_nets()
        nns = self._nns

        # Parse powerflow osil instance (pre-processed by Antigone for bounds)
        self.__parse_powerflow_osil_into_cvx(filename)

        # Add McCorcmick M^2 constraints (linear + SOCP)
        self.__add_mccormick_to_instance()
        x, X, obj_func, constraints = (self._x, self._X, self._objective, self._constraints)
        objective = cvx.Minimize(obj_func)

        # Get the info needed from an SDP decomposition (info is stored for each relevant constraint)
        self.__get_sdp_decomposition(dim)
        agg_list = self._agg_list

        # Round 0 - solve problem instance with McCormick and SOCP constraints added
        prob = cvx.Problem(objective, constraints)
        value = prob.solve(verbose=False, solver=cvx.ECOS)
        # print(str(value))
        obj_values_rounds[0] = value

        ##############################
        # Start rounds of cuts (with selection and generation of linear SDP cuts)
        ##############################
        for cut_round in range(1, nb_rounds_cuts + 1):
            x_vals = list(np.array(x.value)[0])
            X_vals = np.array(X.value)
            rank_list = []
            # For each sub-problem rho
            for idx, (clique, set_idxs, inputNNs) in enumerate(agg_list):
                dim_act = len(clique)
                curr_pt = itemgetter(*clique)(x_vals)
                X_slice = itemgetter(*set_idxs)(X_vals)
                obj_improve = 0
                # Get eigenvalues
                eigvals = super()._get_eigendecomp(dim_act, curr_pt, X_slice, False)
                # Feasibility selection
                if feas_sel:
                    obj_improve = -eigvals[0]
                # Random selection
                elif rand_sel:
                    obj_improve = np.random.random_sample()
                # A variant of combined selection
                elif eigvals[0] < CutSolver._THRES_NEG_EIGVAL:
                    # Flag indicating whether a cut can be selected by the optimality measure
                    sel_by_opt = False
                    # Combined selection with optimality mesures on one constraint at a time all added up
                    if comb_one_cons:
                        for idx2, (input_nn, max_elem, con_idx) in enumerate(inputNNs):
                            mu = prob.constraints[con_idx].dual_value  # Lagrange multiplier
                            # If mu negative reverse sense of constraint coefficients inputted in a neural net
                            # since an improvement in the objective is linked to positive mu
                            input_nn = [-i if mu < 0 else i for i in input_nn]
                            estim = nns[dim_act - 2][0](nns[dim_act - 2][1](*(list(curr_pt) + input_nn))) - \
                                    sum(map(mul, input_nn, X_slice))
                            if estim > CutSolver._THRES_MIN_OPT:
                                obj_improve += estim * max_elem * abs(mu)
                                sel_by_opt = True
                    # Combined selection with one optimality measure on all constraints aggregated
                    elif comb_all_cons:
                        input_nn_sum = [0] * len(set_idxs)
                        # Sum up coefficient of X_rho varaibles from all relevant constraints
                        # (accounting for mu and rescaling)
                        for idx2, (input_nn, max_elem, con_idx) in enumerate(inputNNs):
                            mu = prob.constraints[con_idx].dual_value  # Lagrange multiplier
                            input_nn_sum = [el + input_nn[idx] * max_elem * mu for idx, el in enumerate(input_nn_sum)]
                        # Remove elements that can result from numerical innacuracies
                        input_nn_sum = [i if abs(i) < 10**-5 else 0 for i in input_nn_sum]
                        # Bound domains of eigenvalues/coefficients to [-1,1] via Lemma 4.1.2
                        max_elem = len(clique) * abs(max(input_nn_sum, key=abs))
                        max_elem = max_elem if max_elem > 0 else 1
                        input_nn_sum = list(np.divide(input_nn_sum, max_elem))
                        estim = nns[dim_act - 2][0](nns[dim_act - 2][1](*(list(curr_pt) + input_nn_sum))) - \
                                sum(map(mul, input_nn_sum, X_slice))
                        if estim > CutSolver._THRES_MIN_OPT:
                            obj_improve += estim * max_elem
                            sel_by_opt = True
                    if sel_by_opt:      # If strong cut is selection by an optimality measure
                        obj_improve += CutSolver._BIG_M

                    else:               # If cut not strong but valid
                        obj_improve = -eigvals[0]
                rank_list.append((idx, obj_improve, curr_pt, X_slice, dim_act))
            # Sort sub-problems rho by measure
            rank_list.sort(key=lambda tup: tup[1], reverse=True)

            nb_cuts_sel_by_opt = 0
            # Generate and add selected cuts up to (sel_size) in number
            for ix in range(0, sel_size):
                (idx, obj_improve, curr_pt, X_slice, dim_act) = rank_list[ix]
                if obj_improve > CutSolver._BIG_M:
                    nb_cuts_sel_by_opt += 1
                clique = agg_list[idx][0]
                eigvals, evecs = super()._get_eigendecomp(dim_act, curr_pt, X_slice, True)
                if eigvals[0] < CutSolver._THRES_NEG_EIGVAL:
                    evect = evecs.T[0]
                    evect = np.where(abs(evect) <= -CutSolver._THRES_NEG_EIGVAL, 0, evect)
                    x_rho = list(itemgetter(*clique)(x))
                    clique_lifted = [ind - self._nb_x_unlifted for ind in clique]
                    X_rho = X[np.array(clique_lifted), np.array([[Ind] for Ind in clique_lifted])]
                    mat_var = cvx.hstack([cvx.vstack([1, *x_rho]), cvx.vstack([cvx.bmat([x_rho]), X_rho])])
                    constraints.append(cvx.sum_entries(cvx.mul_elemwise(np.outer(evect, evect), mat_var)) >= 0)
            # Solve relaxations at cut_round
            prob = cvx.Problem(objective, constraints)
            value = prob.solve(verbose=False, solver=cvx.ECOS)
            obj_values_rounds[cut_round] = value
            nb_cuts_opt[cut_round - 1] = nb_cuts_sel_by_opt
        return obj_values_rounds, nb_cuts_opt

    def __parse_powerflow_osil_into_cvx(self, filename):
        """Parsing powerflow file for instance http://www.minlplib.org/lp/powerflow0009r.lp
        after it was pre-processed by the Antigone solver for obtaining bounds and eliminating variables
        """
        dirname = os.path.join(os.path.dirname(__file__), "boxqp_instances", filename + ".osil")
        # Parse xml in a tree structure
        expr_tree = parser.iterparse(dirname)
        for _, el in expr_tree:
            el.tag = el.tag.split('}', 1)[1]  # strip all namespaces
        tree = expr_tree.root
        # Get variables xml element
        vars_xml = tree.find('instanceData/variables')
        # Number of linear variables
        nb_lin_vars = int(vars_xml.attrib['numberOfVariables'])

        # Original bounds of x variables before re-scaling
        x_up = np.array([float(var.attrib['ub']) for idx, var in enumerate(vars_xml.findall('var'))])
        x_low = np.array([float(var.attrib['lb']) for idx, var in enumerate(vars_xml.findall('var'))])
        # x variables in CVX
        x = cvx.Variable(1, nb_lin_vars)

        ######################
        #  Parse OSIL objective
        ######################
        # Get objective xml element
        obj_xml = tree.find('instanceData/objectives/obj')
        obj_func = float(obj_xml.attrib['constant'])
        obj_lin_coeffs = np.array([float(coeff.text) for coeff in obj_xml.findall('coef')])
        if obj_xml.attrib['maxOrMin'] == 'min':
            # for all indexed (by x_idx) variables x that participate in linear objective terms)
            for idx, x_idx in enumerate(np.array([int(coeff.attrib['idx']) for coeff in obj_xml.findall('coef')])):
                lin_coeff = obj_lin_coeffs[idx]
                # Linear part of objective
                # rescale each linear term x_idx from [x_low[x_idx], x_up[x_idx]] to [0,1]
                obj_func += x[x_idx] * lin_coeff * (x_up[x_idx] - x_low[x_idx]) + lin_coeff * x_low[x_idx]

        ######################
        #  Parse OSIL constraints
        ######################
        # Get constraints xml element
        con_xml = tree.find('instanceData/constraints')
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

        # QUADRATIC CONSTRAINTS
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

        ###############
        # Specific to powerflow09r instance (after Antigone pre-processing for variable bounds).
        # Only x variables past index 36 participate in quadratic terms, so build reduced X self._Matrix by
        # not lifting x1 to x36
        nb_x_unlifted = 36
        dim_lifted = nb_lin_vars - nb_x_unlifted
        ###############

        X = cvx.Variable(dim_lifted, dim_lifted)

        ######################
        # Build CVX constraints (linear, linearized quadratic via lifting, and SOCP where possible)
        ######################
        constraints = []
        # List of resulting quadratic lifted constraints (used later for cut selection)
        cons_quad_list = []
        # Keep counter of how many constraints are added
        cons_counter = 0
        # Go through each constraint
        for con_idx in range(0, nb_cons):
            con_expr = 0  # initialize constraint lhs expresssion
            ineq_sign, const_val = cons_sgn_rhs[con_idx]  # type of constraint and rhs constant
            is_con_quad = False  # quadratic constraint?
            socp_norm_term = [(1 - const_val)]  # start norm for an SOCP constraint (if constraint qualifies)
            # Quadratic part for any lifted quadratic constraint, with re-scaling
            if con_idx in quad_term_to_cons:
                is_con_quad = True
                potential_socp = True  # can quad constraint be potentially SOCP? start with yes assumption
                for q_idx, quad_term in enumerate(quad_term_list):
                    # If quad term is in constraint
                    if quad_term[0] == con_idx:
                        [x_idx, x_idx2, coeff] = quad_term[
                                                 1:]  # quadratic term first and second x index, and coefficient
                        # Rescale each linear term x_idx from [x_low[x_idx], x_up[x_idx]] to [0,1] and then lift
                        if x_idx - nb_x_unlifted >= 0 and x_idx2 - nb_x_unlifted >= 0:
                            con_expr += X[x_idx - nb_x_unlifted, x_idx2 - nb_x_unlifted] * coeff * \
                                        (x_up[x_idx] - x_low[x_idx]) * (x_up[x_idx2] - x_low[x_idx2]) + \
                                        x[x_idx] * coeff * (x_up[x_idx] - x_low[x_idx]) * x_low[x_idx2] + \
                                        x[x_idx2] * coeff * (x_up[x_idx2] - x_low[x_idx2]) * x_low[x_idx] + \
                                        coeff * x_low[x_idx] * x_low[x_idx2]
                        # Update coefficient of new re-scaled quadratic term (used later in cut selection)
                        quad_term_list[q_idx][3] = coeff * (x_up[x_idx] - x_low[x_idx]) * (x_up[x_idx2] - x_low[x_idx2])
                        # if we keep encountering squares-only in a <= ineq (ineq_sign == 2) then continue assuming
                        # the inequality is an SOCP and built it up; re-scale x_idx variable to [0,1]
                        if potential_socp and ineq_sign == 2:
                            if x_idx == x_idx2:
                                # Add to the norm for an SOCP constraint
                                socp_norm_term.append(
                                    2 * (x[x_idx] * (x_up[x_idx] - x_low[x_idx]) + x_low[x_idx]) * np.sqrt(coeff))
                            else:  # first non-square element encountered disqualifies constraint as SOCP
                                potential_socp = False
                # Add confirmed SOCP constraint and move to next constraint
                if potential_socp and ineq_sign == 2:
                    constraints.append(cvx.norm(cvx.vstack(socp_norm_term), 2) <= (1 + const_val))
                    cons_counter += 1
                    continue
            # Linear part for any constraint, with re-scaling of x_idx variable
            for term_idx in range(con_start_list[con_idx], con_start_list[con_idx + 1]):
                x_idx = int(lin_col_idx[term_idx])
                con_expr += (x[x_idx] * (x_up[x_idx] - x_low[x_idx]) + x_low[x_idx]) * lin_col_val[term_idx]
            # Add linear or lifted quadratic constraint
            if ineq_sign == 0:  # equality
                constraints.append(con_expr == const_val)
                if is_con_quad:  # lifted quadratic equality (x, X variables)
                    cons_quad_list.append((con_idx, cons_counter, 1))  # Flag with 1 for equality
            elif ineq_sign == 1:  # >= inequality
                constraints.append(-1 * con_expr <= -const_val)
                if is_con_quad:  # lifted quadratic inequality (x, X variables)
                    cons_quad_list.append((con_idx, cons_counter, -1))  # Flag with -1 for <= inequality
            elif ineq_sign == 2:  # <= inequality
                constraints.append(con_expr <= const_val)
                if is_con_quad:  # lifted quadratic inequality (x, X variables)
                    cons_quad_list.append((con_idx, cons_counter, -1))  # Flag with -1 for <= inequality
            cons_counter += 1

        ######################
        # Build CVX quadratic part of the objective - introduce SOCP constraint for square terms
        ######################
        c = cvx.Variable()  # new variable in objective replacing SOCP expression
        obj_func += c
        obj_socp_norm_term = [(1 - c)]  # start norm for SOCP constraint
        obj_has_socp = False
        for q_idx, quad_term in enumerate(quad_term_list):
            if quad_term[0] == -1:  # if quadratic term in objective
                [x_idx, x_idx2, coeff] = quad_term[1:]  # quadratic term first and second x index, and coefficient
                if x_idx != x_idx2:  # non-square term
                    # Rescale each linear term x_idx from [x_low[x_idx], x_up[x_idx]] to [0,1] and then lift
                    obj_func += X[x_idx - nb_x_unlifted, x_idx2 - nb_x_unlifted] * coeff * \
                                (x_up[x_idx] - x_low[x_idx]) * (x_up[x_idx2] - x_low[x_idx2]) + \
                                x[x_idx] * coeff * (x_up[x_idx] - x_low[x_idx]) * x_low[x_idx2] + \
                                x[x_idx2] * coeff * (x_up[x_idx2] - x_low[x_idx2]) * x_low[x_idx] + \
                                coeff * x_low[x_idx] * x_low[x_idx2]
                    # Update coefficient of new re-scaled quadratic term (used later in cut selection)
                    quad_term_list[q_idx][3] = coeff * (x_up[x_idx] - x_low[x_idx]) * (x_up[x_idx2] - x_low[x_idx2])
                else:
                    # Add to the norm for an SOCP constraint
                    obj_socp_norm_term.append(
                        2 * (x[x_idx] * (x_up[x_idx] - x_low[x_idx]) + x_low[x_idx]) * np.sqrt(coeff))
                    obj_has_socp = True
        # If an SOCP constraint can be formed from objective terms, add it
        if obj_has_socp:
            constraints.append(cvx.norm(cvx.vstack(obj_socp_norm_term), 2) <= (1 + c))

        self._nb_x_unlifted, self._dim_lifted, self._quad_term_list, self._cons_quad_list = \
            (nb_x_unlifted, dim_lifted, quad_term_list, cons_quad_list)

        ######################
        # Bound constraints [0,1] for all x variables after re-scaling
        ######################
        for idx, var in enumerate(x):
            constraints.extend([var >= 0,var <= 1])

        # Form adjacency matrix
        Q_adj = np.zeros((dim_lifted, dim_lifted))
        for quad_term in quad_term_list:
            if quad_term[1] >= nb_x_unlifted and quad_term[2] >= nb_x_unlifted:
                Q_adj[quad_term[1] - nb_x_unlifted, quad_term[2] - nb_x_unlifted] = 1
                Q_adj[quad_term[2] - nb_x_unlifted, quad_term[1] - nb_x_unlifted] = 1
        self._Q_adj = Q_adj

        # Save objective, constraints and variables
        self._objective, self._constraints, self._x, self._X  = obj_func, constraints, x, X

    def __add_mccormick_to_instance(self):
        """Add M^2 constraints (McCormick linear and convex SOCP constraints)
        """
        x, X, Q_adj, nb_x_unlifted, dim_lifted = self._x, self._X, self._Q_adj, self._nb_x_unlifted, self._dim_lifted
        constraints = []
        for iX in range(0, dim_lifted):
            i = iX + nb_x_unlifted
            constraints.extend([X[iX, iX] <= x[i], cvx.norm(cvx.vstack([1 - X[iX, iX], 2 * x[i]]), 2) <= 1 + X[iX, iX]])
            for jX in range(0, iX):
                j = jX + nb_x_unlifted
                if Q_adj[iX, jX]:
                    constraints.extend([X[iX, jX] >= 0, X[iX, jX] >= x[i] + x[j] - 1,
                                        X[iX, jX] <= x[i], X[iX, jX] <= x[j], X[iX, jX] == X[jX, iX]])
        self._constraints.extend(constraints)

    def __get_sdp_decomposition(self, dim):
        """Get semidefinite decomposition in cliques (P'_3) and get the index set and sliced coefficients
        for all constraints in Theta_rho for each clique rho (as explained in manuscript)
        """
        assert (dim == 3), "Only SDP decomposition of size 3 is implemented for QCQP!"
        Q_adj, nb_x_unlifted, dim_lifted, quad_term_list, cons_quad_list = \
            (self._Q_adj, self._nb_x_unlifted, self._dim_lifted, self._quad_term_list, self._cons_quad_list)
        # Build SDP decomposition (S_3=S_2 for powerflow009r instance since there are no triple cliques)
        agg_list = []
        # Follows logic for building P'_3 for the QP case
        if dim == 3:
            for i1 in range(0, dim_lifted):
                for i2 in range(i1 + 1, dim_lifted):
                    if Q_adj[i1, i2]:
                        triple_flag = False
                        # look forward (i3>i2) for triple clique to add
                        for i3 in range(i2 + 1, dim_lifted):
                            if Q_adj[i1, i3] and Q_adj[i2, i3]:
                                agg_list.append(((i1, i2, i3), 3))
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
                                agg_list.append(((i1, i2), 2))

        #####################
        # Find which constraints are relevant for each clique as explained in the manuscript and collect info
        # Stores x variable indices in lifted X for each constraint that doesn't only contain squares
        cons_x_idxs_lifted = []
        # Stores x variable indices and coefficient in lifted X for each constraint that doesn't only contain squares
        cons_lifted_list = []
        # Find all x variable indices involved in each lifted quadratic constraint
        for con_idx, _, _ in cons_quad_list:
            con_x_idxs_lifted = []
            con_lifted_list = []
            all_squares = True
            for q_idx, quad_term in enumerate(quad_term_list):
                if quad_term[0] == con_idx:
                    con_x_idxs_lifted.extend([x_idx - nb_x_unlifted for x_idx in quad_term[1:3]])
                    con_lifted_list.append([(quad_term[1] - nb_x_unlifted, quad_term[2] - nb_x_unlifted), quad_term[3]])
                    if quad_term[1] != quad_term[2]:
                        all_squares = False
            if not all_squares:
                cons_x_idxs_lifted.append((con_idx, set(np.unique(con_x_idxs_lifted))))
                cons_lifted_list.append(con_lifted_list)
        # For each decomposed clique, find the associated coefficients in each constraint
        # where the clique appears (not only through square terms)
        for agg_idx, (clique, clique_size) in enumerate(agg_list):
            inputs_nn = []
            set_idxs = list(itertools.combinations_with_replacement(clique, 2))
            for other_idx, (con_idx, x_idxs) in enumerate(cons_x_idxs_lifted):
                if len(set(clique).intersection(x_idxs)) > 1:
                    all_squares = True
                    input_nn = [0] * len(set_idxs)
                    for idx, set_idx in enumerate(set_idxs):
                        for el in cons_lifted_list[other_idx]:
                            if set(el[0]) == set(set_idx):
                                input_nn[idx] = el[1]
                                if el[0][0] != el[0][1]:
                                    all_squares = False
                    max_elem = clique_size * abs(max(input_nn, key=abs))
                    if max_elem == 0 or all_squares:
                        continue
                    inputs_nn.append((list(np.divide(input_nn, max_elem)), max_elem, con_idx))
            clique = [ind + nb_x_unlifted for ind in clique]
            agg_list[agg_idx] = (clique, set_idxs, inputs_nn)
        ######################

        self._agg_list = list(filter(lambda x: x[2], agg_list))


