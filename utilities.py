from timeit import default_timer as timer
import math
from copy import deepcopy
import numpy as np
from scipy.stats import ortho_group
import cvxpy as cvx


def gen_data_ndim(nb_datapoints, dim, savefile, rand_seed=7):
    """Sampling according to Table 1 in manuscript:
    - uniform point positioning (x) in [0,1] for each dimension
    - uniform eigenvalues in [-1,1]
    - ortho-normal basis/matrix (eigvecs) of eigenvectors
    :param nb_datapoints: how many data points to sample
    :param dim: dimensionality of SDP sub-problem to sample
    :param savefile: file to save sampled data in
    :param rand_seed: random seed, pre-set to 7
    :return: None
    """
    np.random.seed(rand_seed)
    X = cvx.Variable(dim, dim)
    data_points = []
    t_init = timer()
    t0 = timer()
    for data_pt_nb in range(1, nb_datapoints + 1):
        # ortho-normal basis/matrix (eigvecs) of eigenvectors
        eigvecs = ortho_group.rvs(dim)
        # uniform eigenvalues in [-1,1]
        eigvals = np.random.uniform(-1, 1, dim).tolist()
        # construct sampled Q from eigen-decomposition
        Q = np.matmul(np.matmul(eigvecs, np.diag(eigvals)), np.transpose(eigvecs))
        # uniform point positioning (x) in [0,1] for each dimension
        x = np.random.uniform(0, 1, dim).tolist()
        # construct cvx SDP sub-problem
        obj_sdp = cvx.Minimize(cvx.sum_entries(cvx.mul_elemwise(Q, X)))
        constraints = [
            cvx.lambda_min(cvx.hstack(cvx.vstack(1, np.array(x)), cvx.vstack(cvx.vec(x).T, X))) >= 0,
            *[X[ids, ids] <= x[ids] for ids in range(0, dim)]]
        prob = cvx.Problem(obj_sdp, constraints)
        # solve it using Mosek SDP solver
        prob.solve(verbose=False, solver=cvx.MOSEK)
        # store upper triangular matrix in array (row major) since it is symmetric
        Q = np.triu(Q, 1) + np.triu(Q, 0)
        # save eigendecomposition, point positioning, matrix Q and solution of SDP sub-problem
        data_points.append([*eigvecs.T.flatten(), *eigvals, *x, *list(Q[np.triu_indices(dim)]), obj_sdp.value])
        # save to file and empty data_points buffer every 1000 entries
        if not data_pt_nb % 1000: 
            t1 = timer()
            with open(savefile, "a") as f:
                for line in data_points:
                    f.write(",".join(str(x) for x in line) + "\n")
                data_points = []
                print(str(data_pt_nb) + " "+str(dim)+"D points, time = " + str(t1 - t0) + "s" + ", total = " + str(t1 - t_init) + "s")
                t0 = t1


def gen_data_3d_q(nb_datapoints, savefile, rand_seed=7):
    """Sampling uniformly directly on the matrix Q itself (uniform Q entries in [-1,1]) for 3-dim sub-problems.
    :param nb_datapoints: how many data points to sample
    :param savefile: file to save sampled data in
    :param rand_seed: random seed, pre-set to 7
    :return: None
    """
    t_init = timer()
    t0 = timer()
    np.random.seed(rand_seed)
    iters_size = 10000
    data_points = [0]*iters_size
    iter = 0
    Q = np.zeros((3, 3))
    triu_inds = np.triu_indices(3)
    for data_pt_nb in range(1, nb_datapoints + 1):
        # generate upper triangular part of 3x3 Q with 6 elements in [-1, 1]
        rand_arr = np.random.uniform(-1, 1, 6).tolist()
        Q[triu_inds] = rand_arr
        data_points[iter] = np.linalg.eigvalsh(Q, UPLO='U')
        iter += 1
        # save to file and empty data_points buffer every 1000 entries
        if not data_pt_nb % iters_size:
            t1 = timer()
            iter = 0
            with open(savefile, "a") as f:
                for line in data_points:
                    f.write(",".join(str(x) for x in line) + "\n")
                data_points = [0]*iters_size
                print(str(data_pt_nb) + ", time = " + str(t1 - t0) + "s" + ", total = " + str(t1 - t_init) + "s")
                t0 = t1


def get_average_bounds_3d(savefile, start=5, stop=65, step=5, nb_instances=30, rand_seed=7):
    """ Get average percentage bounds for sampled problem sizes (Fig.1 in manuscript) - by default between 5-65
    for 30 instances at every size that's a multiple of 5
    :param savefile: .csv file to save results to
    :param start: Lower bound on problem size
    :param stop: Upper bound on problem size
    :param step: Step increment in problem size between [start,stop]
    :param nb_instances: How many instances to sample and solve for each problem size
    :param rand_seed: random seed, pre-set to 7
    :return: None
    """
    np.random.seed(rand_seed)
    for nbVb in range(start, stop, step):
        solve_random_inst_3d(nbVb, nb_instances, savefile)


def solve_random_inst_3d(nb_vars, nb_instances, savefile):
    """Calculates percent gap between M and S+M  closed by M+S_3 for randomly generated (dense BoxQP) instances
    :param nb_vars: size of instances (number of unlifted variables x)
    :param nb_instances: number of problem instances to sample
    :param savefile: file to save results
    :return: None
    """
    # variables for problems of dimension nb_vars
    X = cvx.Variable(nb_vars, nb_vars)
    x = cvx.Variable(nb_vars)
    # (M) McCormick constraints
    constraints_rlt = []
    for i in range(nb_vars):
        constraints_rlt.extend([X[i, i] <= x[i], X[i, i] >= 0, X[i, i] >= 2*x[i] - 1, x[i] <= 1])
        for j in range(i+1, nb_vars):
            constraints_rlt.extend([X[i, j] >= 0, X[i, j] >= x[i] + x[j] - 1,
                                   X[i, j] <= x[i], X[i, j] <= x[j], X[i, j] == X[j, i]])
    # (M+S) McCormick + full SDP constraint
    constraints_sdp = deepcopy(constraints_rlt)
    constraints_sdp.append(cvx.lambda_min(cvx.hstack(cvx.vstack(1, x), cvx.vstack(x.T, X))) >= 0)
    # (M+S_3) McCormick + 3D SDP constraints
    constraints_3d = deepcopy(constraints_rlt)
    for i in range(nb_vars):
        for j in range(i + 1, nb_vars):
            for k in range(j + 1, nb_vars):
                set_inds = [i, j, k]
                constraints_3d += \
                    [cvx.lambda_min(cvx.hstack(
                        cvx.vstack(1, x[i], x[j], x[k]),
                        cvx.vstack(cvx.hstack(x[i], x[j], x[k]),
                                   X[np.array(set_inds), np.array([[Ind] for Ind in set_inds])]))) >= 0]
    # generate random Q and therefore objectives for nb_instances problem instances and solve them
    for sample_nb in range(nb_instances):
        # generate random Q according to Table 1 in manuscript
        eigvecs = ortho_group.rvs(nb_vars)
        eigvals = np.random.uniform(-1, 1, nb_vars).tolist()
        Q = np.matmul(np.matmul(eigvecs, np.diag(eigvals)), np.transpose(eigvecs))
        # Create objective
        obj_expr = 0
        for i in range(0, nb_vars):
            for j in range(0, nb_vars):
                obj_expr += Q[i, j]*X[i, j]
        obj = cvx.Minimize(obj_expr)
        solution = []
        for constraints in [constraints_sdp, constraints_3d, constraints_rlt]:
            prob = cvx.Problem(obj, constraints)
            prob.solve(verbose=False, solver=cvx.MOSEK)
            solution.append(obj.value)
        # Calculate the percent gap between M+S and M closed by M+S_3
        if solution[0]-solution[2] <= 10e-5:
            percent_gap = 1     # if there is no gap, the percent closed is 1
        else:
            percent_gap = (solution[1] - solution[2]) / (solution[0] - solution[2])
        solution.append(min(1, percent_gap))
        print("nb_vars=" + str(nb_vars) + ", sample_nb=" + str(sample_nb) + ", percent=" + str(solution[-1]))
        with open(savefile, "a") as f:
            f.write((str(nb_vars) + "," + str(sample_nb) + ",") + ",".join(str(x) for x in solution) + "\n")


def gen_sdp_surface_2d_fig4(savefile, iters=11):
    """Calls gen_sdp_surface_2d for settings used in Fig. 4 in the manuscript
    """
    Q = np.asmatrix(np.array([[-1, 10], [0, -1]]))
    pt_tangent = [0.5, 0.5]
    gen_sdp_surface_2d(Q, iters, pt_tangent, savefile)


def gen_sdp_surface_2d(Q, iters, pt_tan, savefile):
    """Generate the semidefinite surface/underestimator for semidefinite 2D sub-problems with given Q on a mesh
    with distance between points 1/(iters-1) and finds hyoperplane tangent to the SDP surface at tangent point pt_tan
    """
    dim = 2
    # 2x2 variable X holding the relaxed bilinear/quadratic terms.
    X = cvx.Variable(2, 2)
    x = cvx.Variable(2)
    # Create objective
    obj_expr = 0
    for i in range(dim):
        for j in range(dim):
            obj_expr += Q[i, j] * X[i, j]
    obj = cvx.Minimize(obj_expr)
    # deviation from point pt_tan
    dev = 0.01
    # Create 3 additional points around pt_tan to determine a hyperplane tangent to
    # the SDP surface at approximately pt_tan
    pts = [pt_tan, [pt_tan[0] - dev, pt_tan[1] - dev], [pt_tan[0] - dev, pt_tan[1] + dev],
           [pt_tan[0] + math.sqrt(2 * pow(dev, 2)), pt_tan[1]]]
    res_points = []
    # Calculate for point of SDP surface for pt_tan and the 3 points around it determining hyperplane
    for i in range(0, dim+2):
        pt = np.array(pts[i])
        constraints = [cvx.lambda_min(cvx.bmat([[1, *pt], [pt[0], X[0, :]], [pt[1], X[1, :]]])) >= 0,
                       X[0, 0] <= pt[0], X[1, 1] <= pts[i][1]]
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=False, solver=cvx.MOSEK)
        res_points.append([*pts[i], obj_expr.value])

    # Hyperplane equation calculations
    vectors = [[1, *([0]*dim)]]
    for i in range(2, dim + 2):
        vectors.append([x - y for x, y in zip(res_points[i], res_points[i-1])])
    hp_sols = np.linalg.solve(np.array(vectors), vectors[0])
    coeffs = [*np.ndarray.tolist(hp_sols), -np.dot(hp_sols, res_points[1])]
    # Separating the hyperplane
    obj_hp = obj_expr - (coeffs[0]*x[0] + coeffs[1]*x[1] + coeffs[dim+ 1]) / (-coeffs[dim])
    obj_sdp_hp = cvx.Minimize(obj_hp)
    constraints_hp = [
        cvx.lambda_min(cvx.hstack(cvx.vstack(1, x), cvx.vstack(x.T, X))) >= 0,
        *[X[ids, ids] <= x[ids] for ids in range(0, dim)], x[0]>=0, x[1]>=0, x[0]<=1, x[1]<=1]
    prob = cvx.Problem(obj_sdp_hp, constraints_hp)
    prob.solve(verbose=False, solver=cvx.MOSEK)
    # Constant coordinate minus distance to hyperplane (bring out hyperplane till tangent)
    coeffs[dim+1] += obj_hp.value*(-coeffs[dim])

    # Determine points on SDP surface and on hyperplane tangent to it at pt_tan
    # on a mesh grid, with spaces in between them of 1/(iters-1)
    data_points = []
    for i in range(1, iters + 1):
        for j in range(1, iters + 1):
            pt = [(i - 1) / float(iters - 1), (j - 1) / float(iters - 1)]
            obj_value = 0
            for r in range(0, dim):
                for c in range(0, dim):
                    obj_value += Q[r, c] * pt[r] * pt[c]
            # Mosek fails due to numerical error at pt=[0,0], but solution is obviously 0
            if i == 1 and j == 1:
                hp_value = (np.dot(coeffs[0:dim], pt[0:dim]) + coeffs[dim + 1]) / (-coeffs[dim])
                data_points.append([pt[0], pt[1], obj_value, hp_value, 0])
                continue
            constraints = [cvx.lambda_min(cvx.bmat([[1, *pt], [pt[0], X[0, :]], [pt[1], X[1, :]]])) >= 0,
                           X[0, 0] <= pt[0], X[1, 1] <= pt[1]]
            prob = cvx.Problem(obj, constraints)
            prob.solve(verbose=False, solver=cvx.MOSEK)
            # hyperplane value at pt
            hp_value = (np.dot(coeffs[0:dim], pt[0:dim]) + coeffs[dim + 1]) / (-coeffs[dim])
            data_points.append([pt[0], pt[1], obj_value, hp_value, obj_expr.value])
    # save all data point in .csv file
    with open(savefile, "w") as f:
        for line in data_points:
            f.write(",".join(str(x) for x in line) + "\n")