"""
Collection of methods to solve the min-norm convex minimization problem:

  given:      V = [V1, V2, ... Vn]    # (Vi are k-dimensional vectors)
  minimize:   || Σ ci*Vi ||^2         # Find min-norm point...
  w.r.t:      c = [c1, c2, ... cn]    # (ci are scalars)
  subj. to:   Σ ci = 1                # ... on the convex hull of V.

The following methods are implemented (using PyTorch):
  * MinNormLinearSolver - analytical solver in case of 2 vectors
  * MinNormPlanarSolver - analytical solver in case V lies on a plane
  * MinNormSolver       - iterative general solver (simplex projection)
  * MinNormSolverFW     - iterative general solver (Frank-Wolfe)
"""

import numpy as np
import torch
import torch.nn as nn


class MinNormLinearSolver(nn.Module):
    """Solves the min norm problem in case of 2 vectors (lies on a line):
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, v1v1, v1v2, v2v2):
        """Solver execution on scalar products of 2 vectors

        Args:
          v1v1:  scalar product <V1, V1>
          v1v2:  scalar product <V1, V2>
          v2v2:  scalar product <V2, V2>

        Returns:
          gamma: min-norm solution c = (gamma, 1. - gamma)
          cost:  the norm of min-norm point
        """
        if v1v2 >= v1v1:
            return 1., v1v1
        if v1v2 >= v2v2:
            return 0., v2v2
        gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2 + 1e-8)
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost


class MinNormPlanarSolver(nn.Module):
    """Solves the min norm problem in case the vectors lies on same plane
    """
    def __init__(self, n_tasks):
        super().__init__()
        i_grid = torch.arange(n_tasks)
        j_grid = torch.arange(n_tasks)
        ii_grid, jj_grid = torch.meshgrid(i_grid, j_grid)
        i_triu, j_triu = np.triu_indices(n_tasks, 1)

        self.register_buffer('n', torch.tensor(n_tasks))
        self.register_buffer('i_triu', torch.from_numpy(i_triu))
        self.register_buffer('j_triu', torch.from_numpy(j_triu))
        self.register_buffer('ii_triu', ii_grid[i_triu, j_triu])
        self.register_buffer('jj_triu', jj_grid[i_triu, j_triu])
        self.register_buffer('one', torch.ones(self.ii_triu.shape))
        self.register_buffer('zero', torch.zeros(self.ii_triu.shape))

    @torch.no_grad()
    def line_solver_vectorized(self, v1v1, v1v2, v2v2):
        """Linear case solver, but for collection of vector pairs (Vi, Vj)

        Args:
          v1v1:  vector of scalar product <Vi, Vi>
          v1v2:  vector of scalar product <Vi, Vj>
          v2v2:  vector of scalar product <Vj, Vj>

        Returns:
          gamma: vector of min-norm solution c = (gamma, 1. - gamma)
          cost:  vector of the norm of min-norm point
        """
        gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2 + 1e-8)
        gamma = gamma.where(v1v2 < v2v2, self.zero)
        gamma = gamma.where(v1v2 < v1v1, self.one)

        cost = v2v2 + gamma * (v1v2 - v2v2)
        cost = cost.where(v1v2 < v2v2, v2v2)
        cost = cost.where(v1v2 < v1v1, v1v1)
        return gamma, cost

    @torch.no_grad()
    def forward(self, grammian):
        """Planar case solver, when Vi lies on the same plane

        Args:
          grammian: grammian matrix G[i, j] = [<Vi, Vj>], G is a nxn tensor

        Returns:
          sol: coefficients c = [c1, ... cn] that solves the min-norm problem
        """
        vivj = grammian[self.ii_triu, self.jj_triu]
        vivi = grammian[self.ii_triu, self.ii_triu]
        vjvj = grammian[self.jj_triu, self.jj_triu]

        gamma, cost = self.line_solver_vectorized(vivi, vivj, vjvj)
        offset = torch.argmin(cost)
        i_min, j_min = self.i_triu[offset], self.j_triu[offset]
        sol = torch.zeros(self.n, device=grammian.device)
        sol[i_min], sol[j_min] = gamma[offset], 1. - gamma[offset]
        return sol


class MinNormSolver(nn.Module):
    """Solves the min norm problem in the general case.
    """
    def __init__(self, n_tasks, max_iter=250, stop_crit=1e-6):
        super().__init__()
        self.n = n_tasks
        self.linear_solver = MinNormLinearSolver()
        self.planar_solver = MinNormPlanarSolver(n_tasks)

        n_grid = torch.arange(n_tasks)
        i_grid = torch.arange(n_tasks, dtype=torch.float32) + 1
        ii_grid, jj_grid = torch.meshgrid(n_grid, n_grid)

        self.register_buffer('n_ts', torch.tensor(n_tasks))
        self.register_buffer('i_grid', i_grid)
        self.register_buffer('ii_grid', ii_grid)
        self.register_buffer('jj_grid', jj_grid)
        self.register_buffer('zero', torch.zeros(n_tasks))
        self.register_buffer('stop_crit', torch.tensor(stop_crit))

        self.max_iter = max_iter
        self.two_sol = nn.Parameter(torch.zeros(2))
        self.two_sol.require_grad = False

    @torch.no_grad()
    def projection_to_simplex(self, gamma):
        sorted_gamma, indices = torch.sort(gamma, descending=True)
        tmp_sum = torch.cumsum(sorted_gamma, 0)
        tmp_max = ((tmp_sum - 1.) / self.i_grid)

        non_zeros = torch.nonzero(tmp_max[:-1] > sorted_gamma[1:])
        if non_zeros.shape[0] > 0:
            tmax_f = tmp_max[:-1][non_zeros[0][0]]
        else:
            tmax_f = tmp_max[-1]
        return torch.max(gamma - tmax_f, self.zero)

    @torch.no_grad()
    def next_point(self, cur_val, grad):
        proj_grad = grad - (torch.sum(grad) / self.n_ts)
        lt_zero = torch.nonzero(proj_grad < 0)
        lt_zero = lt_zero.view(lt_zero.numel())
        gt_zero = torch.nonzero(proj_grad > 0)
        gt_zero = gt_zero.view(gt_zero.numel())
        tm1 = -cur_val[lt_zero] / proj_grad[lt_zero]
        tm2 = (1. - cur_val[gt_zero]) / proj_grad[gt_zero]

        t = torch.tensor(1., device=grad.device)
        tm1_gt_zero = torch.nonzero(tm1 > 1e-7)
        tm1_gt_zero = tm1_gt_zero.view(tm1_gt_zero.numel())
        if tm1_gt_zero.shape[0] > 0:
            t = torch.min(tm1[tm1_gt_zero])

        tm2_gt_zero = torch.nonzero(tm2 > 1e-7)
        tm2_gt_zero = tm2_gt_zero.view(tm2_gt_zero.numel())
        if tm2_gt_zero.shape[0] > 0:
            t = torch.min(t, torch.min(tm2[tm2_gt_zero]))

        next_point = proj_grad * t + cur_val
        next_point = self.projection_to_simplex(next_point)
        return next_point

    @torch.no_grad()
    def forward(self, vecs):
        """General case solver using simplex projection algorithm.

        Args:
          vecs:  2D tensor V, where each row is a vector Vi

        Returns:
          sol: coefficients c = [c1, ... cn] that solves the min-norm problem
        """
        if self.n == 1:
            return vecs[0]
        if self.n == 2:
            v1v1 = torch.dot(vecs[0], vecs[0])
            v1v2 = torch.dot(vecs[0], vecs[1])
            v2v2 = torch.dot(vecs[1], vecs[1])
            self.two_sol[0], cost = self.linear_solver(v1v1, v1v2, v2v2)
            self.two_sol[1] = 1. - self.two_sol[0]
            return self.two_sol.clone()

        grammian = torch.mm(vecs, vecs.t())
        sol_vec = self.planar_solver(grammian)

        ii, jj = self.ii_grid, self.jj_grid
        for iter_count in range(self.max_iter):
            grad_dir = -torch.mv(grammian, sol_vec)
            new_point = self.next_point(sol_vec, grad_dir)

            v1v1 = (sol_vec[ii] * sol_vec[jj] * grammian[ii, jj]).sum()
            v1v2 = (sol_vec[ii] * new_point[jj] * grammian[ii, jj]).sum()
            v2v2 = (new_point[ii] * new_point[jj] * grammian[ii, jj]).sum()

            gamma, cost = self.linear_solver(v1v1, v1v2, v2v2)
            new_sol_vec = gamma * sol_vec + (1 - gamma) * new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < self.stop_crit:
                return sol_vec
            sol_vec = new_sol_vec
        return sol_vec


class MinNormSolverFW(nn.Module):
    """Wrapper over series of algorithms for solving min-norm tasks.
    """
    def __init__(self, n_tasks, max_iter=250, stop_crit=1e-6):
        """Stuffs we don't want to re-define too much times
        """
        super().__init__()
        self.n_tasks = n_tasks
        n = torch.tensor(n_tasks)

        self.MAX_ITER = max_iter
        STOP_CRIT = torch.tensor(stop_crit)

        grammian = torch.empty((n_tasks, n_tasks), dtype=torch.float32)
        sol = torch.empty((n_tasks,), dtype=torch.float32)
        new_sol = torch.empty((n_tasks,), dtype=torch.float32)

        self.register_buffer('n', n)
        self.register_buffer('STOP_CRIT', STOP_CRIT)
        self.register_buffer('grammian', grammian)
        self.register_buffer('sol', sol)
        self.register_buffer('new_sol', new_sol)

    @torch.no_grad()
    def line_solver(self, v1v1, v1v2, v2v2):
        """Analytical solution for the min-norm problem
        """
        if v1v2 >= v1v1:
            return 0.999
        if v1v2 >= v2v2:
            return 0.001
        return ((v2v2 - v1v2) / (v1v1+v2v2 - 2*v1v2))

    @torch.no_grad()
    def forward(self, vecs):
        """Computes grammian matrix G_{i,j} = (<v_i, v_j>)_{i,j}.
        """
        if self.n_tasks == 1:
            return vecs[0]
        if self.n_tasks == 2:
            v1v1 = torch.dot(vecs[0], vecs[0])
            v1v2 = torch.dot(vecs[0], vecs[1])
            v2v2 = torch.dot(vecs[1], vecs[1])
            gamma = self.line_solver(v1v1, v1v2, v2v2)
            return gamma * vecs[0] + (1. - gamma) * vecs[1]

        self.sol.fill_(1. / self.n)
        self.new_sol.copy_(self.sol)
        torch.mm(vecs, vecs.t(), out=self.grammian)

        for iter_count in range(self.MAX_ITER):
            gram_dot_sol = torch.mv(self.grammian, self.sol)
            t_iter = torch.argmin(gram_dot_sol)

            v1v1 = torch.dot(self.sol, gram_dot_sol)
            v1v2 = torch.dot(self.sol, self.grammian[:, t_iter])
            v2v2 = self.grammian[t_iter, t_iter]

            gamma = self.line_solver(v1v1, v1v2, v2v2)
            self.new_sol *= gamma
            self.new_sol[t_iter] += 1. - gamma

            change = self.new_sol - self.sol
            if torch.sum(torch.abs(change)) < self.STOP_CRIT:
                return self.new_sol
            self.sol.copy_(self.new_sol)
        return self.sol
