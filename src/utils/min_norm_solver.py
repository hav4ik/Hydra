import numpy as np
import torch
import torch.nn as nn


class MinNormLinearSolver(nn.Module):
    """Solves the min norm problem in case of 2 vectors (lies on a line)
    """
    def __init__(self):
        super().__init__()
        self.one = torch.tensor(1.)
        self.zero = torch.tensor(0.)

    @torch.no_grad()
    def forward(self, v1v1, v1v2, v2v2):
        if v1v2 >= v1v1:
            return self.one, v1v1
        if v1v2 >= v2v2:
            return self.zero, v2v2
        gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2 + 1e-8)
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost


class MinNormPlanarSolver(nn.Module):
    """Solves the min norm problem in case the vectors lies on same plane
    """
    def __init__(self, n_tasks):
        super().__init__()
        self.n = torch.tensor(n_tasks)

        i_grid = torch.arange(n_tasks)
        j_grid = torch.arange(n_tasks)
        ii_grid, jj_grid = torch.meshgrid(i_grid, j_grid)
        i_triu, j_triu = np.triu_indices(self.n, 1)
        self.i_triu = torch.from_numpy(i_triu)
        self.j_triu = torch.from_numpy(j_triu)
        self.ii_triu = ii_grid[i_triu, j_triu]
        self.jj_triu = jj_grid[i_triu, j_triu]

        self.one = torch.ones(self.ii_triu.shape)
        self.zero = torch.zeros(self.ii_triu.shape)

    @torch.no_grad()
    def line_solver_vectorized(self, v1v1, v1v2, v2v2):
        gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2 + 1e-8)
        gamma = gamma.where(v1v2 < v2v2, self.zero)
        gamma = gamma.where(v1v2 < v1v1, self.one)

        cost = v2v2 + gamma * (v1v2 - v2v2)
        cost = cost.where(v1v2 < v2v2, v2v2)
        cost = cost.where(v1v2 < v1v1, v1v1)
        return gamma, cost

    @torch.no_grad()
    def forward(self, grammian, from_grammian=True):
        if not from_grammian:
            grammian = torch.mm(grammian, grammian.t())

        vivj = grammian[self.ii_triu, self.jj_triu]
        vivi = grammian[self.ii_triu, self.ii_triu]
        vjvj = grammian[self.jj_triu, self.jj_triu]

        gamma, cost = self.line_solver_vectorized(vivi, vivj, vjvj)
        offset = torch.argmin(cost)
        i_min, j_min = self.i_triu[offset], self.j_triu[offset]
        sol = torch.zeros(self.n)
        sol[i_min], sol[j_min] = gamma[offset], 1. - gamma[offset]
        return sol


class MinNormSolver(nn.Module):
    def __init__(self, n_tasks, max_iter=250, stop_crit=1e-6):
        super().__init__()
        self.n = n_tasks
        self.n_ts = torch.tensor(n_tasks)

        self.linear_solver = MinNormLinearSolver()
        self.planar_solver = MinNormPlanarSolver(n_tasks)
        self.i_grid = torch.arange(n_tasks, dtype=torch.float32) + 1

        i_grid = torch.arange(n_tasks)
        j_grid = torch.arange(n_tasks)
        self.ii_grid, self.jj_grid = torch.meshgrid(i_grid, j_grid)

        self.one = torch.tensor(1.)
        self.zero = torch.zeros(n_tasks)

        self.max_iter = max_iter
        self.stop_crit = stop_crit

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
        lt_zero = torch.nonzero(proj_grad < 0).squeeze()
        gt_zero = torch.nonzero(proj_grad > 0).squeeze()
        tm1 = -cur_val[lt_zero] / proj_grad[lt_zero]
        tm2 = (1. - cur_val[gt_zero]) / proj_grad[gt_zero]

        t = self.one
        tm1_gt_zero = torch.nonzero(tm1 > 1e-7).squeeze()
        if tm1_gt_zero.shape[0] > 0:
            t = torch.min(tm1[tm1_gt_zero])

        tm2_gt_zero = torch.nonzero(tm2 > 1e-7).squeeze()
        if tm2_gt_zero.shape[0] > 0:
            t = torch.min(t, torch.min(tm2[tm2_gt_zero]))

        next_point = proj_grad * t + cur_val
        next_point = self.projection_to_simplex(next_point)
        return next_point

    @torch.no_grad()
    def forward(self, vecs):
        if self.n == 1:
            return vecs[0]
        if self.n == 2:
            v1v1 = torch.dot(vecs[0], vecs[0])
            v1v2 = torch.dot(vecs[0], vecs[1])
            v2v2 = torch.dot(vecs[1], vecs[1])
            gamma, cost = self.linear_solver(v1v1, v1v2, v2v2)
            return torch.tensor([gamma, 1. - gamma])

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
