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
