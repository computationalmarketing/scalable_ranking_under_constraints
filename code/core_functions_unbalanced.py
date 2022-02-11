
# Copyright (C) 2022 Yegor Tkachenko, Wassim Dhaouadi, Kamel Jedidi
# Code -- Scaling up Ranking under Constraints for Live Recommendations by Replacing Optimization with Prediction
# https://github.com/computationalmarketing/scalable_ranking_under_constraints/

# code implementing dual bipartite weighted matching linear program
# and matching algorithms



import numpy as np
import pandas as pd
import cvxopt
from scipy.optimize import linear_sum_assignment
import time
import cvxpy as cp
from multiprocessing import Pool
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from scipy import sparse
import matplotlib.tri as tri


def cvxpy_solve_dual(U, A_list, b, verbose=True, solver='CBC'):
    # http://people.lids.mit.edu/pari/online_ranking.pdf
    # note that our decision variable vector x is of dim = k + n + n
        
    n = U.shape[0]
    m = U.shape[1]
    k = b.shape[0]
    
    assert len(A_list) == k
    
    lmbd = cp.Variable((k, 1))
    alpha = cp.Variable((m, 1))
    beta = cp.Variable((n, 1))
    
    onevec_n = np.ones((n, 1))
    onevec_m = np.ones((m, 1))

    U = cp.Constant(U)
    A = cp.sum([lmbd[i]*cp.Constant(A_list[i]) for i in range(k)]) # constraint matrix component
    
    objective = cp.Maximize(b.T@lmbd + cp.sum(alpha) + cp.sum(beta))
    constraints = [U + A + onevec_n@alpha.T + beta@onevec_m.T <= 0, lmbd >= 0]
    
    if m < n:
        constraints.append(beta <= 0) 
        # corresponding to primal \sum_j P_ij <= 1 \forall i 
        # (n constraints corresponding to n dual variables coded by beta)
        # in case of =, beta is free, but in case of <=, beta <= 0
    elif m > n:
        constraints.append(alpha <= 0)

    prob = cp.Problem(objective, constraints)

    if solver == 'CBC':
        prob.solve(solver=cp.CBC, verbose=verbose, GomoryCuts=True, MIRCuts=True,
               MIRCuts2=True, TwoMIRCuts=True, ResidualCapacityCuts=True,
               KnapsackCuts=True, FlowCoverCuts=True, CliqueCuts=True,
               LiftProjectCuts=True, AllDifferentCuts=False, OddHoleCuts=True,
               RedSplitCuts=False, LandPCuts=False, PreProcessCuts=False,
               ProbingCuts=True, SimpleRoundingCuts=True) #, maximumSeconds=10
    else:
        prob.solve()
    
    return prob, np.maximum(lmbd.value,0) # to avoid numerical imprecision


def user_dual(U, A_list, b):
    # wrapper to solve dual problem for the user

    solution, lmbd = cvxpy_solve_dual(U, A_list, b, verbose=False)

    if solution.status == "optimal":
        return lmbd.squeeze()
    elif solution.status == "infeasible":
        print('Infeasible problem')
        return None
    elif solution.status == "unbounded":
        print('Unbounded problem')
        return None
    else:
        return None


def greedy_max_match(W):
    # greedy heuristic via sorting
    # https://stackoverflow.com/questions/36072577/complexity-of-a-greedy-assignment-algorithm
    # complexity sorting - nm (n^2 when equal number of vertices)
    # https://link.springer.com/chapter/10.1007/3-540-49116-3_24

    n = W.shape[0]
    m = W.shape[0]
    
    row_deleted = [False] * n
    col_deleted = [False] * m
    
    rows, cols = np.unravel_index(np.argsort(-W, axis=None), W.shape)
    rows, cols = rows.squeeze(), cols.squeeze()
    
    P = np.zeros_like(W)
    
    for i in range(rows.shape[0]):
        ro = rows[i].item()
        co = cols[i].item()
        if not row_deleted[ro] and not col_deleted[co]:
            P[ro,co] = 1
            row_deleted[ro] = True
            col_deleted[co] = True
            
        if (sum(row_deleted) == n) or (sum(col_deleted) == m):
            break
    
    return P


# https://antimatroid.wordpress.com/2017/03/21/a-greedy-approximation-algorithm-for-the-linear-assignment-problem/
# http://www.cs.ust.hk/mjg_lib/bibs/DPSu/DPSu.Files/sdarticle_95.pdf

def optimal_ranking(U, A_list, lmbd, method='hungarian', EPSILON=1.00001):
    
    # variable EPSILON is used to break ties (it denotes 1.0+\epsilon referenced in the paper)

    S = U.copy() # constructing an adjusted (ethical) rating matrix
    for i in range(len(A_list)):
        S += EPSILON * lmbd[i] * A_list[i]
    
    if method == 'rearrangement':
        # gives optimal result if, after sorting on the first column, S is a dot product of two descending vectors 
        # in balanced case: O(m log m)
        # general inverse monge requires special treatment in unbalanced case
        # http://www.cs.ust.hk/mjg_lib/bibs/DPSu/DPSu.Files/sdarticle_95.pdf
        # O((n − m + 1)m)
        
        # https://onlinelibrary.wiley.com/doi/abs/10.1002/net.21507
        P = np.zeros_like(S)
        P[np.argsort(-S[:,0],axis=0).squeeze()[:S.shape[1]], list(range(S.shape[1]))] = 1
    
    elif method == 'monge':
        # greedy heuristic O(n) - only implemeneted for the balanced case
        assert S.shape[0] == S.shape[1]
        P = np.eye(S.shape[0])

    elif method == 'greedy':
        # greedy heuristic O(nm)
        P = greedy_max_match(S)
            
    elif method == 'hungarian':        
        # Hungarian algorithm - optimal linear sum assignment
        # http://zafar.cc/2017/7/19/hungarian-algorithm/
        # O(nm^2) for m <= n (O(min(n, m)^2 max(n, m)) in general) 
        row_ind, col_ind = linear_sum_assignment(-S) # operates on 'costs'
        P = np.zeros_like(S)
        P[row_ind, col_ind] = 1
    
    else:
        print('Unrecognized methods')
        return None
        
    return P, S






