"""
This code contains the linear programming solution used in an adaptive charging
network approach to the distribution of energy.
"""

import scipy.io as scio
import math
import scipy
import numpy as np


def AlgLP2v2(EV, powermax, timeint, time):
    # Input:
    #   EV : m-by-3 matrix where each row is an active (charging) EV, and the
    #           columns specify:
    #               EV(:, 1) = remaining energy demand;
    #               EV(:, 2) = remaining parking time (rpt);
    #               EV(:, 3) = peak charging rate (scalar)
    #   powermax : time-by-1 column vector of power capacity for ACN;
    #   timeint : length of control interval in minutes; used in equality
    #               constraint: sum_t X_i(t) * timeint  =  energydemand
    #   time : time horizon for optimization
    #
    # AlgLP2(...) computes an optimal charging schedule for the m EVs over the
    # horizon [0, 1, 2, ..., time-1] subject to max power constraint vector powermax,
    # using linear programming. The variable of the LP is an m*time
    # dimensional vector X where X(1:m) are the charging rates at
    # time 0, X(m+1:2m) are the charging rates at time 1, ...,
    # X((time-1)*m+1 : time*m) are the charging rates at time time-1.
    #
    # Output:
    #   schedule = m x time matrix of optimal charging schedule if
    #               feasible==1; otherwise, schedule is a matrix of -1's.
    #   feasible = 1 if LP feasible and converges; 0 otherwise
    #   res.get('success') = flag if LP is feasible and converges
    #   res.get('fun') = optimal objective value of LP if feasible==1

    m = len(EV)
    time = int(time)

    # Create A and b for inequality contraints A*x <= b for LP
    #     Sum of rates <= powermax: A1*x <= powermax
    A1 = np.zeros(shape=(time, m*time))
    for c1 in range(time):
        for c2 in range(m):
            A1[c1][c2 + m*c1] = 1
    b1 = powermax

    # Create Aeq and beq for equality constraints for LP
    #    Satisfy all demands exactly
    #      Note that energy in each period is X_i(t) * timeint, not X_i(t)
    Aeq = np.zeros(shape=(m, m*time))
    for c1 in range(time):
        Aeq[:, m*c1:m*(c1+1)] = np.identity(m)
    Aeq = Aeq * timeint
    beq = EV[:, 0]

    # Upper and lower bounds
    #     Individual rates lower bounded by 0: 0 <= x
    #     Individual rates <= max rates: x <= r-bar
    #       for each EV i=1:m, set max rates to EV(:,3) if t < EV(i:m,2)
    bounds_arr = np.zeros(shape=(m*time, 2))
    for c1 in range(time):
        for c2 in range(m):
            if c1 < EV[c2][1]:
                bounds_arr[c2 + m*c1][1] = EV[c2][2]
    bounds = []
    for i in range(len(bounds_arr)):
        bounds.append((bounds_arr[i][0], bounds_arr[i][1]))

    # Set linear cost function
    #   The charging rate r_i(t) of EV i at time t is weighted by (for ONline LP)
    #         t %* laxity(i)
    #   for t = 0, ..., t_i (remaining parking time EV(i, 2)
    f = np.zeros(m*time)
    for c1 in range(time):  # for each time instant c1=1, ..., time-1
        for c2 in range(m):
            if c1 < EV[c2][1]:
                f[c2 + m*c1] = c1 + 1

    # Solve LP
    options = {'disp': False}
    res = scipy.optimize.linprog(f, A1, b1, Aeq, beq, bounds, options=options)
    x = res.get('x')
    # LP is feasible and has converged to a solution
    if res.get('success'):
        schedule = np.zeros(shape=(m, time))
        for c1 in range(time):
            schedule[:, c1] = x[m*c1:m*(c1+1)]
        feasible = 1
    # LP is infeasible
    else:
        schedule = np.ones(shape=(m, time)) * -1
        feasible = 0

    return schedule, feasible, res.get('success'), res.get('fun')
