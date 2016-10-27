import scipy.io as scio
import math
import scipy
import numpy as np


def AlgLP2v2(EV, powermax, timeint, time):
    m = len(EV)
    time = int(time)

    A1 = np.zeros(shape=(time, m*time))
    for c1 in range(time):
        for c2 in range(m):
            A1[c1][c2 + m*c1] = 1
    b1 = powermax

    Aeq = np.zeros(shape=(m, m*time))
    for c1 in range(time):
        Aeq[:, m*c1:m*(c1+1)] = np.identity(m)
    Aeq = Aeq * timeint
    beq = EV[:, 0]

    bounds_arr = np.zeros(shape=(m*time, 2))
    for c1 in range(time):
        for c2 in range(m):
            if c1 < EV[c2][1]:
                bounds_arr[c2 + m*c1][1] = EV[c2][2]
    bounds = []
    for i in range(len(bounds_arr)):
        bounds.append((bounds_arr[i][0], bounds_arr[i][1]))

    f = np.zeros(m*time)
    for c1 in range(time):
        for c2 in range(m):
            if c1 < EV[c2][1]:
                f[c2 + m*c1] = c1 + 1

    options = {'disp': False}
    res = scipy.optimize.linprog(f, A1, b1, Aeq, beq, bounds, options=options)
    x = res.get('x')
    if res.get('success'):
        schedule = np.zeros(shape=(m, time))
        for c1 in range(time):
            schedule[:, c1] = x[m*c1:m*(c1+1)]
        feasible = 1
    else:
        schedule = np.ones(shape=(m, time)) * -1
        feasible = 0

    return schedule, feasible, res.get('success'), res.get('fun')
