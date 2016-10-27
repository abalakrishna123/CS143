import scipy.io as scio
import math
import scipy
import numpy as np


def getACN(fname, timeint, opt_horizon):
    matfile = scio.loadmat(fname)
    AllEV = np.array(matfile['A'])

    if round(timeint) <= 0:
        NotifyAbort(1)
    elif round(timeint) > 1:
        AllEV[:,1] = np.floor(AllEV[:,1]/timeint)
        AllEV[:,2] = np.ceil(AllEV[:,2]/timeint)

    power_cap = 20*np.ones(shape=(1, opt_horizon - 1))

    return AllEV, power_cap

def getActiveEV(charger, numAllEV, numActiveEV, opt_horizon):
    ActiveEV = np.zeros(shape=(numActiveEV, 3))
    chargerID = np.zeros(shape=(numActiveEV, 1))
    aev = 0
    for ev in range(numAllEV):
        if charger[ev].get('active') == 1:
            ActiveEV[aev,:] = charger[ev].get('EV')
            if ActiveEV[aev][1] > opt_horizon:
                ActiveEV[aev][1] = opt_horizon - 1
            chargerID[aev] = ev
            aev += 1
    return ActiveEV, chargerID

def checkEV(EV, timeint, time, ecode):
    # Check #EV > 0
    m = len(EV)
    if m <= 0:
        ecode[0] = 1

    # Check energy demand > 0
    if sum(EV[:,0] <= 0) > 0:  # EV demands <= 0 energy
        ecode[1] = 1

    # Check arrival times < departure times for all EV
    duration = EV[:,2] - EV[:,1]
    if sum(duration <= 0) > 0:
        ecode[2] = 1

    # Check departure times < time for all EV
    deadline = time - EV[:,2]
    if sum(deadline < 0) > 0:
        ecode[3] = 1

    # Check:
    #   laxity = 1 - energy_demand / ((departure - arriva)*max_rate*timeint) >= 0
    laxity = 1 - EV[:,0] / ((EV[:,2] - EV[:,1]) * EV[:,3] * timeint)
    if sum(laxity < 0) > 0 or sum(laxity > 1) > 0:
        ecode[4] = 1

    return ecode

def updateCharger(oldcharger, AllEV, t, current_rate, timeint):
    charger = oldcharger
    numAllEV = len(AllEV)
    for ii in range(numAllEV):
        if t > AllEV[ii][1] and t < AllEV[ii][2]:
            if oldcharger[ii].get('active') == 1:
                charger[ii]['active'] = 1
                charger[ii]['EV'][0] = oldcharger[ii]['EV'][0] - current_rate[ii]*timeint
                charger[ii]['EV'][1] = AllEV[ii][2] - t
                charger[ii]['EV'][2] = AllEV[ii][3]
            else:
                charger[ii]['active'] = 1
                charger[ii]['EV'] = [AllEV[ii][0], AllEV[ii][2] - t, AllEV[ii][3]]
        else:
            charger[ii]['active'] = 0
            charger[ii]['EV'] = [0, 0, 0]
    return charger

def NotifyAbort(ecode):
    print('Problem: NotifyAbort error code = %d\n' % ecode)
