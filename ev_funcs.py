"""
This file contains functions used to interact with the charging network itself,
and get/update non-statistical information about the network.
"""

import scipy.io as scio
import math
import scipy
import numpy as np


def getACN(fname, timeint, opt_horizon):
    # Input:
    #   fname : file that contains an #EV-by-4 matrix A where each row is
    #       an EV, and the columns specify energy demanded, arrival time, departure
    #       time, and peak charging rate; arrival time and departure time are in
    #       minutes.
    #   timeint : duration of each control interval in minute;
    #   opt_horizon : #control intervals for optimization
    #
    # Output:
    #   AllEV : #EV-by-4 matrix A where each row is an EV, and the columns
    #           specify energy demanded, arrival time, departure time, and peak
    #           charging rate; arrival time and departuretime are in #control invervals.
    #   power_cap : 1-by-(opt_horizon-1) vector of time-varying power limit for ACN;

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
    # Input:
    #   1-by-numAllEV struc array charger with fields
    #       charger(ii).active = 1 if there is an active EV charging at charger
    #                           (parking spot) ii, and 0 otherwise;
    #       charger(ii).EV = EVii is a 1x3 row vector
    #                           EVi(1,1) = remaining energy demand;
    #                           EVi(1,2) = remaining parking time (rpt);
    #                           EVi(1,3) = peak charging rate (scalar)
    #   numAllEV : number of chargers;
    #   numActiveEV : number of active (charging) EV;
    #   opt_horizon : time horizon for optimization (in #control intervals)
    #
    # Output:
    #   numActiveEV-by-3 array ActiveEV for each charger i that has an active
    #   (charging) EV, i.e., chargers(i).active==1.
    #       chargerID : numActiveEV-by-1 vector that maps each active EV in the array
    #                   ActiveEV to its charger ID (index of the struct array charger).
    #                   This will be used for updating the charging rates for each
    #                   active EV in ACN.
    #
    #       ActiveEV : numActiveEV-by-3 array where each row aev is an active EV
    #                   and the columns specify:
    #                       ActiveEV(aev, 1) = remaining energy demand;
    #                       ActiveEV(aev, 2) = min(remaining parking time, opt_horizon-1)
    #                       ActiveEV(aev, 3) = peak charging rate (scalar)
    #   Note that ActiveEV(aev,2) = opt_horizon-1 if remaining parking time is longer.

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
    #   Input:
    #       m = #EVs
    #       EV = m by 4 matrix where each row is an EV, and the columns specify
    #            energy demanded, start time, stop time, and maximum charge rate
    #       timeint : duration of each control interval in minute;
    #       time = time horizon for optimization
    #
    #   Output:
    #       ecode is a vector; ecode = zeros means there is no ecode.  Otherwise
    #       ecode[0] = 1: no EV (m<=0)
    #       ecode[1] = 1: there exists EV with energy demand <= 0
    #       ecode[2] = 1: departure time <= arrival time for some EV
    #       ecode[3] = 1: departure time > time for some EV
    #       ecode[4] = 1: laxity<0 or laxity>1 for some EV

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
    # Input:
    #   AllEV : #AllEV-by-4 matrix where each row is an EV and the columns specify
    #       energy demanded, arrival time, departure time, and peak charging rate;
    #       arrive time and departure time are in #control invervals, not minutes.
    #   t : current time in #control intervals (not minutes);
    #   opt_horizon : #control intervals for optimization;
    #   current_rates : #AllEV-by-1 column vector of charging rates at time t;
    #   timeint : length of each control interval in minutes:
    #           remaining energy = old energy - current_rate*timeint
    #
    # Output:
    #   Initialize the 1-by-numAllEV list of dictionaries representning the start of
    #                 current control interval with fields:
    #       active  - 1 if there is an active EV charging at charger ii,
    #                 0 otherwise
    #       EV      - 1x3 row vector with columns remaining energy demand at
    #                 START of current control interval, remaning parking
    #                 time and peak charging rate

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
    print('Problem: NotifyAbort error code =     #d\n'     # ecode)
