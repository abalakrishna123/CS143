"""
This file contains numerous statsitical calculations on a charging network that
can be useful to characterize the macrobehavior of the network.
"""

import numpy as np

def getPercentage(charger, AllEV):
    # Input:
    #     charger     - 1-by-numAllEV list of dictionaries representning the start of
    #                   current control interval with fields:
    #         active  - 1 if there is an active EV charging at charger ii,
    #                   0 otherwise
    #         EV      - 1x3 row vector with columns remaining energy demand at
    #                   START of current control interval, remaning parking
    #                   time and peak charging rate
    #     allEV       - #EV-by-4 matrix where each row is an EV, and the columns
    #                   specify energy demanded, arrival time, departure time, and peak
    #                   charging rate; arrival time and departure time are in #control
    #                   invervals, not minutes
    # Output:
    #     percentage = #chargers-by-1 vector of percentages of energy met
    #                  at chagrger ii. If charger ii is inactive,
    #                  percentage(ii) is set to NaN.
    #     dev = standard deviation of percentages for active chargers

    numChargers = len(charger)
    percentage = np.empty(shape=(numChargers, 1))
    aev = 0
    for ii in range(numChargers):
        if charger[ii].get('active') == 1:
            aev += 1
            percentage[ii] = 1 - (charger[ii].get('EV')[0] / AllEV[ii][0])

    dev = getStandardDev(percentage)
    return percentage, dev


def getAvgPercentage(percentage, numActiveEV):
    # Input:
    #     percentage = #chargers-by-1 vector of percentages of energy met
    #                  at chagrger ii. If charger ii is inactive,
    #                  percentage(ii) is set to NaN.
    #     numActiveEV - Number of currently active EVs
    # Output:
    #     average percentages for active chargers

    if numActiveEV > 0:
        return np.sum(percentage[~np.isnan(percentage)]) / numActiveEV
    else:
        return 1


def getLaxity(charger, timeint):
    # Input:
    #     charger     - 1-by-numAllEV list of dictionaries representning the start of
    #                   current control interval with fields:
    #         active  - 1 if there is an active EV charging at charger ii,
    #                   0 otherwise
    #         EV      - 1x3 row vector with columns remaining energy demand at
    #                   START of current control interval, remaning parking
    #                   time and peak charging rate
    #     timeint = length of control interval
    #
    # Output:
    #     laxity = m-by-1 vector of laxities for the chargers. If a charger
    #              is inactive, laxity(ii) is set to NaN
    #     dev = standard deviation of laxities for active chargers

    numAllEV = len(charger)
    laxity = np.empty(shape=(numAllEV, 1))
    for ii in range(numAllEV):
        if charger[ii].get('active') == 1:
            EV = charger[ii].get('EV')
            laxity[ii] = 1 - EV[0] / (EV[2] * EV[1] * timeint)

    dev = getStandardDev(laxity)
    return laxity, dev


def getAvgLaxity(laxity, numActiveEV):
    # Input:
    #     laxity = m-by-1 vector of laxities for the chargers. If a charger
    #              is inactive, laxity(ii) is set to NaN
    #     numActiveEV - Number of currently active EVs
    # Output:
    #     average laxity for active chargers

    if numActiveEV > 0:
        return np.sum(laxity[~np.isnan(laxity)]) / numActiveEV
    else:
        return 1


def getAvgCharging(schedule, t):
    # Input:
    #   schedule    - numActiveEV-by-(opt_horizon - t) matrix of curent
    #                 schedule of charging rates for active EVs
    #   t           - Last control interval that was analyzed
    #
    # Output:
    #   avgRate = average charging rate at time t (of vehicles that are
    #             currently being charged)
    #   dev = sample standard deviation

    aev = 0
    m = len(schedule)
    tol = np.sum(schedule[:,t-1]) / (100*m)  # Tolerance level
    tolSchedule = []

    # Only include EVs with a charging rate exceeding tol
    for i in range(m):
        if schedule[i][t-1] > tol:
            aev += 1
            tolSchedule.append(schedule[i][t-1])
    if aev > 0:
        avgRate = np.sum(tolSchedule)/aev
        dev = np.std(tolSchedule)
    else:
        # No EVs charging at time t
        avgRate = 0
        dev = 0
    return avgRate, dev


def getAvgRemEnergy(charger):
    # Input:
    #     charger     - 1-by-numAllEV list of dictionaries representning the start of
    #                   current control interval with fields:
    #         active  - 1 if there is an active EV charging at charger ii,
    #                   0 otherwise
    #         EV      - 1x3 row vector with columns remaining energy demand at
    #                   START of current control interval, remaning parking
    #                   time and peak charging rate
    #
    # Output:
    #     avgRemEnergy = average remaning energy demand
    #     dev = standard deviation

    numChargers = len(charger)
    remEnergy = []
    aev = 0
    for ii in range(numChargers):
        if charger[ii].get('active') == 1:
            remEnergy.append(charger[ii].get('EV')[0])
            aev += 1
    if aev > 0:
        avgRemEnergy = np.sum(remEnergy)/aev
        dev = np.std(remEnergy)
    else:
        avgRemEnergy = 0
        dev = 0
    return avgRemEnergy, dev


def getPredictReady(charger, schedule, timeint):
    # Input:
    #     charger     - 1-by-numAllEV list of dictionaries representning the start of
    #                   current control interval with fields:
    #         active  - 1 if there is an active EV charging at charger ii,
    #                   0 otherwise
    #         EV      - 1x3 row vector with columns remaining energy demand at
    #                   START of current control interval, remaning parking
    #                   time and peak charging rate
    #     schedule    - numActiveEV-by-(opt_horizon - t) matrix of curent
    #                   schedule of charging rates for active EVs
    #     timeint     - Length of each control interval used in the OLP
    #                   alogirthm (in minutes)
    # Output:
    #     predReady = #chargers-by-1 vector of predicted times when the EV
    #                 at charger ii is fully charged (to 99 %) in minutes. If
    #                 charger ii is not active or EV alreay fully charger,
    #                 predReady(ii) is set to NaN. If the energy demand is not
    #                 met with the current schedule, predReady(ii) is set to -1.
    #
    # Assuming that the EVs in schedule are in the same order as in charger

    numAllEV = len(charger)
    predReady = np.empty(shape=(numAllEV, 1))
    aev = 1
    for ii in range(numAllEV):
        # Check if still need charge
        if charger[ii].get('active') == 1:
            demand = charger[ii].get('EV')[0]
            # Already fully charged
            if demand < 0.1:
                aev += 1
            # Demand will be met with current schedule
            elif np.sum(timeint * schedule[aev - 1,:]) < demand * 0.99:
                predReady[ii] = -1
                aev += 1
            # Find time instance when fully charged (99%)
            else:
                charged = 0
                t = 0
                while demand*0.99 >= charged:
                    charged += schedule[aev - 1][t] * timeint
                    t += 1
                predReady[ii] = t * timeint
                aev += 1
    return predReady


def getSuccessRate(predictReady):
    # Input:
    #   predictReady - #chargers-by-1 vector of predicted times (in control intervals)
    #                  where element ii correspnds to the EV at charger ii.
    #                  If charger ii is not active or the EV alreay fully charger,
    #                  predictReady(ii) is NaN. If the energy demand is not met
    #                  with the current schedule, predReady(ii) is -1
    #
    # Output:
    #   fraction of active chargers that will be ready on current schedule

    aReady = predictReady[~np.isnan(predictReady)]
    if len(aReady) != 0:
        return sum(x > 0 for x in aReady) / len(aReady)
    else:
        return 1


def getAvgTotEnergy(allEV, chargerID, numActiveEV):
    # Input:
    #   allEV       - #EV-by-4 matrix where each row is an EV, and the columns
    #                 specify energy demanded, arrival time, departure time, and peak
    #                 charging rate; arrival time and departure time are in #control
    #                 invervals, not minutes
    #   charger     - 1-by-numAllEV list of dictionaries representning the start of
    #                 current control interval with fields:
    #       active  - 1 if there is an active EV charging at charger ii,
    #                 0 otherwise
    #       EV      - 1x3 row vector with columns remaining energy demand at
    #                 START of current control interval, remaning parking
    #                 time and peak charging rate
    #   numActiveEV - Number of currently active EVs
    #
    # Output:
    #   avg - avarge amount of energy being used over the active EV's
    #   dev - standard deviation in amount of energy being used over the active EV's

    if numActiveEV > 0:
        c_id = int(chargerID[0])
        avg = np.sum(allEV[c_id][0]) / numActiveEV
        dev = np.std(allEV[c_id][0])
    else:
        avg = 0
        dev = 0
    return avg, dev


def getTotalPower(schedule, t):
    # Input:
    #     schedule = matrix of charging rates
    #     t = time instant (in control intervals)
    #
    # Output:
    #     total power consumed at time t accordning to schedule

    return np.sum(schedule[:,t-1])


def getStandardDev(data):
    # Input:
    #     data - some list of values
    #
    # Output:
    #     standard deviation over the non NaN elements in the data

    data = data[~np.isnan(data)]
    if len(data) != 0:
        return np.std(data)
    else:
        return 0
