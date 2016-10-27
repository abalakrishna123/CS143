import numpy as np

def getPercentage(charger, AllEV):
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
    if numActiveEV > 0:
        return np.sum(percentage[~np.isnan(percentage)]) / numActiveEV
    else:
        return 1

def getLaxity(charger, timeint):
    numAllEV = len(charger)
    laxity = np.empty(shape=(numAllEV, 1))
    for ii in range(numAllEV):
        if charger[ii].get('active') == 1:
            EV = charger[ii].get('EV')
            laxity[ii] = 1 - EV[0] / (EV[2] * EV[1] * timeint)

    dev = getStandardDev(laxity)
    return laxity, dev

def getAvgLaxity(laxity, numActiveEV):
    if numActiveEV > 0:
        return np.sum(laxity[~np.isnan(laxity)]) / numActiveEV
    else:
        return 1

def getAvgCharging(schedule, t):
    aev = 0
    m = len(schedule)
    tol = np.sum(schedule[:,t-1]) / (100*m)
    tolSchedule = []
    for i in range(m):
        if schedule[i][t-1] > tol:
            aev += 1
            tolSchedule.append(schedule[i][t-1])
    if aev > 0:
        avgRate = np.sum(tolSchedule)/aev
        dev = np.std(tolSchedule)
    else:
        avgRate = 0
        dev = 0
    return avgRate, dev

def getAvgRemEnergy(charger):
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
    numAllEV = len(charger)
    predReady = np.empty(shape=(numAllEV, 1))
    aev = 1
    for ii in range(numAllEV):
        if charger[ii].get('active') == 1:
            demand = charger[ii].get('EV')[0]
            if demand < 0.1:
                aev += 1
            elif np.sum(timeint * schedule[aev - 1,:]) < demand * 0.99:
                predReady[ii] = -1
                aev += 1
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
    aReady = predictReady[~np.isnan(predictReady)]
    if len(aReady) != 0:
        return sum(x > 0 for x in aReady) / len(aReady)
    else:
        return 1

def getAvgTotEnergy(allEV, chargerID, numActiveEV):
    if numActiveEV > 0:
        c_id = int(chargerID[0])
        avg = np.sum(allEV[c_id][0]) / numActiveEV
        dev = np.std(allEV[c_id][0])
    else:
        avg = 0
        dev = 0
    return avg, dev

def getTotalPower(schedule, t):
    return np.sum(schedule[:,t-1])

def getStandardDev(data):
    data = data[~np.isnan(data)]
    if len(data) != 0:
        return np.std(data)
    else:
        return 0
