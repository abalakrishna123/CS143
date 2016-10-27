import os
import glob
import fnmatch
import lp_alg
import ev_funcs
import calcs
import numpy as np

class EVError(Exception):
    pass

class EVModel(object):

    def __init__(self, dataSet, setName, fn, startTime, tstep):
        self.timeint = 2
        self.tmax = round(60/2 * 12)
        self.opt_horizon = 60/2 * 12

        self.startTime = startTime
        self.tstep = tstep

        if dataSet == 'CAGarage':
            path = 'CAGarage2016-initial_cond-clean'
        else:
            if dataSet == 'Sunnyvale':
                path = os.path.join('Google2016-initialcond-stepwise-clean/Sunnyvale/', setName)
            else:
                path = os.path.join('Google2016-initialcond-stepwise-clean/Mountain View/', setName)

        filenames = rdir(os.path.join(os.getcwd(), 'cleandata2', path))
        fnlen = len(filenames)

        if fnlen == 0:
            filenames = rdir(os.path.join(os.getcwd(), 'cleandata2', 'Google2016-initialcond-stepwise-clean', 'Mountain View', 'MTV-2000'))

        if fn > fnlen:
            fn = 0

        self.allEV, _ = ev_funcs.getACN(filenames[fn], self.timeint, self.opt_horizon)
        self.numAllEV = len(self.allEV)

        np.random.shuffle(self.allEV)

        # Check EV set
        ecode = np.zeros(shape=(5,1))
        ecode = ev_funcs.checkEV(self.allEV, self.timeint, self.opt_horizon, ecode)

        if np.sum(ecode) > 0:
            print(ecode)
            raise EVError('Error found when checking EV Set.')

        # Set power capacity
        self.powerCap = 100 * np.ones(shape=(1, self.opt_horizon))

        # Initialize rate matrix
        self.rate = np.zeros(shape=(self.numAllEV, self.tmax))

        # Initialize chargers (saved as list of dictionaries)
        self.chargerNew = []
        # The struct array charger will be continuously updated.
        for i in range(self.numAllEV):
            if 0 >= self.allEV[i][1] and 0 < self.allEV[i][2]:
                self.chargerNew.append(
                    {"active" : 1,
                    "EV" : [self.allEV[i][0], self.allEV[i][2], self.allEV[i][3]]})
            else:
                self.chargerNew.append(
                    {"active" : 0,
                    "EV" : [0, 0, 0]})

        self.t = -1  # Next control interval starts at 0

        self.update()


    def update(self):
        for ti in range(self.t + 1, self.t + self.tstep + 1):
            self.numActiveEV = 0

            for ii in range(self.numAllEV):
                self.numActiveEV += self.chargerNew[ii].get('active')

            self.chargerIDNew = []
            self.schedule = np.zeros(shape=(self.numActiveEV, self.opt_horizon))

            if self.numActiveEV > 0:
                self.activeEV, self.chargerIDNew = ev_funcs.getActiveEV(
                    self.chargerNew, self.numAllEV, self.numActiveEV, self.opt_horizon)
                self.activeEV[:,0] = np.round(self.activeEV[:,0], decimals=6)

                self.schedule, feasible, _, _ = lp_alg.AlgLP2v2(
                    self.activeEV, self.powerCap, self.timeint, self.opt_horizon)

                if feasible:
                    for aev in range(self.numActiveEV):
                        self.rate[int(self.chargerIDNew[aev])][ti] = self.schedule[aev][0]
                else:
                    print('Infeasible OLP!')

            self.charger = self.chargerNew
            self.chargerID = self.chargerIDNew
            self.chargerNew = ev_funcs.updateCharger(
                self.chargerNew, self.allEV, ti + 1, self.rate[:,ti], self.timeint)

            self.t += 1

        self.lastUpdate = self.startTime + self.timeint * self.t

        self.schedule = self.schedule[:, 0:(len(self.schedule[0]) - self.t)]
        self.histSchedule = self.rate[:, 0:self.t]

        self.numChargers = len(self.charger)
        self.percentage, self.devPercentage = calcs.getPercentage(self.charger, self.allEV)
        self.avgPercentage = calcs.getAvgPercentage(self.percentage, self.numActiveEV)
        self.laxity, self.devLaxity = calcs.getLaxity(self.charger, self.timeint)
        self.avgLaxity = calcs.getAvgLaxity(self.laxity, self.numActiveEV)
        self.avgCharg, self.devCharg = calcs.getAvgCharging(self.schedule, 1)
        self.avgRemEnergy, self.devRemEnergy = calcs.getAvgRemEnergy(self.charger)
        self.predictReady = calcs.getPredictReady(self.charger, self.schedule, self.timeint)
        self.successRate = calcs.getSuccessRate(self.predictReady)
        self.avgTotEnergy, self.devTotEnergy = calcs.getAvgTotEnergy(self.allEV, self.chargerID, self.numActiveEV)
        self.totalPower = calcs.getTotalPower(self.schedule, 1)

        # FOR DEBUGGING/OUTPUT:
        # print('start print')
        # print(self.percentage, self.devPercentage, self.avgPercentage)
        # print(self.laxity, self.devLaxity, self.avgLaxity)
        # print(self.avgCharg, self.devCharg)
        # print(self.avgRemEnergy, self.devRemEnergy)
        # print(self.predictReady, self.successRate)
        # print(self.avgTotEnergy, self.devTotEnergy)
        # print(self.totalPower)


def rdir(rootpath):
    matches = []
    for root, dirnames, filenames in os.walk(rootpath):
        for filename in fnmatch.filter(filenames, '*.mat'):
            matches.append(os.path.join(root, filename))
    return matches
