"""
Model class for the EV charging network. Loads data of car arrivals, departures,
and energy needs, assigns cars to charging stations, and measures Statistical
properties of the charging network over time.
"""

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

    # Model class for the ACN Analysis Application. EVModel loads an
    # existing file of EV data and simulates the real-time charging
    # using the online LP algorithm.
    #
    # EVModel Parameters:
    #   startTime   - value representing the start time of the control period
    #                 for the EV set, set in name
    #   lastUpdate  - value representing the time of the last update, as
    #                 compared to startTime
    #   timeint     - Length of each control interval used in the OLP
    #                 alogirthm (in minutes)
    #   tmax        - Max #control intervals to run OLP over
    #   opt_horizon - Number of control intervals for optimization
    #   tstep       - Number of control intervals to advance when updating
    #   allEV       - #EV-by-4 matrix where each row is an EV, and the columns
    #                 specify energy demanded, arrival time, departure time, and peak
    #                 charging rate; arrival time and departure time are in #control
    #                 invervals, not minutes
    #   activeEV    - numActiveEV-by-3 matrix where each row is an active EV,
    #                 and the columns specify remaning energy demand,
    #                 reamning parking time and peak cahrging rate; arrival
    #                 time and departure time are in #control invervals, not minutes
    #   numAllEV    - Total number of EVs in the EV set
    #   numActiveEV - Number of currently active EVs
    #   numChargers - Number of chargers
    #   powerCap    - 1-by-opt_horizon vector representning the power capacity
    #                 of the ACN
    #   rate        - numAllEV-by-tmax matrix of historically optimal charging
    #                 rates calculated using the OLP algorithm. If the online
    #                 LP problem is infeasible, rate is not a vailid charging schedule.
    #   t           - Last control interval that was analyzed
    #   charger     - 1-by-numAllEV list of dictionaries representning the start of
    #                 current control interval with fields:
    #       active  - 1 if there is an active EV charging at charger ii,
    #                 0 otherwise
    #       EV      - 1x3 row vector with columns remaining energy demand at
    #                 START of current control interval, remaning parking
    #                 time and peak charging rate
    #   chargerNew  - Same as charger but for the start of the next control
    #                 interval
    #   chargerID   - numActiveEV-by-1 vector that maps each active EV in the array
    #                 ActiveEV to its charger ID (index of the struct array charger)
    #                 during this current control interval.
    #   chargerIDNew - Same as chargerID but for the start of the next control interval.
    #   schedule    - numActiveEV-by-(opt_horizon - t) matrix of curent
    #                 schedule of charging rates for active EVs
    #   histSchedule - numAllEV-by-t matrix of charging rates that has been
    #                 used to charge the set of all EV. Each row corresponds
    #                 to one of the EVs in the EV set that is studdied.
    #   percentage  - #chargers-by-1 vector of percentages of energy met
    #                at chagrger ii. If charger ii is inactive, percentage(ii) is NaN.
    #   devPercentage - Standard deviation of percentages for active EVs
    #   avgPercentage - Average percentage of energy met for the current
    #                   set of active EVs
    #   laxity      - #chargers-by-1 vector where element ii is the current
    #                 laxity of the EV at chagrger ii. If charger ii is inactive,
    #                 laxity(ii) is NaN.
    #   devLaxity   - Standard deviation of laxities for active EVs
    #   avgLaxity   - Average laxity for the current set of active EVs
    #   avgCharg    - Average charging rate for the current set of active EVs
    #   devCharg    - Standard deviation of charging rates for active EVs
    #   avgTotEnergy - Averge total energy demand for the current set of active EVs
    #   devTotEnergy - Standard deviation of total energy demand for active EVs
    #   avgRemEnergy - Average remaning energy demand for the current set of active EVs
    #   devTotEnergy - Standard deviation of remaning energy demand for active EVs
    #   predictReady - #chargers-by-1 vector of predicted times (in control intervals)
    #                  where element ii correspnds to the EV at charger ii.
    #                  If charger ii is not active or the EV alreay fully charger,
    #                  predictReady(ii) is NaN. If the energy demand is not met
    #                  with the current schedule, predReady(ii) is -1.
    #   successRate - Rate of success with the current charging schedule
    #   totalPower  - Current power consumption (in kW)
    #
    # EVModel Methods:
    #   update - Advances the EVModel tstep control intervals

    def __init__(self, dataSet, setName, fn, startTime, tstep):

        # ---------------- CONSTRUCTOR ------------------
        #
        # Input:
        #   dataSet - Name of data set, e.g. 'Mountain View', 'CAGarage'
        #   setName - Name of the specific set, e.g. 'MTV-2000', 'CAGarage'
        #   fn - File number in the specified set, e.g. 1, 2, ...
        #   startTime - Datetime object representing the start time of the
        #               control period for the EV set
        #   tstep - Number of control intervals to advance when updating
        #
        # EVModel(...) constructs a new class instance to represent a
        # model of charging EVs for the specified data set.
        #
        # ------------------------------------------------

        self.timeint = 2
        self.tmax = round(60/2 * 12)
        self.opt_horizon = 60/2 * 12

        self.startTime = startTime
        self.tstep = tstep

        # Loading data
        if dataSet == 'CAGarage':
            path = 'CAGarage2016-initial_cond-clean'
        else:
            if dataSet == 'Sunnyvale':
                path = os.path.join('Google2016-initialcond-stepwise-clean/Sunnyvale/', setName)
            else:
                path = os.path.join('Google2016-initialcond-stepwise-clean/Mountain View/', setName)

        filenames = rdir(os.path.join(os.getcwd(), 'cleandata2', path))
        fnlen = len(filenames)

        # Default dataset
        if fnlen == 0:
            filenames = rdir(os.path.join(os.getcwd(),
                             'cleandata2',
                             'Google2016-initialcond-stepwise-clean',
                             'Mountain View',
                             'MTV-2000'))
        # If want file number not in range, then just default to first file number
        if fn > fnlen:
            fn = 0

        self.allEV, _ = ev_funcs.getACN(filenames[fn], self.timeint, self.opt_horizon)
        self.numAllEV = len(self.allEV)

        # TODO (anshul): remove this random shuffle with assignments to parking spots
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
        # ------- Update EV Model ----------

        # Online LP iteration

        for ti in range(self.t + 1, self.t + self.tstep + 1):
            self.numActiveEV = 0

            for ii in range(self.numAllEV):
                self.numActiveEV += self.chargerNew[ii].get('active')

            self.chargerIDNew = []
            self.schedule = np.zeros(shape=(self.numActiveEV, self.opt_horizon))

            if self.numActiveEV > 0:
                # Extract from the 1-by-numAllEV struct array charger and store into
                # the #activeEV-by-3 array ActiveEV for each charger i that has an active
                # (charging) EV, i.e., chargers(i).active==1.
                #   chargerID : numActiveEV-by-1 vector that maps each active EV in the array
                #           ActiveEV to its charger ID (index of the struct array charger).
                #           This will be used for updating the charging rates for each
                #           active EV in ACN.
                #
                #   ActiveEV : numActiveEV-by-3 array where each row aev is an active EV
                #           and the columns specify:
                #               ActiveEV(aev, 1) = remaining energy demand;
                #               ActiveEV(aev, 2) = min(remaining parking time,
                #                                   opt_horizon-1)
                #               ActiveEV(aev, 3) = peak charging rate (scalar)
                # Note that ActiveEV(aev,2) = opt_horizon-1 if remaining parking
                # time is longer.

                self.activeEV, self.chargerIDNew = ev_funcs.getActiveEV(
                    self.chargerNew, self.numAllEV, self.numActiveEV, self.opt_horizon)
                self.activeEV[:,0] = np.round(self.activeEV[:,0], decimals=6)

                # Compute charging rates for all active EVs for all t=1, ..,
                # opt_horizon-1 and store them into the
                # numActiveEV-by-(opt_horizon-1) matrix schedule.
                self.schedule, feasible, _, _ = lp_alg.AlgLP2v2(
                    self.activeEV, self.powerCap, self.timeint, self.opt_horizon)

                if feasible:
                    # update charging rates for active EVs for time t by setting
                    # rate(chargerID(aev), t) = schedule(aev,1);
                    # This is MPC (model predictive control)
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

        # Reformat schedule and histSchedule
        self.schedule = self.schedule[:, 0:(len(self.schedule[0]) - self.t)]
        self.histSchedule = self.rate[:, 0:self.t]

        # Statistical calculations
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
    """
    Returns all .mat files that appear in any subdirectory of the rootpath directory.
    """
    matches = []
    for root, dirnames, filenames in os.walk(rootpath):
        for filename in fnmatch.filter(filenames, '*.mat'):
            matches.append(os.path.join(root, filename))
    return matches
