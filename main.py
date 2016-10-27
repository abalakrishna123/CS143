"""
This file is the main control file for the EV simulator. Generates a model
and then sees how the model behaves over time given a dataset of EV arrivals,
departures, and energy need.
"""

import evmodel

# Data set group name (Mountain View, CAGarage, etc)
dataSet = 'Mountain View'
# Particular set name (MTV-41, CaGarage, etc)
setName = 'MTV-41'
# File number in dataSet/setName
fn = 0
# Start time of control period
startTime = 0

# Number of control intervals to advance when updating
tstep = 1
# Time in seconds between updates
timerPeriod = 1

def main():
    # Create and setup model parameters
    model = evmodel.EVModel(dataSet, setName, fn, startTime, tstep)

main()
