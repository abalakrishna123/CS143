import evmodel


dataSet = 'Mountain View'
setName = 'MTV-41'
fn = 0
startTime = 0
tstep = 1
timerPeriod = 1

def main():
    model = evmodel.EVModel(dataSet, setName, fn, startTime, tstep)


main()
