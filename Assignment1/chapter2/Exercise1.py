from Python3Code.Chapter2.CreateDataset import CreateDataset

def plotSensorData -> None:
    GRANULARITIES = [60000, 250]

    for milliseconds_per_instance in GRANULARITIES:
        dataset = CreateDataset('./d', milliseconds_per_instance)

if __name__ == '__main__':
    pass