from Python3Code.Chapter2.CreateDataset import CreateDataset
from definitions import ROOT_DIR

def plotSensorData() -> None:
    GRANULARITIES = [60000, 250]

    for milliseconds_per_instance in GRANULARITIES:
        dataset: CreateDataset = CreateDataset(f'{ROOT_DIR}/Python3Code/phyphox_exports', milliseconds_per_instance)

        dataset.add_numerical_dataset('gyroscope_noise.csv', 'Time (s)', ['Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)'], 'avg', 'acc_phone_noise_')

if __name__ == '__main__':
    plotSensorData()
