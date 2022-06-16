import os
import pandas as pd

from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from definitions import ROOT_DIR
from pathlib import Path
from typing import Union, List, Dict


def plotSensorData(_base_path: str, _numerical_datasets: List[Dict[str, Union[str, List[str]]]]) -> None:
    GRANULARITY: List[int] = [1000, 250]
    DataViz: VisualizeDataset = VisualizeDataset(__file__)

    for milliseconds_per_instance in GRANULARITY:
        dataset: CreateDataset = CreateDataset(Path(_base_path), milliseconds_per_instance)

        for numerical_dataset in _numerical_datasets:
            dataset.add_numerical_dataset(file=f'Transformed{numerical_dataset["filename"]}',
                                          timestamp_col=numerical_dataset['col_name_time'],
                                          value_cols=numerical_dataset['attributes'],
                                          aggregation='avg',
                                          prefix=numerical_dataset['prefix'])
        df_dataset: pd.DataFrame = dataset.data_table

        df_dataset_annotate: pd.DataFrame = df_dataset.copy(deep=True) # For annotation purposes
        df_dataset_annotate.insert(0, 'duration_sec', (df_dataset_annotate.index - df_dataset_annotate.index[0]).astype('timedelta64[s]') % 60)
        df_dataset_annotate.insert(0, 'duration_min', (df_dataset_annotate.index - df_dataset_annotate.index[0]).astype('timedelta64[m]'))
        df_dataset_annotate.insert(0, 'timestamp', df_dataset_annotate.index.values.astype(int))

        DataViz.plot_dataset_boxplot(df_dataset, [_ for _ in df_dataset.columns])
        dataset.add_event_dataset(file='labels.csv',
                                  start_timestamp_col='label_start',
                                  end_timestamp_col='label_end',
                                  value_col='label',
                                  aggregation='binary')
        df_dataset: pd.DataFrame = dataset.data_table
        DataViz.plot_dataset(df_dataset,
                             [numerical_dataset['prefix'] for numerical_dataset in _numerical_datasets] + ['label'],
                             ['like' for _ in range(len(_numerical_datasets) + 1)],
                             ['line' for _ in range(len(_numerical_datasets))] + ['points'])
    chapter2_result_path: str = f'{_base_path}/chapter2_result.csv'

    if not os.path.isfile(chapter2_result_path):
        # Finally, store the last dataset we generated (250 ms).
        df_dataset.to_csv(Path(chapter2_result_path))

def transformNumDatasets(_base_path: str, _path_metadata: str, _numerical_datasets: List[Dict[str, Union[str, List[str]]]]) -> None:
    df_metadata: pd.DataFrame = pd.read_csv(_path_metadata)

    for numerical_dataset in _numerical_datasets:
        df_dataset: pd.DataFrame = pd.read_csv(f'{ROOT_DIR}/phyphox_exports/{numerical_dataset["filename"]}')
        export_path: str = f'{_base_path}/Transformed{numerical_dataset["filename"]}'
        start_time_experiment: float = df_metadata[df_metadata['event'] == 'START']['system time'].iloc[0]

        if os.path.isfile(export_path):
            continue
        df_dataset[numerical_dataset['col_name_time']] = (df_dataset[numerical_dataset['col_name_time']] + start_time_experiment) * 1000 * 1000 * 1000
        df_dataset: pd.DataFrame = df_dataset.rename(columns=dict(zip(numerical_dataset['col_names_numerical'],
                                                                      numerical_dataset['attributes'])))
        df_dataset.to_csv(export_path, index=False)

if __name__ == '__main__':
    base_path: str = f'{ROOT_DIR}/phyphox_exports'
    attributes: List[str] = ['x', 'y', 'z']
    col_name_time: str = 'Time (s)'
    numerical_datasets: List[Dict[str, Union[str, List[str]]]] = [
        {
            'filename': 'Accelerometer.csv',
            'prefix': 'acc_',
            'attributes': attributes,
            'col_names_numerical': ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'],
            'col_name_time': col_name_time
        },
        {
            'filename': 'Gyroscope.csv',
            'prefix': 'gyr_',
            'attributes': attributes,
            'col_names_numerical': ['Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)'],
            'col_name_time': col_name_time
        },
        {
            'filename': 'Linear Acceleration.csv',
            'prefix': 'lin_acc_',
            'attributes': attributes,
            'col_names_numerical': ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)'],
            'col_name_time': col_name_time
        },
        {
            'filename': 'Magnetometer.csv',
            'prefix': 'mag_',
            'attributes': attributes,
            'col_names_numerical': ['Magnetic field x (µT)', 'Magnetic field y (µT)', 'Magnetic field z (µT)'],
            'col_name_time': col_name_time
        }
    ]

    transformNumDatasets(_base_path=base_path, _path_metadata=f'{base_path}/meta/time.csv', _numerical_datasets=numerical_datasets)
    plotSensorData(_base_path=base_path, _numerical_datasets=numerical_datasets)
