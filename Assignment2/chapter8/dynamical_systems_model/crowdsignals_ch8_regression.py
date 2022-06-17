# ##############################################################
# #                                                            #
# #    Mark Hoogendoorn and Burkhardt Funk (2017)              #
# #    Machine Learning for the Quantified Self                #
# #    Springer                                                #
# #    Chapter 8                                               #
# #                                                            #
# ##############################################################
#
from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter8.LearningAlgorithmsTemporal import TemporalRegressionAlgorithms
from definitions import ROOT_DIR

import copy
import pandas as pd
from pathlib import Path

DATA_PATH = Path(f'{ROOT_DIR}/intermediate_datafiles')
DATASET_FNAME = 'chapter5_result.csv'
#
DataViz = VisualizeDataset(__file__)
#
try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

prepare = PrepareDatasetForLearning()

learner = TemporalRegressionAlgorithms()

train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression(copy.deepcopy(dataset), ['hr_watch_rate'], 0.9, filter=False, temporal=True)
# train_X, test_X, train_y, test_y = prepare.split_single_dataset_regression(copy.deepcopy(dataset), ['acc_phone_x', 'acc_phone_y'], 0.9, filter=False, temporal=True)

output_sets = learner.dynamical_systems_model_nsga_2(train_X, train_y, test_X, test_y, ['self.hr_watch_rate'],
                                                     ['self.a * self.hr_watch_rate + self.b * self.hr_watch_rate'],
                                                     ['self.hr_watch_rate'],
                                                     ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                     pop_size=10, max_generations=10, per_time_step=True)
# output_sets = learner.dynamical_systems_model_nsga_2(train_X, train_y, test_X, test_y, ['self.acc_phone_x', 'self.acc_phone_y', 'self.acc_phone_z'],
#                                                      ['self.a * self.acc_phone_x + self.b * self.acc_phone_y', 'self.c * self.acc_phone_y + self.d * self.acc_phone_z', 'self.e * self.acc_phone_x + self.f * self.acc_phone_z'],
#                                                      ['self.acc_phone_x', 'self.acc_phone_y'],
#                                                      ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
#                                                      pop_size=10, max_generations=10, per_time_step=True)
DataViz.plot_pareto_front(output_sets)

DataViz.plot_numerical_prediction_versus_real_dynsys_mo(train_X.index, train_y, test_X.index, test_y, output_sets, 0, 'hr_watch_rate')
# DataViz.plot_numerical_prediction_versus_real_dynsys_mo(train_X.index, train_y, test_X.index, test_y, output_sets, 0, 'acc_phone_x')

regr_train_y, regr_test_y = learner.dynamical_systems_model_ga(train_X, train_y, test_X, test_y, ['self.hr_watch_rate'],
                                                     ['self.a * self.hr_watch_rate + self.b * self.hr_watch_rate'],
                                                     ['self.hr_watch_rate'],
                                                     ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                     pop_size=5, max_generations=10, per_time_step=True)
# regr_train_y, regr_test_y = learner.dynamical_systems_model_ga(train_X, train_y, test_X, test_y, ['self.acc_phone_x', 'self.acc_phone_y', 'self.acc_phone_z'],
#                                                      ['self.a * self.acc_phone_x + self.b * self.acc_phone_y', 'self.c * self.acc_phone_y + self.d * self.acc_phone_z', 'self.e * self.acc_phone_x + self.f * self.acc_phone_z'],
#                                                      ['self.acc_phone_x', 'self.acc_phone_y'],
#                                                      ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
#                                                      pop_size=5, max_generations=10, per_time_step=True)

DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y['hr_watch_rate'], regr_train_y['hr_watch_rate'], test_X.index, test_y['hr_watch_rate'], regr_test_y['hr_watch_rate'], 'hr_watch_rate')
# DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y['acc_phone_x'], regr_train_y['acc_phone_x'], test_X.index, test_y['acc_phone_x'], regr_test_y['acc_phone_x'], 'acc_phone_x')

regr_train_y, regr_test_y = learner.dynamical_systems_model_sa(train_X, train_y, test_X, test_y, ['self.hr_watch_rate'],
                                                     ['self.a * self.hr_watch_rate + self.b * self.hr_watch_rate'],
                                                     ['self.hr_watch_rate'],
                                                     ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
                                                     max_generations=10, per_time_step=True)
# regr_train_y, regr_test_y = learner.dynamical_systems_model_sa(train_X, train_y, test_X, test_y, ['self.acc_phone_x', 'self.acc_phone_y', 'self.acc_phone_z'],
#                                                      ['self.a * self.acc_phone_x + self.b * self.acc_phone_y', 'self.c * self.acc_phone_y + self.d * self.acc_phone_z', 'self.e * self.acc_phone_x + self.f * self.acc_phone_z'],
#                                                      ['self.acc_phone_x', 'self.acc_phone_y'],
#                                                      ['self.a', 'self.b', 'self.c', 'self.d', 'self.e', 'self.f'],
#                                                      max_generations=10, per_time_step=True)

DataViz.plot_numerical_prediction_versus_real(train_X.index, train_y['hr_watch_rate'], regr_train_y['hr_watch_rate'], test_X.index, test_y['hr_watch_rate'], regr_test_y['hr_watch_rate'], 'hr_watch_rate')
