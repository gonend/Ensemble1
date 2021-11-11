import read_csv as reader
from data_preparation import DataPreparation
from models import DTC


def iterate_files():
    """
    for each csv file - read, prepare, fit and predict data.
    :return:
    """

    file_names = reader.read_file_names('data')
    for name in file_names:
        dp = DataPreparation()
        dp.df = reader.read_file(name)
        prepare_data(dp)
        return dp


def prepare_data(dp):
    """
    calls all relevant functions from data_preparation class
    :param dp:
    :return:
    """
    dp.add_missing_data_10_percent()
    dp.fill_na()
    dp.discretization()
    dp.partition_data_sets()


def train_model(dp, class_model):
    raise NotImplemented
    class_model.model.fit()  # dp.x, dp.y, etc...


def run_complete_model():  # gonen working on this
    raise NotImplemented

    preparedData = iterate_files()

    dtc = DTC()  # 'normal' DecisionTreeClassifier
    train_model(dp, dtc)


def run_missing_values_in_prediction_model():  # guy working on this
    raise NotImplemented

    preparedData = iterate_files()
    train_model()


def run_missing_values_in_training_model():
    raise NotImplemented


if __name__ == '__main__':
    run_complete_model()
    run_missing_values_in_prediction_model()
    run_missing_values_in_training_model()