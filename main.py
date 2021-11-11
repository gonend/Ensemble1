import read_csv as reader
from data_preparation import DataPreparation
from models import DTC


def iterate_files(impute=True):
    """
    for each csv file - read, prepare, fit and predict data.
    :return:
    """

    file_names = reader.read_file_names('data')
    for name in file_names:
        dp = DataPreparation()
        dp.df = reader.read_file(name)
        prepare_data(dp, impute)
        return dp


def prepare_data(dp, impute):
    """
    calls all relevant functions from data_preparation class
    :param impute:
    :param dp:
    :return:
    """
    dp.add_missing_data_10_percent()
    if impute:
        dp.fill_na()
        dp.discretization()
    dp.partition_data_sets()


def train_model(class_model, x_train, x_test, y_train, y_test):
    raise NotImplemented
    class_model.model.fit()
    class_model.model.predict()


def run_complete_model():  # gonen working on this
    raise NotImplemented
    preparedData = iterate_files(impute=True)

    dtc = DTC()  # 'normal' DecisionTreeClassifier
    train_model(dtc, preparedData.x_train, preparedData.x_test, preparedData.y_train, preparedData.y_test)


def run_missing_values_in_prediction_model():  # guy working on this
    raise NotImplemented
    prepared_data = iterate_files(impute=True)
    train_model(model, prepared_data.x_train, prepared_data.x_test, prepared_data.y_train, prepared_data.y_test)


def run_missing_values_in_training_model():
    raise NotImplemented
    prepared_data = iterate_files(impute=False)

    # process y_test
    y_dp = DataPreparation()
    y_dp.df = prepared_data.y_test.to_frame()
    y_dp.fill_na()
    y_dp.discretization()
    complete_y_test = y_dp.df.copy()

    # process y_train
    y_dp.df = prepared_data.y_train.to_frame()
    y_dp.fill_na()
    y_dp.discretization()
    complete_y_train = y_dp.df.copy()


    complete_y_train = y_dp.y_test
    train_model(model, prepared_data.x_train, prepared_data.x_test, complete_y_train, complete_y_test)


if __name__ == '__main__':
    # run_complete_model()
    # run_missing_values_in_prediction_model()
    run_missing_values_in_training_model()