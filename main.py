import read_csv as reader
from data_preparation import DataPreparation
from models import DTC, MissingDataTrainDTC, MissingDataPredictionDTC


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
    dp.add_missing_data_10_percent(dp.df)
    dp.fill_na()
    dp.discretization()
    dp.partition_data_sets()


def train_model(class_model, x_train, y_train):
    class_model.model.fit(x_train, y_train)


def run_complete_model():
    prepared_data = iterate_files(impute=True)

    dtc = DTC()  # 'normal' DecisionTreeClassifier
    train_model(dtc, prepared_data.x_train, prepared_data.y_train)
    y_prediction = dtc.model.predict(prepared_data.x_test)
    return y_prediction, prepared_data.y_test  # return both for evaluation


def run_missing_values_in_prediction_model():  # gonen working on this

    # TODO:
    # make sure that after data is prepared (prepared_data instance)
    # we need to omit values in y_test, x_test.
    # then, override predict() method -> and predict

    raise NotImplemented
    class_model = MissingDataPredictionDTC()
    prepared_data = iterate_files(impute=True)
    train_model(class_model.model, prepared_data.x_train, prepared_data.y_train)
    y_prediction = class_model.model.predict(prepared_data.x_test)
    return y_prediction, prepared_data.y_test  # return both for evaluation



def run_missing_values_in_training_model():  # guy working on this

    # TODO:
    # make sure that after data is prepared (prepared_data instance)
    # we need to complete values in y_test, x_test.
    # then, override fit() method -> and train

    class_model = MissingDataTrainDTC()
    prepared_data = iterate_files(impute=False)
    prepared_data.add_missing_data_10_percent(prepared_data.y_train.to_frame())
    prepared_data.add_missing_data_10_percent(prepared_data.x_train)

    train_model(class_model.model, prepared_data.x_train, prepared_data.y_train)
    y_prediction = class_model.model.predict(prepared_data.x_test)
    return y_prediction, prepared_data.y_test  # return both for evaluation


if __name__ == '__main__':
    # y_prediction_complete_model, y_test_complete_model = run_complete_model()
    # y_prediction_missing_predicion_values_model, y_test_missing_prediction_values_model = run_missing_values_in_prediction_model()
    y_prediction_missing_train_values_model, y_test_missing_train_values_model = run_missing_values_in_training_model()