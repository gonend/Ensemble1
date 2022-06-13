from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import read_csv as reader
from data_preparation import DataPreparation
from models import DTC, MissingDataTrainDTC, MissingDataPredictionDTC
from sklearn.metrics import roc_auc_score

SKF = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

def iterate_files():
    """
    for each csv file - read, prepare, fit and predict data.
    :return:
    """

    file_names = reader.read_file_names('data')
    total_data = {}
    for name in file_names:
        dp = DataPreparation()
        dp.df = reader.read_file(name)
        prepare_data(dp)
        total_data[name] = dp

    return total_data


def prepare_data(dp):
    """
    calls all relevant functions from data_preparation class
    :param dp:
    :return:
    """

    dp.add_missing_data_10_percent(dp.df)
    dp.fill_na()
    dp.discretization()
    dp.partition_data_sets()


def train_model(class_model, x_train, y_train):
    """
    run model's fit function
    :param class_model:
    :param x_train:
    :param y_train:
    :return:
    """

    class_model.fit(x_train, y_train)


def run_complete_model():
    """
    run decision tree model.
    apply stratified k fold validation and evaluate the model using AUC ROC metric.
    :return:
    """

    total_data = iterate_files()
    class_model = DTC()
    scores = []

    for name, prepared_data in total_data.items():
        temp_scores = []
        for train_index, test_index in SKF.split(prepared_data.x, prepared_data.y):
            X_train, X_test = prepared_data.x[~prepared_data.x.index.isin(train_index)], \
                              prepared_data.x[~prepared_data.x.index.isin(test_index)]
            y_train, y_test = prepared_data.y[~prepared_data.y.index.isin(train_index)], \
                              prepared_data.y[~prepared_data.y.index.isin(test_index)]
            train_model(class_model, X_train, y_train)
            y_prediction = class_model.predict(X_test)
            temp_scores.append(evaluate(y_prediction, y_test))
        scores.append([name.split('/')[1], np.mean(temp_scores)])
    return scores


def run_missing_values_in_training_model():
    """
    run decision tree model wrapped by a pipeline for dealing with missing values during training,
    not in a manner of pre-processing.
    apply stratified k fold validation and evaluate the model using AUC ROC metric.
    :return:
    """

    class_model = MissingDataTrainDTC()
    total_data = iterate_files()
    scores = []

    for name, prepared_data in total_data.items():
        temp_scores = []
        for train_index, test_index in SKF.split(prepared_data.x, prepared_data.y):
            X_train, X_test = prepared_data.x[~prepared_data.x.index.isin(train_index)], \
                              prepared_data.x[~prepared_data.x.index.isin(test_index)]
            y_train, y_test = prepared_data.y[~prepared_data.y.index.isin(train_index)], \
                              prepared_data.y[~prepared_data.y.index.isin(test_index)]
            X_train = prepared_data.add_missing_data_10_percent(X_train)
            train_model(class_model, X_train, y_train)
            y_prediction = class_model.predict(X_test)
            temp_scores.append(evaluate(y_prediction, y_test))
        scores.append([name.split('/')[1], np.mean(temp_scores)])
    return scores


def run_missing_values_in_prediction_model():
    """
    run decision tree model wrapped by a pipeline for dealing with missing values during prediction,
    not in a manner of pre-processing.
    apply stratified k fold validation and evaluate the model using AUC ROC metric.
    :return:
    """

    class_model = MissingDataPredictionDTC()
    total_data = iterate_files()
    scores = []

    for name, prepared_data in total_data.items():
        temp_scores = []
        for train_index, test_index in SKF.split(prepared_data.x, prepared_data.y):
            X_train, X_test = prepared_data.x[~prepared_data.x.index.isin(train_index)], \
                              prepared_data.x[~prepared_data.x.index.isin(test_index)]
            y_train, y_test = prepared_data.y[~prepared_data.y.index.isin(train_index)], \
                              prepared_data.y[~prepared_data.y.index.isin(test_index)]
            X_test = prepared_data.add_missing_data_10_percent(X_test)
            train_model(class_model, X_train, y_train)
            y_prediction = class_model.predict(X_test)
            temp_scores.append(evaluate(y_prediction, y_test))
        scores.append([name.split('/')[1], np.mean(temp_scores)])
    return scores


def evaluate(y_test, y_pred):
    """
    apply under the ROC area metric.
    :param y_test:
    :param y_pred:
    :return:
    """

    try:
        return roc_auc_score(y_test, y_pred)
    except ValueError:
        pass


if __name__ == '__main__':

    for i in tqdm(range(1)):
        complete_model_scores = run_complete_model()
        print(f'\nComplete model AUROC: {complete_model_scores}\n')

        missing_train_values_model_scores = run_missing_values_in_training_model()
        print(f'Missing values during train model AUROC: {missing_train_values_model_scores}\n')

        missing_test_values_model_scores = run_missing_values_in_prediction_model()
        print(f'Missing values during test model AUROC: {missing_test_values_model_scores}')
