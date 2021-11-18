from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

import read_csv as reader
from data_preparation import DataPreparation
from models import DTC, MissingDataTrainDTC, MissingDataPredictionDTC
from sklearn.metrics import roc_auc_score


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


def apply_grid_search(class_model, prepared_data, special=False):
    """

    :param class_model:
    :param prepared_data:
    :return:
    """
    if special:
        params = {'cls__max_depth': [5], 'cls__min_samples_split': [2]}
    else:
        params = {'max_depth': [5], 'min_samples_split': [2]}
    grid_search_results = GridSearchCV(class_model.model, params, cv=5).fit(prepared_data.x_train,
                                                                            prepared_data.y_train)
    return grid_search_results.best_params_


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
    :return:
    """

    total_data = iterate_files()
    class_model = DTC()
    scores = []

    for name, prepared_data in total_data.items():
        best_params = apply_grid_search(class_model, prepared_data)
        class_model = DTC(best_params['max_depth'], best_params['min_samples_split'])
        train_model(class_model, prepared_data.x_train, prepared_data.y_train)
        y_prediction = class_model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])
    return scores


def run_missing_values_in_training_model():
    """
    run decision tree model wrapped by a pipeline for dealing with missing values during training,
    not in a manner of pre-processing.
    :return:
    """

    class_model = MissingDataTrainDTC()
    total_data = iterate_files()
    scores = []

    for name, prepared_data in total_data.items():
        prepared_data.x_train = prepared_data.add_missing_data_10_percent(prepared_data.x_train)
        best_params = apply_grid_search(class_model, prepared_data, special=True)
        class_model.model.set_params(**best_params)
        train_model(class_model, prepared_data.x_train, prepared_data.y_train)
        y_prediction = class_model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])
    return scores


def run_missing_values_in_prediction_model():
    """
    run decision tree model wrapped by a pipeline for dealing with missing values during prediction,
    not in a manner of pre-processing.
    :return:
    """

    class_model = MissingDataPredictionDTC()
    total_data = iterate_files()
    scores = []

    for name, prepared_data in total_data.items():
        prepared_data.x_test = prepared_data.add_missing_data_10_percent(prepared_data.x_test)
        best_params = apply_grid_search(class_model, prepared_data, special=True)
        class_model.model.set_params(**best_params)
        train_model(class_model, prepared_data.x_train, prepared_data.y_train)
        y_prediction = class_model.predict(prepared_data.x_test)
        scores.append([name.split('/')[1], evaluate(y_prediction, prepared_data.y_test)])
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
