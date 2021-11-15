import read_csv as reader
from data_preparation import DataPreparation
from models import DTC, MissingDataTrainDTC, MissingDataPredictionDTC
from sklearn.metrics import roc_auc_score

def iterate_files(impute=True):
    """
    for each csv file - read, prepare, fit and predict data.
    :return:
    """

    file_names = reader.read_file_names('data')
    total_data = {}
    for name in file_names:
        dp = DataPreparation()
        dp.df = reader.read_file(name)
        prepare_data(dp, impute)
        total_data[name] = dp

    return total_data


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
    class_model.fit(x_train, y_train)


def run_complete_model():

    total_data = iterate_files(impute=True)
    dtc = DTC()  # 'normal' DecisionTreeClassifier
    scores = []

    for name, prepared_data in total_data.items():
        counter = 0
        for i in range(len(prepared_data.train_index)):
            pd_train_inx = prepared_data.train_index.__getitem__(i)
            pd_test_inx = prepared_data.test_index.__getitem__(i)
            train_model(dtc, prepared_data.x[~prepared_data.x.index.isin(pd_train_inx)], prepared_data.y[~prepared_data.x.index.isin(pd_train_inx)])
            y_prediction = dtc.predict(prepared_data.x[~prepared_data.x.index.isin(pd_test_inx)])
            counter += evaluate(y_prediction, prepared_data.y[~prepared_data.y.index.isin(pd_test_inx)])
        scores.append([name,counter/len(prepared_data.train_index)])  # return both for evaluation
    return scores


def run_missing_values_in_prediction_model():

    # TODO: gonen working on this
    # make sure that after data is prepared (prepared_data instance)
    # we need to omit values in y_test, x_test.
    # then, override predict() method -> and predict

    # raise NotImplemented
    class_model = MissingDataPredictionDTC()
    total_data = iterate_files(impute=True)
    scores = []

    for name, prepared_data in total_data.items():
        prepared_data.x_test = prepared_data.add_missing_data_10_percent(prepared_data.x_test)
        train_model(class_model, prepared_data.x_train, prepared_data.y_train)
        y_prediction = class_model.predict(prepared_data.x_test)
        scores.append([name,evaluate(y_prediction,prepared_data.y_test)])
    return scores
    #y_prediction, prepared_data.y_test  # return both for evaluation


def run_missing_values_in_training_model():  # guy working on this

    # TODO:
    # make sure that after data is prepared (prepared_data instance)
    # we need to complete values in y_test, x_test.
    # then, override fit() method -> and train

    class_model = MissingDataTrainDTC()
    total_data = iterate_files(impute=False)
    scores = []

    for name, prepared_data in total_data.items():
        prepared_data.x_train = prepared_data.add_missing_data_10_percent(prepared_data.x_train)
        train_model(class_model, prepared_data.x_train, prepared_data.y_train)
        y_prediction = class_model.predict(prepared_data.x_test)
        scores.append([name, evaluate(y_prediction, prepared_data.y_test)])

    return scores


def evaluate(y_test, y_pred):
    try:
        return roc_auc_score(y_test, y_pred)
    except ValueError:
        pass


if __name__ == '__main__':
    complete_model_scores = run_complete_model()
    # y_prediction_missing_predicion_values_model, y_test_missing_prediction_values_model = run_missing_values_in_prediction_model()
    missing_train_values_model_scores = run_missing_values_in_training_model()
    missing_test_values_model_scores = run_missing_values_in_prediction_model()

    print('')
    print(f'Complete model AUROC: {complete_model_scores}\n')
    print(f'Missing values during train model AUROC: {missing_train_values_model_scores}')
    print(f'Missing values during test model AUROC: {missing_test_values_model_scores}')