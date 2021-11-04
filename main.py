import read_csv as reader
from data_preparation import DataPreparation


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


if __name__ == '__main__':
    iterate_files()
