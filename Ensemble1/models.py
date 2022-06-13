from sklearn.tree import DecisionTreeClassifier
from pyearth import Earth
from sklearn.pipeline import Pipeline


class DTC:

    def __init__(self, max_depth=None, min_samples_split=2):
        self.model = DecisionTreeClassifier(random_state=0, max_depth=max_depth, min_samples_split=min_samples_split)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)


class MissingDataTrainDTC(DTC):

    def __init__(self):
        super().__init__()
        self.model = Pipeline([('earth', Earth(allow_missing=True)),
                               ('cls', DecisionTreeClassifier())])


class MissingDataPredictionDTC(DTC):

    def __init__(self):
        super().__init__()
        self.model = Pipeline([('earth', Earth(allow_missing=True)),
                               ('cls', DecisionTreeClassifier())])
