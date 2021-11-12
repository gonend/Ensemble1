from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from pyearth import Earth
from sklearn.pipeline import Pipeline


class DTC:

    def __init__(self):

        self.model = DecisionTreeClassifier(random_state=0)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

class MissingDataTrainDTC:

    def __init__(self):

        self.model = None  # override .fit function

    def fit(self, x_train, y_train):
        earth_classifier = Pipeline([('earth', Earth(allow_missing=True)),
                                     ('cls', DecisionTreeClassifier())])

        earth_classifier.fit(x_train, y_train)
        self.model = earth_classifier

    def predict(self, x_test):
        return self.model.predict(x_test)


class MissingDataPredictionDTC:

    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=0)  # override .predict function

    def fit(self, x_train, y_train):
        earth_classifier = Pipeline([('earth', Earth(allow_missing=True)),
                                     ('cls', DecisionTreeClassifier())])

        earth_classifier.fit(x_train, y_train)
        self.model = earth_classifier

    def predict(self, x_test):
        return self.model.predict(x_test)
