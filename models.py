from sklearn.tree import DecisionTreeClassifier


class DTC:

    def __init__(self):

        self.model = DecisionTreeClassifier(random_state=0)


class MissingDataTrainDTC:

    def __init__(self):

        self.model = DecisionTreeClassifier(random_state=0)  # override .fit function


class MissingDataPredictionDTC:

    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=0)  # override .predict function
