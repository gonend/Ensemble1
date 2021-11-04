from sklearn.tree import DecisionTreeClassifier


class DTC:

    def __init__(self):

        self.model = DecisionTreeClassifier(random_state=0)
