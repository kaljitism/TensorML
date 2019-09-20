import dependencies


class DecisionTree:
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __call__(self, *args, **kwargs):
        pass

    def feature_count(self):
        return list(set(self.inputs))

    def label_count(self):
        return list(set(self.labels))

    def value_count(self, lbl):
        count = 0
        for j in self.inputs:
            if lbl == j:
                count += 1
        return count

    def gini(self):
        pass

    def split(self):
        pass

    def tree(self):
        pass


dt = DecisionTree([0, 0, 1, 2, 44, 22, 22, 44, 1, 1, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8,9, 0], )
dt()
print(dt.value_count(22))