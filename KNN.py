import dependencies


class KNN:
    def __init__(self, test_p, inputs, labels, k=3):
        self.k = k
        self.test_p = test_p
        self.inputs = inputs
        self.labels = labels
        self.dist = []
        self.decider_dist = []

    def __call__(self, *args, **kwargs):
        self.knn()

    def calculate_distance(self, test_point, data_point):
        distance = 0
        for i in range(2):
            distance += dependencies.tf.square(test_point[i] - data_point[i])
        return dependencies.math.sqrt(distance)

    def knn(self):
        count = 0
        for point in self.inputs:
            dst = self.calculate_distance(self.test_p, point)
            self.dist.append([self.labels[count], dst])
            count += 1

        lst = self.sort()
        self.decider_dist.append(lst[:self.k])

        count0 = 0
        count1 = 0
        for item in self.decider_dist:
            if item[0] == 0:
                count0 += 1
            if count1 == 1:
                count1 += 1

        if count1 > count0:
            print("Point belongs to Class 1")
        else:
            print("Point belongs to Class 0")

    def sort(self):
        return sorted(self.dist)


knn = KNN([3, 3],
          [[1, 2], [1, 2], [2, 2], [2, 3], [4, 5], [6, 7], [7, 7], [8, 7], [6, 8], [8, 9]],
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
knn()
