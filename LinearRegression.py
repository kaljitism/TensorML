import dependencies


class LinearRegression:
    def __init__(self, inputs, labels, epochs=1, learning_rate=0.01):
        self.inputs = inputs
        self.labels = labels
        self.weights = dependencies.tf.Variable(0.0)
        self.bias = dependencies.tf.Variable(0.0)
        self.epochs = epochs
        self.learning_rate = learning_rate

    def __call__(self, *args, **kwargs):
        self.gradient_descent()

    def model(self):
        return self.inputs * self.weights + self.bias

    def error(self):
        return dependencies.tf.reduce_mean(dependencies.tf.square(self.model() - self.labels))

    def gradient_descent(self):
        for epoch in range(self.epochs):
            with dependencies.tf.GradientTape() as t:
                cost = self.error()
                dw, db = t.gradient(cost, [self.weights, self.bias])
                self.weights.assign_sub(self.learning_rate * dw)
                self.bias.assign_sub(self.learning_rate * db)
                print('Epoch {}: Weight={}, Bias={}, Loss={}'.format(epoch, self.weights.numpy(), self.bias.numpy(), cost))


model = LinearRegression(inputs=[1, 2, 3, 4, 5],
           labels=[3, 5, 7, 9, 11],
           epochs=10)
model()
print(model.model().numpy())
