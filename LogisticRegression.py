import dependencies


class LogisticRegression:
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
        return self.sigmoid(self.inputs * self.weights + self.bias)

    def sigmoid(self, x):
        return 1 / (1 + dependencies.tf.exp(-1 * x))

    def error(self):
        return - dependencies.tf.reduce_mean(self.labels * dependencies.tf.math.log(self.model()) + (dependencies.tf.ones_like(self.labels).numpy() - self.labels) * dependencies.tf.math.log((dependencies.tf.ones_like(self.labels).numpy() - self.model())))

    def gradient_descent(self):
        for epoch in range(self.epochs):
            with dependencies.tf.GradientTape() as t:
                cost = self.error()
                dw, db = t.gradient(cost, [self.weights, self.bias])
                self.weights.assign_sub(self.learning_rate * dw)
                self.bias.assign_sub(self.learning_rate * db)
                print('Epoch {}: Weight={}, Bias={}, Loss={}'.format(epoch, self.weights.numpy(), self.bias.numpy(), cost))


model = LogisticRegression(inputs=[1, 2, 3, 23, 24, 25],
           labels=[0, 0, 0, 1, 1, 1],
           epochs=5)
model()
print(model.model().numpy())
