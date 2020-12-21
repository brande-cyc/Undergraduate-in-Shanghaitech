import nn
import time
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        pred_node = self.run(x)
        pred_value = nn.as_scalar(pred_node)
        return 1 if pred_value >=0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        convergence = 1
        while convergence != 0:
            convergence = 0
            batch_size = 1
            for x, y in dataset.iterate_once(batch_size):
                pred = self.get_prediction(x)
                label = nn.as_scalar(y)
                if pred != label:
                    convergence += 1
                    self.w.update(x, label)


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = -1e-2
        self.hidden_size1 = 100
        self.hidden_size2 = 50

        self.w1 = nn.Parameter(1, self.hidden_size1)
        self.b1 = nn.Parameter(1, self.hidden_size1)
        self.w2 = nn.Parameter(self.hidden_size1, 1)
        self.b2 = nn.Parameter(1, 1)
        
        self.params_list = [self.w1, self.b1, self.w2, self.b2]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        z_1 = nn.Linear(x, self.w1)
        z_2 = nn.AddBias(z_1, self.b1)
        z_3 = nn.ReLU(z_2)
        z_4 = nn.Linear(z_3, self.w2)
        z_5 = nn.AddBias(z_4, self.b2)

        return z_5

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred_node = self.run(x)
        loss_node = nn.SquareLoss(pred_node, y)
        return loss_node

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(1):
            # Check break
            total_loss_node = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            total_loss_value = nn.as_scalar(total_loss_node)
            if total_loss_value <= 0.02:
                break
            # Update params
            loss_node = self.get_loss(x, y)
            grad_list = list(nn.gradients(loss_node, self.params_list))
            for i, param in enumerate(self.params_list):
                param.update(grad_list[i], self.learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = -4e-1
        self.hidden_size1 = 100
        self.hidden_size2 = 100

        self.w1 = nn.Parameter(784, self.hidden_size1)
        self.b1 = nn.Parameter(1, self.hidden_size1)
        self.w2 = nn.Parameter(self.hidden_size1, 10)
        self.b2 = nn.Parameter(1, 10)
        
        self.params_list = [self.w1, self.b1, self.w2, self.b2]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        z_1 = nn.Linear(x, self.w1)
        z_2 = nn.AddBias(z_1, self.b1)
        z_3 = nn.ReLU(z_2)
        z_4 = nn.Linear(z_3, self.w2)
        z_5 = nn.AddBias(z_4, self.b2)

        return z_5

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred_node = self.run(x)
        loss_node = nn.SoftmaxLoss(pred_node, y)
        
        return loss_node

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        acc = 0
        while acc < 0.98:
            count = 0
            for x, y in dataset.iterate_once(60):
                # Check break
                count += 1
                if count % 10 == 0:
                    acc = dataset.get_validation_accuracy()
                    if acc > 0.98:
                        break
                # Update params
                loss_node = self.get_loss(x, y)
                grad_list = list(nn.gradients(loss_node, self.params_list))
                for i, param in enumerate(self.params_list):
                    param.update(grad_list[i], self.learning_rate)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = -0.08
        self.hidden_size = 200
        self.w1 = nn.Parameter(self.num_chars, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.w3 = nn.Parameter(self.hidden_size, 5)
        self.b2 = nn.Parameter(1, 5)

        self.params_list = [self.w1, self.w2, self.b1, self.w3, self.b2]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        initial_word = xs[0]
        h_i = nn.Linear(initial_word, self.w1)

        for mini_batch in xs[1:]:
            f_1 = nn.Linear(mini_batch, self.w1)
            f_2 = nn.Linear(h_i, self.w2)
            h_i = nn.AddBias(nn.Add(f_1, f_2), self.b1)
            h_i = nn.ReLU(h_i)
        h_i = nn.Linear(h_i, self.w3)
        h_i = nn.AddBias(h_i, self.b2)
        return h_i

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        logits_node = self.run(xs)
        loss_node = nn.SoftmaxLoss(logits_node, y)
        return loss_node

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        start_time = time.time()
        acc = 0
        while acc < 0.88:
            count = 0
            batch_size = 100
            for x, y in dataset.iterate_once(batch_size):
                count += 1
                # Check break
                if count % 10 == 0:
                    acc = dataset.get_validation_accuracy()
                    if acc > 0.88:
                        break
                # Update params
                loss_node = self.get_loss(x, y)
                grad_list = list(nn.gradients(loss_node, self.params_list))
                for i, param in enumerate(self.params_list):
                    param.update(grad_list[i], self.learning_rate)

        end_time = time.time()
        print(end_time-start_time)