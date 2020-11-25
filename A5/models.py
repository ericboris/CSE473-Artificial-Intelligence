import nn


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
		# Let x represent the features
		# Let w represent the weights
		return nn.dot_product(x, self.w)

	def get_prediction(self, x):
		"""
		Calculates the predicted class for a single data point `x`.

		Returns: 1 or -1
		"""
		# Let cross product be the cross between features x and weights w
		cross_product = self.run(x)

		# "Flatten" the result of cross product to -1 or 1
		return -1 if cross_product.item() < 0 else 1

	def train(self, dataset):
		"""
		Train the perceptron until convergence.
		"""
		# Let batch size represent the number of iterations to make over the dataset.
		# Set to 1 since we iterate until convergence.
		batch_size = 1

		# Let trained be a flag representing whether the weights have converged.
		trained = False

		# Iterate over the dataset until the weights converge.
		while not trained:
			trained = True

			# Let x be the features and let y be the expected training result.
			for x, y in dataset.iterate_once(batch_size):
				# Let prediction be the perceptron's predicted output value.
				prediction = self.get_prediction(x)

				# Let expected be the perceptron's expected output value.
				expected = y.item()

				# If the prediction doesn't match the expected value then
				if prediction != expected:
					# update the weights and flag that training should continue.
					self.w.update(x, expected)
					trained = False
			

class RegressionModel(object):
	"""
	A neural network model for approximating a function that maps from real
	numbers to real numbers. The network should be sufficiently large to be able
	to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
	"""

	def __init__(self):
		# Let the following serve as controls for tuning the model.
		self.batch_size = 50
		self.learning_rate = -0.2
		self.maximum_loss = 0.02

		# Let the following represent the dimensionalities of the respective layers.
		layer_0 = 1
		layer_1 = 20
		layer_2 = 30
		layer_3 = 20
		layer_4 = 1	
		
		# Let each wi represent a (n x m) layer of weights in the network
		# where n is the dimensionality of the previous layer and
		# where m is the dimensionality of the current layer.
		w1 = nn.Parameter(layer_0, layer_1)
		w2 = nn.Parameter(layer_1, layer_2)
		w3 = nn.Parameter(layer_2, layer_3)
		w4 = nn.Parameter(layer_3, layer_4)

		# Let each bi represent a (1 x m) layer of biases in the network
		# such that for each i in bi is associated with a hidden layer wi and
		# such that m is the size of the hidden layer.
		b1 = nn.Parameter(1, layer_1)
		b2 = nn.Parameter(1, layer_2)
		b3 = nn.Parameter(1, layer_3)
		b4 = nn.Parameter(1, layer_4)

		# Let the following hold each layer's weights and biases as tuple pairs.
		self.weights_and_biases = [(w1, b1), (w2, b2), (w3, b3), (w4, b4)]

	def run(self, x):
		"""
		Runs the model for a batch of examples.

		Inputs:
			x: a node with shape (batch_size x 1)
		Returns:
			A node with shape (batch_size x 1) containing predicted y-values
		"""
		# Return the predicted value y = f(X).

		# Given a starting activation layer X, if X = A_0 and 
		# if activation layer A_i+1 = relu(A_i * W_i + B_i) where i = [0, n-1).
		# Then the following computes the function f(X) = A_n-1 * W_n + B_n.
		i = 0
		n = len(self.weights_and_biases)
		a = x

		# Compute A_i+1 = relu(A_i * W_i + B_i) while i < n-1.
		while i < n - 1:
			w, b = self.weights_and_biases[i]
			a_cross_w = nn.matmul(a, w)
			a_cross_w_plus_b = nn.add_bias(a_cross_w, b)
			a = nn.relu(a_cross_w_plus_b)
			i += 1
		
		# Compute f(X) = A_n-1 * W_n + B_n.
		w, b = self.weights_and_biases[i]
		a_cross_w = nn.matmul(a, w)
		a_cross_w_plus_b = nn.add_bias(a_cross_w, b)
	
		# The last term represents the predicted y value. 
		predicted_y = a_cross_w_plus_b

		return predicted_y 

	def get_loss(self, x, y):
		"""
		Computes the loss for a batch of examples.

		Inputs:
			x: a node with shape (batch_size x 1)
			y: a node with shape (batch_size x 1), containing the true y-values
				to be used for training
		Returns: a loss node
		"""
		# Let the following compute the square loss between the
		# predicted outputs given x and the expected outputs given x
		predicted_y = self.run(x)
		loss = nn.square_loss(predicted_y, y)
	
		# and return the result.
		return loss

	def train(self, dataset):
		"""
		Trains the model.
		"""
		# Let trained be a flag representing whether the weights have converged.
		trained = False

		# Let parameters be a flat list of the weights and biases, i.e.
		# let parameters = [w1, b1, w2, b2, ..., wn, bn]
		parameters = [p for pair in self.weights_and_biases for p in pair]

		# Iterate over the dataset until the weights converge.
		while not trained:
			trained = True
			
			for x, y in dataset.iterate_once(self.batch_size):
				# Compute the square loss from the current model's predictions.
				loss = self.get_loss(x, y)
				
				# Continue training if the model poorly predicts a test case.
				if loss.item() >= self.maximum_loss:
					trained = False	
				
					# Compute the gradients.
					gradients = nn.gradients(loss, parameters)

					# Update each of the weights and biases.				
					for i, p in enumerate(parameters):
						p.update(gradients[i], self.learning_rate)


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
		# Let the following serve as controls for tuning the model.
		self.batch_size = 50
		self.learning_rate = -0.5
		self.minimum_accuracy = 0.98
		self.decay = 0.5

		# Let the following represent the dimensionalities of the respective layers.
		layer_0 = 784
		layer_1 = 392
		layer_2 = 196	
		layer_3 = 98
		layer_4 = 10

		# Let each wi represent a (n x m) layer of weights in the network
		# where n is the dimensionality of the previous layer and
		# where m is the dimensionality of the current layer.
		w1 = nn.Parameter(layer_0, layer_1)
		w2 = nn.Parameter(layer_1, layer_2)
		w3 = nn.Parameter(layer_2, layer_3)
		w4 = nn.Parameter(layer_3, layer_4)

		# Let each bi represent a (1 x m) layer of biases in the network
		# such that for each i in bi is associated with a hidden layer wi and
		# such that m is the size of the hidden layer.
		b1 = nn.Parameter(1, layer_1)
		b2 = nn.Parameter(1, layer_2)
		b3 = nn.Parameter(1, layer_3)
		b4 = nn.Parameter(1, layer_4)

		# Let the following hold each layer's weights and biases as tuple pairs.
		self.weights_and_biases = [(w1, b1), (w2, b2), (w3, b3), (w4, b4)]

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
		# Return the predicted value y = f(X).

		# Given a starting activation layer X, if X = A_0 and 
		# if activation layer A_i+1 = relu(A_i * W_i + B_i) where i = [0, n-1).
		# Then the following computes the function f(X) = A_n-1 * W_n + B_n.
		i = 0
		n = len(self.weights_and_biases)
		a = x

		# Compute A_i+1 = relu(A_i * W_i + B_i) while i < n-1.
		while i < n - 1:
			w, b = self.weights_and_biases[i]
			a_cross_w = nn.matmul(a, w)
			a_cross_w_plus_b = nn.add_bias(a_cross_w, b)
			a = nn.relu(a_cross_w_plus_b)
			i += 1
		
		# Compute f(X) = A_n-1 * W_n + B_n.
		w, b = self.weights_and_biases[i]
		a_cross_w = nn.matmul(a, w)
		a_cross_w_plus_b = nn.add_bias(a_cross_w, b)
	
		# The last term represents the predicted y value. 
		predicted_y = a_cross_w_plus_b

		return predicted_y 


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
		# Let the following compute the square loss between the
		# predicted outputs given x and the expected outputs given x
		predicted_y = self.run(x)
		loss = nn.softmax_loss(predicted_y, y)
	
		# and return the result.
		return loss


	def train(self, dataset):
		"""
		Trains the model.
		"""
		# Let trained be a flag representing whether the weights have converged.
		trained = False

		# Let parameters be a flat list of the weights and biases, i.e.
		# let parameters = [w1, b1, w2, b2, ..., wn, bn]
		parameters = [p for pair in self.weights_and_biases for p in pair]

		# Iterate over the dataset until the weights converge.
		while not trained:
			trained = True
			
			for x, y in dataset.iterate_once(self.batch_size):
				# Compute the square loss from the current model's predictions.
				loss = self.get_loss(x, y)
			
				# Compute the gradients.
				gradients = nn.gradients(loss, parameters)

				# Update each of the weights and biases.				
				for i, p in enumerate(parameters):
					p.update(gradients[i], self.learning_rate)

			# Reduce the learning rate to avoid over-stepping convergence.
			self.learning_rate *= self.decay

			# Continue training if the model accuracy is below the minimum threshold. 
			if dataset.get_validation_accuracy() < self.minimum_accuracy:
				trained = False


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

		# Let the following serve as controls for tuning the model.
		self.batch_size = 50
		self.learning_rate = -0.5
		self.minimum_accuracy = 0.86
		self.decay = 0.5

		# Let the following represent the dimensionalities of the respective layers.
		layer_input = self.num_chars
		layer_hidden = 235
		layer_output = len(self.languages)
		
		# Let each wi represent a (n x m) layer of weights in the network
		# where n is the dimensionality of the previous layer and
		# where m is the dimensionality of the current layer.
		self.w_input = nn.Parameter(layer_input, layer_hidden)
		self.w_hidden = nn.Parameter(layer_hidden, layer_hidden)
		self.w_output = nn.Parameter(layer_hidden, layer_output)

		# Let each bi represent a (1 x m) layer of biases in the network
		# such that for each i in bi is associated with a hidden layer wi and
		# such that m is the size of the hidden layer.
		self.b_input = nn.Parameter(1, layer_hidden)
		self.b_hidden = nn.Parameter(1, layer_hidden)
		self.b_output = nn.Parameter(1, layer_output)

		# Let the following hold each layer's weights and biases as tuple pairs.
		self.weights_and_biases = [(self.w_input, self.b_input), 
								   (self.w_hidden, self.b_hidden), 
								   (self.w_output, self.b_output)]

	def run(self, xs):
		"""
		Runs the model for a batch of examples.

		Although words have different lengths, our data processing guarantees
		that within a single batch, all words will be of the same length (L).

		Here `xs` will be a list of length L. Each element of `xs` will be a
		node with shape (batch_size x self.num_chars), where every row in the
		array is a one-hot vector encoding of a character. For example: 

		Given:
			xs = [[batch_size X num_chars] * L]
			batch_size	- The number of words in the input matrices
			num_chars	- A one-hot encoding of the possible letters
			L			- The number of letters in the words in the batch
		If:
			batch_size = 8
			num_chars = 47
			L = 3
			"cat" is the last word in the batch
		Then:
			xs[0] has a 1 at postition (7, 2) 
				i.e. The first letter, xs[0]
				of the last word in the batch, 7 
				starts with a "c", 2 
				since "c" is the third letter in the alphabet. 
			xs[1] has a 1 at position (7, 0)
				i.e. The second letter, xs[1]
				of the last word in the batch, 7
				has an "a" as it's middle letter, 0
				since "a" is the first letter of the alphabet.
			xs[2] has a 1 at position (7, 19)
				ie. the third letter, xs[2]
				of the last word in the batch, 7
				has a "t" as it's last letter, 19
				since "t" is the nineteenth letter of the alphabet.
					
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
		# Return the predicted value y = f(X).

		# Given an activation layer x representing the first letters in the words
		# a weights layer W and a bias b
		# then the base case y_0 of the recurrence is given by
		# y_0 = relu(x * W + b).
		x_cross_w = nn.matmul(xs[0], self.w_input)
		x_cross_w_plus_b = nn.add_bias(x_cross_w, self.b_input)
		y = nn.relu(x_cross_w_plus_b)
		
		# The remaining iterations of the recurrence for y_i for i in range [1, n]
		# where n is the length of the input words, n = L
		# are given by y_i = relu((x_i * W) + (y_i-1 * W_hidden) + b_hidden)
		# where x_i represents the remaining letters in the words. 
		n = len(xs)
		for i in range(1, n):
			x_cross_w = nn.matmul(xs[i], self.w_input)
			y_cross_w = nn.matmul(y, self.w_hidden)
			y = nn.relu(nn.add_bias(nn.add(x_cross_w, y_cross_w), self.b_hidden))

		# Map the predictions made over the letters onto a language prediction.  
		# y = relu(y_n * W_last_layer + b_last_layer)
		y_cross_w = nn.matmul(y, self.w_output)
		y_cross_w_plus_b = nn.add_bias(y_cross_w, self.b_output)

		# The previous term represents the predicted value. 
		predicted_y = y_cross_w_plus_b

		return predicted_y

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
		# Let the following compute the square loss between the
		# predicted outputs given x and the expected outputs given x
		predicted_y = self.run(xs)
		loss = nn.softmax_loss(predicted_y, y)
	
		# and return the result.
		return loss

	def train(self, dataset):
		"""
		Trains the model.
		"""
		# Let trained be a flag representing whether the weights have converged.
		trained = False

		# Let parameters be a flat list of the weights and biases, i.e.
		# let parameters = [w1, b1, w2, b2, ..., wn, bn]
		parameters = [p for pair in self.weights_and_biases for p in pair]

		# Iterate over the dataset until the weights converge.
		while not trained:
			trained = True
			
			for x, y in dataset.iterate_once(self.batch_size):
				# Compute the square loss from the current model's predictions.
				loss = self.get_loss(x, y)
			
				# Compute the gradients.
				gradients = nn.gradients(loss, parameters)

				# Update each of the weights and biases.				
				for i, p in enumerate(parameters):
					p.update(gradients[i], self.learning_rate)

			# Reduce the learning rate to avoid over-stepping convergence.
			self.learning_rate *= self.decay

			# Continue training if the model accuracy is below the minimum threshold. 
			if dataset.get_validation_accuracy() < self.minimum_accuracy:
				trained = False

