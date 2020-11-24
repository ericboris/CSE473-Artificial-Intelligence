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
		# Let the following represent the dimensionalities of the respective layers.
		layer_0 = 1
		layer_1 = 350
		layer_2 = 150	
		layer_3 = 1	
		
		# Let the following serve as controls for tuning the model.
		self.batch_size = 1
		self.learning_rate = -0.09

		# Let each wi represent a (n x m) layer of weights in the network
		# where n is the dimensionality of the previous layer and
		# where m is the dimensionality of the current layer.
		self.w1 = nn.Parameter(layer_0, layer_1)
		self.w2 = nn.Parameter(layer_1, layer_2)
		self.w3 = nn.Parameter(layer_2, layer_3)

		# Let each bi represent a (1 x m) layer of biases in the network
		# such that for each i in bi is associated with a hidden layer wi and
		# such that m is the size of the hidden layer.
		self.b1 = nn.Parameter(1, layer_1)
		self.b2 = nn.Parameter(1, layer_2)
		self.b3 = nn.Parameter(1, layer_3)

		# Let the following hold each layer's weights and biases as a tuple.
		self.weights_and_biases = [(self.w1, self.b1), (self.w2, self.b2), (self.w3, self.b3)]

	def run(self, x):
		"""
		Runs the model for a batch of examples.

		Inputs:
			x: a node with shape (batch_size x 1)
		Returns:
			A node with shape (batch_size x 1) containing predicted y-values
		"""
		'''
		x_cross_w1 = nn.matmul(x, self.w1)
		x_cross_w1_plus_b1 = nn.add_bias(x_cross_w1, self.b1)
		relu = nn.relu(x_cross_w1_plus_b1)
		relu_cross_w2 = nn.matmul(relu, self.w2)
		relu_cross_w2_plus_b2 = nn.add_bias(relu_cross_w2, self.b2)
		'''
		# Return the predicted y value y = f(X).

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
			'''	
			print('i= ', i)
			print('w= ', w, ' b= ', b)
			'''
			i += 1
		#print('\n\n')
		
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
		trained = False

		while not trained:
			trained = True
			
			for x, y in dataset.iterate_once(self.batch_size):
				loss = self.get_loss(x, y)
				
				# Continue training if necessary.
				if loss.item() > 0.02:
					trained = False	
				
					# Let parameters be a flat list of weights and biases, i.e.
					# parameters = [w1, b1, w2, b2, ..., wn, bn] from [(w1, b1), (w2, b2), ... (wn, bn)]
					parameters = []
					for w, b in self.weights_and_biases:
						parameters.append(w)
						parameters.append(b)

					'''grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = nn.gradients(loss, parameters)'''
					'''gradients = nn.gradients(loss, parameters)'''
					gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])

					# Update each of the weights and biases.				
					self.w1.update(gradients[0], self.learning_rate)
					self.b1.update(gradients[1], self.learning_rate)
					self.w2.update(gradients[2], self.learning_rate)
					self.b2.update(gradients[3], self.learning_rate)
					self.w3.update(gradients[4], self.learning_rate)
					self.b3.update(gradients[5], self.learning_rate)

					print('w1= ', self.w1)
					print('b1= ', self.b1)
					print('w2= ', self.w2)
					print('b2= ', self.b2)
					print('loss', loss)
					print('\n\n')
					'''
					i = 0	
					for w, b in self.weights_and_biases:
						print('start w and b ', self.weights_and_biases)
						w.update(gradients[i], self.learning_rate)
						i += 1
						b.update(gradients[i], self.learning_rate)
						print('end   w and b ', self.weights_and_biases)
					print('\n\n')
					'''
					'''
					self.weights_and_biases[0][0].update(grad_wrt_w1, self.learning_rate)
					self.weights_and_biases[0][1].update(grad_wrt_b1, self.learning_rate)
					self.weights_and_biases[1][0].update(grad_wrt_w2, self.learning_rate)	
					self.weights_and_biases[0][1].update(grad_wrt_b2, self.learning_rate)
					'''	
				'''
				print('w		', self.w1)
				print('b		', self.b1)
				print('w		', self.w2)
				print('b		', self.b2)
				print('x		', x)
				print('y		', y)
				print('gw		', grad_wrt_w1)
				print('gb		', grad_wrt_b1)
				#print('gw		', grad_wrt_w2)
				#print('gb		', grad_wrt_b2)
				print('loss		', loss)
				print('\n\n')	'''

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
		## TODO Q3

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
		## TODO Q3

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
		## TODO Q3

	def train(self, dataset):
		"""
		Trains the model.
		"""
		"*** YOUR CODE HERE ***"
		## TODO Q3

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
		## TODO Q4

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
		## TODO Q4

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
		## TODO Q4

	def train(self, dataset):
		"""
		Trains the model.
		"""
		"*** YOUR CODE HERE ***"
		## TODO Q4

