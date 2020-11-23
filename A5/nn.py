import inspect

import numpy as np


def add(x, y):
    """Matrix addition.

    Args:
        x: a Node with shape \\((\\text{batch_size} \\times \\text{num_features})\\)
        y: a Node with the same shape as `x`

    Returns:
        a Node with shape \\((\\text{batch_size} \\times \\text{num_features})\\)
    """
    return Add(x, y)


def add_bias(features, bias):
    """Adds a bias vector to each feature vector

    Args:
        features: a Node with shape \\((\\text{batch_size} \\times \\text{num_features})\\)
        bias: a Node with shape \\((1 \\times \\text{num_features})\\)

    Returns:
        a Node with shape \\((\\text{batch_size} \\times \\text{num_features})\\)
    """
    return AddBias(features, bias)


def dot_product(features, weights):
    """Batched dot product.

    Args:
        features: a Node with shape \\((\\text{batch_size} \\times \\text{num_features})\\)
        weights: a Node with shape \\((1 \\times \\text{num_features})\\)

    Returns:
        a Node with shape \\((\\text{batch_size} \\times 1)\\)
    """
    return DotProduct(features, weights)


def matmul(x, y):
    """Matrix multiplication.

    Can be used to apply a linear transformation to some input,
    e.g., you can do something like`nn.matmul(features ,weights)`,
    where features has shape
    \\((\\text{batch_size} \\times \\text{input_features})\\)
    and weights has shape
    \\((\\text{batch_size} \\times \\text{output_features})\\)
    then the output will have shape
    \\((\\text{batch_size} \\times \\text{output_features})\\).

    Args:
        x: a Node with shape \\((k \\times n)\\)
        y: a Node with shape \\((n \\times m)\\)

    Returns:
        a node with shape \\((k \\times m)\\)
    """
    return MatMul(x, y)


def relu(x):
    """An element-wise Rectified Linear Unit nonlinearity: \\(\\max(x, 0)\\).
    This nonlinearity replaces all negative entries in its input with zeros.

    Args:
        x: a Node with shape \\((\\text{batch_size} \\times \\text{num_features})\\)

    Returns:
        a Node with the same shape as `x`, but no negative entries
    """
    return ReLU(x)


def tanh(x):
    """Apply \\(\\tanh(\\cdot)\\) nonlinearity element-wise.

    Args:
        x: a Node with shape \\((\\text{batch_size} \\times \\text{num_features})\\)

    Returns:
        a Node with the same shape as `x`
    """
    return Tanh(x)


def square_loss(a, b):
    """Square loss function.
    This first computes \\(0.5 \\cdot (a[i,j] - b[i,j])^2\\) at all positions
    \\((i,j)\\) in the inputs, which creates a
    \\((\\text{batch_size} \\times \\text{dim})\\) matrix.
    It then calculates and returns the mean of all elements in this matrix.

    Args:
        a: a Node with shape \\((\\text{batch_size} \\times \\text{dim})\\)
        b: a Node with shape \\((\\text{batch_size} \\times \\text{dim})\\)

    Returns:
        a 1x1 scalar Node (containing a single floating-point number)
    """
    return SquareLoss(a, b)


def softmax_loss(logits, labels):
    """A batched softmax loss, used for classification problems.

    **IMPORTANT: do not swap the order of the inputs to this node!**

    Args:
        logits: a Node with shape (batch_size x num_classes). Each row
            represents the scores associated with that example belonging to a
            particular class. A score can be an arbitrary real number.
        labels: a Node with shape (batch_size x num_classes) that encodes the
            correct labels for the examples. All entries must be non-negative
            and the sum of values along each row should be 1.

    Returns:
        a 1x1 scalar Node (containing a single floating-point number)
    """
    return SoftmaxLoss(logits, labels)


__pdoc__ = {}
_NODE_API_MAP = {}


def register_api(api_name, hide=True):

    def wrap(cls):
        _NODE_API_MAP[cls] = api_name
        if hide:
            __pdoc__[cls.__name__] = False
        return cls

    return wrap


def format_shape(shape):
    return "x".join(map(str, shape)) if shape else "()"


class Node(object):
    data: np.ndarray

    def __repr__(self):
        return "<{} shape={} at {}\n{}>".format(
            type(self).__name__, format_shape(self.data.shape), hex(id(self)),
            np.array2string(self.data, precision=4, threshold=100))

    def item(self):
        """Returns the value of a Node as a standard Python number.
        This only works for nodes wth one element (e.g. output from
        `nn.square_loss`, `nn.softmax_loss`,
        `nn.dot_product` with a batch size of 1 element etc.)
        """
        if self.data.size != 1:
            raise ValueError(
                "Node has shape {}, cannot convert to a scalar".format(
                    format_shape(self.data.shape)))
        return self.data.item()


class DataNode(Node):
    """
    DataNode is the parent class for Parameter and Constant nodes.

    You should not need to use this class directly.
    """
    def __init__(self, data):
        self.parents = []
        self.data = data

    def _forward(self, *inputs):
        return self.data

    @staticmethod
    def _backward(gradient, *inputs):
        return []


class Parameter(DataNode):
    """
    A Parameter node stores parameters used in a neural network (or perceptron).

    Use the the `update` method to update parameters when training the
    perceptron or neural network.
    """
    def __init__(self, *shape):
        assert len(shape) == 2, (
            "Shape must have 2 dimensions, instead has {}".format(len(shape)))
        assert all(isinstance(dim, int) and dim > 0 for dim in shape), (
            "Shape must consist of positive integers, got {!r}".format(shape))
        limit = np.sqrt(3.0 / np.mean(shape))
        data = np.random.uniform(low=-limit, high=limit, size=shape)
        super().__init__(data)

    def update(self, direction, multiplier):
        assert isinstance(direction, Constant), (
            "Update direction must be a {} node, instead has type {!r}".format(
                Constant.__name__, type(direction).__name__))
        assert direction.data.shape == self.data.shape, (
            "Update direction shape {} does not match parameter shape "
            "{}".format(
                format_shape(direction.data.shape),
                format_shape(self.data.shape)))
        assert isinstance(multiplier, (int, float)), (
            "Multiplier must be a Python scalar, instead has type {!r}".format(
                type(multiplier).__name__))
        self.data += multiplier * direction.data
        assert np.all(np.isfinite(self.data)), (
            "Parameter contains NaN or infinity after update, cannot continue")


class Constant(DataNode):
    """
    A Constant node is used to represent:

    * Input features
    * Output labels
    * Gradients computed by back-propagation

    You should not need to construct any Constant nodes directly; they will
    instead be provided by either the dataset or when you call `nn.gradients`.
    """
    def __init__(self, data):
        assert isinstance(data, np.ndarray), (
            "Data should be a numpy array, instead has type {!r}".format(
                type(data).__name__))
        assert np.issubdtype(data.dtype, np.floating), (
            "Data should be a float array, instead has data type {!r}".format(
                data.dtype))
        super().__init__(data)


class FunctionNode(Node):
    """
    A FunctionNode represents a value that is computed based on other nodes.
    The FunctionNode class performs necessary book-keeping to compute gradients.
    """
    def __init__(self, *parents):
        prev_frame = inspect.currentframe().f_back
        caller_file = inspect.getframeinfo(prev_frame)[0]
        assert caller_file == __file__, (
            "Please use function {}() instead of using class {}() "
            "directly".format(_NODE_API_MAP[type(self)], type(self).__name__))
        assert all(isinstance(parent, Node) for parent in parents), (
            "Inputs must be node objects, instead got types {!r}".format(
                tuple(type(parent).__name__ for parent in parents)))
        self.parents = parents
        self.data = self._forward(*(parent.data for parent in parents))


@register_api("add")
class Add(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[0].shape == inputs[1].shape, (
            "Input shapes should match, instead got {} and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        return inputs[0] + inputs[1]

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape == inputs[0].shape
        return [gradient, gradient]


@register_api("add_bias")
class AddBias(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[1].shape[0] == 1, (
            "First dimension of second input should be 1, instead got shape "
            "{}".format(format_shape(inputs[1].shape)))
        assert inputs[0].shape[1] == inputs[1].shape[1], (
            "Second dimension of inputs should match, instead got shapes {} "
            "and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        return inputs[0] + inputs[1]

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape == inputs[0].shape
        return [gradient, np.sum(gradient, axis=0, keepdims=True)]


@register_api("dot_product")
class DotProduct(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[1].shape[0] == 1, (
            "First dimension of second input should be 1, instead got shape "
            "{}".format(format_shape(inputs[1].shape)))
        assert inputs[0].shape[1] == inputs[1].shape[1], (
            "Second dimension of inputs should match, instead got shapes {} "
            "and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        return np.dot(inputs[0], inputs[1].T)

    @staticmethod
    def _backward(gradient, *inputs):
        # assert gradient.shape[0] == inputs[0].shape[0]
        # assert gradient.shape[1] == 1
        # return [np.dot(gradient, inputs[1]), np.dot(gradient.T, inputs[0])]
        raise NotImplementedError(
            "Backpropagation through DotProduct nodes is not needed in this "
            "assignment")


@register_api("matmul")
class MatMul(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[0].shape[1] == inputs[1].shape[0], (
            "Second dimension of first input should match first dimension of "
            "second input, instead got shapes {} and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        return np.dot(inputs[0], inputs[1])

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape[0] == inputs[0].shape[0]
        assert gradient.shape[1] == inputs[1].shape[1]
        return [np.dot(gradient, inputs[1].T), np.dot(inputs[0].T, gradient)]


@register_api("relu")
class ReLU(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 1, "Expected 1 input, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "Input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        return np.maximum(inputs[0], 0)

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape == inputs[0].shape
        return [gradient * np.where(inputs[0] > 0, 1.0, 0.0)]


@register_api("tanh")
class Tanh(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 1, "Expected 1 input, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "Input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        return np.tanh(inputs[0])

    @staticmethod
    def _backward(gradient, *inputs):
        assert gradient.shape == inputs[0].shape
        return [gradient * (1.0 - np.tanh(inputs[0])**2)]


@register_api("square_loss")
class SquareLoss(FunctionNode):
    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[0].shape == inputs[1].shape, (
            "Input shapes should match, instead got {} and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        return np.mean(np.square(inputs[0] - inputs[1]) / 2)

    @staticmethod
    def _backward(gradient, *inputs):
        assert np.asarray(gradient).ndim == 0
        return [
            gradient * (inputs[0] - inputs[1]) / inputs[0].size,
            gradient * (inputs[1] - inputs[0]) / inputs[0].size
        ]


@register_api("softmax_loss")
class SoftmaxLoss(FunctionNode):
    @staticmethod
    def log_softmax(logits):
        log_probs = logits - np.max(logits, axis=1, keepdims=True)
        log_probs -= np.log(np.sum(np.exp(log_probs), axis=1, keepdims=True))
        return log_probs

    @staticmethod
    def _forward(*inputs):
        assert len(inputs) == 2, "Expected 2 inputs, got {}".format(len(inputs))
        assert inputs[0].ndim == 2, (
            "First input should have 2 dimensions, instead has {}".format(
                inputs[0].ndim))
        assert inputs[1].ndim == 2, (
            "Second input should have 2 dimensions, instead has {}".format(
                inputs[1].ndim))
        assert inputs[0].shape == inputs[1].shape, (
            "Input shapes should match, instead got {} and {}".format(
                format_shape(inputs[0].shape), format_shape(inputs[1].shape)))
        assert np.all(inputs[1] >= 0), (
            "All entries in the labels input must be non-negative")
        assert np.allclose(np.sum(inputs[1], axis=1), 1), (
            "Labels input must sum to 1 along each row")
        log_probs = SoftmaxLoss.log_softmax(inputs[0])
        return np.mean(-np.sum(inputs[1] * log_probs, axis=1))

    @staticmethod
    def _backward(gradient, *inputs):
        assert np.asarray(gradient).ndim == 0
        log_probs = SoftmaxLoss.log_softmax(inputs[0])
        return [
            gradient * (np.exp(log_probs) - inputs[1]) / inputs[0].shape[0],
            gradient * -log_probs / inputs[0].shape[0]
        ]


def gradients(loss, parameters):
    """
    Computes and returns the gradient of the loss with respect to the provided
    parameters.

    Args:
        loss: a `SquareLoss` or `SoftmaxLoss` node returned by
            `nn.square_loss` or `nn.softmax_loss`
        parameters: a list (or iterable) containing Parameter nodes

    Returns:
        a list of Constant objects, representing the gradient of the loss
        with respect to each provided parameter.
    """

    assert isinstance(loss, (SquareLoss, SoftmaxLoss)), (
        "Loss must be a loss node, instead has type {!r}".format(
            type(loss).__name__))
    assert all(isinstance(parameter, Parameter) for parameter in parameters), (
        "Parameters must all have type {}, instead got types {!r}".format(
            Parameter.__name__,
            tuple(type(parameter).__name__ for parameter in parameters)))
    assert not hasattr(loss, "used"), (
        "Loss node has already been used for backpropagation, cannot reuse")

    loss.used = True

    nodes = set()
    tape = []

    def visit(node):
        if node not in nodes:
            for parent in node.parents:
                visit(parent)
            nodes.add(node)
            tape.append(node)

    visit(loss)
    nodes |= set(parameters)

    grads = {node: np.zeros_like(node.data) for node in nodes}
    grads[loss] = 1.0

    for node in reversed(tape):
        parent_grads = node._backward(
            grads[node], *(parent.data for parent in node.parents))
        for parent, parent_grad in zip(node.parents, parent_grads):
            grads[parent] += parent_grad

    return [Constant(grads[parameter]) for parameter in parameters]


__pdoc__[register_api.__name__] = False
__pdoc__[format_shape.__name__] = False
for c in ["Node", "Parameter", "DataNode", "Constant"]:
    __pdoc__[f"{c}.data"] = False
__pdoc__[FunctionNode.__name__] = False
