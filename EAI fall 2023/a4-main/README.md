

## PART 1 
### 1.Objective:
- Task is to implement a K-Nearest Neighbors (KNN) classifier from scratch. The codebase consists of two files, utils.py and k_nearest_neighbors.py. 
- The former includes utility functions for calculating Euclidean and Manhattan distances, while the latter defines the KNearestNeighbors class to implement the KNN algorithm.

### 2. Code Explaination:
- The code consists of two Python files, k_nearest_neighbors.py and utils.py, which together implement a K-Nearest Neighbors (KNN) classifier from scratch. 
  Below is a detailed explanation of each file and its functions:
  
  1. k_nearest_neighbors.py
    - Class: KNearestNeighbors has below mentioned attributes:
		- n_neighbors: An integer representing the number of neighbors considered when predicting target class values.
		- weights: A string ('uniform' or 'distance') representing the weight function used during prediction.
		- _X: A numpy array representing the input data used for fitting and predicting.
		- _y: A numpy array representing the true class values for each sample.
		- _distance: A function representing the distance metric used ('euclidean_distance' or 'manhattan_distance').
		
	- __init__(self, n_neighbors=5, weights='uniform', metric='l2') function :
		- Initializes the KNearestNeighbors object with specified parameters.
		- Checks if the provided arguments are valid.
	- fit(self, X, y):
		- Fits the model to the provided data matrix X and targets y.
		- Sets _X and _y attributes.
	- predict(self, X):
		- Predicts class target values for the given test data matrix X using the fitted classifier model.
		- Computes distances between test samples and training samples.
		- Considers nearest neighbors based on distances and predicts the class.
		
  2. utils.py
	- euclidean_distance(x1, x2):
		- This function calculates the Euclidean distance between two vectors x1 and x2.		
	- manhattan_distance(x1, x2):
		- This function calculates the Manhattan distance between two vectors x1 and x2. 
	- identity(x, derivative=False):
		- Computes the identity activation function of the given input data x.
		- If derivative=True, it returns the derivative of the identity function, which is always 1.
	- sigmoid(x, derivative=False):
		- Computes the sigmoid (logistic) activation function of the given input data x.
		- If derivative=True, it returns the derivative of the sigmoid function.
	- tanh(x, derivative=False):
		- Computes the hyperbolic tangent activation function of the given input data x.
		- If derivative=True, it returns the derivative of the tanh function.
	- relu(x, derivative=False):
		- Computes the rectified linear unit (ReLU) activation function of the given input data x.
		- If derivative=True, it returns the derivative of the ReLU function.
	- softmax(x, derivative=False):
		- Computes the softmax activation function for a given input data x.
		- If derivative=True, it returns the derivative of the softmax function. Note that the softmax derivative is computed using the softmax function.
	- cross_entropy(y, p):
		- This function computes and returns the cross-entropy loss between true class values (y) and predicted probabilities (p). It uses the negative 
		  log-likelihood of a logistic model.
		- y: A numpy array representing the one-hot encoded target class values.
		- p: A numpy array representing the predicted probabilities from the softmax output activation function.
	- one_hot_encoding(y):
		- Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding. It creates new binary-valued columns, 
		  each indicating the presence of each possible value from the original data.
		- y: A numpy array representing the target class values for each sample in the input data.

### 4. Assumptions, Simplifications, and Design Decisions:
- Activation Functions: The choice of activation functions indicates a general-purpose set, allowing users to experiment with different functions based on 
  their use case.
- One-Hot Encoding: The one-hot encoding function assumes that the target class values are categorical.
- Commented-Out Code: The additional activation functions are commented out, possibly to simplify the code and focus on the core functionality.

### 5.Conclusion:
- The K-Nearest Neighbors (KNN) classifier and utility functions have been successfully implemented from scratch.
- The testing approach, comparing the custom KNN with scikit-learn's implementation, provides a reliable benchmark for accuracy.



## PART 2 

### 1. Problem Formulation:
- The problem addressed in this implementation is the creation of a Multilayer Perceptron (MLP) classifier from scratch. An MLP is a type of artificial neural 
  network with at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. The MLP is trained using a supervised learning 
  algorithm called backpropagation.
- The main components of the problem are as follows:
	- Design an MLP class capable of training on a dataset and making predictions.
	- Implement the forward and backward propagation steps of the backpropagation algorithm.
	- Use activation functions for the hidden layer, output layer, and their derivatives.
	- Optimize the model's weights and biases using gradient descent.
	
### 2. Code Explaination:
- numpy is imported for numerical operations.
- utils is assumed to be a custom module containing various utility functions. These include activation functions (identity, sigmoid, tanh, relu), softmax activation, 
  cross-entropy loss, and one-hot encoding.
- The class MultilayerPerceptron is defined with the following parameters:
	- n_hidden: Number of neurons in the hidden layer.
	- hidden_activation: Activation function for the hidden layer.
	- n_iterations: Number of iterations for training.
	- learning_rate: Learning rate for gradient descent.
- Initialization __init__:
	- The __init__ method initializes the MLP with parameters such as the number of hidden neurons, activation functions, number of iterations, and learning rate.
	- It also defines activation functions for hidden and output layers, loss function, and initializes weights and biases.
- Initialization _initialize method:
	- This method initializes the input data, target output, and randomizes weights and biases.
	- It performs one-hot encoding on the target output for proper training.
- Training fit method:
	- The fit method trains the MLP using forward and backward propagation.The forward propagation calculates the output of the network, and backward propagation 
	  updates weights and biases based on the error.
	- It iteratively updates weights and biases to minimize the error between predicted and target output.
	- The loss is computed and stored every 20 iterations.
	- The training process is controlled by the number of iterations.
- Prediction predict method:
	- The predict method makes predictions on new input data.
	- It performs forward propagation to obtain the output layer's predictions and returns the indices of the maximum values (argmax) as the predicted classes.
  
### 3. Problems Faced:
- Activation Functions:Implementing various activation functions and their derivatives correctly was crucial for proper backpropagation. 
  Debugging and testing were performed to ensure accuracy.
- Backpropagation:Properly implementing the backward propagation step, including computing errors and deltas, required careful attention to avoid errors.

### 4. Assumptions, Simplifications, and Design Decisions:
- Activation Functions:The code assumes that the user will provide a valid activation function for the hidden layer from a predefined set.
- Initialization:We initialize weights and biases randomly, assuming that this provides a reasonable starting point for training.
- One-Hot Encoding:One-hot encoding is assumed to be necessary for the target output to facilitate training.
- Loss Computation:The code computes and stores the loss every 20 iterations, providing insight into the model's training progress.

### 5.Conclusion:
- The implementation successfully trains on data and makes predictions, demonstrating the core concepts of forward and backward propagation in neural network training.
  Ongoing testing and potential optimizations could further enhance the robustness and efficiency of the MLP implementation.
  
  

