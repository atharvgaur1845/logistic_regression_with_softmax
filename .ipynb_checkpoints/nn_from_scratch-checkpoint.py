import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize neural network
        Parameters:
        layer_sizes (list): Number of neurons in each layer [input_size, hidden_size, output_size]
        learning_rate (float): Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # He initialization for weights
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)

    def forward(self, X):
        """
        Forward propagation
        Parameters:
        X (array): Input data
        """
        self.activations = [X]
        
        # Compute activations for each layer
        for i in range(len(self.weights)):
            net = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(self.sigmoid(net))
        
        return self.activations[-1]

    def backward(self, X, y, output):
        """
        Backward propagation
        Parameters:
        X (array): Input data
        y (array): True labels
        output (array): Predicted output
        """
        m = X.shape[0]
        delta = output - y
        
        # Update weights and biases for each layer
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate gradients
            weight_grad = np.dot(self.activations[i].T, delta) / m
            bias_grad = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * weight_grad
            self.biases[i] -= self.learning_rate * bias_grad
            
            # Calculate delta for next layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])

    def train(self, X, y, epochs=1000, batch_size=32, verbose=True):
        """
        Train the neural network
        Parameters:
        X (array): Training data
        y (array): Training labels
        epochs (int): Number of training epochs
        batch_size (int): Size of mini-batches
        verbose (bool): Whether to print progress
        """
        m = X.shape[0]
        losses = []

        for epoch in range(epochs):
            # Mini-batch gradient descent
            indices = np.random.permutation(m)
            for i in range(0, m, batch_size):
                batch_indices = indices[i:min(i + batch_size, m)]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Forward and backward propagation
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)
            
            # Calculate and store loss
            output = self.forward(X)
            loss = np.mean(-y * np.log(output + 1e-15) - (1 - y) * np.log(1 - output + 1e-15))
            losses.append(loss)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses

    def predict(self, X):
        """
        Make predictions
        Parameters:
        X (array): Input data
        """
        output = self.forward(X)
        return output

# Example usage with wine quality data
def prepare_data(df):
    """
    Prepare data for neural network
    Parameters:
    df (DataFrame): Wine quality dataset
    """
    # Normalize features
    X = df.drop('quality', axis=1).values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Convert quality to binary classification (high quality vs not high quality)
    y = (df['quality'] >= 7).astype(int)
    y = y.values.reshape(-1, 1)
    
    return X, y

def main(file_path):
    """
    Main function to run neural network on wine data
    Parameters:
    file_path (str): Path to wine data CSV
    """
    # Load and prepare data
    df = pd.read_csv(file_path)
    X, y = prepare_data(df)
    
    # Create and train neural network
    input_size = X.shape[1]
    nn = NeuralNetwork([input_size, 8, 1])  # Input layer, hidden layer with 8 neurons, output layer
    
    # Split data into train and test sets
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    
    X_train = X[indices[:train_size]]
    y_train = y[indices[:train_size]]
    X_test = X[indices[train_size:]]
    y_test = y[indices[train_size:]]
    
    # Train the network
    losses = nn.train(X_train, y_train, epochs=1000, verbose=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # Evaluate on test set
    predictions = nn.predict(X_test)
    predictions = (predictions >= 0.5).astype(int)
    accuracy = np.mean(predictions == y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

# Example usage:
# main('path_to_your_wine_data.csv')