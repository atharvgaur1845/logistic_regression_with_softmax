import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KNNClassifier:
    def __init__(self, k=3):
        """
        Initialize KNN classifier
        Parameters:
        k (int): Number of neighbors to use
        """
        self.k = k
        
    def fit(self, X, y):
        """
        Fit the classifier (for KNN, this just stores the training data)
        Parameters:
        X (array): Training features
        y (array): Training labels
        """
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        """
        Predict classes for new data points
        Parameters:
        X (array): Test features
        """
        predictions = []
        
        for x in X:
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]
            
            # Predict class (majority vote)
            prediction = pd.Series(k_nearest_labels).mode()[0]
            predictions.append(prediction)
            
        return np.array(predictions)

def prepare_data(df):
    """
    Prepare data for classification
    Parameters:
    df (DataFrame): Wine quality dataset
    """
    # Create quality categories
    conditions = [
        (df['quality'] <= 4),
        (df['quality'] > 4) & (df['quality'] <= 7),
        (df['quality'] > 7)
    ]
    choices = ['low', 'medium', 'high']
    df['quality_category'] = np.select(conditions, choices)
    
    # Prepare features
    X = df.drop(['quality', 'quality_category'], axis=1).values
    y = df['quality_category'].values
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    return X, y

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split data into training and test sets
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Generate random indices
    indices = np.random.permutation(len(X))
    test_size = int(test_size * len(X))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def calculate_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy
    """
    return np.mean(y_true == y_pred)

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix
    """
    # Get unique classes
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes))
    
    # Fill confusion matrix
    for i in range(len(y_true)):
        true_idx = np.where(classes == y_true[i])[0][0]
        pred_idx = np.where(classes == y_pred[i])[0][0]
        cm[true_idx, pred_idx] += 1
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    plt.xticks(range(n_classes), classes, rotation=45)
    plt.yticks(range(n_classes), classes)
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, int(cm[i, j]),
                    ha="center", va="center")
    
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()

def main(file_path):
    """
    Main function to run the classification pipeline
    Parameters:
    file_path (str): Path to wine data CSV
    """
    # Load and prepare data
    df = pd.read_csv(file_path)
    X, y = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train classifier
    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Calculate and print accuracy
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Print class distribution
    class_distribution = pd.Series(y).value_counts()
    print("\nClass Distribution:")
    print(class_distribution)
    
    # Identify outlier classes
    mean_count = class_distribution.mean()
    std_count = class_distribution.std()
    outlier_threshold = mean_count + 2 * std_count
    
    print("\nOutlier Classes (classes with counts > 2 std from mean):")
    outlier_classes = class_distribution[class_distribution > outlier_threshold]
    print(outlier_classes)

# Example usage:
# main('path_to_your_wine_data.csv')