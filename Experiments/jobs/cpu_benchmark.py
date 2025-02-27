import time
import random
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import torch
import gpytorch

def cpu_benchmark():
    """Test CPU performance by performing mathematical operations."""
    print("CPU Benchmark")
    print("Performing 10 million loops with mathematical operations...")
    
    start = time.time()
    
    for i in range(10**7):
        random.random()
    
    end = time.time()
    elapsed = end - start
    print(f"CPU Benchmark completed in {elapsed:.2f} seconds")
    print(f"Operations per second: {1e7 / elapsed:.2f}\n")

def numpy_cpu_benchmark():
    """Perform a series of complex CPU operations using NumPy."""
    print("CPU Benchmark (NumPy)")

    # Define the size of the matrices
    matrix_size = 10000  # Adjust based on your system's capabilities

    print(f"Matrix operations with {matrix_size}x{matrix_size} matrices")
    print("-" * 50)

    # Matrix Multiplication
    print("Performing matrix multiplication...")
    start = time.time()
    
    # Generate two random matrices
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)
    
    # Multiply the matrices
    matrix_c = np.dot(matrix_a, matrix_b)
    
    end = time.time()
    elapsed = end - start
    print(f"Matrix multiplication completed in {elapsed:.2f} seconds\n")

    # Element-wise Operations
    print("Performing element-wise operations...")
    start = time.time()
    
    # Perform element-wise square root
    matrix_sqrt = np.sqrt(matrix_c)
    
    # Perform element-wise multiplication
    matrix_mult = matrix_sqrt * matrix_c
    
    # Perform element-wise division
    if np.any(matrix_c == 0):
        # Avoid division by zero
        matrix_div = np.True_divide(matrix_sqrt, matrix_c, where=matrix_c != 0)
    else:
        matrix_div = matrix_sqrt / matrix_c
        
    end = time.time()
    elapsed = end - start
    print(f"Element-wise operations completed in {elapsed:.2f} seconds\n")

    
def generate_data(n_samples=1000000, n_features=100):
    """Generate a synthetic dataset for benchmarking."""
    # Generate a large dataset with 1,000,000 samples and 100 features
    X, y = datasets.make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
    return X, y

def linear_regression_benchmark(X, y):
    """Benchmark Linear Regression model."""
    print("\nLinear Regression Benchmark")
    print("---------------------------")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and fit the model
    start = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    end = time.time()
    
    print(f"Training Time: {end - start:.2f} seconds")

def svm_benchmark(X, y):
    """Benchmark SVM model."""
    print("\nSVM Benchmark")
    print("------------")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize and fit the model
    start = time.time()
    model = SVC(kernel='linear', C=1)
    model.fit(X_train, y_train)
    end = time.time()
    
    print(f"Training Time: {end - start:.2f} seconds")

def kmeans_benchmark(X):
    """Benchmark K-means clustering."""
    print("\nK-means Clustering Benchmark")
    print("---------------------------")
    
    # Initialize and fit the model
    start = time.time()
    model = KMeans(n_clusters=10, random_state=42)
    model.fit(X)
    end = time.time()
    
    print(f"Clustering Time: {end - start:.2f} seconds")

def pca_benchmark(X):
    """Benchmark PCA dimensionality reduction."""
    print("\nPCA Benchmark")
    print("------------")
    
    # Initialize and fit the model
    start = time.time()
    model = PCA(n_components=50)
    model.fit_transform(X)
    end = time.time()
    
    print(f"Transformation Time: {end - start:.2f} seconds")

def neural_network_benchmark(X, y):
    """Benchmark Neural Network (MLPClassifier)."""
    print("\nNeural Network Benchmark")
    print("-------------------------")

    # map y to integers range(0, 10)
    y = np.digitize(y, np.linspace(y.min(), y.max(), 10))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize and fit the model
    start = time.time()
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    end = time.time()
    
    print(f"Training Time: {end - start:.2f} seconds")

def gaussian_process_benchmark(X, y):
    """Benchmark Gaussian Process Regression."""
    print("\nGaussian Process Benchmark")
    print("--------------------------")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and fit the model
    start = time.time()
    # Use RBF kernel for Gaussian Process
    kernel = RBF(length_scale=1.0)
    model = GaussianProcessRegressor(kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    end = time.time()
    
    print(f"Training Time: {end - start:.2f} seconds")


def gpytorch_benchmark():
    n_train = 5000  
    input_dim = 5   
    num_epochs = 100 
    learning_rate = 0.1 

    torch.manual_seed(123) 

    train_x = torch.randn(n_train, input_dim)
    train_y = torch.sin(train_x.sum(dim=-1)) + torch.randn(n_train) * 0.1 

    class SimpleGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SimpleGPModel(train_x, train_y, likelihood)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    start_time = time.time() 

    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0: 
            print(f'Epoch {i+1}/{num_epochs}, Loss: {loss.item():.4f}')

    end_time = time.time() 
    training_time = end_time - start_time

    print("\n--- Benchmark Results ---")
    print(f"Number of training data points (n_train): {n_train}")
    print(f"Input dimension (input_dim): {input_dim}")
    print(f"Number of epochs (num_epochs): {num_epochs}")
    print(f"Total Training Time: {training_time:.4f} seconds")
    print("\nBenchmark completed!")

if __name__ == "__main__":
    # cpu_benchmark()
    numpy_cpu_benchmark()

    # X, y = generate_data(n_samples=10000, n_features=100)
    # neural_network_benchmark(X=X, y=y)

    # X, y = generate_data(n_samples=4000, n_features=10)
    # gaussian_process_benchmark(X=X, y=y)

    # gpytorch_benchmark()

