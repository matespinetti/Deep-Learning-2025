import numpy as np
import matplotlib.pyplot as plt

class MulticlassRN:
    def __init__(self, learning_rate=0.01, max_iter=1000, activation='sigmoid', tolerance=1e-6, verbose=False, cost='bce'):
        """
        Logistic Regression with different activation functions
        
        Parameters:
        - learning_rate: Learning rate
        - max_iter: Maximum number of iterations
        - activation: 'sigmoid' or 'tanh'
        - tolerance: Convergence tolerance
        - verbose: Show detailed information during training
        - cost: 'bce' (binary cross-entropy) or 'mse' (mean squared error)
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.activation = activation
        self.tolerance = tolerance
        self.verbose = verbose
        self.cost = cost
        if self.activation not in ('sigmoid', 'tanh'):
            raise ValueError("activation must be 'sigmoid' or 'tanh'")
        if self.cost not in ('bce', 'mse'):
            raise ValueError("cost must be 'bce' or 'mse'")
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _sigmoid(self, z):
        """Sigmoid function"""
        # Avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _tanh(self, z):
        """Hyperbolic tangent function"""
        return np.tanh(z)
    
    def _activation_function(self, z):
        """Apply the selected activation function"""
        if self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        else:
            raise ValueError("Activation must be 'sigmoid' or 'tanh'")
    
    def _derivative_sigmoid(self, z):
        """Derivative of sigmoid function"""
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _derivative_tanh(self, z):
        """Derivative of hyperbolic tangent"""
        return 1 - np.tanh(z)**2
    
    def _derivative_activation(self, z):
        """Derivative of activation function"""
        if self.activation == 'sigmoid':
            return self._derivative_sigmoid(z)
        elif self.activation == 'tanh':
            return self._derivative_tanh(z)
    
    def _binary_crossentropy(self, y_true, y_pred):
        """
        Binary cross-entropy loss calculated sample by sample
        
        For each sample i:
        BCE_i = -[y_i * log(p_i) + (1-y_i) * log(1-p_i)]
        
        Returns:
        - Individual losses for each sample
        - Total loss (sum of individual losses)
        """
        # Avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if self.activation == 'sigmoid':
            # For sigmoid: standard binary cross-entropy
            # BCE_i = -[y_i * log(p_i) + (1-y_i) * log(1-p_i)]
            individual_losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:  # tanh
            # For tanh: adapt the loss
            # Convert predictions from [-1,1] to [0,1] for calculation
            y_pred_01 = (y_pred + 1) / 2
            y_pred_01 = np.clip(y_pred_01, epsilon, 1 - epsilon)
            individual_losses = -(y_true * np.log(y_pred_01) + (1 - y_true) * np.log(1 - y_pred_01))
        
        total_loss = np.sum(individual_losses)
        return individual_losses, total_loss
    
    def _normalize_predictions(self, predictions):
        """Normalize predictions according to activation function"""
        if self.activation == 'sigmoid':
            # For sigmoid: already in [0,1]
            return predictions
        else:  # tanh
            # For tanh: convert from [-1,1] to [0,1]
            return (predictions + 1) / 2

    def _compute_loss_and_grad_z_per_sample(self, y_i, z_i):
        """Compute loss scalar and dL/dz for a single sample."""
        eps = 1e-15
        if self.activation == 'sigmoid':
            a = self._sigmoid(z_i)
            if self.cost == 'bce':
                p = np.clip(a, eps, 1 - eps)
                loss = - (y_i * np.log(p) + (1 - y_i) * np.log(1 - p))
                dLdz = (a - y_i)
            else:  # mse
                loss = 0.5 * (a - y_i) ** 2
                dLdz = (a - y_i) * a * (1 - a)
        else:  # tanh
            a = self._tanh(z_i)
            if self.cost == 'bce':
                p = np.clip((a + 1) / 2, eps, 1 - eps)
                loss = - (y_i * np.log(p) + (1 - y_i) * np.log(1 - p))
                dLdz = (p - y_i) * 0.5 * (1 - a ** 2)
            else:  # mse
                y_t = 2 * y_i - 1 if (y_i in (0, 1)) else y_i
                loss = 0.5 * (a - y_t) ** 2
                dLdz = (a - y_t) * (1 - a ** 2)
        return float(loss), float(dLdz)
    
    def fit(self, X, y):
        """
        Train the model
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Label vector (0 or 1)
        """
        if self.verbose:
            print(f"=== STARTING TRAINING ===")
            print(f"Activation function: {self.activation}")
            print(f"Cost function: {self.cost}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Max iterations: {self.max_iter}")
            print(f"Tolerance: {self.tolerance}")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape if hasattr(y, 'shape') else len(y)}")
            print("-" * 50)
        
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        if self.verbose:
            print(f"Initial weights: {self.weights}")
            print(f"Initial bias: {self.bias}")
            print()
        
        # Convert y to numpy array if necessary
        y = np.array(y).flatten()
        
        # Training loop
        prev_loss = None
        for i in range(self.max_iter):
            total_loss = 0.0
            # loop over samples, SGD-style updates
            for e in range(n_samples):
                x_i = X[e]
                z_i = float(np.dot(x_i, self.weights) + self.bias)
                loss_i, dLdz_i = self._compute_loss_and_grad_z_per_sample(y[e], z_i)
                total_loss += loss_i

                # gradients per sample
                dw = x_i * dLdz_i
                db = dLdz_i

                # update
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            self.cost_history.append(total_loss)

            # Verbose progress
            if self.verbose:
                if i == 0:
                    print(f"{'Iter':<6} {'Total Loss':<14} {'Avg Loss':<12} {'Bias':<10}")
                    print("-" * 60)
                if i % 10 == 0 or i < 5:
                    avg_loss = total_loss / n_samples
                    print(f"{i:<6} {total_loss:<14.6f} {avg_loss:<12.6f} {self.bias:<10.4f}")

            # Convergence check
            if prev_loss is not None and abs(prev_loss - total_loss) < self.tolerance:
                if self.verbose:
                    print(f"\n✓ Convergence reached at iteration {i}")
                    print(f"  Loss difference: {abs(prev_loss - total_loss):.2e}")
                break
            prev_loss = total_loss
        
        if self.verbose:
            print(f"\n=== TRAINING COMPLETED ===")
            print(f"Total iterations: {i+1}")
            print(f"Final total loss: {self.cost_history[-1]:.6f}")
            print(f"Final average loss: {self.cost_history[-1]/n_samples:.6f}")
            print(f"Final weights: {self.weights}")
            print(f"Final bias: {self.bias:.3f}")
            print("-" * 50)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = np.dot(X, self.weights) + self.bias
        predictions = self._activation_function(z)
        return self._normalize_predictions(predictions)
    
    def predict(self, X, threshold=0.5):
        """Predict classes"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """Compute accuracy"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        
        if self.verbose:
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Correct predictions: {np.sum(predictions == y)}/{len(y)}")
        
        return accuracy
    
    def plot_cost_history(self):
        """Plot cost history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title(f'Total Loss History - {self.activation.capitalize()} - {self.cost.upper()}')
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.grid(True)
        plt.show()
    
    def plot_individual_losses(self, X, y):
        """Plot individual losses for each sample"""
        z = np.dot(X, self.weights) + self.bias
        predictions = self._activation_function(z)
        individual_losses, _ = self._binary_crossentropy(y, predictions)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(individual_losses)
        plt.title('Individual Losses per Sample')
        plt.xlabel('Sample Index')
        plt.ylabel('Binary Cross-Entropy Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.hist(individual_losses, bins=20, alpha=0.7)
        plt.title('Distribution of Individual Losses')
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_weights(self):
        """Get weights and bias"""
        return self.weights, self.bias

# Example usage
if __name__ == "__main__":
    # Create example data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Split into train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train with verbose
    print("=== LOGISTIC REGRESSION WITH BINARY CROSS-ENTROPY ===")
    lr = MulticlassRN(learning_rate=0.1, max_iter=100, activation='sigmoid', verbose=True)
    lr.fit(X_train, y_train)
    lr.score(X_test, y_test)
    
    # Plot individual losses
    lr.plot_individual_losses(X_test, y_test)