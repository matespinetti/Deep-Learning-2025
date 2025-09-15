import numpy as np
import time
import matplotlib.pyplot as plt

class RNMulticlase(object):
    """
    Parameters
    ------------
    alpha : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    cotaE : float
        minimum error threshold
    FUN : string
        activation function: 'sigmoid', 'tanh', 'softmax', otherwise linear
    COSTO : string
        'ECM' (MSE), 'EC_binaria' (BCE), 'EC' (cross-entropy multiclase)
    random_state : int
        Random number generator seed for random weight initialization.
    verbose : bool
        If True, prints training logs each epoch.
        
    Attributes
    -----------
    w_ : 2d-array (nOut, nIn)
        Weights after fitting.
    b_ : 2d-array (nOut, 1)
        Biases after fitting.
    errors_ : list
        Cost per epoch (promedio por muestra).
    accuracy_ : list
        Accuracy por epoch.
    """
    def __init__(self, alpha=0.01, n_iter=50, cotaE=10e-07, FUN='sigmoid', COSTO='ECM', random_state=None, verbose=False):
        self.alpha = alpha
        self.n_iter = n_iter
        self.cotaE = cotaE
        self.FUN = FUN
        self.COSTO = COSTO
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        X : shape [n_examples, n_features]
        y : shape [n_examples, n_class] (one-hot)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        nRow = X.shape[0]
        nIn  = X.shape[1]
        nOut = y.shape[1]

        self.w_ = np.random.uniform(-0.5, 0.5, [nOut, nIn])
        self.b_ = np.random.uniform(-0.5, 0.5, [nOut, 1])

        self.errors_ = []
        self.accuracy_ = []
        ErrorAnt = 0.0
        ErrorAct = 1.0

        if self.verbose:
            print(f"Training start: FUN={self.FUN}, COSTO={self.COSTO}, alpha={self.alpha}, epochs={self.n_iter}")

        i = 0
        t0 = time.time()
        while (i < self.n_iter) and (np.abs(ErrorAnt - ErrorAct) > self.cotaE):
            ErrorAnt = ErrorAct
            ErrorAct = 0.0

            for e in range(nRow):
                xi = X[e:e+1, :]               # (1, nIn)
                salida = self.predict_proba(xi).T  # (nOut, 1)
                errorXi = (y[e:e+1, :].T - salida)  # (nOut, 1)

                # Regla de actualización:
                # - Para softmax + EC: gradiente es (y - y_hat) sin derivada adicional
                # - Para sigmoid/tanh/lineal: usar derivar(salida) como estaba
                if self.FUN == 'softmax' and self.COSTO in ('EC', 'EC_binaria'):
                    update = self.alpha * errorXi
                else:
                    update = self.alpha * errorXi * self._activate_deriv(salida)

                self.w_ += update @ xi          # (nOut,1)@(1,nIn) -> (nOut,nIn)
                self.b_ += update               # (nOut,1)

                ErrorAct += self._loss(y[e:e+1, :].T, salida)

            ErrorAct = ErrorAct / nRow
            self.errors_.append(ErrorAct)
            acc = self.score(X, y)
            self.accuracy_.append(acc)

            if self.verbose:
                print(f"Epoch {i+1:3d} | loss={ErrorAct:.6f} | acc={acc:.4f}")

            i += 1

        if self.verbose:
            dt = time.time() - t0
            print(f"Finished at epoch {i} | loss={ErrorAct:.6f} | acc={self.accuracy_[-1]:.4f} | time={dt:.2f}s")

        return self

    # Standard name
    def _loss(self, y, y_hat):
        EPS = np.finfo(float).eps
        if self.COSTO == 'ECM':
            return np.sum((y - y_hat) ** 2)

        if self.COSTO == 'EC_binaria':
            # Para tanh, mapear predicción a [0,1]
            if self.FUN == 'tanh':
                y_hat = (y_hat + 1.0) / 2.0
            y_hat = np.clip(y_hat, EPS, 1 - EPS)
            return np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))

        if self.COSTO == 'EC':
            # Multiclase: usar softmax implícito. Si FUN es softmax ya está en probas.
            # Si no es softmax, asumimos que y_hat son scores -> softmax estable
            if self.FUN != 'softmax':
                # transformar a probabilidades con softmax estable por fila (aquí y_hat (nOut,1))
                s = y_hat - np.max(y_hat, axis=0, keepdims=True)
                exp_s = np.exp(s)
                y_hat = exp_s / np.sum(exp_s, axis=0, keepdims=True)
            y_hat = np.clip(y_hat, EPS, 1 - EPS)
            return -np.sum(y * np.log(y_hat))

        # Por defecto, ECM
        return np.sum((y - y_hat) ** 2)

    # Backward-compatible alias
    def fCosto(self, y, y_hat):
        return self._loss(y, y_hat)

    # Standard name
    def decision_function(self, X):
        """Linear scores before activation (nRow, nOut)"""
        netas = self.w_ @ X.T + self.b_  # (nOut, nIn)@(nIn, nRow)^T + (nOut,1)
        return netas.T                    # (nRow, nOut)

    # Backward-compatible alias
    def net_input(self, X):
        return self.decision_function(X)
    
    # Standard name
    def _activate(self, x):
        if self.FUN == 'tanh':
            return (2.0 / (1 + np.exp(-2 * x)) - 1)
        elif self.FUN == 'sigmoid':
            return (1.0 / (1 + np.exp(-x)))
        elif self.FUN == 'softmax':
            # Softmax estable por filas
            x = x - np.max(x, axis=1, keepdims=True)
            ex = np.exp(x)
            return ex / np.sum(ex, axis=1, keepdims=True)
        else:
            return x
        
    # Backward-compatible alias
    def evaluar(self, x):
        return self._activate(x)

    # Standard name
    def _activate_deriv(self, x):
        # x es la salida activada (y_hat), como en tu implementación original
        if self.FUN == 'tanh':
            return (1 - x**2)
        elif self.FUN == 'sigmoid':
            return (x * (1 - x))
        else:
            # lineal o softmax (softmax + CE se maneja aparte en fit)
            return 1.0

    # Backward-compatible alias
    def derivar(self, x):
        return self._activate_deriv(x)

    # Standard name
    def predict_proba(self, X):
        """Activated outputs (nRow, nOut)"""
        return self._activate(self.decision_function(X))

    # Backward-compatible alias
    def predict_nOut(self, X):
        return self.predict_proba(X)
    
    def predict(self, X):
        """Retorna un entero con el índice de la clase más probable"""
        y_hat = self.predict_nOut(X)
        if self.FUN == 'tanh':
            y_hat = (y_hat > 0) * 1
        if self.FUN == 'sigmoid':
            y_hat = (y_hat > 0.5) * 1
        # softmax y lineal quedan como scores/probas y se toma argmax
        return np.argmax(y_hat, axis=1)
            
    # Standard name
    def score(self, X, y):
        y_pred = self.predict(X)
        OK = np.sum(np.argmax(y, axis=1) == y_pred)
        return (OK / X.shape[0])

    # Backward-compatible alias
    def accuracy(self, X, y):
        return self.score(X, y)

    def save(self, archivo):
        np.savez(archivo, matriz1=self.w_, matriz2=self.b_)

    def load(self, archivo):
        with np.load(archivo) as data:
            claves = list(data.keys())
            if (len(claves) != 2):
                print("ERROR --> Formato de archivo incorrecto")
            self.w_ = data['matriz1']
            self.b_ = data['matriz2']

    # Métodos nuevos de utilidad

    def plot_training(self):
        """Grafica costo (promedio por muestra) y accuracy por época"""
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(self.errors_, label='Loss')
        ax[0].set_title('Training Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].grid(True)
        ax[0].legend()

        ax[1].plot(self.accuracy_, color='orange', label='Accuracy')
        ax[1].set_title('Training Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_ylim(0, 1)
        ax[1].grid(True)
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    def plot_outputs(self, X, y, sample_size=100):
        """Grafica barras de salidas promedio por clase en una muestra"""
        idx = np.random.choice(X.shape[0], size=min(sample_size, X.shape[0]), replace=False)
        y_hat = self.predict_nOut(X[idx])
        clases = np.argmax(y[idx], axis=1)
        nOut = y.shape[1]
        means = []
        for c in range(nOut):
            if np.any(clases == c):
                means.append(y_hat[clases == c].mean(axis=0))
            else:
                means.append(np.zeros(nOut))
        means = np.vstack(means)
        plt.figure(figsize=(8, 5))
        for c in range(nOut):
            plt.bar(np.arange(nOut) + c*(0.8/nOut), means[c], width=(0.8/nOut), label=f'Class {c}')
        plt.xticks(np.arange(nOut), [f'Out {i}' for i in range(nOut)])
        plt.title('Average outputs per true class (sample)')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.show()