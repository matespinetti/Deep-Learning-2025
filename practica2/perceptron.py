import numpy as np

class Perceptron:
    """
    Perceptrón binario implementado desde cero.
    
    El perceptrón es un algoritmo de aprendizaje supervisado para clasificación binaria.
    Aprende un hiperplano que separa las dos clases.
    
    Regla de decisión: y_pred = 1 si w·x + b > 0, 0 en caso contrario
    
    Parámetros:
    -----------
    lr : float, default=0.1
        Tasa de aprendizaje (learning rate)
    max_iter : int, default=1000
        Número máximo de iteraciones (épocas)
    tol : float, default=0.0
        Margen de tolerancia para la convergencia
    shuffle : bool, default=True
        Si mezclar los datos en cada época
    random_state : int, default=None
        Semilla para reproducibilidad
    verbose : bool, default=False
        Si mostrar progreso durante el entrenamiento
    """

    def __init__(self, lr=0.1, max_iter=1000, tol=0.0, shuffle=True, random_state=None, verbose=False):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

        # Atributos que se inicializan después de fit()
        self.w_ = None          # pesos (n_features,)
        self.b_ = None          # sesgo (bias)
        self.n_iter_ = 0        # número de iteraciones realizadas
        self.errors_per_epoch_ = []  # errores por época
        self.is_fitted_ = False

    def _validate_input(self, X, y):
        """Valida y convierte las entradas a formato numpy."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        
        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D")
        if y.ndim != 1:
            raise ValueError("y debe ser un vector 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de muestras")
        
        # Validar que y sea binario
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"y debe tener exactamente 2 clases, encontré {len(unique_classes)}")
        
        # Convertir a {0, 1} si es necesario
        if set(unique_classes) == {-1, 1}:
            y = (y == 1).astype(int)
        elif set(unique_classes) != {0, 1}:
            raise ValueError(f"Las clases deben ser {{0,1}} o {{-1,1}}, encontré {unique_classes}")
        
        return X, y

    def fit(self, X, y):
        """
        Entrena el perceptrón.
        
        Parámetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Datos de entrenamiento
        y : array-like, shape (n_samples,)
            Etiquetas de clase (0 o 1)
        
        Retorna:
        --------
        self : Perceptron
            El objeto perceptrón entrenado
        """
        # Validar y convertir entradas
        X, y = self._validate_input(X, y)
        
        n_samples, n_features = X.shape
        
        # Inicializar pesos y sesgo
        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b_ = 0.0  # Inicializar sesgo en 0 es más común
        
        # Lista para guardar errores por época
        self.errors_per_epoch_ = []
        
        # Entrenamiento
        indices = np.arange(n_samples)
        
        for epoch in range(1, self.max_iter + 1):
            if self.shuffle:
                rng.shuffle(indices)
            
            errors = 0
            
            # Iterar sobre todas las muestras
            for i in indices:
                xi = X[i]
                yi = y[i]
                
                # Calcular activación: w·x + b
                activation = np.dot(xi, self.w_) + self.b_
                
                # Predicción: 1 si activation > 0, 0 en caso contrario
                prediction = 1 if activation > 0 else 0
                
                # Actualizar pesos si hay error de clasificación
                if prediction != yi:
                    # Regla de actualización del perceptrón
                    error = yi - prediction
                    self.w_ += self.lr * error * xi
                    self.b_ += self.lr * error
                    errors += 1
            
            # Guardar número de errores en esta época
            self.errors_per_epoch_.append(errors)
            self.n_iter_ = epoch
            
            if self.verbose and epoch % 100 == 0:
                print(f"Época {epoch:4d} | Errores: {errors:3d} | Accuracy: {(n_samples-errors)/n_samples:.3f}")
            
            # Convergencia: si no hay errores, el algoritmo convergió
            if errors == 0:
                if self.verbose:
                    print(f"Convergencia alcanzada en época {epoch}")
                break
        
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        """
        Calcula la función de decisión para las muestras.
        
        Parámetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Datos de entrada
        
        Retorna:
        --------
        scores : array, shape (n_samples,)
            Puntuaciones de decisión (w·x + b)
        """
        if not self.is_fitted_:
            raise ValueError("El perceptrón debe ser entrenado antes de hacer predicciones")
        
        X = np.asarray(X, dtype=float)
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """
        Predice las clases para las muestras.
        
        Parámetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Datos de entrada
        
        Retorna:
        --------
        predictions : array, shape (n_samples,)
            Predicciones de clase (0 o 1)
        """
        scores = self.decision_function(X)
        return (scores > 0).astype(int)

    def score(self, X, y):
        """
        Calcula la precisión (accuracy) en el conjunto dado.
        
        Parámetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Datos de entrada
        y : array-like, shape (n_samples,)
            Etiquetas verdaderas
        
        Retorna:
        --------
        accuracy : float
            Precisión del modelo
        """
        y = np.asarray(y, dtype=int)
        y_pred = self.predict(X)
        return (y_pred == y).mean()

    def get_params(self):
        """Retorna los parámetros del modelo."""
        return {
            'w_': self.w_,
            'b_': self.b_,
            'n_iter_': self.n_iter_,
            'errors_per_epoch_': self.errors_per_epoch_
        }
