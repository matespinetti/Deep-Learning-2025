import numpy as np

class LinearSGDRegressor:
    """
    Regresión lineal entrenada con Descenso de Gradiente Estocástico (SGD) sin mini-batches.

    Modelo:
        y_hat = w·x + b

    Parámetros
    ----------
    lr : float (default=0.01)
        Tasa de aprendizaje.
    epochs : int (default=1000)
        Cantidad de épocas (pasadas completas por el dataset).
    shuffle : bool (default=True)
        Si mezclar o no las muestras en cada época.
    random_state : int | None
        Semilla para reproducibilidad del barajado e inicialización.
    fit_intercept : bool (default=True)
        Si aprender o no el término independiente b.
    l2 : float (default=0.0)
        Regularización L2 (Ridge). 0 desactiva la regularización.
    tol : float | None
        Si se especifica, detiene temprano cuando |ECM_prev - ECM_actual| < tol.
    verbose : bool (default=False)
        Si imprimir el progreso por época.

    Atributos tras fit()
    --------------------
    w_ : ndarray de forma (n_features,)
        Vector de pesos aprendido.
    b_ : float
        Intercepto (si fit_intercept=True, caso contrario 0.0).
    history_ : list[float]
        Historia del ECM por época.

    Métodos
    -------
    fit(X, y)         -> entrena el modelo
    partial_fit(X, y) -> una pasada (1 época) de SGD
    predict(X)        -> predicciones
    score(X, y)       -> R²
    mse(X, y)         -> error cuadrático medio
    get_params()      -> dict con w_ y b_
    set_params(**kw)  -> actualiza hiperparámetros
    """

    def __init__(self, lr=0.01, epochs=1000, shuffle=True, random_state=None,
                 fit_intercept=True, l2=0.0, tol=None, verbose=False):
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.shuffle = bool(shuffle)
        self.random_state = random_state
        self.fit_intercept = bool(fit_intercept)
        self.l2 = float(l2)
        self.tol = tol
        self.verbose = bool(verbose)

        self.w_ = None
        self.b_ = 0.0
        self.history_ = []

        self._rng = np.random.default_rng(random_state)

    # ---------- utils internos ----------
    def _check_Xy(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener la misma cantidad de filas.")
        return X, y

    def _init_params(self, n_features):
        if self.w_ is None:
            # Inicialización pequeña (mejor que grande para estabilidad numérica)
            self.w_ = self._rng.normal(loc=0.0, scale=0.01, size=n_features)
        if not self.fit_intercept:
            self.b_ = 0.0
        else:
            # Inicializar b pequeño
            if not hasattr(self, "b_") or self.b_ is None:
                self.b_ = float(self._rng.normal(0.0, 0.01))

    # ---------- API pública ----------
    def fit(self, X, y):
        X, y = self._check_Xy(X, y)
        n, d = X.shape
        self._init_params(d)
        self.history_ = []

        indices = np.arange(n)

        prev_ecm = None
        for epoch in range(self.epochs):
            if self.shuffle:
                self._rng.shuffle(indices)

            # SGD: actualizamos por muestra
            for i in indices:
                xi = X[i]
                yi = y[i]

                yhat = xi @ self.w_ + (self.b_ if self.fit_intercept else 0.0)
                error = yhat - yi

                # Gradientes (ECM por muestra: 1/2 * error^2 → grad = error * deriv(lineal))
                # L2 sólo para w (no para b)
                grad_w = error * xi + self.l2 * self.w_
                self.w_ -= self.lr * grad_w

                if self.fit_intercept:
                    grad_b = error
                    self.b_ -= self.lr * grad_b

            # Al final de la época, registrar ECM y chequear tol
            ecm = self.mse(X, y)
            self.history_.append(ecm)
            if self.verbose:
                print(f"Época {epoch+1:4d}/{self.epochs}  ECM={ecm:.6f}")

            if self.tol is not None and prev_ecm is not None:
                if abs(prev_ecm - ecm) < self.tol:
                    if self.verbose:
                        print(f"Early stopping por tol={self.tol}.")
                    break
            prev_ecm = ecm

        return self

    def partial_fit(self, X, y):
        """Una sola época de entrenamiento (útil para flujos en línea)."""
        X, y = self._check_Xy(X, y)
        n, d = X.shape
        self._init_params(d)

        indices = np.arange(n)
        if self.shuffle:
            self._rng.shuffle(indices)

        for i in indices:
            xi = X[i]
            yi = y[i]
            yhat = xi @ self.w_ + (self.b_ if self.fit_intercept else 0.0)
            error = yhat - yi

            grad_w = error * xi + self.l2 * self.w_
            self.w_ -= self.lr * grad_w

            if self.fit_intercept:
                self.b_ -= self.lr * error

        # opcional: registrar ECM de esta pasada
        self.history_.append(self.mse(X, y))
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.w_ is None:
            raise RuntimeError("El modelo no está entrenado. Llamá a fit() primero.")
        yhat = X @ self.w_ + (self.b_ if self.fit_intercept else 0.0)
        return yhat

    def mse(self, X, y):
        X, y = self._check_Xy(X, y)
        yhat = self.predict(X)
        return float(np.mean((y - yhat) ** 2))

    def score(self, X, y):
        """Coeficiente de determinación R^2."""
        X, y = self._check_Xy(X, y)
        yhat = self.predict(X)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def get_params(self):
        return {"w_": None if self.w_ is None else self.w_.copy(),
                "b_": float(self.b_)}

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Parámetro desconocido: {k}")
            setattr(self, k, v)
        return self
