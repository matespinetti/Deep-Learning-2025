
import numpy as np
# Función para calcular estadísticas y outliers
def resumen_outliers(data, atributo="Age"):
    Q1 = data[atributo].quantile(0.25)
    Q2 = data[atributo].quantile(0.50)
    Q3 = data[atributo].quantile(0.75)
    IQR = Q3 - Q1
    
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    
    lim_inf_ext = Q1 - 3 * IQR
    lim_sup_ext = Q3 + 3 * IQR
    
    out_leves = data[(data[atributo] < lim_inf) | (data[atributo] > lim_sup)]
    out_extremos = data[(data[atributo] < lim_inf_ext) | (data[atributo] > lim_sup_ext)]
    
    return {
        "Q1": Q1,
        "Q2": Q2,
        "Q3": Q3,
        "IQR": IQR,
        "lim_inf": lim_inf,
        "lim_sup": lim_sup,
        "out_leves": out_leves[atributo].tolist(),
        "out_extremos": out_extremos[atributo].tolist()
    }



def sgd_regresion_lineal(X, y, lr=0.01, epochs=1000):
    """
    Entrena un modelo de regresión lineal simple usando SGD (sin batches).
    
    Parámetros
    ----------
    X : array 1D o 2D
        Variable independiente (características). Si es 1D, se convierte en columna.
    y : array 1D
        Variable dependiente (target).
    lr : float
        Tasa de aprendizaje.
    epochs : int
        Cantidad de épocas (pasadas completas por el dataset).
        
    Retorna
    -------
    w : float
        Peso (pendiente).
    b : float
        Bias (intercepto).
    history : list
        Lista con el error cuadrático medio por época.
    """
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    n = len(y)

    # Inicialización aleatoria
    w = np.random.randn()
    b = np.random.randn()
    
    history = []

    for epoch in range(epochs):
        mse_epoch = 0
        # recorremos los datos uno por uno (SGD puro)
        for i in range(n):
            xi = X[i][0]
            yi = y[i]
            y_pred = w * xi + b
            error = y_pred - yi

            # Gradientes
            dw = error * xi
            db = error

            # Actualización
            w -= lr * dw
            b -= lr * db

            mse_epoch += error**2
        
        # guardamos el ECM promedio de la época
        history.append(mse_epoch / n)

    return w, b, history
