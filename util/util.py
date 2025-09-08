

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

