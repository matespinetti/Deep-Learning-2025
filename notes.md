# 📋 Guía de Preprocesamiento de Datos

## 1. 🔍 EXPLORACIÓN INICIAL
- [ ] **Cargar y visualizar los datos**
  - [ ] `df.head()`, `df.tail()`, `df.info()`
  - [ ] `df.describe()` para estadísticas básicas
  - [ ] `df.shape` para dimensiones
- [ ] **Identificar tipos de variables**
  - [ ] Cuantitativas (continuas/discretas)
  - [ ] Cualitativas (nominales/ordinales)
- [ ] **Verificar valores únicos**
  - [ ] `df.nunique()` por columna
  - [ ] `df.value_counts()` para categóricas

## 2. 🚨 DETECCIÓN DE PROBLEMAS
- [ ] **Valores faltantes**
  - [ ] `df.isnull().sum()` - cantidad por columna
  - [ ] `df.isnull().sum() / len(df) * 100` - porcentaje
  - [ ] Visualizar con `sns.heatmap(df.isnull())`
- [ ] **Valores duplicados**
  - [ ] `df.duplicated().sum()`
  - [ ] `df.drop_duplicates()` si es necesario
- [ ] **Outliers**
  - [ ] Boxplots para variables numéricas
  - [ ] `sns.boxplot(data=df, y='variable')`
  - [ ] Z-score o IQR para detectar

## 3. 🧹 LIMPIEZA DE DATOS
- [ ] **Manejar valores faltantes**
  - [ ] Eliminar: `df.dropna()` (si pocos casos)
  - [ ] Imputar: `df.fillna()` o `SimpleImputer`
  - [ ] Crear categoría: `df['col'].fillna('Unknown')`
- [ ] **Eliminar outliers** (si es necesario)
  - [ ] IQR method: `Q1 - 1.5*IQR` y `Q3 + 1.5*IQR`
  - [ ] Z-score: `|z| > 3`
- [ ] **Estandarizar formatos**
  - [ ] Fechas: `pd.to_datetime()`
  - [ ] Texto: `.str.lower()`, `.str.strip()`
  - [ ] Categorías: `.str.capitalize()`

## 4. 🔄 TRANSFORMACIÓN DE VARIABLES
- [ ] **Variables categóricas**
  - [ ] **Nominales**: One-Hot Encoding (`pd.get_dummies()` o `OneHotEncoder`)
  - [ ] **Ordinales**: Label Encoding (`LabelEncoder` o `OrdinalEncoder`)
  - [ ] **Binarias**: Mapeo directo (`{'yes': 1, 'no': 0}`)
- [ ] **Variables numéricas**
  - [ ] **Discretas**: Verificar si necesitan transformación
  - [ ] **Continuas**: Estandarización/normalización
- [ ] **Crear nuevas variables**
  - [ ] Binning: `pd.cut()`, `pd.qcut()`
  - [ ] Interacciones: `df['A'] * df['B']`
  - [ ] Transformaciones: `np.log()`, `np.sqrt()`

## 5. 📊 NORMALIZACIÓN/ESTANDARIZACIÓN
- [ ] **Elegir método apropiado**
  - [ ] **StandardScaler**: `(x - μ) / σ` (media=0, std=1)
  - [ ] **MinMaxScaler**: `(x - min) / (max - min)` (rango [0,1])
  - [ ] **RobustScaler**: usa mediana e IQR (resistente a outliers)
- [ ] **Aplicar SOLO a variables numéricas**
- [ ] **Verificar distribución después**

## 6. ✂️ DIVISIÓN DE DATOS
- [ ] **Dividir ANTES de estandarizar**
  - [ ] `train_test_split(X, y, test_size=0.2, random_state=42)`
  - [ ] Validación adicional si es necesario
- [ ] **Verificar balance de clases**
  - [ ] `y_train.value_counts()`
  - [ ] Estratificar si hay desbalance: `stratify=y`

## 7. 🎯 APLICAR TRANSFORMACIONES
- [ ] **Estandarizar usando SOLO train**
  - [ ] `scaler.fit(X_train)`
  - [ ] `X_train_scaled = scaler.transform(X_train)`
  - [ ] `X_test_scaled = scaler.transform(X_test)`
- [ ] **Aplicar encoding usando SOLO train**
  - [ ] `encoder.fit(X_train)`
  - [ ] `X_train_encoded = encoder.transform(X_train)`
  - [ ] `X_test_encoded = encoder.transform(X_test)`

## 8. ✅ VALIDACIÓN FINAL
- [ ] **Verificar dimensiones**
  - [ ] `X_train.shape`, `X_test.shape`
  - [ ] `y_train.shape`, `y_test.shape`
- [ ] **Verificar tipos de datos**
  - [ ] `X_train.dtypes`
  - [ ] No hay valores faltantes: `X_train.isnull().sum()`
- [ ] **Verificar distribuciones**
  - [ ] Comparar train vs test
  - [ ] `sns.histplot()` para visualizar
- [ ] **Guardar transformadores**
  - [ ] `joblib.dump(scaler, 'scaler.pkl')`
  - [ ] `joblib.dump(encoder, 'encoder.pkl')`

## 9. 📈 ANÁLISIS ADICIONAL
- [ ] **Matriz de correlación**
  - [ ] `corr_matrix = X_train.corr()`
  - [ ] `sns.heatmap(corr_matrix)`
- [ ] **Selección de características**
  - [ ] Eliminar variables altamente correlacionadas
  - [ ] Feature importance si es necesario
- [ ] **Documentar decisiones**
  - [ ] ¿Por qué eliminaste ciertas variables?
  - [ ] ¿Qué método de imputación usaste?
  - [ ] ¿Por qué elegiste ese scaler?

## 10. 🚀 PREPARACIÓN PARA MODELADO
- [ ] **Verificar que todo esté listo**
  - [ ] Datos limpios y transformados
  - [ ] Sin valores faltantes
  - [ ] Tipos de datos correctos
  - [ ] Dimensiones consistentes
- [ ] **Backup de datos originales**
  - [ ] Guardar versión sin procesar
  - [ ] Documentar todos los pasos

---

## 💡 TIPS IMPORTANTES:

1. **Siempre divide ANTES de estandarizar**
2. **Usa SOLO datos de train para calcular parámetros**
3. **Documenta cada decisión de preprocesamiento**
4. **Guarda los transformadores para usar en producción**
5. **Verifica que no haya data leakage**
6. **Visualiza antes y después de cada transformación**

---

## 📝 CÓDIGO DE EJEMPLO:

```python
# 1. Exploración
df.head()
df.info()
df.describe()

# 2. Detección de problemas
df.isnull().sum()
df.duplicated().sum()

# 3. División ANTES de estandarizar
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Estandarización usando SOLO train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Encoding categóricas
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train_categorical)
X_test_encoded = encoder.transform(X_test_categorical)
```
