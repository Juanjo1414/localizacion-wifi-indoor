# Plan de implementación — Localización WiFi indoor (UJIIndoorLoc)

Oct 1, 2026 · @Juan Jose

## 1. Resumen y decisiones tomadas

Vamos a predecir en qué **edificio y piso** está un celular a partir de la intensidad de 520 señales WiFi, como un problema de clasificación de 13 clases, y lo vamos a mostrar en una app Gradio.

**Dataset:** UJIIndoorLoc ([UCI, id 310](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)). Trae dos archivos: `trainingData.csv` (\~19.937 filas) y `validationData.csv` (\~1.111 filas), con 529 columnas.

**Variable objetivo:** `BF`, una etiqueta nueva que combina `BUILDINGID` y `FLOOR`, por ejemplo `B0_P2`. Según el artículo original, los edificios 0 y 1 tienen 4 pisos (0 a 3) y el edificio 2 tiene 5 (0 a 4), así que esperamos 13 clases. Esto se confirma en el EDA, no se da por hecho.

**Decisiones ya tomadas:**

- Se trabaja en Colab o en local con Claude Code; el mismo código sirve para ambos cambiando una sola variable de ruta.
- Se usan **tres notebooks** más un archivo `app.py` (ver sección 2). Así cada integrante puede avanzar en paralelo y cada notebook se parece a uno de los de clase.
- `trainingData.csv` se usa para entrenar y para validación cruzada. `validationData.csv` se usa **una sola vez**, al final, como prueba. Se tomó meses después y con otros usuarios, así que mide si el modelo generaliza de verdad.
- La publicación es una **app Gradio**. Se elige Gradio y no Streamlit porque corre dentro de Colab con un enlace público (`share=True`) y en local con uv run `python app.py`, sin cambiar nada.
- Entregables: notebooks, app, informe y presentación.

**Estilo:** todo el código sigue los notebooks de la profesora (función `fit_and_eval`, `Pipeline`, `StratifiedKFold`, `GridSearchCV`, tabla de resultados con `rows`). Nada de list comprehensions ni construcciones avanzadas; ciclos `for` simples. Los markdown se escriben en voz grupal ("probamos", "notamos").

## 2. Estructura del proyecto y entorno

El proyecto vive en el repositorio de GitHub `localizacion-wifi-indoor` y el entorno local se administra con **uv** (`pyproject.toml` + `uv.lock`), igual que en el trabajo de Waze; en Colab se usa una copia de la carpeta en Drive.

```
localizacion-wifi-indoor/
├── data/                         # no se sube a GitHub (excepto ejemplo_app.csv)
│   ├── trainingData.csv          # original, no se modifica
│   ├── validationData.csv        # original, no se modifica
│   ├── train_procesado.csv       # sale del notebook 01
│   ├── test_procesado.csv        # sale del notebook 01
│   └── ejemplo_app.csv           # sale del notebook 03, para probar la app
├── models/                       # no se sube a GitHub
│   ├── modelo_final.joblib       # sale del notebook 03
│   ├── label_encoder.joblib      # sale del notebook 01
│   └── rasgos_usados.joblib      # sale del notebook 02
├── figuras/                      # gráficas para informe y presentación
├── referencia/                   # notebooks de clase, para imitar el estilo
├── 01_EDA_Preprocesamiento.ipynb
├── 02_PCA_Seleccion_Rasgos.ipynb
├── 03_Modelos_Evaluacion.ipynb
├── app.py
├── PLAN.md                       # este plan exportado a Markdown
├── CLAUDE.md                     # instrucciones para Claude Code (sección 8)
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
└── .gitignore
```

Los CSV y los `.joblib` están en `.gitignore` por tamaño: cada integrante descarga el dataset de UCI y corre los notebooks. Si el grupo decide subir el modelo final, se quita `models/*.joblib` del `.gitignore` (límite de GitHub: 100 MB por archivo).

### Celda de configuración (primera celda de cada notebook)

Es la única parte que cambia entre Colab y local. En Drive la carpeta se llama `Proyecto_UJIIndoorLoc`.

```python
EN_COLAB = True   # poner False cuando se corra en local

if EN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_BASE = '/content/drive/MyDrive/Proyecto_UJIIndoorLoc/'
else:
    RUTA_BASE = './'

RUTA_DATOS = RUTA_BASE + 'data/'
RUTA_MODELOS = RUTA_BASE + 'models/'
RUTA_FIGURAS = RUTA_BASE + 'figuras/'
```

### Librerías

Se agregan con uv y quedan registradas en `pyproject.toml` y `uv.lock`: pandas, numpy, matplotlib, seaborn, scikit-learn, phik, shap, joblib, gradio, ipykernel y nbconvert (este último para que Claude Code pueda ejecutar los notebooks desde la terminal).

En Colab ya vienen casi todas. Solo hay que instalar al inicio: `!pip install phik shap gradio -q`.

### Creación del proyecto (solo lo hace una persona, una vez)

En PowerShell:

```powershell
mkdir localizacion-wifi-indoor
cd localizacion-wifi-indoor

uv init --bare --python 3.11
uv python pin 3.11
uv add pandas numpy matplotlib seaborn scikit-learn phik shap joblib gradio ipykernel nbconvert

mkdir data, models, figuras, referencia
New-Item models\.gitkeep, figuras\.gitkeep, data\.gitkeep -ItemType File
```

Luego se copian `README.md`, `.gitignore`, `PLAN.md`, `CLAUDE.md` y los notebooks de clase en `referencia/`, y se hace el primer push.

### Instalación para el resto del grupo (Windows 11, PowerShell, VS Code)

1. Instalar uv: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`, cerrar y abrir la terminal, y verificar con `uv --version`.
2. Clonar el repositorio y entrar a la carpeta.
3. Correr `uv sync`. Crea `.venv` con Python 3.11 e instala exactamente las versiones de `uv.lock`.
4. Descargar el dataset de UCI y copiar los dos CSV en `data/`.
5. En VS Code, abrir el notebook y elegir el kernel `.venv\Scripts\python.exe`.
6. Poner `EN_COLAB = False` en la celda de configuración.

Para agregar una librería nueva se usa `uv add nombre-del-paquete` desde la terminal, nunca `pip install` dentro del notebook. Después se hace commit de `pyproject.toml` y `uv.lock`.

### Celda de importaciones (segunda celda de cada notebook)

Cada notebook importa solo lo que usa, igual que en clase. Base común:

```python
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
```

Las importaciones de `sklearn` se agregan en cada notebook (se listan en las secciones 4, 5 y 6).

## 3. Cronograma de 12 días

El modelo debe quedar guardado el día 7 para tener cinco días de margen para la app, el informe y la presentación. Los días 4 a 6 se pueden trabajar en paralelo si el grupo se reparte los notebooks 02 y 03.

| Día | Qué se hace | Se considera terminado cuando |
| --- | --- | --- |
| 1 | Montar carpeta y entorno (Colab y local). Descargar el dataset. Crear `CLAUDE.md`. | Los dos CSV cargan sin error en Colab y en local. |
| 2 | Notebook 01: carga, revisión general, crear `BF`, distribución de clases, mapa de puntos. | Se confirma cuántas clases hay en train y en test. |
| 3 | Notebook 01: tratamiento del valor 100, duplicados, filas vacías, WAPs que nunca se detectan, LabelEncoder. Guardar procesados. | Existen `train_procesado.csv`, `test_procesado.csv` y `label_encoder.joblib`. |
| 4 | Notebook 02: escalado (MinMax vs Standard), PCA con varianza acumulada y visualización 2D. | Se sabe cuántas componentes dan el 95%. |
| 5 | Notebook 02: selección de rasgos (VarianceThreshold, SelectKBest, SelectFromModel, SHAP) y tabla comparativa. | Hay una lista final de rasgos con su F1 macro en CV. |
| 6 | Notebook 03: comparación de 7 clasificadores con validación cruzada. | Tabla de resultados con media y desviación por modelo. |
| 7 | Notebook 03: GridSearch de los 2 o 3 mejores, ensambles, elegir modelo, evaluación única en test, guardar. | Existe `modelo_final.joblib` y la matriz de confusión de test. |
| 8 | App Gradio en local y en Colab. | La app predice correctamente una fila de test. |
| 9 | Informe: borrador completo con figuras. | Todas las secciones tienen contenido. |
| 10 | Presentación y ensayo de la demo de la app. | Diapositivas listas y demo probada. |
| 11 | Revisión cruzada: cada integrante corre los notebooks de cero ("Reiniciar y ejecutar todo"). | Todo corre sin errores en un entorno limpio. |
| 12 | Margen para imprevistos, correcciones y entrega. | Entregado. |

## 4. Notebook 01 — Carga, EDA y preprocesamiento

Este notebook entrega dos CSV limpios con la etiqueta `BF` y el `LabelEncoder` guardado; su modelo de referencia es `IA_ML_Example_Dif_Classifiers_CV.ipynb` (análisis de datos y label encoding) y `ML_Normalización.ipynb` (revisión con `describe` y box plots).

Importaciones extra: `from sklearn import preprocessing` y `import phik`.

### 4.1 Encabezado

Celda markdown con el título, el problema en dos líneas y la fuente del dataset. Igual que los notebooks de clase: corto y directo.

### 4.2 Cargar los datos

```python
train = pd.read_csv(RUTA_DATOS + 'trainingData.csv')
test = pd.read_csv(RUTA_DATOS + 'validationData.csv')

print(train.shape, test.shape)
train.head()
```

Verificar: 529 columnas en ambos; las primeras 520 se llaman `WAP001` a `WAP520`; las últimas 9 son `LONGITUDE`, `LATITUDE`, `FLOOR`, `BUILDINGID`, `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP`. Revisar nulos con `train.isnull().sum().sum()`.

### 4.3 Separar columnas de señal y metadatos

```python
cols_wap = []
for col in train.columns:
    if col.startswith('WAP'):
        cols_wap.append(col)

cols_meta = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID', 'SPACEID',
             'RELATIVEPOSITION', 'USERID', 'PHONEID', 'TIMESTAMP']
print(len(cols_wap))
train[cols_meta].describe()
```

### 4.4 Crear la variable objetivo BF

```python
train['BF'] = 'B' + train['BUILDINGID'].astype(str) + '_P' + train['FLOOR'].astype(str)
test['BF'] = 'B' + test['BUILDINGID'].astype(str) + '_P' + test['FLOOR'].astype(str)

print(train['BF'].value_counts().sort_index())
print(test['BF'].value_counts().sort_index())
```

- Gráfica de barras de `BF` en train y en test (como el pie chart de clases de clase, pero en barras porque son 13).
- Confirmar que todas las clases de test existen en train. Si alguna falta, **parar y avisar al grupo** antes de seguir.
- Comentar el desbalance: qué clase tiene más y cuál menos ejemplos.

### 4.5 Mapa de los puntos de medición

Scatter de `LONGITUDE` contra `LATITUDE` coloreado por `BUILDINGID`, y un segundo scatter coloreado por `FLOOR` para un solo edificio. Es la gráfica que mejor explica el problema en la presentación. Guardarla en `figuras/` con `plt.savefig`.

### 4.6 Análisis del valor 100 (no detectado)

1. Porcentaje de celdas con 100 en la matriz WAP: `(train[cols_wap] == 100).sum().sum() / train[cols_wap].size`. Se espera un porcentaje muy alto (la matriz es casi vacía).
2. Histograma de los valores detectados (todos los distintos de 100), que deben estar entre -104 y 0 dBm.
3. Cantidad de WAPs detectados por fila: `(train[cols_wap] != 100).sum(axis=1)` e histograma. Contar filas con 0 detecciones.
4. WAPs que nunca se detectan en train y WAPs que nunca se detectan en test, con un ciclo `for` sobre `cols_wap`. Anotar cuántos son; se usan en el notebook 02.

### 4.7 Duplicados

`train.duplicated().sum()` y luego `train = train.drop_duplicates()`. Reportar cuántas filas se quitaron.

### 4.8 Correlación

phik sobre 520 columnas es impracticable. Se hace en dos partes:

- phik solo sobre los metadatos y `BF`, usando `plot_correlation_matrix` como en clase. Sirve para mostrar cómo se relacionan `USERID`, `PHONEID` y la ubicación.
- Mapa de calor de correlación normal (`.corr()`) de los 20 WAPs con más detecciones.

### 4.9 Transformar el valor 100

El 100 no es una señal: significa "no detectado". Si se deja, el modelo lo trata como una señal más fuerte que cualquier señal real (0 dBm es la máxima). Se reemplaza por -105, un valor apenas por debajo del mínimo real.

```python
train[cols_wap] = train[cols_wap].replace(100, -105)
test[cols_wap] = test[cols_wap].replace(100, -105)
```

Después, quitar las filas de train sin ninguna detección (todas en -105). En test no se quita nada: es la prueba final y debe quedar como viene.

### 4.10 Quitar columnas que no deben ser rasgos

`USERID`, `PHONEID`, `TIMESTAMP`, `SPACEID` y `RELATIVEPOSITION` se eliminan. `LONGITUDE`, `LATITUDE`, `BUILDINGID` y `FLOOR` **se guardan en el CSV** (los usa la app para dibujar el mapa), pero **nunca entran a X**: delatan la respuesta. Esto se explica en un markdown.

### 4.11 Label encoding

```python
le = preprocessing.LabelEncoder()
le.fit(train['BF'])
print(le.classes_)

train['BF_int'] = le.transform(train['BF'])
test['BF_int'] = le.transform(test['BF'])

joblib.dump(le, RUTA_MODELOS + 'label_encoder.joblib')
```

### 4.12 Guardar y concluir

```python
train.to_csv(RUTA_DATOS + 'train_procesado.csv', index=False)
test.to_csv(RUTA_DATOS + 'test_procesado.csv', index=False)
```

Markdown final con las conclusiones del EDA en voz grupal: número de clases, desbalance, porcentaje de celdas vacías, WAPs inútiles y filas eliminadas.

## 5. Notebook 02 — Escalado, PCA y selección de rasgos

Este notebook decide **con qué rasgos y con qué escalado** se entrenan los modelos, y guarda esa lista en `rasgos_usados.joblib`. Sus modelos de referencia son `ML_Normalización.ipynb`, `PCA_Wine.ipynb` y `ML_FeatureSelection_comparacion_metodos_SHAP.ipynb`.

Aquí **no se toca `test_procesado.csv`**. Todo se mide con validación cruzada sobre train.

Importaciones extra:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel
import sklearn.feature_selection as fs
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import shap
```

### 5.1 Cargar datos procesados y armar X, y

```python
train = pd.read_csv(RUTA_DATOS + 'train_procesado.csv')

cols_wap = []
for col in train.columns:
    if col.startswith('WAP'):
        cols_wap.append(col)

X_train = train[cols_wap]
y_train = train['BF_int']
```

### 5.2 Validación cruzada y función de evaluación

Se usa `StratifiedKFold` con **5 particiones** (en clase se usaron 10, pero con 520 rasgos y \~19.000 filas cada corrida tardaría el doble; se explica en un markdown). Se crea una función al estilo de `fit_and_eval` de la profesora, pero con CV:

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def eval_cv(pipeline, nombre, X, y):
    inicio = time.time()
    res = cross_validate(pipeline, X, y, cv=cv,
                         scoring=['accuracy', 'f1_macro', 'recall_macro', 'precision_macro'],
                         n_jobs=-1)
    duracion = time.time() - inicio
    return {"Modelo": nombre,
            "Accuracy": res['test_accuracy'].mean(),
            "F1_macro": res['test_f1_macro'].mean(),
            "F1_std": res['test_f1_macro'].std(),
            "Recall_macro": res['test_recall_macro'].mean(),
            "Precision_macro": res['test_precision_macro'].mean(),
            "Tiempo_s": duracion}
```

Cada resultado se agrega a una lista `rows` y al final se muestra con `pd.DataFrame(rows)`, igual que en `PCA_Wine`. La métrica principal es **F1 macro**, porque las clases están desbalanceadas.

### 5.3 Escalado: sin escalar vs MinMax vs Standard

Igual que en `ML_Normalización`, pero con validación cruzada. Se prueban KNN y SVC, que son los más sensibles al escalado. Son 6 combinaciones; cada una en un `Pipeline` (`scaler` + `clf`), y para "sin escalar" el pipeline solo tiene `clf`. Antes, box plots de 10 WAPs antes y después de escalar.

Resultado esperado: decidir qué escalador se usa en el resto del trabajo.

### 5.4 PCA

1. Escalar con el escalador elegido y ajustar `PCA()` completo. Graficar la varianza acumulada con líneas horizontales en 0.90, 0.95 y 0.99, como en `PCA_Wine`.
2. Reportar cuántas componentes se necesitan para cada umbral. Se espera una reducción grande respecto a 520; es uno de los resultados más vistosos del trabajo.
3. Visualización 2D con las dos primeras componentes: una gráfica coloreada por edificio (deberían separarse bien) y otra por `BF` (deberían mezclarse los pisos). Guardar ambas en `figuras/`.
4. Comparar con `eval_cv`: todos los rasgos, PCA al 95% y PCA con 2 componentes, con `LogisticRegression(max_iter=2000)` y KNN. Incluir el tiempo para mostrar si PCA acelera el entrenamiento.

### 5.5 Selección de rasgos

**a) VarianceThreshold.** `VarianceThreshold(threshold=0)` elimina los WAPs que nunca se detectan en train (constantes en -105). Reportar cuántos quedan. Esta lista base (`rasgos_var`) es la entrada de los siguientes métodos.

**b) SelectKBest.** Como el ciclo `for k in range(1, 9)` de clase, pero con valores salteados porque hay cientos de rasgos:

```python
valores_k = [10, 25, 50, 100, 150, 200, 300]
f1_list = []
for k in valores_k:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kbest", SelectKBest(score_func=fs.f_classif, k=k)),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])
    res = eval_cv(pipe, "KBest_" + str(k), X_train[rasgos_var], y_train)
    f1_list.append(res["F1_macro"])
```

Graficar F1 contra k y elegir el mejor k.

**c) SelectFromModel.** Con `RandomForestClassifier(n_estimators=100, random_state=42)` y `threshold='median'`. Reportar cuántos rasgos deja y su F1 en CV.

**d) RFE y Relief (opcional).** Con 500 rasgos son muy lentos. Si sobra tiempo, `RFE` con `step=50`; si no, se menciona en el markdown por qué no se usó.

**e) SHAP.** Como en el notebook de clase, pero con un detalle: el `TreeExplainer` de SHAP puede no aceptar `GradientBoostingClassifier` con más de dos clases, así que se usa un `RandomForestClassifier` pequeño. Si en la prueba GradientBoosting sí funciona, se usa ese para quedar igual al de clase.

```python
X_sub = X_train[rasgos_var].sample(n=3000, random_state=42)
y_sub = y_train.loc[X_sub.index]

rf_shap = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
rf_shap.fit(X_sub, y_sub)

explainer = shap.TreeExplainer(rf_shap)
shap_values = explainer(X_sub.iloc[0:500])

values = shap_values.values            # forma: (filas, rasgos, clases)
shap_importance = np.abs(values).mean(axis=0).mean(axis=1)
feature_importance_shap = pd.Series(shap_importance, index=rasgos_var).sort_values(ascending=False)
```

- Gráfica de barras de los 20 WAPs más importantes.
- Seleccionar los rasgos que acumulan el 90% de la importancia, con un ciclo `for` que vaya sumando, igual que en clase.
- Antes de correr esto, imprimir `values.shape` para confirmar que tiene 3 dimensiones. Si tiene otra forma (cambia entre versiones de shap), ajustar el promedio.

**f) Tabla comparativa.** Una fila por método: nombre, número de rasgos, F1 macro, desviación, accuracy y tiempo. Todos con el mismo clasificador (KNN) para que sea justo.

### 5.6 Decisión y guardado

Elegir el conjunto de rasgos con mejor equilibrio entre F1 y número de rasgos. Si PCA gana, la decisión es "usar PCA dentro del pipeline" y se guarda la lista de `rasgos_var`.

```python
joblib.dump(rasgos_elegidos, RUTA_MODELOS + 'rasgos_usados.joblib')
```

Markdown de conclusiones: escalador elegido, componentes del 95%, método de selección ganador y por qué.

## 6. Notebook 03 — Modelos, evaluación final y guardado

Este notebook compara clasificadores, ajusta los mejores, evalúa **una sola vez** en test y guarda `modelo_final.joblib`. Su modelo de referencia es `IA_ML_Example_Dif_Classifiers_CV.ipynb`.

Importaciones extra:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier)
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             classification_report, ConfusionMatrixDisplay)
```

### 6.1 Cargar datos, rasgos y encoder

```python
train = pd.read_csv(RUTA_DATOS + 'train_procesado.csv')
test = pd.read_csv(RUTA_DATOS + 'test_procesado.csv')
rasgos = joblib.load(RUTA_MODELOS + 'rasgos_usados.joblib')
le = joblib.load(RUTA_MODELOS + 'label_encoder.joblib')

X_train = train[rasgos]
y_train = train['BF_int']
X_test = test[rasgos]
y_test = test['BF_int']
```

Se copia la misma `cv` y la misma función `eval_cv` del notebook 02, para que los números sean comparables.

### 6.2 Comparar clasificadores con validación cruzada

Siete modelos con parámetros por defecto, cada uno en un `Pipeline` con el escalador elegido (y PCA si ganó en el notebook 02):

```python
modelos = [
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('NaiveBayes', GaussianNB()),
    ('Arbol', DecisionTreeClassifier(random_state=42)),
    ('RegLogistica', LogisticRegression(max_iter=2000)),
    ('SVM', SVC()),
    ('MLP', MLPClassifier(max_iter=500, random_state=42)),
    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
]

rows = []
for nombre, modelo in modelos:
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", modelo)])
    res = eval_cv(pipe, nombre, X_train, y_train)
    rows.append(res)
    print(nombre, round(res["F1_macro"], 4), round(res["Tiempo_s"], 1), "s")

tabla_modelos = pd.DataFrame(rows).sort_values("F1_macro", ascending=False)
tabla_modelos
```

- Box plot del F1 macro por partición para cada modelo, como en clase. Para eso, `eval_cv` debe devolver también el arreglo `res['test_f1_macro']`.
- SVM y MLP son los más lentos (varios minutos cada uno). Correrlos una vez y no repetir a la ligera.

### 6.3 GridSearchCV de los 2 o 3 mejores

Grillas pequeñas a propósito, para que cada búsqueda no pase de unos 15 minutos. Los nombres llevan el prefijo `clf__` porque el modelo está dentro de un pipeline.

| Modelo | Parámetros a probar |
| --- | --- |
| KNN | `n_neighbors`: 1, 3, 5, 7, 9 · `weights`: uniform, distance · `metric`: euclidean, manhattan |
| SVM | `C`: 1, 10, 100 · `gamma`: scale, 0.01, 0.001 |
| RandomForest | `n_estimators`: 100, 300 · `max_depth`: None, 20 · `min_samples_split`: 2, 5 |
| MLP | `hidden_layer_sizes`: (100,), (200,), (200, 100) · `alpha`: 0.0001, 0.001 |

```python
param_grid = {'clf__n_neighbors': [1, 3, 5, 7, 9],
              'clf__weights': ['uniform', 'distance'],
              'clf__metric': ['euclidean', 'manhattan']}

knn_pipe = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
knn_cv = GridSearchCV(knn_pipe, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
knn_cv.fit(X_train, y_train)
print(knn_cv.best_params_, knn_cv.best_score_)
```

Solo se ajustan los que quedaron arriba en 6.2; la tabla de arriba es una guía, no hay que correrlas todas.

### 6.4 Ensambles

Como en la sección de multiclasificadores de clase, evaluados con `eval_cv`:

- `BaggingClassifier` con el mejor KNN como estimador base.
- `AdaBoostClassifier` con `DecisionTreeClassifier(max_depth=5)`.
- `GradientBoostingClassifier(n_estimators=50)`. Con 13 clases es lento; si pasa de 20 minutos, se reporta y se deja por fuera.
- `StackingClassifier` con los 3 mejores modelos ajustados y `LogisticRegression` como estimador final, `cv=3` para que no tarde demasiado.

### 6.5 Elegir el modelo final

Tabla única con todos los modelos (base, ajustados y ensambles). Se elige por **F1 macro en validación cruzada**, nunca por el resultado en test. En un markdown se justifica la elección, considerando también el tiempo de predicción porque la app lo va a usar.

### 6.6 Evaluación única en test

```python
modelo_final = knn_cv.best_estimator_   # o el que haya ganado
modelo_final.fit(X_train, y_train)
y_pred = modelo_final.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 macro:", f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred, target_names=le.classes_))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_,
                                        cmap=plt.cm.Blues, xticks_rotation=45)
```

Análisis adicional que le da valor al trabajo, separando edificio y piso de la etiqueta combinada:

```python
real = pd.Series(le.inverse_transform(y_test))
pred = pd.Series(le.inverse_transform(y_pred))

acc_edificio = (real.str[1] == pred.str[1]).mean()   # 'B0_P2' -> '0'
acc_piso = (real.str[4] == pred.str[4]).mean()       # 'B0_P2' -> '2'
print(acc_edificio, acc_piso)
```

Se espera que el edificio salga casi perfecto y que los errores estén en el piso. También se compara el F1 de CV con el de test: si cae, se explica por qué (otros usuarios, otros celulares, meses después).

### 6.7 Guardar y volver a cargar

```python
joblib.dump(modelo_final, RUTA_MODELOS + 'modelo_final.joblib')

modelo_cargado = joblib.load(RUTA_MODELOS + 'modelo_final.joblib')
print(le.inverse_transform(modelo_cargado.predict(X_test.iloc[0:5])))
print(le.inverse_transform(y_test.iloc[0:5]))
```

Esto replica las secciones "Guardar un modelo" y "Cargar el modelo entrenado" de clase. El pipeline guardado ya incluye el escalador, así que la app no tiene que escalar nada aparte.

### 6.8 Conclusiones

Markdown en voz grupal: mejor modelo, F1 en CV y en test, accuracy de edificio y de piso, qué pisos se confunden más y qué se haría con más tiempo.

## 7. App Gradio (publicación del modelo)

La app carga el modelo guardado y tiene dos pestañas: probar con una medición real del set de prueba (con mapa) y subir un CSV propio. No se pide escribir 520 valores a mano porque nadie podría usarla así.

### 7.1 Qué muestra

**Pestaña 1 — "Probar con una medición real".** Un slider para escoger una fila de test y un botón "Elegir una al azar". Muestra la predicción, la ubicación real, si acertó, las 3 clases más probables y un mapa con todos los puntos de entrenamiento en gris y la ubicación real como una estrella roja.

**Pestaña 2 — "Subir un CSV".** Se sube un archivo con el mismo formato del dataset original (columnas WAP, con 100 para no detectado). La app hace la misma transformación del 100 y devuelve una tabla con la predicción por fila. En el notebook 03 se guarda `data/ejemplo_app.csv` con 5 filas de `validationData.csv` original para probarla.

### 7.2 Código de `app.py`

```python
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

EN_COLAB = False
if EN_COLAB:
    RUTA_BASE = '/content/drive/MyDrive/Proyecto_UJIIndoorLoc/'
else:
    RUTA_BASE = './'

modelo = joblib.load(RUTA_BASE + 'models/modelo_final.joblib')
le = joblib.load(RUTA_BASE + 'models/label_encoder.joblib')
rasgos = joblib.load(RUTA_BASE + 'models/rasgos_usados.joblib')
test = pd.read_csv(RUTA_BASE + 'data/test_procesado.csv')
train = pd.read_csv(RUTA_BASE + 'data/train_procesado.csv')


def texto_etiqueta(etiqueta):
    # 'B0_P2' -> 'Edificio 0, piso 2'
    return 'Edificio ' + etiqueta[1] + ', piso ' + etiqueta[4]


def dibujar_mapa(fila):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(train['LONGITUDE'], train['LATITUDE'], s=2, c='lightgray', label='Puntos de entrenamiento')
    ax.scatter(fila['LONGITUDE'], fila['LATITUDE'], s=200, c='red', marker='*', label='Ubicación real')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.legend()
    return fig


def predecir_fila(indice):
    indice = int(indice)
    fila = test.iloc[indice]
    X = test[rasgos].iloc[[indice]]

    pred = modelo.predict(X)[0]
    etiqueta_pred = le.inverse_transform([pred])[0]
    etiqueta_real = fila['BF']

    if etiqueta_pred == etiqueta_real:
        resultado = 'Acertó'
    else:
        resultado = 'Falló'

    probs = modelo.predict_proba(X)[0]
    dic_probs = {}
    for i in range(len(probs)):
        dic_probs[texto_etiqueta(le.classes_[i])] = float(probs[i])

    return texto_etiqueta(etiqueta_pred), texto_etiqueta(etiqueta_real), resultado, dic_probs, dibujar_mapa(fila)


def elegir_azar():
    indice = np.random.randint(0, len(test))
    pred, real, resultado, probs, mapa = predecir_fila(indice)
    return indice, pred, real, resultado, probs, mapa


def predecir_csv(archivo):
    datos = pd.read_csv(archivo)
    faltan = []
    for col in rasgos:
        if col not in datos.columns:
            faltan.append(col)
    if len(faltan) > 0:
        raise gr.Error('Al archivo le faltan ' + str(len(faltan)) + ' columnas WAP')

    X = datos[rasgos].replace(100, -105)
    preds = modelo.predict(X)
    etiquetas = le.inverse_transform(preds)

    textos = []
    for e in etiquetas:
        textos.append(texto_etiqueta(e))

    resultado = pd.DataFrame()
    resultado['Fila'] = range(1, len(textos) + 1)
    resultado['Predicción'] = textos
    return resultado


with gr.Blocks(title='¿En qué edificio y piso estoy?') as app:
    gr.Markdown('# ¿En qué edificio y piso estoy?\nLocalización en interiores con señales WiFi (UJIIndoorLoc)')

    with gr.Tab('Probar con una medición real'):
        indice = gr.Slider(0, len(test) - 1, step=1, value=0, label='Medición del set de prueba')
        with gr.Row():
            boton = gr.Button('Predecir', variant='primary')
            boton_azar = gr.Button('Elegir una al azar')
        with gr.Row():
            salida_pred = gr.Textbox(label='Predicción')
            salida_real = gr.Textbox(label='Ubicación real')
            salida_res = gr.Textbox(label='Resultado')
        salida_probs = gr.Label(num_top_classes=3, label='Clases más probables')
        salida_mapa = gr.Plot(label='Mapa')

        salidas = [salida_pred, salida_real, salida_res, salida_probs, salida_mapa]
        boton.click(predecir_fila, inputs=indice, outputs=salidas)
        boton_azar.click(elegir_azar, outputs=[indice] + salidas)

    with gr.Tab('Subir un CSV'):
        archivo = gr.File(type='filepath', file_types=['.csv'], label='CSV con columnas WAP')
        boton_csv = gr.Button('Predecir', variant='primary')
        salida_tabla = gr.Dataframe(label='Resultados')
        boton_csv.click(predecir_csv, inputs=archivo, outputs=salida_tabla)

app.launch(share=EN_COLAB)
```

**Importante:** `predict_proba` existe en KNN, Random Forest, regresión logística y MLP. Si el modelo ganador es `SVC`, en el notebook 03 hay que entrenarlo con `SVC(probability=True)` antes de guardarlo; si no, la app falla en esa línea.

### 7.3 Cómo correrla

**Local:** desde la raíz del repositorio, en PowerShell, uv run `python app.py`. Se abre en `http://127.0.0.1:7860`.

**Colab:** poner `EN_COLAB = True` en `app.py`, subirlo a la carpeta del proyecto en Drive y en una celda correr `%cd /content/drive/MyDrive/Proyecto_UJIIndoorLoc/` y luego `%run app.py`. Gradio imprime un enlace público `*.gradio.live` que dura 72 horas y sirve para la presentación.

**Opcional, si sobra tiempo:** subir la app a Hugging Face Spaces para tener un enlace permanente. Requiere copiar `app.py`, `requirements.txt`, la carpeta `models/` y los CSV procesados.

Hugging Face instala con pip, así que antes de subir se genera el `requirements.txt` desde uv con `uv export --format requirements-txt --no-hashes -o requirements.txt`.

### 7.4 Pruebas de la app

- [ ] La fila 0 de test da la misma predicción en la app y en el notebook 03.
- [ ] El botón "Elegir una al azar" mueve el slider y actualiza todo.
- [ ] Subir `ejemplo_app.csv` devuelve 5 predicciones.
- [ ] Subir un CSV sin columnas WAP muestra el mensaje de error en vez de romperse.
- [ ] Funciona en local y en Colab con el enlace público.

## 8. Trabajar con Claude Code en local

Claude Code lee `CLAUDE.md` al iniciar en la carpeta, así que ahí van las reglas del proyecto. Este plan se exporta a Markdown como `PLAN.md` y se pone en la misma carpeta, junto con los notebooks de clase en una subcarpeta `referencia/` para que imite su estilo.

### 8.1 Contenido de `CLAUDE.md`

```markdown
# Proyecto: Localización WiFi indoor (UJIIndoorLoc) - Fundamentos de IA, EIA

## Contexto
- Clasificación de edificio + piso combinados (variable BF, ~13 clases) a partir de 520 señales WiFi.
- El plan completo está en PLAN.md. Seguirlo en orden y no saltarse pasos.
- Los notebooks de clase están en referencia/. Imitar su estructura, nombres y forma de escribir.

## Entorno
- Windows 11, PowerShell, VS Code. El entorno se administra con uv (pyproject.toml + uv.lock), Python 3.11.
- Para correr Python o Jupyter usar siempre `uv run ...` (por ejemplo `uv run python app.py`).
- NO usar pip. Si hace falta una librería nueva, preguntar primero y luego agregarla con `uv add nombre-del-paquete`.
- Los CSV de data/ y los .joblib de models/ están en .gitignore. No cambiar eso sin preguntar.

## Reglas de código
- Código de nivel básico-intermedio, como en los notebooks de referencia.
- NO usar list comprehensions, dict comprehensions, generadores ni lambdas. Usar ciclos for simples.
- Usar Pipeline, StratifiedKFold, GridSearchCV y joblib como en clase.
- La primera celda de cada notebook es la de configuración con EN_COLAB y RUTA_BASE.
- Nunca usar LONGITUDE, LATITUDE, BUILDINGID, FLOOR, SPACEID, RELATIVEPOSITION, USERID, PHONEID ni TIMESTAMP como rasgos.
- validationData.csv / test_procesado.csv solo se usa en la sección de evaluación final del notebook 03.
- random_state=42 en todo.

## Reglas de escritura
- Todo en español.
- Celdas markdown cortas, en voz grupal ("probamos", "notamos", "decidimos"), naturales, como las escribiría un estudiante.
- Nada de tablas en markdown: explicar los resultados en prosa.

## Forma de trabajar
- Si algo no está claro o un resultado no cuadra con lo esperado en PLAN.md, preguntar antes de seguir. No suponer.
- Después de editar un notebook, ejecutarlo completo para verificar que corre:
  uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb --ExecutePreprocessor.timeout=-1
- No hacer commits ni push sin que se lo pidamos.
```

### 8.2 Flujo recomendado

1. Abrir PowerShell en la carpeta del repositorio, correr `uv sync` (por si alguien agregó librerías) y luego `claude`.
2. Pedir un notebook a la vez, nombrando la sección del plan. Por ejemplo: "Crea 01\_EDA\_Preprocesamiento.ipynb siguiendo la sección 4 de PLAN.md".
3. Pedirle que lo ejecute de punta a punta con `uv run jupyter nbconvert --to notebook --execute --inplace 01_EDA_Preprocesamiento.ipynb --ExecutePreprocessor.timeout=-1` y que reporte los números clave (clases, filas eliminadas, WAPs inútiles).
4. Revisar el notebook en VS Code antes de pasar al siguiente. Los resultados de un notebook alimentan al siguiente, así que un error en el 01 se arrastra.
5. Para el 02 y el 03, avisarle que algunas celdas tardan varios minutos (SVM, MLP, GridSearch).
6. Cuando el notebook esté aprobado, hacer commit en la rama de quien lo trabaja (una persona por notebook) y abrir un Pull Request hacia `main`.
7. Al terminar todo, correr los notebooks una vez en Colab con `EN_COLAB = True` para confirmar que funcionan en ambos lados.

## 9. Informe

El informe cuenta el proceso y las decisiones, no repite el código; cada sección sale de un notebook. Si la profesora dio un formato o una extensión, ese formato manda sobre esta propuesta.

1. **Introducción.** El problema del GPS en interiores, por qué importa (centros comerciales, hospitales, universidades) y el objetivo: predecir edificio y piso.
2. **Dataset.** Origen (Universitat Jaume I, competencia IPIN), tamaño, qué es una huella WiFi, el valor 100 y la separación entre entrenamiento y validación tomada meses después. Cita del artículo de Torres-Sospedra et al. (2014).
3. **Análisis exploratorio.** Distribución de las 13 clases, mapa de puntos, porcentaje de señales vacías y WAPs inútiles. Sale del notebook 01.
4. **Preprocesamiento.** Por qué el 100 se cambió por -105, duplicados, filas vacías, columnas eliminadas y por qué las coordenadas no se usan como rasgos.
5. **Reducción de dimensionalidad y selección de rasgos.** Comparación de escaladores, varianza acumulada de PCA, proyección 2D, comparación de métodos de selección y SHAP. Sale del notebook 02.
6. **Modelos.** Comparación de los 7 clasificadores con CV, ajuste de hiperparámetros, ensambles y elección del modelo final.
7. **Resultados en el set de prueba.** Métricas, matriz de confusión, accuracy de edificio vs de piso y diferencia entre CV y test. Sale del notebook 03.
8. **App.** Captura de las dos pestañas y cómo se usa.
9. **Conclusiones y trabajo futuro.** Qué aprendimos, limitaciones (otros celulares, cambios en los routers con el tiempo) y qué se podría hacer después (predecir coordenadas con regresión, recolectar datos en la EIA).
10. **Referencias.** UCI, artículo original, documentación de scikit-learn, SHAP y Gradio.

Las figuras salen de la carpeta `figuras/`, que se va llenando con `plt.savefig` en cada notebook. Así no hay que volver a correr nada para armar el informe.

## 10. Presentación

Se proponen 11 diapositivas para unos 12 a 15 minutos, cerrando con la demo en vivo de la app; si la profesora fijó otra duración, se recortan las de métodos.

1. **Título:** "¿En qué edificio y piso estoy?" e integrantes.
2. **El problema:** el GPS no funciona dentro de edificios. Una pregunta al público: "¿cómo sabe Google Maps en qué piso de un centro comercial están?"
3. **El dataset:** qué es una huella WiFi y el mapa de puntos de los tres edificios.
4. **El reto:** 520 rasgos casi vacíos, 13 clases desbalanceadas, prueba con otros usuarios meses después.
5. **Preprocesamiento:** el truco del 100 y por qué importa.
6. **PCA:** varianza acumulada y la proyección 2D donde se ven los edificios separados.
7. **Selección de rasgos y SHAP:** cuántos WAPs bastan y cuáles pesan más.
8. **Comparación de modelos:** box plot del F1 macro por modelo.
9. **Resultado final:** matriz de confusión y accuracy de edificio vs piso.
10. **Demo en vivo** de la app.
11. **Conclusiones y trabajo futuro.**

Para la demo, tener abierto el enlace de Gradio antes de empezar y un video corto de respaldo por si falla internet.

## 11. Riesgos y checklist final

El riesgo más probable no es que el modelo salga mal, sino que la validación cruzada salga casi perfecta y el test bastante más bajo; eso hay que explicarlo, no esconderlo.

### 11.1 Trampas conocidas

- **CV demasiado optimista.** En train hay muchas mediciones repetidas en los mismos puntos y por los mismos usuarios, así que las particiones de CV se parecen mucho entre sí. En test los usuarios son otros. Si sobra tiempo, un análisis extra con `GroupKFold` agrupando por `USERID` muestra una estimación más realista y es un punto fuerte del informe.
- **Fuga de información.** Las coordenadas y los IDs nunca entran a X, y el escalador y PCA siempre van dentro del `Pipeline` para que se ajusten solo con los datos de entrenamiento de cada partición.
- **Usar test más de una vez.** Si se prueba en test, se ve el resultado y se cambia el modelo, el resultado ya no vale. Test se usa una vez, en la sección 6.6.
- **Tiempos largos.** Si un GridSearch pasa de 20 minutos, se corre sobre una muestra estratificada del 50% de train y se aclara en el markdown.
- **Colab se desconecta.** Guardar en Drive lo que tarda en calcularse (tablas de resultados con `to_csv`, modelos ajustados con `joblib`) para no repetirlo.
- **Versiones de librerías.** La forma de los valores SHAP y algunos parámetros de Gradio cambian entre versiones. Si algo falla, imprimir `shap.__version__` y `gr.__version__` antes de cambiar código.
- **SVM sin probabilidades.** Si gana SVM, entrenarlo con `probability=True` antes de guardarlo, o la app falla.

### 11.2 Checklist de entrega

- [ ] Los tres notebooks corren de cero con "Reiniciar y ejecutar todo", en Colab y en local.
- [ ] Ningún notebook usa list comprehensions ni construcciones avanzadas.
- [ ] Todos los markdown están en español y en voz grupal.
- [ ] `test_procesado.csv` solo se usa en la sección 6.6.
- [ ] Existen `modelo_final.joblib`, `label_encoder.joblib` y `rasgos_usados.joblib`.
- [ ] La app pasa las 5 pruebas de la sección 7.4.
- [ ] Todas las figuras del informe están en `figuras/`.
- [ ] El informe tiene todas las secciones y las referencias.
- [ ] La presentación está ensayada con la demo y hay video de respaldo.
