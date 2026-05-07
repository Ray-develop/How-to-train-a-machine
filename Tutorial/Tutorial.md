# Tutorial Machine-Learning

Un problema de machine learning considera un conjunt indeterminat de dades originades d'una base de dades i després intenta predir altres propietats de dades desconegudes. Hi han diversos tipus d'algoritmes de machine learning:

- **Algoritmes supervisats**: Amb el model supervisat, la màquina aprèn per l'exemple. A partir de dades amb les respostes correctes que l'operador la entrena, la màquina aprèn els patrons que les relacionen.

- **Algoritmes no supervisats**: Dins del model no supervisat, la màquina no rep dades etiquetades, sinó que haurà d'aprendre els patrons per si mateixa sense saber què és què.

---

# Algoritmes supervisats

Dins dels algoritmes supervisats, existeixen altres subtipus:

- **Regressió (Regression)**: Si el resultat consisteix en una o més variables continues, aquesta tasca es diu regressió. Per exemple, predir la longitud d'un peix en funció de la seva edat i pes. Els models dins de Regression són: linear regression, support vector regression i decision tree regression.

- **Classificació (Classification)**: Les dades donades tenen més d'una classe i el que volem aprendre, a partir de dades etiquetades, és com predir altres dades sense etiquetar, i aquesta màquina intentarà classificar-les amb la categoria o classe correcta. Els models dins de Classification són: logistic regression, decision trees, random forests, support vector machines (*SVMs*) i gradient boosting.

---

## Exemple Regressió (LinearRegression)

**Importar llibreries:**

Primer de tot, s'ha d'importar les llibreries que s'utilitzaran i el dataset per part de sklearn.

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
```

**Definir dataset:**

Seguidament, s'haurà de definir el dataset com una variable.

```python
breast_cancer = load_breast_cancer()
```

**Separar variables:**

Després, s'haurà de dividir el dataset en X, que seran les dades. La y seran els tipus que es prediran.

```python
X = breast_cancer.data
y = breast_cancer.target
```

**Dividir el dataset en entrenament i prova:**

Importar llibreria per separar la part de les dades amb la que s'entrenarà la màquina i la part amb la que predirà el tipus.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

**Crear model de Regressió Lineal:**

Importar el model de ML i definir-lo.

```python
from sklearn.linear_model import LinearRegression

model_linear = LinearRegression()
```

**Entrenar model:**

Entrenar el model amb la part de X i y amb la que s'ha distribuït abans.

```python
model_linear.fit(X_train, y_train)
```

**Predir amb el model:**

Fer les prediccions amb la part de test.

```python
prediction_linear = model_linear.predict(X_test)
```

**Evaluar model:**

Importar la part per evaluar per saber el percentatge d'encerts.

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, prediction_linear)
r2 = r2_score(y_test, prediction_linear)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Coeficiente R2: {r2:.4f}")
```

**Convertir en matrius de confussió:**

Fer la matriu de confussió.

```python
pred_binary = np.where(prediction_linear >= 0.5, 1, 0)

from sklearn.metrics import confusion_matrix

print("\nMatriz de Confusión (Regresión Lineal):")
print(confusion_matrix(y_test, pred_binary))
```

---

## Exemple Classificació (LogisticRegression)

**Importar dataset i llibreries:**

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
```

**Definir dataset:**

```python
breast_cancer = load_breast_cancer()
```

**Separar variables:**

```python
X = breast_cancer.data
y = breast_cancer.target
```

**Dividir el dataset en entrenament i prova:**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

**Crear models:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

model_logistic = LogisticRegression(max_iter=3000)
model_tree = DecisionTreeClassifier(random_state=42)
```

**Entrenar models:**

```python
model_logistic.fit(X_train, y_train)
model_tree.fit(X_train, y_train)
```

**Predir amb els models:**

```python
prediction_logistic = model_logistic.predict(X_test)
prediction_tree = model_tree.predict(X_test)
```

**Evaluar models:**

```python
from sklearn.metrics import accuracy_score

acc_log = accuracy_score(y_test, prediction_logistic)
acc_tree = accuracy_score(y_test, prediction_tree)

print(f"Acuracy Logistic Regression: {acc_log*100:.2f}%")
print(f"Acuracy Decision Tree: {acc_tree*100:.2f}%")
```

**Crear matriu:**

```python
from sklearn.metrics import confusion_matrix

print("\nMatriz de Confusión (Regresión Logística):")
print(confusion_matrix(y_test, prediction_logistic))
```

---

# Algoritmes no supervisats

Dins dels algoritmes no supervisats, la màquina no rep dades etiquetades. Els subtipus principals són:

- **Clustering**: Agrupar dades similars sense saber les categories prèviament. Els models principals són **K-Means**, **DBSCAN** i **Hierarchical Clustering**.
- **Reducció de dimensionalitat**: Reduir el nombre de variables mantenint la màxima informació possible. El model principal és **PCA (Principal Component Analysis)**.

---

## Exemple Clustering (K-Means)

**Importar llibreries:**

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
```

**Definir dataset:**

```python
breast_cancer = load_breast_cancer()
```

**Separar variables:**

En el model no supervisat, només s'utilitza X, ja que la màquina no coneix les etiquetes y.

```python
X = breast_cancer.data
y = breast_cancer.target 
```

**Escalar les dades:**

És important escalar les dades abans d'aplicar K-Means, ja que és sensible a les escales de les variables.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Crear i entrenar el model K-Means:**

`n_clusters=2` indica que volem 2 grups (benigne i maligne), però la màquina no sap això.

```python
from sklearn.cluster import KMeans

model_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
model_kmeans.fit(X_scaled)
```

**Obtenir les prediccions (clústers):**

```python
clusters = model_kmeans.labels_
```

**Evaluar el model:**

Com que no supervisat no té etiquetes, s'utilitza el **Silhouette Score** per mesurar la qualitat dels clústers (de -1 a 1, com més alt millor).

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Acurracy: {score*100:.4f}")
```

**Comparar clústers amb les etiquetes reals:**

Podem comparar els clústers trobats amb les etiquetes reals  per veure com de bé ha separat la màquina les dades sense supervisió.

```python
from sklearn.metrics import confusion_matrix

print("\nMatriu de Confusió (K-Means vs etiquetes reals):")
print(confusion_matrix(y, clusters))
```

---

## Exemple Reducció de Dimensionalitat (PCA)

**Importar llibreries:**

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
```

**Aplicar PCA per reduir a 2 dimensions:**

Reduïm les 30 variables originals del dataset a només 2 components principals per poder visualitzar-les.

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Variança explicada per cada component: {pca.explained_variance_ratio_}")
print(f"Variança total explicada: {sum(pca.explained_variance_ratio_)*100:.2f}%")
```

**Visualitzar els resultats:**

```python
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title("Visualització PCA + K-Means")
plt.xlabel("Component Principal 1")
plt.ylabel("Component Principal 2")
plt.colorbar(label="Clúster")
plt.show()
```

**Combinar PCA + K-Means:**

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('kmeans', KMeans(n_clusters=2, random_state=42, n_init=10))
])

pipeline.fit(X)
labels_pipeline = pipeline.named_steps['kmeans'].labels_

print(f"Silhouette Score (Pipeline PCA+KMeans): {silhouette_score(X_scaled, labels_pipeline):.4f}")
```

---
