**Tutorial Machine-Learning**

Un problema de machine learning considera unn conjunt indeterminat de dades originades d'una base de dades i despres intenta predir altres propietats de dades desconegudes. Hi han diversos tipus d'algoritmes de machine learning:

- Algoritmes supervisats: Amb el model supervisat, la maquina aprèn per el exemple. A partir de dades amb les respostes correctes que l’operador la entrena, la maquina aprèn els patrons que les relacionen.
No supervisat: Dins del model no supervisat, la maquina no rep dades etiquetades, sinó que haurà d’aprendre els patrons per si mateixa sense saber que es que.

- Algoritmes no supervisats: Dins del model no supervisat, la maquina no rep dades etiquetades, sinó que haurà d’aprendre els patrons per si mateixa sense saber que es que.


Algoritmes supervisats

Dins dels algoritmes supervisats, existeixen altres subtipus:

- Regressió(Regression): Si el resultat consisteix en una o mes variables continues, aquesta tasca es diu regressio. Per exemple, predir la longitud d'un peix en funcio de la seva edat i pes. Els models dins de Regression són  linear regression, support vector regression and decision tree regression.

- Classification: Les dades donades tenen mes d'una classe i el que volem aprendre, a partir de dades etiquetades, com predir altres dades sense etiquetar, i aquesta maquina intentara classificarles amb a categoria o classe correcta. Els models dins de Classification són logistic regression, decision trees, random forests, support vector machines (*SVMs*) and gradient boosting.

# Exemple Regressió (Linear regression):

**Importar llibreries:**

Primer de tot, s'ha de importar les llibreries que s'utilitzaran i el dataset per part de sklearn

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
```
**Definir dataset:**

Seguidament, s'haura de definir el dataset com una variable.
```
breast_cancer = load_breast_cancer()
```
**Separar variables:**

Despres, s'haura de dividir el dataset en X, que seran les dades. La y seran els tipus que es prediran.
```
X = breast_cancer.data
y = breast_cancer.target
```

**Dividir el dataset en entrenament i prova:**

importar llibreria per separar la part de les dades amb la que s'entrenara la maquina y la part amb la que predirá el tipus.
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```
Crear model de Regresió Lineal

Importar el model de ML i definirlo.
```
from sklearn.linear_model import LinearRegression

model_linear = LinearRegression()
```
**Entrenar model:**

Entrenar el model amb la part de X i y amb la que s'ha distribuit abans.
```
model_linear.fit(X_train, y_train)
```
**Predir amb el model:**

Fer les prediccions amb la part de test.
```
prediction_linear = model_linear.predict(X_test)
```
**Evaluar model:**

Importar la part per evaluar per saber el percentatge de acerts.
```
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_**Convertir en matrius de confussió**test, prediction_linear)
r2 = r2_score(y_test, prediction_linear)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Coeficiente R2: {r2:.4f}")
```
**Convertir en matrius de confussió:**

Fer la matriu de confussió.

```
pred_binary = np.where(prediction_linear >= 0.5, 1, 0)

from sklearn.metrics import confusion_matrix

print("\nMatriz de Confusión (Regresión Lineal):")
print(confusion_matrix(y_test, pred_binary))
```
# Exemple Classificació (LogisticRegression)

**Importar dataset i llibreries**

```
from sklearn.datasets import load_breast_cancer
import pandas as pd
```

**Definir dataset**
```
breast_cancer = load_breast_cancer()
```
**separar variables**
```
X = breast_cancer.data
y= breast_cancer.target
```
**Dividir el dataset en entrenament i prova**
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
**crear modelos**
```
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

model_logistic = LogisticRegression(max_iter = 3000)
model_tree = DecisionTreeClassifier(random_state=42)
```
#Entrenar modelos
```
model_logistic.fit(X_train, y_train)
model_tree.fit(X_train, y_train)
```
#predecir con los modelos
```
prediction_logistic = model_logistic.predict(X_test)
prediction_tree = model_tree.predict(X_test)
```
#Evaluar modelos
```
from sklearn.metrics import accuracy_score

acc_log = accuracy_score(y_test, prediction_logistic)
acc_tree = accuracy_score(y_test, prediction_tree)

print(f"Acuracy Logistic Regression: {acc_log*100:.2f}%")
print(f"Acuracy Decision Tree: {acc_tree*100:.2f}%")
```
#Matrix
```
from sklearn.metrics import confusion_matrix
 
print("\nMatriz de Confusión (Regresión Logística):")
print(confusion_matrix(y_test, prediction_logistic))
```
