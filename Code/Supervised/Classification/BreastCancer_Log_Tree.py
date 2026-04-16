#Importar dataset y librerías

from sklearn.datasets import load_breast_cancer
import pandas as pd

#definir dataset
breast_cancer = load_breast_cancer()

#separar variables

X = breast_cancer.data
y= breast_cancer.target

#Dividir el dataset en entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#crear modelos

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

model_logistic = LogisticRegression(max_iter = 3000)
model_tree = DecisionTreeClassifier(random_state=42)

#Entrenar modelos

model_logistic.fit(X_train, y_train)
model_tree.fit(X_train, y_train)

#predecir con los modelos

prediction_logistic = model_logistic.predict(X_test)
prediction_tree = model_tree.predict(X_test)

#Evaluar modelos
from sklearn.metrics import accuracy_score

acc_log = accuracy_score(y_test, prediction_logistic)
acc_tree = accuracy_score(y_test, prediction_tree)

print(f"Acuracy Logistic Regression: {acc_log*100:.2f}%")
print(f"Acuracy Decision Tree: {acc_tree*100:.2f}%")

#Matrix

from sklearn.metrics import confusion_matrix
 
print("\nMatriz de Confusión (Regresión Logística):")
print(confusion_matrix(y_test, prediction_logistic))
