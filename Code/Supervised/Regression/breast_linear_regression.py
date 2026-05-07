from sklearn.datasets import load_breast_cancer
import pandas as pd

breast_cancer = load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

prediction_linear = model_linear.predict(X_test)


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, prediction_linear)
r2 = r2_score(y_test, prediction_linear)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Coeficiente R2: {r2:.4f}")

pred_binary = np.where(prediction_linear >= 0.5, 1, 0)

from sklearn.metrics import confusion_matrix

print("\nMatriz de Confusión (Regresión Lineal):")
print(confusion_matrix(y_test, pred_binary))
