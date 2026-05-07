from sklearn.datasets import load_breast_cancer
import pandas as pd

breast_cancer = load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans

model_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
model_kmeans.fit(X_scaled)

clusters = model_kmeans.labels_

from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Acurracy: {score*100:.4f}")


from sklearn.metrics import confusion_matrix

print("\nMatriu de Confusió (K-Means vs etiquetes reals):")
print(confusion_matrix(y, clusters))

