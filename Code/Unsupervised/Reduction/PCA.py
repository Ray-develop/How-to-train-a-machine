from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Variança explicada per cada component: {pca.explained_variance_ratio_}")
print(f"Variança total explicada: {sum(pca.explained_variance_ratio_)*100:.2f}%")

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title("Visualització PCA + K-Means")
plt.xlabel("Component Principal 1")
plt.ylabel("Component Principal 2")
plt.colorbar(label="Clúster")
plt.show()

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('kmeans', KMeans(n_clusters=2, random_state=42, n_init=10))
])

pipeline.fit(X)
labels_pipeline = pipeline.named_steps['kmeans'].labels_

print(f"Silhouette Score (Pipeline PCA+KMeans): {silhouette_score(X_scaled, labels_pipeline):.4f}")
