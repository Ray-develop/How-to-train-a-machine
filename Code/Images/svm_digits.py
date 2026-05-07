from sklearn.datasets import load_digits

digits = load_digits()
#Tranformar el 8x8 en 64 numeros
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
# Abans: (1797, 8, 8)
# Desprès: (1797, 64)

X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False) 
from sklearn import svm
img = svm.SVC(gamma=0.001)

# Entrenament
img.fit(X_train, y_train)

pred_img = img.predict(X_test)

from sklearn.metrics import accuracy_score

acc_img = accuracy_score(y_test, pred_img)

print("Accuracy score:", acc_img*100)
