import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Load Olivetti faces dataset
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

# PCA with 2 components and visualization
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'red', 'green',
          'blue', 'purple', 'brown', 'pink', 'black']
lw = 2

for color, i in zip(colors, range(40)):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=i)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Olivetti faces dataset')
plt.show()

# Display first 10 images
fig, ax = plt.subplots(2, 5, figsize=(10, 5),
                       subplot_kw={'xticks': [], 'yticks': []},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[i // 5, i % 5].imshow(faces.images[i + 200], cmap='bone')

plt.show()

# Function to evaluate classifiers using cross-validation
def evaluate_classifiers_cv(n_components):
    # Naive Bayes pipeline
    nb_pipeline = make_pipeline(PCA(n_components=n_components), GaussianNB())
    acc_nb = cross_val_score(nb_pipeline, X, y, cv=5, scoring='accuracy').mean()

    # SVM pipeline
    svm_pipeline = make_pipeline(PCA(n_components=n_components), SVC())
    acc_svm = cross_val_score(svm_pipeline, X, y, cv=5, scoring='accuracy').mean()

    # Logistic Regression pipeline
    lr_pipeline = make_pipeline(PCA(n_components=n_components), LogisticRegression(max_iter=10000))
    acc_lr = cross_val_score(lr_pipeline, X, y, cv=5, scoring='accuracy').mean()

    return acc_nb, acc_svm, acc_lr

# Experiment with different PCA dimensions
components_list = [2, 10, 50, 100]
results = {}

for n in components_list:
    results[n] = evaluate_classifiers_cv(n)

# Print results for each PCA component setting
for n in components_list:
    print(f"PCA components: {n}")
    print(f"Naive Bayes Accuracy: {results[n][0]:.4f}")
    print(f"SVM Accuracy: {results[n][1]:.4f}")
    print(f"Logistic Regression Accuracy: {results[n][2]:.4f}")
    print("\n")

# Compare results and identify the best classifier
best_accuracy = 0
best_classifier = None
best_pca_components = None

for n, (nb_acc, svm_acc, lr_acc) in results.items():
    accuracies = {
        'Naive Bayes': nb_acc,
        'SVM': svm_acc,
        'Logistic Regression': lr_acc
    }
    current_best_classifier = max(accuracies, key=accuracies.get)
    current_best_accuracy = accuracies[current_best_classifier]

    if current_best_accuracy > best_accuracy:
        best_accuracy = current_best_accuracy
        best_classifier = current_best_classifier
        best_pca_components = n

# Print overall best classifier and configuration
print(f"Best Classifier: {best_classifier}")
print(f"Highest Accuracy: {best_accuracy:.4f}")
print(f"Viable PCA number: {best_pca_components}")
