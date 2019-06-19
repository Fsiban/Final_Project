from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

# loading breast cancer data set from sklearn
breast_cancer = datasets.load_breast_cancer()

#transforming breast cancer_data into a DataFrame with all the features as columns
features = pd.DataFrame(data = breast_cancer['data'],columns=breast_cancer['feature_names'])
breast_cancer_df = features

# bringing in 'target' column into breast cancer_df DataFrame from breast cancer data
breast_cancer_df['target']=breast_cancer['target']

#assigning variables to data and target
X = breast_cancer.data
y = breast_cancer.target

# Splitting dataset into: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.35)

# Training Logistic Regression model with fit()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=7000)
lr.fit(X_train, y_train)

#Training KNeighbours Classifier model with fit()
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3,weights='uniform',algorithm='auto')
kn.fit(X_train,y_train)

# Output of the Logistic Regression training is a model: a + b*X0 + c*X1 + d*X2 ...
print(f"LogisticRegression Intercept per class: {lr.intercept_}\n")
print(f"LogisticRegression Coeficients per class: {lr.coef_}\n")
print(f"LogisticRegression Available classes: {lr.classes_}\n")
print(f"LogisticRegression Number of iterations generating model: {lr.n_iter_}\n")

# Predicting the Logistic Regression results for test data set
predicted_values = lr.predict(X_test)

# Printing the Logistic Regression residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real}, LogisticRegression pred: {predicted} {"Oops Missed!!" if real != predicted else ""}\n')


# Printing Logistic Regression accuracy score(mean accuracy) from 0 - 1
print(f'LogisticRegression Accuracy score is {lr.score(X_test, y_test):.2f}/ \n')

# Printing Logistic Regression classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score
print('LogisticRegression Classification Report')
print(classification_report(y_test, predicted_values))

# Printing Logistic Regression classification confusion matrix (diagonal is true)
print('LogisticRegression Confusion Matrix')
print(confusion_matrix(y_test, predicted_values))

# Printing Logistic Regression classification
print('LogisticRegression Overall f1-score')
print(f1_score(y_test, predicted_values, average="macro"))

# Cross validation Logistic Regression using cross_val_score
from sklearn.model_selection import cross_val_score, ShuffleSplit
print(f'LogisticRegression Cross validation score:{cross_val_score(lr, X, y, cv=5)}\n')

# Cross validation Logic Regression using shuffle split
cv = ShuffleSplit(n_splits=5)
print(f'LogisticRegression ShuffleSplit val_score:{cross_val_score(lr, X, y, cv=cv)}\n')
print()

# Predicting the results for test dataset using KNeighbours
kn_predicted_values = kn.predict(X_test)

# Printing the residuals: difference between real and KN predicted
for (real, predicted) in list(zip(y_test, kn_predicted_values)):
    print(f'Value: {real}, KNeighbours pred: {predicted} {"Oops Missed!!" if real != predicted else ""}\n')

# Printing KN accuracy score(mean accuracy) from 0 - 1
print(f'KNeighbours Accuracy score is {kn.score(X_test, y_test):.2f}/ \n')

# Printing the KN classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score
print('KNeighbours Classification Report')
print(classification_report(y_test, kn_predicted_values))

# Printing the KN classification confusion matrix (diagonal is true)
print('KNeighbours Confusion Matrix')
print(confusion_matrix(y_test, kn_predicted_values))

print('KNeighbours Overall f1-score')
print(f1_score(y_test, kn_predicted_values, average="macro"))

# Cross validation KN using cross_val_score
from sklearn.model_selection import cross_val_score, ShuffleSplit
print(f'KNeighbours Cross validation score:{cross_val_score(kn, X, y, cv=5)}\n')

# Cross validation KN using shuffle split
cv = ShuffleSplit(n_splits=5)
print(f'KNeighbours ShuffleSplit val_score:{cross_val_score(kn, X, y, cv=cv)}\n')

#Visualisation
# create directory for plots
os.makedirs('plots/project1160', exist_ok=True)

# visualization of each feature distribution on the two targets/classes
for each_target in breast_cancer_df.target.unique():
    sns.distplot(breast_cancer_df['mean radius'][breast_cancer_df.target==each_target], kde = 1, label ='{}'.format(each_target))
    plt.legend()
plt.savefig(f'plots/project1160/mean_radius_distribution.png', dpi=300)
plt.clf()

# redefining the data set header to only display 10 columns
breast_cancer = breast_cancer_df.drop(['radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness','worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension'], axis=1)

sns.pairplot(breast_cancer,hue='target', diag_kind='hist')
plt.savefig(f'plots/project1160/pairplot.png', dpi = 300)
plt.clf()