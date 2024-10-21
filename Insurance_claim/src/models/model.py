from tkinter.font import names

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, confusion_matrix


#prepare data for build model

#split data into data test(20%) and data train(80%)

names = ['model_DT', 'model_RDF', 'model_KN', 'model_XGBC', 'model_MLCP', 'model_GAUS','model_Stak']
data = pd.read_csv('./data/processed/insurance_claims.csv')
columns = data.columns.drop('fraud_reported')
x = data[columns]
y = data['fraud_reported']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

models = list()
models.append(DecisionTreeClassifier(max_depth=10))
models.append(RandomForestClassifier(n_estimators=500))
models.append(KNeighborsClassifier(5))
models.append(XGBClassifier(objective= 'binary:logistic'))
models.append(MLPClassifier(max_iter=1000))
models.append(GaussianNB())
models.append(StackingClassifier(
    estimators=[
    ('xgbc',XGBClassifier(objective= 'binary:logistic')),
    ('knn',KNeighborsClassifier(5)),
    ('gaus',GaussianNB())
    ],
    final_estimator=RandomForestClassifier(n_estimators=500)
))



# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=5)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


def test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)


    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)


results= list()
print(y_test.value_counts())
for name, model in zip(names,models):
    scores = evaluate_model(model, X_train, y_train)
    results.append(scores)
    print('>%s %.2f (%.2f)' % (name, mean(scores), std(scores)))

plt.boxplot(results, labels=names, showmeans=True)
plt.show()

for name, model in zip(names, models):
    print(f"Testing {name}...")
    test_model(model, X_train, y_train, X_test, y_test)





