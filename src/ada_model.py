from sklearn import datasets, tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib

_author_ = 'Stanley Hardt'

class AdaModel:
    def learn(self, X, y, file_name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)

        model_tree = tree.DecisionTreeClassifier(max_features = "auto",max_depth = None)
        abc = AdaBoostClassifier(base_estimator = model_tree)

        tuned_parameters = {
            "base_estimator__criterion" : ["gini", "entropy"],
            "base_estimator__splitter" :   ["best", "random"],
            "n_estimators": [1, 5]
         }

        clf = RandomizedSearchCV(abc, tuned_parameters, scoring='recall', n_iter=5, random_state=0)

        clf.fit(X_train, y_train)

        clf.predict(X_test)
        joblib.dump(clf,  "klasyfikatory/ada/" + file_name)

        print 'grid_scores:', clf.grid_scores_
        print 'best_estimator:', clf.best_estimator_
        print 'best_score:', clf.best_score_
        print 'best_params:', clf.best_params_
        return clf
    def predict(self, clf, X_test, file_name):
        y_pred = clf.predict(X_test)
        y_pred.tofile('wyniki_przewidywan/ada/'+file_name, sep=';')