from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib

_author_ = 'Stanley Hardt'

class RandomForestModel:
    def learn(self, X, y, file_name):
        tuned_parameters = {"n_estimators" : range(1, 31),
              "criterion" : ["gini","entropy"]
             }
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        clf = RandomizedSearchCV(RandomForestClassifier(), tuned_parameters, scoring='recall_macro', n_iter=2, random_state=5)
        clf.fit(X_train, y_train)


        clf.predict(X_test)
        joblib.dump(clf, "klasyfikatory/random_forest/" + file_name)

        print 'grid_scores:', clf.grid_scores_
        print 'best_estimator:', clf.best_estimator_
        print 'best_score:', clf.best_score_
        print 'best_params:', clf.best_params_
        return clf
    def predict(self, clf, X_test, file_name):
        y_pred = clf.predict(X_test)
        y_pred.tofile('wyniki_przewidywan/random_forest/'+file_name, sep=';')