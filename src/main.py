from ada_model import AdaModel
from random_forest_model import RandomForestModel
from loader import Loader


_author_ = 'Stanley Hardt'

class Main:
    def run(self):

        ##
        # Ladowanie danych
        X_train,y_train = Loader().load_train_set('~/ligandy_dane/all_summary.txt')
        X_test = Loader().load_test_set('~/ligandy_dane/test_data.txt')

        ##
        # Pojedynczy res_name

        #Ada
        clf = AdaModel().learn(X_train, y_train, 'single.pk')
        AdaModel().predict(clf, X_test, 'single.txt')
        #RandomForest
        clf = RandomForestModel().learn(X_train, y_train, 'single.pk')
        RandomForestModel().predict(clf, X_test, 'single.txt')

        ##
        # Zgrupowany res_name

        #ADA
        clf = AdaModel().learn(X_train, y_train, 'group.pk')
        AdaModel().predict(clf, X_test, 'group.txt')

        #RandomForest
        clf = RandomForestModel().learn(X_train, y_train, 'group.pk')
        RandomForestModel().predict(clf, X_test, 'group.txt')


Main().run()