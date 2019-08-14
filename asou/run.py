from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn import svm

from sample.utils import data_loader

XSS_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS_TEST_FILE = 'dataset/test_level_1.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'
STOP_WORDS = ['']


def run():
    """
    データ作成
    """
    xss_train_data, xss_train_label = data_loader(XSS_TRAIN_FILE, 'xss')
    xss_test_data, xss_test_label = data_loader(XSS_TEST_FILE, 'xss')
    normal_train_data, normal_train_label = data_loader(NORMAL_TRAIN_FILE, 'normal')
    normal_test_data, normal_test_label = data_loader(NORMAL_TEST_FILE, 'normal')

    y_train = xss_train_label + normal_train_label
    y_test = xss_test_label + normal_test_label

    X_train = np.zeros((len(xss_train_data + normal_train_data), 3))
    for i, s in enumerate(xss_train_data + normal_train_data):
        X_train[i,0] = s.count('<')
        X_train[i,1] = s.count('>')
        X_train[i,2] = s.count('!')

    X_test = np.zeros((len(xss_test_data + normal_test_data), 3))
    for i, s in enumerate(xss_test_data + normal_test_data):
        X_test[i,0] = s.count('<')
        X_test[i,1] = s.count('>')
        X_test[i,2] = s.count('!')


    """
    データ前処理・学習機作成
    """
    clf = svm.SVC(kernel='linear', random_state=None)
    clf.fit(X_train, y_train)


    """
    テスト
    """
    pred = clf.predict(X_test)
    acc_score = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(
        pred, y_test, labels=['xss', 'normal']
    )
    print("acc: \n", acc_score)
    print("confusion matrix: \n", conf_mat)


if __name__ == '__main__':
    run()
