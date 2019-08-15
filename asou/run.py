from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

from sample.utils import data_loader

XSS1_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS1_TEST_FILE = 'dataset/test_level_1.csv'
XSS2_TRAIN_FILE = 'dataset/train_level_2.csv'
XSS2_TEST_FILE = 'dataset/test_level_2.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'
NORMAL2_TRAIN_FILE = 'dataset/train_level_4.csv'
NORMAL2_TEST_FILE = 'dataset/test_level_4.csv'
STOP_WORDS = ['']


def run():
    """
    データ作成
    """
    xss1_train_data, xss1_train_label = data_loader(XSS1_TRAIN_FILE, 'xss')
    xss1_test_data, xss1_test_label = data_loader(XSS1_TEST_FILE, 'xss')
    xss2_train_data, xss2_train_label = data_loader(XSS2_TRAIN_FILE, 'xss')
    xss2_test_data, xss2_test_label = data_loader(XSS2_TEST_FILE, 'xss')
    normal_train_data, normal_train_label = data_loader(NORMAL_TRAIN_FILE, 'normal')
    normal_test_data, normal_test_label = data_loader(NORMAL_TEST_FILE, 'normal')
    normal2_train_data, normal2_train_label = data_loader(NORMAL2_TRAIN_FILE, 'normal')
    normal2_test_data, normal2_test_label = data_loader(NORMAL2_TEST_FILE, 'normal')

    X_train_all = xss1_train_data + xss2_train_data + normal_train_data + normal2_train_data
    y_train_all = xss1_train_label + xss2_train_label + normal_train_label + normal2_train_label
    X_test_all = xss1_test_data + xss2_test_data + normal_test_data + normal2_test_data
    y_test_all = xss1_test_label + xss2_test_label + normal_test_label + normal2_test_label

    X = X_train_all + X_test_all
    y = y_train_all + y_test_all

    """
    for i, s in enumerate(X):
        s = s.replace('<br>', '')
        s = s.replace('<td>', '')
        s = s.replace('</td>', '')
        s = s.replace('</td>', '')
        s = s.replace('</strong>', '')
        s = s.replace('<strong>', '')
        s = s.replace('<title>', '')
        s = s.replace('</title>', '')
        X[i] = s
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    X_vec_train = np.zeros((len(X_train), 4))
    for i, s in enumerate(X_train):
        X_vec_train[i,0] = s.count('<')
        X_vec_train[i,1] = s.count('>')
        X_vec_train[i,2] = s.count('!')
        X_vec_train[i,3] = s.count(' ')

    X_vec_test = np.zeros((len(X_test), 4))
    for i, s in enumerate(X_test):
        X_vec_test[i,0] = s.count('<')
        X_vec_test[i,1] = s.count('>')
        X_vec_test[i,2] = s.count('!')
        X_vec_test[i,3] = s.count(' ')

    """
    データ前処理・学習機作成
    """
    # clf = svm.SVC(kernel='linear', random_state=None)
    clf = svm.SVC(kernel='rbf', random_state=None)
    clf.fit(X_vec_train, y_train)

    """
    テスト
    """
    pred = clf.predict(X_vec_test)
    acc_score = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(
        pred, y_test, labels=['xss', 'normal']
    )
    print("acc: \n", acc_score)
    print("confusion matrix: \n", conf_mat)

    for i in range(len(y_test)):
        if(y_test[i] != pred[i]):
            print('label:', y_test[i], ', prediction:',pred[i], X_test[i], '\n')

if __name__ == '__main__':
    run()
