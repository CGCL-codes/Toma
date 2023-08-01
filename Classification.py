import argparse
import csv
from itertools import islice
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import joblib
from sklearn.linear_model import LogisticRegression



def parse_options():
    parser = argparse.ArgumentParser(description='Malware Detection.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains benign and malware feature csv.', required=True, type=str)
    parser.add_argument('-o', '--out', help='The dir_path of output', required=True, type=str)
    args = parser.parse_args()

    return args


def feature_extraction_all(feature_csv):
    features = []

    with open(feature_csv, 'r') as f:
        data = csv.reader(f)
        for line in islice(data, 1, None):
            try:
                feature = [float(i) for i in line[2:]]
                features.append(feature)
            except Exception as e:
                # print(line[0:2])
                # print(e)
                pass
    print('len:')
    print(len(features))
    return features


def obtain_dataset(dir_path):

    # clone_featureCSV_1 = dir_path + 'type-1_sim.csv'
    # clone_featureCSV_2 = dir_path + 'type-2_sim.csv'
    # clone_featureCSV_3 = dir_path + 'type-3_sim.csv'
    # clone_featureCSV_4 = dir_path + 'type-4_sim.csv'
    # clone_featureCSV_5 = dir_path + 'type-5_sim.csv'
    # clone_featureCSV_6 = dir_path + 'type-6_sim.csv'
    nonclone_featureCSV = dir_path + 'noclone_sim.csv'
    clone_featureCSV = dir_path + 'clone_sim.csv'


    Vectors = []
    Labels = []

    # clone_features1 = feature_extraction_all(clone_featureCSV_1)
    # clone_features2 = feature_extraction_all(clone_featureCSV_2)
    # clone_features3 = feature_extraction_all(clone_featureCSV_3)
    # clone_features4 = feature_extraction_all(clone_featureCSV_4)
    # clone_features5 = feature_extraction_all(clone_featureCSV_5)
    # clone_features6 = feature_extraction_all(clone_featureCSV_6)
    nonclone_features = feature_extraction_all(nonclone_featureCSV)
    clone_features = feature_extraction_all(clone_featureCSV)


    # Vectors.extend(clone_features1)
    # Labels.extend([1 for i in range(len(clone_features1))])
    # Vectors.extend(clone_features2)
    # Labels.extend([1 for i in range(len(clone_features2))])
    # Vectors.extend(clone_features3)
    # Labels.extend([1 for i in range(len(clone_features3))])
    # Vectors.extend(clone_features4)
    # Labels.extend([1 for i in range(len(clone_features4))])
    # Vectors.extend(clone_features5)
    # Labels.extend([1 for i in range(len(clone_features5))])
    # Vectors.extend(clone_features6)
    # Labels.extend([1 for i in range(len(clone_features6))])

    Vectors.extend(nonclone_features)
    Labels.extend([0 for i in range(len(nonclone_features))])

    Vectors.extend(clone_features)
    Labels.extend([1 for i in range(len(clone_features))])

    print('len of Vectors:')
    print(len(Vectors))
    print('len of Labels:')
    print(len(Labels))

    return Vectors, Labels


def random_features(vectors, labels):
    Vec_Lab = []

    for i in range(len(vectors)):
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec)

    random.shuffle(Vec_Lab)

    return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]



def voting(vectors, labels, n):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=5)

    Precisions = []
    Recalls = []
    F1s = []

    Precisions1 = []
    Recalls1 = []
    F1s1 = []

    Precisions2 = []
    Recalls2 = []
    F1s2 = []

    Precisions3 = []
    Recalls3 = []
    F1s3 = []

    Precisions4 = []
    Recalls4 = []
    F1s4 = []

    Precisions5 = []
    Recalls5 = []
    F1s5 = []

    Precisions6 = []
    Recalls6 = []
    F1s6 = []

    Precisions7 = []
    Recalls7 = []
    F1s7 = []

    # Precisions8 = []
    # Recalls8 = []
    # F1s8 = []
    #
    Precisions9 = []
    Recalls9 = []
    F1s9 = []
    #
    # Precisions10 = []
    # Recalls10 = []
    # F1s10 = []
    #
    # Precisions11 = []
    # Recalls11 = []
    # F1s11 = []
    #
    # Precisions12 = []
    # Recalls12 = []
    # F1s12 = []

    Precisions13 = []
    Recalls13 = []
    F1s13 = []

    i = 1
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]


        clf1 = KNeighborsClassifier(n_neighbors=5)
        clf1.fit(train_X, train_Y)
        joblib.dump(clf1, 'clf_knn_5.pkl')
        y_pred1 = clf1.predict(test_X)
        print("5NN")
        precision1 = precision_score(y_true=test_Y, y_pred=y_pred1)
        recall1 = recall_score(y_true=test_Y, y_pred=y_pred1)
        f11 = f1_score(y_true=test_Y, y_pred=y_pred1)
        print(f11, precision1, recall1)
        Precisions1.append(precision1)
        Recalls1.append(recall1)
        F1s1.append(f11)

        clf2 = KNeighborsClassifier(n_neighbors=1)
        clf2.fit(train_X, train_Y)
        joblib.dump(clf2, 'clf_knn_1.pkl')
        y_pred2 = clf2.predict(test_X)
        print("1NN")
        precision2 = precision_score(y_true=test_Y, y_pred=y_pred2)
        recall2 = recall_score(y_true=test_Y, y_pred=y_pred2)
        f12 = f1_score(y_true=test_Y, y_pred=y_pred2)
        print(f12, precision2, recall2)
        Precisions2.append(precision2)
        Recalls2.append(recall2)
        F1s2.append(f12)

        clf3 = KNeighborsClassifier(n_neighbors=3)
        clf3.fit(train_X, train_Y)
        joblib.dump(clf3, 'clf_knn_3.pkl')
        y_pred3 = clf3.predict(test_X)
        print("3NN")
        precision3 = precision_score(y_true=test_Y, y_pred=y_pred3)
        recall3 = recall_score(y_true=test_Y, y_pred=y_pred3)
        f13 = f1_score(y_true=test_Y, y_pred=y_pred3)
        print(f13, precision3, recall3)
        Precisions3.append(precision3)
        Recalls3.append(recall3)
        F1s3.append(f13)

        clf4 = tree.DecisionTreeClassifier(max_depth=64)
        clf4.fit(train_X, train_Y)
        joblib.dump(clf4, 'clf_decision_tree.pkl')
        y_pred4 = clf4.predict(test_X)
        print("DT")
        precision4 = precision_score(y_true=test_Y, y_pred=y_pred4)
        recall4 = recall_score(y_true=test_Y, y_pred=y_pred4)
        f14 = f1_score(y_true=test_Y, y_pred=y_pred4)
        print(f14, precision4, recall4)
        Precisions4.append(precision4)
        Recalls4.append(recall4)
        F1s4.append(f14)

        clf5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=256), random_state=0)
        clf5.fit(train_X, train_Y)
        joblib.dump(clf5, 'clf_adaboost.pkl')
        y_pred5 = clf5.predict(test_X)
        print("ADABOOST")
        precision5 = precision_score(y_true=test_Y, y_pred=y_pred5)
        recall5 = recall_score(y_true=test_Y, y_pred=y_pred5)
        f15 = f1_score(y_true=test_Y, y_pred=y_pred5)
        print(f15, precision5, recall5)
        Precisions5.append(precision5)
        Recalls5.append(recall5)
        F1s5.append(f15)

        clf6 = GradientBoostingClassifier(max_depth=16, random_state=0)
        clf6.fit(train_X, train_Y)
        joblib.dump(clf6, 'clf_gdbt.pkl')
        y_pred6 = clf6.predict(test_X)
        print("GDBT")
        precision6 = precision_score(y_true=test_Y, y_pred=y_pred6)
        recall6 = recall_score(y_true=test_Y, y_pred=y_pred6)
        f16 = f1_score(y_true=test_Y, y_pred=y_pred6)
        print(f16, precision6, recall6)
        Precisions6.append(precision6)
        Recalls6.append(recall6)
        F1s6.append(f16)

        # start2 = time.time()
        clf7 = XGBClassifier(max_depth=256, random_state=0)
        clf7.fit(train_X, train_Y)
        joblib.dump(clf7, 'clf_xgboost.pkl')
        y_pred7 = clf7.predict(test_X)
        print("XGBOOST")
        precision7 = precision_score(y_true=test_Y, y_pred=y_pred7)
        recall7 = recall_score(y_true=test_Y, y_pred=y_pred7)
        f17 = f1_score(y_true=test_Y, y_pred=y_pred7)
        print(f17, precision7, recall7)
        Precisions7.append(precision7)
        Recalls7.append(recall7)
        F1s7.append(f17)
        # end2 = time.time()
        # t2 = end2 - start2
        # print(t2)

        # clf8 = GaussianNB()
        # clf8.fit(train_X, train_Y)
        # #joblib.dump(clf7, 'clf_xgboost.pkl')
        # y_pred8 = clf8.predict(test_X)
        # print("GaussianNB")
        # precision8 = precision_score(y_true=test_Y, y_pred=y_pred8)
        # recall8 = recall_score(y_true=test_Y, y_pred=y_pred8)
        # f18 = f1_score(y_true=test_Y, y_pred=y_pred8)
        # print(precision8, recall8, f18)
        # Precisions8.append(precision8)
        # Recalls8.append(recall8)
        # F1s8.append(f18)
        #

        clf9 = LogisticRegression()
        clf9.fit(train_X, train_Y)
        joblib.dump(clf9, 'clf_lr.pkl')
        y_pred9 = clf9.predict(test_X)
        print("LogisticRegression")
        precision9 = precision_score(y_true=test_Y, y_pred=y_pred9)
        recall9 = recall_score(y_true=test_Y, y_pred=y_pred9)
        f19 = f1_score(y_true=test_Y, y_pred=y_pred9)
        print(f19, precision9, recall9)
        Precisions9.append(precision9)
        Recalls9.append(recall9)
        F1s9.append(f19)
        #
        # clf10 = NearestCentroid()
        # clf10.fit(train_X, train_Y)
        # #joblib.dump(clf7, 'clf_xgboost.pkl')
        # y_pred10 = clf10.predict(test_X)
        # print("NearestCentroid")
        # precision10 = precision_score(y_true=test_Y, y_pred=y_pred10)
        # recall10 = recall_score(y_true=test_Y, y_pred=y_pred10)
        # f110 = f1_score(y_true=test_Y, y_pred=y_pred10)
        # print(precision10, recall10, f110)
        # Precisions10.append(precision10)
        # Recalls10.append(recall10)
        # F1s10.append(f110)
        #
        # clf11 = RidgeClassifier()
        # clf11.fit(train_X, train_Y)
        # #joblib.dump(clf7, 'clf_xgboost.pkl')
        # y_pred11 = clf11.predict(test_X)
        # print("RidgeClassifier")
        # precision11 = precision_score(y_true=test_Y, y_pred=y_pred11)
        # recall11 = recall_score(y_true=test_Y, y_pred=y_pred11)
        # f111 = f1_score(y_true=test_Y, y_pred=y_pred11)
        # print(precision11, recall11, f111)
        # Precisions11.append(precision11)
        # Recalls11.append(recall11)
        # F1s11.append(f111)
        #
        # clf12 = QuadraticDiscriminantAnalysis()
        # clf12.fit(train_X, train_Y)
        # #joblib.dump(clf7, 'clf_xgboost.pkl')
        # y_pred12 = clf12.predict(test_X)
        # print("QuadraticDiscriminantAnalysis")
        # precision12 = precision_score(y_true=test_Y, y_pred=y_pred12)
        # recall12 = recall_score(y_true=test_Y, y_pred=y_pred12)
        # f112 = f1_score(y_true=test_Y, y_pred=y_pred12)
        # print(precision12, recall12, f112)
        # Precisions12.append(precision12)
        # Recalls12.append(recall12)
        # F1s12.append(f112)

        # start1 = time.time()
        clf13 = RandomForestClassifier(max_depth=32, random_state=0)
        clf13.fit(train_X, train_Y)
        joblib.dump(clf13, 'clf_randomforest.pkl')
        y_pred13 = clf13.predict(test_X)
        print("RF")
        precision13 = precision_score(y_true=test_Y, y_pred=y_pred13)
        recall13 = recall_score(y_true=test_Y, y_pred=y_pred13)
        f113 = f1_score(y_true=test_Y, y_pred=y_pred13)
        print(f113, precision13, recall13)
        Precisions13.append(precision13)
        Recalls13.append(recall13)
        F1s13.append(f113)
        # end1 = time.time()
        # t1 = end1 - start1
        # print(t1)


        y_pred = [0 for i in range(len(y_pred7))]

        for i in range(len(y_pred7)):
            sum = 0
            sum += y_pred1[i]
            sum += y_pred2[i]
            sum += y_pred3[i]
            sum += y_pred4[i]
            sum += y_pred5[i]
            sum += y_pred6[i]
            sum += y_pred7[i]
            sum += y_pred9[i]
            sum += y_pred13[i]
            if sum >= n:
                y_pred[i] = 1

        y_pred = np.array(y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        print(f1, precision, recall)
        Precisions.append(precision)
        Recalls.append(recall)
        F1s.append(f1)

        # result = pd.DataFrame({'5nn': y_pred1, '1NN': y_pred2, '3NN': y_pred3, 'DT': y_pred4,
        #                        'ADABOOST': y_pred5, 'GDBT': y_pred6,
        #                        'GaussianNB': y_pred8, 'LR': y_pred9, 'NC': y_pred10, 'RC': y_pred11,
        #                        'QD': y_pred12, 'true': test_Y})# 'XGBOOST': y_pred7,
        # result.to_csv(str(i)+'pred.csv', index=False)
        i += 1
        #break
    print('all')
    print(np.mean(F1s1), np.mean(Precisions1), np.mean(Recalls1))
    print(np.mean(F1s2), np.mean(Precisions2), np.mean(Recalls2))
    print(np.mean(F1s3), np.mean(Precisions3), np.mean(Recalls3))
    print(np.mean(F1s4), np.mean(Precisions4), np.mean(Recalls4))
    print(np.mean(F1s5), np.mean(Precisions5), np.mean(Recalls5))
    print(np.mean(F1s6), np.mean(Precisions6), np.mean(Recalls6))
    print(np.mean(F1s7), np.mean(Precisions7), np.mean(Recalls7))
    # print(np.mean(F1s8), np.mean(Precisions8), np.mean(Recalls8))
    print(np.mean(F1s9), np.mean(Precisions9), np.mean(Recalls9))
    # print(np.mean(F1s10), np.mean(Precisions10), np.mean(Recalls10))
    # print(np.mean(F1s11), np.mean(Precisions11), np.mean(Recalls11))
    # print(np.mean(F1s12), np.mean(Precisions12), np.mean(Recalls12))
    print(np.mean(F1s13), np.mean(Precisions13), np.mean(Recalls13))
    print(np.mean(F1s), np.mean(Precisions), np.mean(Recalls))

    #return np.mean(Precisions)


def randomforest(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=5)
    F1s = []
    Precisions = []
    Recalls = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = RandomForestClassifier(max_depth=32, random_state=0)
        clf.fit(train_X, train_Y)
        joblib.dump(clf, 'clf_randomforest.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        print(f1, precision, recall)
        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        break

    print(np.mean(F1s), np.mean(Precisions), np.mean(Recalls))
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls)]


def weight():
    dir_path = '/output/'
    Vectors, Labels = obtain_dataset(dir_path)

    vectors, labels = random_features(Vectors, Labels)
    x_train = np.array(vectors)
    y_train = np.array(labels)

    feat_labels = ['Jaccard', 'Dice', 'Jaro', 'Jaro_winkler', 'Levenshtein_sim', 'Levenshtein_ratio']
    forest = RandomForestClassifier(max_depth=32, random_state=0)  # (n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(x_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


def main1():
    dir_path = '/home/data4T2/fsy/token/output1/'
    Vectors, Labels = obtain_dataset(dir_path)
    vectors, labels = random_features(Vectors, Labels)

    start = time.time()
    randomforest(vectors, labels)
    end = time.time()
    t = end - start
    print(t)


def ml(train_X, train_Y, test_X, test_Y, k, n):
    start1 = time.time()
    if k == 'RF':
        clf13 = RandomForestClassifier(max_depth=n, random_state=0)  #
    elif k == 'DT':
        clf13 = tree.DecisionTreeClassifier(max_depth=n, random_state=0)  #
    elif k == 'GDBT':
        clf13 = GradientBoostingClassifier(max_depth=n, random_state=0)  #
    elif k == 'XGBOOST':
        clf13 = XGBClassifier(max_depth=n, random_state=0)  #
    else:
        clf13 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=n), random_state=0)  #

    print(k, n)
    clf13.fit(train_X, train_Y)
    y_pred13 = clf13.predict(test_X)
    precision13 = precision_score(y_true=test_Y, y_pred=y_pred13)
    recall13 = recall_score(y_true=test_Y, y_pred=y_pred13)
    f113 = f1_score(y_true=test_Y, y_pred=y_pred13)
    print(f113, precision13, recall13)

    end1 = time.time()
    t1 = end1 - start1
    print(t1)


def parameters():
    dir_path = '/output/'
    Vectors, Labels = obtain_dataset(dir_path)
    vectors, labels = random_features(Vectors, Labels)
    X = np.array(vectors)
    Y = np.array(labels)
    parameter = [8, 16, 32, 64, 128, 256]  #
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]
        for i in parameter:
            ml(train_X, train_Y, test_X, test_Y, 'RF', i)
            ml(train_X, train_Y, test_X, test_Y, 'DT', i)
            ml(train_X, train_Y, test_X, test_Y, 'Adaboost', i)
            ml(train_X, train_Y, test_X, test_Y, 'GDBT', i)
            ml(train_X, train_Y, test_X, test_Y, 'XGBOOST', i)
        break


if __name__ == '__main__':
    main1()
    #weight()
    #parameters()

