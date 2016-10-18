# coding: utf-8
import speech2vec as s
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from collections import namedtuple
from gensim.models import doc2vec
import multiprocessing
import csv

Run = namedtuple('Run', 'epochs, pretrained, size, iter')
cores = multiprocessing.cpu_count()
plan = [
    Run(1, False, 100, 5),
    Run(1, False, 100, 10),
    Run(1, False, 100, 15),
    Run(3, False, 100, 15),
    Run(5, False, 50, 10),
    Run(5, False, 50, 15),
    Run(5, False, 50, 20),
    Run(5, False, 100, 10),
    Run(5, False, 100, 15),
    Run(5, False, 100, 20),
    Run(5, False, 300, 15),
    Run(10, False, 100, 15),
    Run(1, True, 300, 15),
    Run(3, True, 300, 15),
    Run(5, True, 300, 15),
    Run(10, True, 300, 15),
    Run(1, True, 300, 20),
    Run(3, True, 300, 20),
    Run(5, True, 300, 10),
    Run(5, True, 300, 20),
    Run(10, True, 300, 20),
]

runs = []
vecs = s.sou()
vecs.parse_speeches()

for p in plan:

    model = doc2vec.Doc2Vec(dm=1, dm_mean=1, size=p.size, window=8, negative=2,
                            hs=0, min_count=3, workers=cores, iter=p.iter)

    vecs.speech2vec(epochs=p.epochs, pretrained=p.pretrained)
    X, y = vecs.get_df()
    run = dict(p._asdict())
    run.update(vecs.run_settings)

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Reduce independent variables
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_reduced = sel.fit_transform(X)
    X_normalized = preprocessing.normalize(X_reduced, norm='l2')

    # Logistic regression
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_reduced = sel.fit_transform(X)
    # X_normalized = preprocessing.normalize(X.values, norm='l2')
    X_normalized = preprocessing.normalize(X_reduced, norm='l2')

    # Cross validate logistic regression
    clf = LogisticRegression()
    scores = cross_val_score(clf, X_normalized, y, scoring='accuracy', cv=10)
    run['logistic'] = scores.mean()

    # Random forest
    clf = RandomForestClassifier(
        max_depth=3, min_samples_split=3, random_state=0)
    scores = cross_val_score(clf, X, y)
    run['random forest'] = scores.mean()

    # GBM
    clf = GradientBoostingClassifier(
        n_estimators=30, learning_rate=1.0, max_depth=3, random_state=0)
    scores = cross_val_score(clf, X, y)
    run['GBM'] = scores.mean()

    print run
    runs.append(run)

# Save run to CSV
myfile = open('grid_search.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(runs)
