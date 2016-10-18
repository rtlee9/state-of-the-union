# coding: utf-8

import collections
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import re
import matplotlib.pyplot as plt
import mpld3
import pandas as pd
from gensim.models import doc2vec
import multiprocessing
import random
from datetime import datetime
import sys
import numpy as np


class sou:

    # Declare speech named tuple
    Speech = collections.namedtuple(
        'Speech', 'speech_type, speaker, party, date, body')

    # Load president details
    def __init__(self):
        self.prez_dets = pd.read_csv('prez_list.csv', index_col='president_no')
        self.prez_dedup = self.prez_dets.drop_duplicates('president_name')
        self.speeches_clean = None
        self.model = None
        self.dlist = None
        self.docs = None
        self.run_settings = None

    # Parse speeches
    def parse_speeches(self):

        # Read text file and split into speeches
        with open('pg5050.txt') as f:
            raw = f.read()
        speeches = raw.split('***')

        sou_list = filter(lambda x: x is not '', speeches[5].split('\r\n'))[3:]

        speeches_clean = filter(
            lambda x: x is not None,
            [self._clean_speech(s) for s in speeches[6:]])
        assert(len(speeches_clean) == len(sou_list))

        self.speeches_clean = speeches_clean
        self.speeches_dt = pd.DataFrame(
            speeches_clean, columns=self.Speech._fields)
        return '{} speeches found'.format(len(sou_list))

    # Clean SOU speeches: get key fields
    def _clean_speech(self, s):
        try:
            s_paragraphs = filter(lambda x: x is not '', s.split('\r\n\r\n'))
            s_header = s_paragraphs[0].split('\r\n')
            s_body = '\n\n'.join(filter(
                lambda x: len(x.split()) > 5,
                [' '.join(p.split('\r\n')) for p in s_paragraphs[1:]]))
            if s_header[0] != 'State of the Union Address':
                return None

            speech_type = s_header[0]
            date_str = s_header[2]
            speech_dt = datetime.strptime(date_str, '%B %d, %Y')
            speech_yr = speech_dt.year

            president = self.prez_dets[
                (self.prez_dets['term_start'] < speech_yr) &
                (self.prez_dets['term_end'] >= speech_yr)].iloc[0]
            president_party = president['party']
            president_name = president['president_name']

            return self.Speech(
                speech_type, president_name,
                president_party, date_str, s_body)

        except:
            return None

    # Convert text to lower-case and strip punctuation/symbols from words
    def _normalize_text(self, text):
        # Replace special characters with spaces
        norm_text = text.lower()
        norm_text = re.sub(r'\d', '0', norm_text)
        norm_text = norm_text.replace('0.0', '00')
        norm_text = norm_text.replace('0,0', '00')
        norm_text = '0'.join(filter(None, norm_text.split('0')))
        norm_text = norm_text.replace('<br />', ' ')
        norm_text = norm_text.replace('\n', ' ')
        norm_text = norm_text.replace('\t', ' ')
        norm_text = norm_text.replace('\t', ' ')
        # Pad punctuation with spaces on both sides
        for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
            norm_text = norm_text.replace(char, ' ' + char + ' ')
        # Consolidate consecutive spaces
        norm_text = ' '.join(norm_text.split())
        return norm_text

    # Vectorize speeches
    def speech2vec(self, epochs=3, pretrained=True, model=None,
                   save_ind=True, seed=400):

        # Confirm speeches were cleaned
        if self.speeches_clean is None:
            raise ValueError(
                'Clean speeches were not initialized; '
                'please call the parse_speeches function before proceeding')

        # Normalize text
        docs = []
        i = 0
        for s in self.speeches_clean:
            words = self._normalize_text(s.body.decode('utf-8')).split()
            tags = ['{}; {}'.format(s.speaker, s.date)]
            docs.append(doc2vec.TaggedDocument(words, tags))
            i += 1
        self.docs = docs

        # Initialize model
        random.seed(seed)
        cores = multiprocessing.cpu_count()
        print "Training {} epochs on {} core(s)".format(epochs, cores)
        if model is None:
            model = doc2vec.Doc2Vec(
                dm=1, dm_mean=1, size=300, window=8, negative=2,
                hs=0, min_count=3, workers=cores, iter=15)
        model.build_vocab(docs)

        if pretrained:
            # Load pre-trained word vectors from Google news corpus
            model.intersect_word2vec_format(
                'GoogleNews-vectors-negative300.bin', binary=True, lockf=1.0)

        # Train paragraph vectors
        for epoch in range(epochs):

            # Update status
            sys.stdout.write('.')
            sys.stdout.flush()

            # Shuffle and train
            random.shuffle(docs)
            model.train(docs)
            model.alpha -= 0.002
            model.min_alpha = model.alpha

        sys.stdout.write('\n')
        sys.stdout.flush()

        # Check and save model
        assert model.docvecs.count == len(docs)
        self.run_settings = {
            'model': str(model),
            'epochs': epochs,
            'pretrained': pretrained}
        self.model = model
        if save_ind:
            model.save('doc2vec_dm1')

        # Clean results
        dlist = []
        for d in docs:
            tag = d.tags[0]
            try:
                dv = model.docvecs[tag]
                dlist.append(len(dv))
            except:
                pass

        self.dlist = dlist

    # Reduce dimensions with t-SNE
    def reduce_dims(self, model):
        vectors = [model.docvecs[v.tags][0] for v in self.docs]
        X_embedded = TSNE(
            n_components=2, perplexity=5).fit_transform(vectors)
        return X_embedded

    # Get reduced doc vec dims
    def get_reduced(self):
        return self.reduce_dims(self.model)

    # Get plot labels
    def get_labels(self):
        prez_list = [s.speaker for s in self.speeches_clean]
        date_list = [int(s.date[-4:]) for s in self.speeches_clean]
        party_list = [s.party for s in self.speeches_clean]
        assert len(prez_list) == len(date_list)

        labels = zip(prez_list, party_list, date_list)
        labels = ['; '.join([str(j) for j in l]) for l in labels]
        return labels

    # Get plot colors
    def get_colors(self):
        i = 1
        party_no = {}
        party_list = [s.party for s in self.speeches_clean]
        for p in list(set(party_list)):
            party_no[p] = i
            i += 1
        return party_no

    # Get interactive scatter plo.display()t
    def get_plot(self):
        labels = self.get_labels()
        party_no = self.get_colors()
        party_list = [s.party for s in self.speeches_clean]

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(frameon=False)
        plt.setp(ax, xticks=(), yticks=())
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                            wspace=0.0, hspace=0.0)

        scatter = plt.scatter(self.X_embedded[:, 0], self.X_embedded[:, 1],
                              c=[party_no[p] for p in party_list], marker="x")
        tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
        mpld3.plugins.connect(fig, tooltip)
        return mpld3

    # Pull a specific speech
    def get_speech(self, speeches_dt, speaker, year):
        return self.speeches_dt[
            (speeches_dt['speaker'] == speaker) &
            (speeches_dt['date'].str[-4:] == str(year))]['body'].iloc[0]

    # Get full dataframe for modeling
    def get_df(self):

        # Confirm speeches were vectorized
        if self.model is None:
            raise ValueError(
                'Speeches were not vectorized; '
                'please call the speech2vec function before proceeding')

        dv = list(self.model.docvecs)
        dt = list(self.model.docvecs.doctags)
        X = pd.DataFrame(dv, index=dt)
        X['year'] = X.index.str[-4:]

        events = pd.DataFrame(
            [{'key': d, 'name': d.split(';')[0], 'date': d.split(';')[1]}
             for d in dt])
        events.set_index('key', inplace=True)
        Y = pd.merge(
            events, self.prez_dedup, how='left', left_on='name',
            right_on='president_name')
        Y.groupby('name').size().sort_values(ascending=False)[:5]
        y = (Y['party'] == 'Republican') * 1

        assert len(X) == len(y)
        return X, y

if __name__ == '__main__':

    # Train wordvecs
    vecs = sou()
    vecs.parse_speeches()
    vecs.speech2vec(pretrained=False)
    X, y = vecs.get_df()

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_reduced = sel.fit_transform(X)
    X_normalized = preprocessing.normalize(X_reduced, norm='l2')

    model2 = LogisticRegression()
    model2.fit(X, y)
    np.sort(model2.coef_)

    # Logistic regression
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_reduced = sel.fit_transform(X)
    # X_normalized = preprocessing.normalize(X.values, norm='l2')
    X_normalized = preprocessing.normalize(X_reduced, norm='l2')

    # Cross validate logistic regression
    clf = LogisticRegression()
    scores = cross_val_score(clf, X_normalized, y, scoring='accuracy', cv=10)
    print 'Mean logistic out of sample accuracy: {}'.format(scores.mean())

    # Random forest
    clf = RandomForestClassifier(
        max_depth=3, min_samples_split=3, random_state=0)
    scores = cross_val_score(clf, X, y)
    print 'Mean random forest out of sample accuracy: {}'.format(scores.mean())

    # GBM
    clf = GradientBoostingClassifier(
        n_estimators=30, learning_rate=1.0, max_depth=3, random_state=0)
    scores = cross_val_score(clf, X, y)
    print 'Mean GBM out of sample accuracy: {}'.format(scores.mean())
