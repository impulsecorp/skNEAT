import numpy as np
import random as rnd
from sklearn.feature_selection import *
from sklearn.feature_extraction import *
from sklearn.svm import SVC
from sklearn.decomposition import DictionaryLearning, FactorAnalysis, FastICA, LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression, PLSSVD
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import *

# dummy arrays so I don't get errors all the time
dx = np.zeros((100,100))
dy = np.zeros((100,1))

####################
# Base Trait Class
####################

class BaseObject:
    def __init__(self):
        pass

    def mate(self, other):
        baby = BaseObject()
        return baby

    def mutate(self):
        baby = BaseObject()
        return baby

    def distance_to(self, other):
        return rnd.uniform(0, 1)

    def fit(self, x, y):
        pass

    def transform(self, x, y):
        return np.array(x)


#########################
# Derived Classes
#########################

class_distance = 3.0
replace_class_prob = 0.02
mutation_prob = 0.2
derived_list = []

#######
# PCA
#######

class genePCA(BaseObject):
    def __init__(self):
        self.ndim = rnd.randint(1, dx.shape[1] - 1)
        self.pca = PCA(n_components=self.ndim)

    def __repr__(self):
        return "PCA(%d)" % self.ndim

    def mate(self, other):
        if isinstance(other, genePCA):
            baby = genePCA()
            baby.ndim = (self.ndim + other.ndim) // 2
            baby.pca = PCA(n_components=baby.ndim)
            return baby
        else:
            return rnd.choice([self, other])

    def mutate(self):
        baby = genePCA()
        baby.ndim += rnd.randint(-3, 3)
        baby.ndim = np.clip(baby.ndim, 1, dx.shape[1] - 1)
        baby.pca = PCA(n_components=baby.ndim)

        if rnd.uniform(0, 1) < replace_class_prob:
            return rnd.choice(derived_list)()

        return baby

    def distance_to(self, other):
        if isinstance(other, genePCA):
            return float(np.abs(self.ndim - other.ndim))
        else:
            return class_distance

    def fit(self, x, y=None):
        self.pca.fit(x, y=y)

    def transform(self, x, y):
        xt = self.pca.transform(x)
        return np.array(xt)




##################
# SelectKBest
##################

class geneSelectKBest(BaseObject):
    def __init__(self):
        self.k = rnd.randint(1, dx.shape[1] - 1)
        self.clf = SelectKBest(score_func=f_classif, k=self.k)

    def __repr__(self):
        return "SelectKBest(%d)" % self.k

    def mate(self, other):
        if isinstance(other, geneSelectKBest):
            baby = geneSelectKBest()
            baby.k = (self.k + other.k) // 2
            baby.clf = SelectKBest(score_func=f_classif, k=baby.k)
            return baby
        else:
            return rnd.choice([self, other])

    def mutate(self):
        baby = geneSelectKBest()
        baby.k += rnd.randint(-3, 3)
        baby.k = np.clip(baby.k, 1, dx.shape[1] - 1)
        baby.clf = SelectKBest(score_func=f_classif, k=baby.k)

        if rnd.uniform(0, 1) < replace_class_prob:
            return rnd.choice(derived_list)()

        return baby

    def distance_to(self, other):
        if isinstance(other, geneSelectKBest):
            return float(np.abs(self.k - other.k))
        else:
            return class_distance

    def fit(self, x, y=None):
        self.clf.fit(x, y=y)

    def transform(self, x, y):
        xt = self.clf.transform(x)
        return np.array(xt)



#######################
# LogisticRegression
#######################
class geneLogisticRegression(BaseObject):
    def __init__(self):
        ps = {}
        ps['penalty'] = rnd.choice(['l1', 'l2'])
        ps['dual'] = rnd.choice([True, False])
        ps['fit_intercept'] = rnd.choice([True, False])
        ps['solver'] = rnd.choice(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        ps['C'] = rnd.choice([0.5, 0.75, 0.95, 1.0])

        self.ps = ps
        self.clf = LogisticRegression(**ps, max_iter=10000)

    def __repr__(self):
        return "LGRG(%s,%s,%s)" % (self.ps['penalty'], self.ps['dual'], self.ps['fit_intercept'])

    def mate(self, other):
        if isinstance(other, geneLogisticRegression):
            baby = geneLogisticRegression()
            ps = self.ps
            if rnd.uniform(0, 1) < 0.5: ps['penalty'] = rnd.choice([self.ps['penalty'], other.ps['penalty']])
            if rnd.uniform(0, 1) < 0.5: ps['dual'] = rnd.choice([self.ps['dual'], other.ps['dual']])
            if rnd.uniform(0, 1) < 0.5: ps['fit_intercept'] = rnd.choice(
                [self.ps['fit_intercept'], other.ps['fit_intercept']])
            if rnd.uniform(0, 1) < 0.5: ps['solver'] = rnd.choice([self.ps['solver'], other.ps['solver']])
            if rnd.uniform(0, 1) < 0.5: ps['C'] = rnd.choice([self.ps['C'], other.ps['C']])
            baby.clf = LogisticRegression(**ps)
            return baby
        else:
            return rnd.choice([self, other])

    def mutate(self):
        baby = geneLogisticRegression()
        ps = self.ps
        if rnd.uniform(0, 1) < mutation_prob: ps['penalty'] = rnd.choice(['l1', 'l2'])
        if rnd.uniform(0, 1) < mutation_prob: ps['dual'] = rnd.choice([True, False])
        if rnd.uniform(0, 1) < mutation_prob: ps['fit_intercept'] = rnd.choice([True, False])
        if rnd.uniform(0, 1) < mutation_prob: ps['solver'] = rnd.choice(
            ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        if rnd.uniform(0, 1) < mutation_prob: ps['C'] = rnd.choice([0.5, 0.75, 0.95, 1.0])
        baby.clf = LogisticRegression(**ps)

        if rnd.uniform(0, 1) < replace_class_prob:
            return rnd.choice(derived_list)()

        return baby

    def distance_to(self, other):
        if isinstance(other, geneLogisticRegression):
            return 0.0
        else:
            return class_distance

    def fit(self, x, y):
        self.clf.fit(x, y)

    def transform(self, x, y):
        return self.clf.predict(x)

