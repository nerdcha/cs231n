'''
This script fits a basic linear SVM to the CIFAR-10 dataset.

The goal is to complete Assignment 1 with as few code lines as possible.
Using SKLearn's SGDClassifier, the only parameter to control is alpha, because by default it uses
an 'optimal' heuristic for the learning rate.
I fix an L2 penalty because that seems much more sensible than an L1 penalty for image recognition.

Author: Jamie Hall
License: GPL2
'''


import numpy as np
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from scipy.stats import lognorm as sp_lognormal

import cs231n.cifar10 as cf

np.random.seed(31337)

X_train, y_train, X_test, y_test, scaler = cf.get_normalised_data()

basic_svm = SGDClassifier(loss="hinge", penalty="l2", l1_ratio=0.0, random_state=31337, n_jobs=5)

# From the Scipy docs: to sample a random variable Y such that Y=exp(X) where X~N(mu,sigma), use
# scipy.stats.lognormal(s=sigma, scale=np.exp(mu))
random_search = RandomizedSearchCV(basic_svm,
                       param_distributions={'alpha': sp_lognormal(s=2, scale=np.exp(-4))},
                       n_iter=20, verbose=1)

random_search.fit(X_train, y_train)

print("Chosen: ", random_search.best_params_["alpha"])
print("Best CV score: ", random_search.best_score_)
chosen_svm = random_search.best_estimator_

os.makedirs("output/svc", exist_ok=True)
labels = cf.get_label_names()
for i in range(10):
    # Don't forget to rescale the hyperplanes to get human-readable versions---the l2 penalty makes
    # them close to the origin, so they look indistinguishable.
    this_hyperplane = 127*(chosen_svm.coef_[i]/np.max(np.abs(chosen_svm.coef_[i]))) + scaler.mean_
    cf.plot_image(this_hyperplane, "output/svc/archetype " + labels[i] + ".png")

preds = chosen_svm.predict(X_test)
print("Test score: %.3f" % chosen_svm.score(X_test, y_test))

