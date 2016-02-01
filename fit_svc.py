'''
This script fits a basic linear SVM to the CIFAR-10 dataset.

I tried using SKLearn's "optimal" learning rate, but I was interested to follow
the Bottou recipe more closely than they seem to have.

Also, I originally used GridSearchCV, but decided to use the 1-sd rule of thumb, whereas
the SKLearn version seems to only allow a greedy choice of the best CV score.

The source for the Bottou recipe is:
Bottou, Léon. “Stochastic Gradient Descent Tricks.” In Neural Networks: Tricks of the Trade,
edited by Grégoire Montavon, Geneviève B. Orr, and Klaus-Robert Müller, 421–36.
Lecture Notes in Computer Science 7700. Springer Berlin Heidelberg, 2012.

Author: Jamie Hall
License: GPL2
'''


import numpy as np
import os
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.linear_model import SGDClassifier

import cs231n.cifar10 as cf


X_train0, y_train0, X_test, y_test, scaler = cf.get_normalised_data()

# Usually at this point you'd shuffle the data before running SGD. But
# the CIFAR10 comes preshuffled, so meh.

n_train = X_train0.shape[0]
n_pretest = 2000
batch_size = 1000
batches_for_cv_performance = 10

X_train = X_train0[n_pretest:(n_train-n_pretest)]
y_train = y_train0[n_pretest:(n_train-n_pretest)]
X_pretest = X_train0[0:n_pretest]
y_pretest = y_train0[0:n_pretest]


def evaluate_svm(alpha):
    # Note: n_iter gets switched to 1 by sklearn whenever you call partial_fit(). This initial
    # setting is for the pretesting of eta0.
    basic_svm = SGDClassifier(loss="hinge", penalty="l2", l1_ratio=0.0, random_state=31337, n_jobs=5,
                              n_iter=5, alpha=alpha)

    learning_rate_grid = [ 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7 ]
    pretest_svm = GridSearchCV(basic_svm,
                               {"learning_rate": ["constant"],
                                "eta0": learning_rate_grid}).fit(X_pretest, y_pretest)
    bottou_gamma0 = pretest_svm.best_params_["eta0"]
    basic_svm.eta0 = bottou_gamma0
    basic_svm.learning_rate = "constant"

    basic_svm = basic_svm.partial_fit(X_pretest, y_pretest, classes = np.unique(y_train))

    progressive_val = []
    train_score = []
    for dp in range(0, X_train.shape[0], batch_size):
        t = dp + n_pretest
        basic_svm.eta0 = bottou_gamma0/(1 + bottou_gamma0*alpha*t)
        X_batch = X_train[dp:dp+batch_size]
        y_batch = y_train[dp:dp+batch_size]
        progressive_val.append(basic_svm.score(X_batch, y_batch))
        basic_svm = basic_svm.partial_fit(X_batch, y_batch)
        train_score.append(basic_svm.score(X_batch, y_batch))

    scores = progressive_val[-batches_for_cv_performance:]
    return np.mean(scores), np.std(scores), basic_svm


alpha_grid = [ 5.0, 1.0, 0.5, 1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6 ]
cv_results = []
for alpha in alpha_grid:
    print(alpha)
    mean_score, std_score, svm = evaluate_svm(alpha)
    cv_results.append({"alpha": alpha, "mean": mean_score, "std": std_score, "svm": svm})

best_score = np.max([x['mean'] for x in cv_results])
best_score_index = np.where([x['mean'] >= best_score for x in cv_results])[0][0]
onesd_at_best = cv_results[best_score_index]['std']
score_to_beat = best_score - onesd_at_best
for i in range(best_score_index-1, -1, -1):
    if cv_results[i]['mean'] > score_to_beat:
        best_score_index = i

chosen_alpha = cv_results[best_score_index]["alpha"]
print("Chosen: ", chosen_alpha)
print("Best CV score: ", cv_results[best_score_index]["mean"])
chosen_svm = cv_results[best_score_index]["svm"]

os.makedirs("output/svc", exist_ok=True)

for i in range(10):
    this_hyperplane = 127*(chosen_svm.coef_[i]/np.max(np.abs(chosen_svm.coef_[i]))) + scaler.mean_
    cf.plot_image(this_hyperplane, "output/svc/archetype " + str(i) + ".png")

preds = chosen_svm.predict(X_test)
print("Test score: %.3f" % chosen_svm.score(X_test, y_test))

