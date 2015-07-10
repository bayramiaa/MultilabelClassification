from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import os
import sys


store = pd.HDFStore('data.h5')

resp = store['resp']
xtrain = store['comb_clean']
xtrain =  np.array(xtrain[0])

yr = np.array(resp['r'])
ymachine = np.array(resp['machine'])
ymath = np.array(resp['math'])
ynump = np.array(resp['numpy'])
ystat = np.array(resp['stat'])
store.close()

sys.exit('Features loaded')


###Tune parameters with 
###gridsearch and pipeline
##########################

pipeline = Pipeline([('vect', CountVectorizer(ngram_range = (1,2))), 
                   ('tfidf', TfidfTransformer(sublinear_tf = True)),
                  ('clf', SGDClassifier())])

parameters ={
    'clf__n_iter': (70),
    'clf__sublinear_tf': (True,),
			 }


grid_search =  GridSearchCV(pipeline, param_grid = parameters,verbose = 1)
grid_search.fit(xtrain, ytrain)
grid_search.best_score_

best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
        print  "\t%s: %r" % (param_name, best_parameters[param_name])

kf = KFold(len(yr), n_folds = 5, shuffle = True)


### Loading Kaggle set
######################
store.open()
xtest = store['xkaggle']
xtest = np.array(xtest[0])
store.close()



### Binary case for R .95
##########################
text_clf = Pipeline([('vect', CountVectorizer(max_df = .7, min_df = 2,
                    ngram_range = (1,2))), 
                     ('tfidf', TfidfTransformer(sublinear_tf = True)),
                     ('clf', SGDClassifier(alpha = 1e-06, 
                        n_iter = 30, loss = 'log'))])



l = []
idr = []
for i,j in kf:
    t = text_clf.fit(xtrain[i],yr[i]).predict(xtrain[j])
    l.append( t )
    idr.append(j)


### Binary case for Machine .98
###############################
text_clf = Pipeline([('vect', CountVectorizer(max_df = .25 ,ngram_range = (1,2))), 
                     ('tfidf', TfidfTransformer( )),
                     ('clf', SGDClassifier(alpha = 1e-05,
                        n_iter = 30))])


lmachine = []
for i,j in kf:
    t = text_clf.fit(xtrain[i],ymachine[i]).predict(xtrain[j])
    lmachine.append(t)
    print 'machine',np.mean(t == ymachine[j])




### Binary case for numpy .98
#############################
text_clf = Pipeline([('vect', CountVectorizer(max_df = .25 ,ngram_range = (1,2))), 
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(alpha = 1e-05,
                        n_iter = 75))])

lnump = []
for i,j in kf:
    t = text_clf.fit(xtrain[i],ynump[i]).predict(xtrain[j])
    lnump.append(t)
    print 'numpy',np.mean(t == ynump[j])


### Binary case for math .947
###############################
text_clf = Pipeline([('vect', CountVectorizer(max_df = .8,
                        ngram_range = (1,2))), 
                     ('tfidf', TfidfTransformer(sublinear_tf = True)),
                     ('clf', SGDClassifier(alpha = 1e-07, loss = 'log',
                        n_iter = 25))])

lmath = []
for i,j in kf:
    t = text_clf.fit(np.array(xtrain)[i],ymath[i]).predict(xtrain[j])
    lmath.append(t)
    print 'math',np.mean(t == ymath[j])


###Binary case for stat .954
#############################


text_clf = Pipeline([('vect', CountVectorizer(max_df = .65 ,ngram_range = (1,2))), 
                     ('tfidf', TfidfTransformer(sublinear_tf = True)),
                     ('clf', SGDClassifier( alpha = 1e-06,n_iter = 50))])


lstat = []
for i,j in kf:
    t = text_clf.fit(xtrain[i],ystat[i]).predict(xtrain[j])
    lstat.append(t)
    print 'stat', np.mean(t == ystat[j])



### Kaggle Submission
#####################


q10pred = pd.DataFrame({
    'statistics':lstat[1],
    'machine-learning':lmachine[1],
    'r':l[1],
    'numpy':lnump[1],
    'math':lmath[1] })

q10val = pd.DataFrame({
    'statistics':ystat[idr[1]],
    'machine-learning':ymachine[idr[1]],
    'r':yr[idr[1]],
    'numpy':ynump[idr[1]],
    'math':ymath[idr[1]],
    })

q10val.to_csv('q10val1')
q10pred.to_csv('q10pred1')


resp = store['resp'].ix[,:]










