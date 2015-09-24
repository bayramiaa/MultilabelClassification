from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from nltk import bigrams, trigrams
from data_utils import *
import pandas as pd 
import numpy as np 



### Load Data
##############
docs_train = np.array(pd.read_csv('train.csv'))[:,1:]
docs_test = np.array(pd.read_csv('XtestKaggle2.csv'))[:,1:]


## train
Y = np.array(response(docs_train[:,2]))
x_train = map(lambda x: ''.join(x), docs_train[:,:2])
x_train = map(clean, x_train)

## test
X_test = map(lambda x: ''.join(x), docs_test)
X_test = map(clean, X_test)



### WC Matrix
#############
vec = CountVectorizer(ngram_range=(1,2), stop_words='english', max_features = 35000)
tfidf = TfidfTransformer()
vocab = vec.fit(x_train).get_feature_names()
X_train = vec.fit_transform(x_train)
X_train =  tfidf.fit_transform(X_train).toarray()



vec = CountVectorizer(ngram_range = (1,2), stop_words = 'english', vocabulary = vocab)
X_test = vec.fit_transform(X_test).toarray()
X_test = tfidf.fit_transform(X_test)



val = np.array([Y[20000:,0],Y[20000:,1],Y[20000:,2],Y[20000:,3],Y[20000:,4]])

### RFC
#########
from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier(n_estimators = 100, n_jobs = -1,verbose = 1)
p0 = rf1.fit(X_train,Y[:,0]).predict(X_test)
p1 = rf1.fit(X_train,Y[:,1]).predict(X_test)
p2 = rf1.fit(X_train,Y[:,2]).predict(X_test)
p3 = rf1.fit(X_train,Y[:,3]).predict(X_test)
p4 = rf1.fit(X_train,Y[:,4]).predict(X_test)

rf_pred = np.array([p0,p1,p2,p3,p4])

#for i in range(5):
#	print accuracy_score(rf_pred[i], val[i])

np.savetxt("preds/rf_test.csv", rf_pred, delimiter=",")


### SVM
#######
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier( alpha = 1e-06,n_iter = 50)
p0_svm = svm.fit(X_train,Y[:,0]).predict(X_test)
svm = SGDClassifier(alpha = 1e-05, n_iter = 30)
p1_svm = svm.fit(X_train,Y[:,1]).predict(X_test)
svm = SGDClassifier(alpha = 1e-06, n_iter = 30, loss = 'log')
p2_svm = svm.fit(X_train,Y[:,2]).predict(X_test)
svm = SGDClassifier(alpha = 1e-05, n_iter = 75)
p3_svm = svm.fit(X_train,Y[:,3]).predict(X_test)
svm = SGDClassifier(alpha = 1e-07, loss = 'log', n_iter = 25)
p4_svm = svm.fit(X_train,Y[:,4]).predict(X_test)

svm_pred = np.array([p0_svm, p1_svm, p2_svm, p3_svm, p4_svm])



#for i in range(5):
#	print accuracy_score(svm_pred[i], val[i])

np.savetxt("preds/SVM_test.csv", svm_pred, delimiter=",")

### Logistic
############
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()

p0_log = logr.fit(X_train,Y[:,0]).predict(X_test)
p1_log = logr.fit(X_train,Y[:,1]).predict(X_test)
p2_log = logr.fit(X_train,Y[:,2]).predict(X_test)
p3_log = logr.fit(X_train,Y[:,3]).predict(X_test)
p4_log = logr.fit(X_train,Y[:,4]).predict(X_test)

logr_pred = np.array([p0_log, p1_log, p2_log, p3_log, p4_log])
#for i in range(5):
#	print accuracy_score(logr_pred[i], val[i])

np.savetxt("preds/logr_test.csv", logr_pred, delimiter=",")



###### voting on classes
########################
def ensemble(m1 = logr_pred, m2 = svm_pred, m3= rf_pred):
	l = []
	for i in range(5):
		vote = np.zeros(m1.shape[1], dtype = 'int')
		for ID, tup in enumerate(np.array(zip(m1,m2,m3))[i].T):
			if sum(tup) >= 2:
				vote[ID] = int(1)
		l.append(vote)		

	return np.array(l)

vote_pred = ensemble()



tags = ['statistics', 'machine-learning', 'r', 'numpy', 'math']
pred = pd.DataFrame(vote_pred.T, columns = tags, index = range(1,26426))
pred.to_csv('preds/test_sub.csv')

pred.to_csv('preds/poop.csv')
### GBC
#######

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators =5, verbose = 1)
p0_gbc = gbc.fit(X_train[:20000],Y[:20000,0]).predict(X_train[20000:])
p1_gbc = gbc.fit(X_train[:20000],Y[:20000,1]).predict(X_train[20000:])
p2_gbc = gbc.fit(X_train[:20000],Y[:20000,2]).predict(X_train[20000:])
p3_gbc = gbc.fit(X_train[:20000],Y[:20000,3]).predict(X_train[20000:])
p4_gbc = gbc.fit(X_train[:20000],Y[:20000,4]).predict(X_train[20000:])

gbc_pred = np.array([p0_gbc, p1_gbc, p2_gbc, p3_gbc, p4_gbc])
for i in range(5):
	print accuracy_score(gbc_pred[i], val[i])

np.savetxt('preds/gbc_pred.csv', gbc_pred, delimiter = ",")


