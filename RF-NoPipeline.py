
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
import sys



#### Import Data
################
store  = pd.HDFStore('data.h5')
resp = store['resp']

xtrain = store['xtrain_trim']
ytrain = resp['r']

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)
store.close()



#### loading Power feature table
################################
xpower = store['PowerFeatures']
resp = store['resp']
ytrain = resp['r']


########################
### Tunning parameters
########################


#### tfidf transformation
#########################
tf = TfidfTransformer()
tf_fit = tf.fit(xpower)
trans = tf.fit_transform(xpower)
xtrain = trans.toarray()
sys.exit('Stop before rerunning gridsearch')




### RF
##########
rf = RandomForestClassifier(, n_jobs = -1)

parameters ={'n_estimators': (100, 300, 450),
			'max_depth': (100, 300),
			 'max_features': (10,20, 30)}

gridsearch = GridSearchCV(rf, param_grid = parameters,verbose = 1)



#### CheckingResults
######################
gridsearch.fit(xtrain[:10000], ytrain[:10000])
gridsearch.best_score_

best_parameters = gridsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
        print  "\t%s: %r" % (param_name, best_parameters[param_name])



### Running model with tuned
############################

rf = RandomForestClassifier(n_estimators = 30, n_jobs = -1, verbose = 1)
kf= KFold(len(rpower[:10000]), n_folds=10, random_state=4)


### Conducting kfold CV
#######################

l = []
ind = []
for i,j in kf:
	ind.append(j)
	q = rf.fit(xtrain[i],ytrain[i])
	l.append(rf.predict(xtrain[j]))
	print rf.predict(xtrain[j])


### CHecking Accuracy
####################
for i in range(0,len(l)):
	print np.mean(l[i] == ytrain[ind[i]])
	

###Outputting Data
####################


pred_q6_power = pd.DataFrame({
	'preds1':l[0],
	'preds2':l[1],
	'preds3':l[2],
	'preds4':l[3],
	'preds5':l[4],
	'preds6':l[5],
	'preds7':l[6],
	'preds8':l[7],
	'preds9':l[8],
	'preds10':l[9],
	})

val_q6_power = pd.DataFrame({
	'val1':ytrain[ind[0]],
	'val2':ytrain[ind[1]],
	'val3':ytrain[ind[2]],
	'val4':ytrain[ind[3]],
	'val5':ytrain[ind[4]],
	'val6':ytrain[ind[5]],
	'val7':ytrain[ind[6]],
	'val8':ytrain[ind[7]],
	'val9':ytrain[ind[8]],
	'val10':ytrain[ind[9]],
	})


pred_q6_power.to_csv('pred_q6_power.csv')
val_q6_power.to_csv('val_q6_power.csv')

