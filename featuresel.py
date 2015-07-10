import pandas as pd 
import numpy as np 
from ggplot import *
import sys



### load data
###############
store = pd.HDFStore('data.h5')
xtrain = store['xtrain']
store.close()


### Unsupervised feature selection
##################################
def nonzero(datas):
	count = []
	for i in range(0,datas.shape[1]):
		count.append(sum(datas.ix[:,i]))
	return count

count = nonzero(xtrain)


graphs = pd.DataFrame({'counts':count,'names':list(xtrain)})

### Scatterplot
###############
scatter = ggplot(data = graphs, aes( x = range(0,len(count)), y ='counts')) +\
geom_point() + xlab("Index") + ylab("Non-Zero Counts") +\
ggtitle("Scatter plot of \n Non-Zero Appearances") + \
xlim(0,len(count)) + ylim(0,max(count))

### Histogram (Bug right now, need to do in R)
##############################################
#ggplot( aes(x ='counts'), data = graphs)
#geom_histogram(binwidth = 250)
 
### Choosing threshold and then removing features
#################################################
thresh = [10 < i  <= 1000 for i in count]
sum(thresh)

### Subsetting data
####################
xtrain2 = xtrain.ix[:, thresh]
store['xtrain_trim'] = xtrain2










