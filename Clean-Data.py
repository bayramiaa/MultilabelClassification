from collections import Counter
import pandas as pd 
import numpy as np 
import string
import re
import time
start_time = time.time()


'''
Removing patterns within w1 and w2
StripInline: stays within a line (No /n in the middle)
StripBlock: spans multiple lines
'''


def StripInline(s, w1 = '<code', w2 = '</code>'):
    return re.sub(w1 + '((?!' + w1 + ').)*?' + w2, '', s, 
                flags = re.MULTILINE | re.S)

def StripBlock(s, w1 = '<pre', w2 = '</pre>'):
    return re.sub('^' + w1 + '((?!' + w1 + ').)*' + w2 + '$', '', s, 
                flags = re.MULTILINE | re.S)

###############################
### Data and cleaning data
###############################

data = pd.read_csv('train.csv')

body = list(data['Body'])
title = list(data['Title'])
tag = list(data['Tags'])

def clean(lists):
	output = []
	for i in lists:
		s1 = re.sub(r"\n|<p>|<\p>|<br>|\d", " ", i.lower())
		s2 = StripInline(s1, w1 = "<code>", w2 = '</code>')
		s3 = StripInline(s2, w1 = "<a", w2 = '</a>')
		s4 = StripInline(s3, w1 = "<pre>", w2 = '</pre>')
		s5 = StripInline(s4, w1 = "<img", w2 = '>')
		s6 = StripInline(s5, w1 = "<strong>", w2 = '</strong>')
		s7 = StripInline(s6, w1 = "\$\$", w2 = '\$\$')
		s8 = StripInline(s7, w1 = "\$", w2 = '\$')
		s9 = StripInline(s8, w1 = "<h1>", w2 = '</h1>')
		s10 = StripInline(s9, w1 = "<", w2 = '>')
		s11 = s10.translate(string.maketrans("",""), string.punctuation)
		s12 = re.sub(r"\s+",' ', s11)
		output.append(s12)
	return output

bodies = clean(body)
titles = clean(title)
tags = clean(tag)
###############################
### individual responses + muli
###############################

def indivy(tag):
	df = pd.DataFrame({'stat':[0]*len(tag),'machine':[0]*len(tag),\
	'r':[0]*len(tag),'numpy':[0]*len(tag),'math':[0]*len(tag), })
	for i in range(0,len(tag)):
		for j in tag[i].lower().split():
			if j == 'statistics':
				df.ix[i,'stat'] =1
			elif j == 'machine-learning':
				df.ix[i,'machine'] = 1
			elif j == 'r':
				df.ix[i,'r'] = 1
			elif j == 'numpy':
				df.ix[i,'numpy'] = 1
			elif j == 'math':
				df.ix[i,'math'] = 1
	return(df)
responses = indivy(tag)


### Combining body and title
############################

combined = []
for i in range(0,len(bodies)):
	combined.append(titles[i] + " " + bodies[i])


### Word counts and list dictionaries
######################################
counts = [Counter(re.findall(r'\w+', x)).most_common(12) for x in combined]
l = [dict(i) for i in counts]


### Data frame of word counts
#############################
df = pd.DataFrame(l)
df  = df.fillna(0)


### Raw Data Frame/ cleaned title body tags
###########################################
dfraw = pd.DataFrame({'Title':title, 'Body':body,"Tags":tag})
cleaned = pd.DataFrame({'Title':titles, 'Body':bodies,"Tags":tags})

### storing data as HDFS5 files
################################

store = pd.HDFStore('data.h5')
store['xtrain'] = df
store['ymulti'] = pd.DataFrame(ymulti)
store['dfraw'] = dfraw
store['cleaned'] = cleaned
store.close()

print("%f seconds" % (time.time() - start_time))


