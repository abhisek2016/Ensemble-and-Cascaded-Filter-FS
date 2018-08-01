

import pandas as pd
import numpy as np

abcd=pd.read_csv("C:/Users/Arunava/Desktop/Feature Selection/New Datasets/Leukaemia.csv")


print("Shape")
print(abcd.shape)


array2=abcd.values


X1=array2[:,:5147]
Y1=array2[:,5147]


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import SelectFdr
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS

#from __future__ import division
from scipy import stats
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


# MI Top 100

test2=SelectKBest(score_func=mutual_info_classif)
fit2=test2.fit(X1,Y1)
p=fit2.scores_
org=[]
S=fit2.scores_
for q in range(0,5147):
    org.append(p[q])
S.sort()
top=[]
for d in range(0,100):
    top.append(S[5146-d])
#print(top)
W=[]
for h in range(0,100):
    for i in range(0,5148):
        if(org[i]==top[h]):
            W.append(i)
            break
#print("Mutual info top100")
#print(W)


# F_classif

test5=SelectKBest(score_func=f_classif)
fit5=test5.fit(X1,Y1)
r=fit5.scores_
l=[]
s=fit5.scores_
for f in range(0,5147):
    l.append(r[f])
s.sort()
top5=[]
for d in range(0,100):
    top5.append(s[5146-d])
#print(top5)
w=[]
for h in range(0,100):
    for i in range(0,5148):
        if(l[i]==top5[h]):
            w.append(i)
            break
#print("F class_if top100")
#print(w)


# Ttest

ttest=stats.ttest_ind(X1,Y1)
pr=ttest.statistic.tolist()
pr.sort(reverse=True)
topPr=[]
for i in range(0,100):
    topPr.append(pr[i])
u=ttest.statistic
u1=u.tolist()
WT=[]
for h in range(0,100):
    for i in range(0,5147):
        if(u1[i]==topPr[h]):
            WT.append(i)
            break
#print("T test top100")
#print(WT)

# new 
union = []
union = W+w+WT
print("Total Length")
print(len(union))
union = np.unique(union)
print("Unique Length")
print('\n', len(union))
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
x_train1, x_test1, y_train1, y_test1 = train_test_split(X1,Y1,test_size= 0.2,random_state=5)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(x_train1,y_train1) 
pred=neigh.predict(x_test1)
print("Normal Accuracy")
print(accuracy_score(y_test1,pred))
final = abcd.iloc[:,union]

# new one
x_train2, x_test2, y_train2, y_test2 = train_test_split(final,Y1,test_size= 0.2,random_state=5)
neigh2 = KNeighborsClassifier(n_neighbors=7)
neigh2.fit(x_train2,y_train2) 
pred2=neigh2.predict(x_test2)
print("Filtered Accuracy")
print(accuracy_score(y_test2,pred2))

###########Cascaded Filtering First

test7=SelectKBest(score_func=mutual_info_classif)
fit7=test7.fit(X1,Y1)
r7=fit7.scores_
l7=[]
s7=fit7.scores_
for f in range(0,1000):
    l7.append(r7[f])
s7.sort()
top7=[]
for d in range(0,1000):
    top7.append(s7[999-d])
W7=[]
for h in range(0,700):
    for i in range(0,1000):
        if(l7[i]==top7[h]):
            W7.append(i)
            break
##2nd

X8= abcd.iloc[:,W7]
ttest2=stats.ttest_ind(X8,Y1)
pr2=ttest2.statistic.tolist()
#print(len(pr2))
pr2.sort(reverse=True)
topPr2=[]
for i in range(0,700):
    topPr2.append(pr2[i])
u2=ttest2.statistic
u3=u2.tolist()
WT2=[]
for h in range(0,700):
    for i in range(0,5147):
        if(u3[i]==topPr2[h]):
            WT2.append(i)
            break
#3rd filter

X3=abcd.iloc[:,WT2]
test6=SelectKBest(score_func=f_classif)
fit6=test6.fit(X3,Y1)
p6=fit6.scores_
org6=[]
S6=fit6.scores_
for q in range(0,700):
    org6.append(p6[q])
S6.sort()
top6=[]
for d in range(0,300):
    top6.append(S6[299-d])
W6=[]
for h in range(0,300):
    for i in range(0,5148):
        if(org6[i]==top6[h]):
            W6.append(i)
            break
#######
final3 = abcd.iloc[:,W6]
x_trainC, x_testC, y_trainC, y_testC = train_test_split(final3,Y1,test_size= 0.2,random_state=5 )
neighC = KNeighborsClassifier(n_neighbors=7)
neighC.fit(x_trainC,y_trainC) 
predC=neighC.predict(x_testC)
print("Cascaded Filter Accuracy")
print(accuracy_score(y_testC,predC))


 
