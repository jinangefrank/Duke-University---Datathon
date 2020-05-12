import pandas as pd
import ast
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn import metrics

def load(file):

    df = pd.read_csv(file)
    
    # convert the column values from literal string to dictionary
    df['ltiFeatures'] = df['ltiFeatures'].apply(ast.literal_eval)
    df['stiFeatures'] = df['stiFeatures'].apply(ast.literal_eval)

    return df

# load all the data
training1 = load('/Academics/Fall 2/Datathon/datathon_dataset/training.csv')
validation = load('/Academics/Fall 2/Datathon/datathon_dataset/validation.csv')
interest_topics = pd.read_csv("/Academics/Fall 2/Datathon/datathon_dataset/interest_topics.csv")
training = pd.concat([training1,validation])

# 初步处理数据
se=training.iloc[:,1]
user=se.to_frame(name='Inaudience')
lti=training['ltiFeatures'].apply(pd.Series)
sti=training['stiFeatures'].apply(pd.Series)
stiname=sti.keys()
stiname=stiname+"s"
stinames=stiname.tolist()
sti.columns=stinames
data=pd.concat([user,lti, sti], axis=1, sort=False)
#----------------------------------------------------------------------------------------------
dataframes = data.fillna(0)

X = dataframes.drop(["Inaudience"],axis=1).values
Y = dataframes["Inaudience"].to_frame(name='Inaudience')
Y=Y+0
Y=Y.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

forest=RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=0)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
threshold=0.001
X_selected = X_train[:, importances > threshold]
dataset=dataframes[0:96406]
featurename=dataset.keys()
featurename=featurename[0:2998]
dictionary = dict(zip(featurename, importances))
selected_names= {key: value for key, value in dictionary.items() if value > threshold}
select_name=selected_names.keys()
select_names=[]
for i in select_name:
    select_names.append(i)

sorted(dictionary)

validation_size = 0.70
seed = 9

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_selected,y_train,test_size=validation_size, random_state=seed)

# Test options and evaluation metric

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
scoring = 'roc_auc'
results = []
names = []

for name, model in models:
 kfold = model_selection.KFold(n_splits=10, random_state=seed)
 cv_results = model_selection.cross_val_score(model, X_train2, Y_train2, cv=kfold, scoring=scoring)

 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)

# performance boxplot
ggwp = pd.DataFrame(results).T
ggwp.columns=names
boxplot = ggwp.boxplot()
plt.show()


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
output=lda.predict_proba(X_test)[:,1]

fpr, tpr, thresholds =roc_curve(y_test, lda.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='LDA (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

lr = LogisticRegression()
lr.fit(X_train, y_train)

fpr, tpr, thresholds =roc_curve(y_test, lr.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='LR (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()