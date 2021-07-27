import pandas as pd
import os
from imblearn.under_sampling import  RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

#print(os.getcwd())

df_review = pd.read_csv('IMDB Dataset.csv')

# sample 10000 rows
df_positive = df_review[df_review['sentiment']=='positive'][:9000]
df_negative = df_review[df_review['sentiment']=='negative'][:1000]
df_review_imb = pd.concat([df_positive, df_negative])

# imbalanced dataset!
# resample using imblearn 
rus = RandomUnderSampler(random_state=0)
df_review_bal, df_review_bal['sentiment']=rus.fit_resample(df_review_imb[['review']], df_review_imb['sentiment'])

print(df_review_imb.value_counts('sentiment'))
print(df_review_bal.value_counts('sentiment'))


train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)

train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

# text to numerical representation using BOW TF-IDF - care about weighted freq of words
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)

# diplay 1340 reviews by 20625 words matrix
print(pd.DataFrame.sparse.from_spmatrix(train_x_vector,index=train_x.index,columns=tfidf.get_feature_names()))

# Models
# SVM
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

# Decision Tree
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

# Naive Bayes
gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

# Model Evaluation

# svc.score('Test samples', 'True labels')
svc_score = svc.score(test_x_vector, test_y)
dec_score = dec_tree.score(test_x_vector, test_y)
naive_score = gnb.score(test_x_vector.toarray(), test_y)
logreg_score = log_reg.score(test_x_vector, test_y)
print('SVM: ', svc_score, 'Decision tree: ' , dec_score, 'Naive Bayes: ', naive_score, 'Logistic Regression: ', logreg_score)

# f1 score
f1_score(test_y, svc.predict(test_x_vector),labels=['positive', 'negative'],average=None)

# classification report
print(classification_report(test_y, svc.predict(test_x_vector),labels=['positive', 'negative']))

# confusion matrix
conf_mat = confusion_matrix(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'])
print(conf_mat)

# tuning the model

#set the parameters
parameters = {'C': [1,4,8,16,32] ,'kernel':['linear', 'rbf']}
svc = SVC()
svc_grid = GridSearchCV(svc,parameters, cv=5)

svc_grid.fit(train_x_vector, train_y)
print(svc_grid.best_params_)
print(svc_grid.best_estimator_)