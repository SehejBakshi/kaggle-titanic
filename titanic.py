import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df=pd.read_csv('train.csv')
#print(train_df.head())
test_df=pd.read_csv('test.csv')
#print(test_df.head())
combine=[train_df, test_df]

print(train_df.columns.values)

print(train_df.info())
print('_'*100)
print(test_df.info())
print(train_df.describe())



print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))


print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))


print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))


print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print('Before', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df=train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df=test_df.drop(['Ticket', 'Cabin'], axis=1)
combine=[train_df, test_df]

print('After', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

for d in combine:
    d['Title']=d.Name.str.extract('([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))

for s in combine:
    d['Title']=d['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don' 'Dr', 'Major', 'Rev', 'Sir', 'Jokheer', 'Dona'], 'Rare')
    d['Title']=d['Title'].replace('Mlle', 'Miss')
    d['Title']=d['Title'].replace('Ms', 'Miss')
    d['Title']=d['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping={'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
for d in combine:
    d['Title']=d['Title'].map(title_mapping)
    d['Title']=d['Title'].fillna(0)

print(train_df.head())


train_df=train_df.drop(['Name', 'PassengerId'], axis=1)
test_df=test_df.drop(['Name'], axis=1)
combine=[train_df, test_df]

print(train_df.shape, test_df.shape)

for d in combine:
    d['Sex']=d['Sex'].map({'female':1, 'male':0}).astype(int)

print(train_df.head())


guess_ages=np.zeros((2,3))
print(guess_ages)


for d in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df=d[(d['Sex']==i) & \
                        (d['Pclass']==j+1)]['Age'].dropna()
            age_guess=guess_df.median()
            guess_ages[i,j]=int(age_guess/0.5+0.5)*0.5

    for i in range(0, 2):
        for j in range(0,3):
            d.loc[(d.Age.isnull()) & (d.Sex==i) & (d.Pclass==j+1), \
            'Age']=guess_ages[i,j]

    d['Age']=d['Age'].astype(int)

print(train_df.head())

train_df['AgeBand']=pd.cut(train_df['Age'], 5)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))


for d in combine:
    d.loc[d['Age']<=16, 'Age']=0
    d.loc[(d['Age']>16) & (d['Age']<=32), 'Age']=1
    d.loc[(d['Age'] > 32) & (d['Age'] <= 48), 'Age'] = 2
    d.loc[(d['Age'] > 48) & (d['Age'] <= 64), 'Age'] = 3
    d.loc[d['Age']>64, 'Age']
print(train_df.head())

train_df=train_df.drop(['AgeBand'], axis=1)
combine=[train_df, test_df]
print(train_df.head())


for d in combine:
    d['FamilySize']=d['SibSp']+d['Parch']+1

print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

for d in combine:
    d['IsAlone']=0
    d.loc[d['FamilySize']==1, 'IsAlone']=1
print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

train_df=train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df=test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine=[train_df, test_df]

print(train_df.head())


for d in combine:
    d['Age*Class']=d.Age*d.Pclass

print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))


freq_port=train_df.Embarked.dropna().mode()[0]
print(freq_port)


for d  in combine:
    d['Embarked']=d['Embarked'].fillna(freq_port)

print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

for d in combine:
    d['Embarked']=d['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

print(train_df.head())

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
print(test_df.head())

train_df['FareBand']=pd.qcut(train_df['Fare'], 4)
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))


for d in combine:
    d.loc[ d['Fare'] <= 7.91, 'Fare'] = 0
    d.loc[(d['Fare'] > 7.91) & (d['Fare'] <= 14.454), 'Fare'] = 1
    d.loc[(d['Fare'] > 14.454) & (d['Fare'] <= 31), 'Fare']   = 2
    d.loc[ d['Fare'] > 31, 'Fare'] = 3
    d['Fare'] = d['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print(train_df.head(10))
print(test_df.head(10))


x_train=train_df.drop('Survived', axis=1)
y_train=train_df['Survived']
x_test=test_df.drop('PassengerId', axis=1).copy()

print(x_train.shape, y_train.shape, x_test.shape)

logreg=LogisticRegression()
logreg.fit(x_train, y_train)
y_pred=logreg.predict(x_test)
acc_log=round(logreg.score(x_train, y_train)*100, 2)
print(acc_log)

coeff_df=pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns=['Feature']
coeff_df['Correlation']=pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))


svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
acc_svc=round(svc.score(x_train, y_train)*100, 2)
print(acc_svc)


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)
acc_knn=round(knn.score(x_train, y_train) * 100, 2)
print(acc_knn)


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression'],
    'Score': [acc_svc, acc_knn, acc_log]})
print(models.sort_values(by='Score', ascending=False))


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)
