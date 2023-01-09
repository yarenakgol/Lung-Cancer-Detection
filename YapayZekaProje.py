#Gerekli Kütüphanelerin Eklenmesi
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import svm,metrics,linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.metrics import classification_report,f1_score,jaccard_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#Veri Seti Okuma İşlemi yapıldı ve ayraç olarak virgül kullanıldı


filename = "dataSet.csv"
cancer_patient = pd.read_csv(filename,sep=',')
cancer_patient.head()


#Tahmin sütunumuzda bulunan kategorik değerler sayısal değerlere dönüştürüldü


label_map = {'Medium':1,'High':2,'Low':0}
labels = cancer_patient.Level
donusum =pd.Series(np.array([label_map[label]for label in labels]))
cancer_patient.Level = donusum


#Tahmin edeceğimiz Sütundaki Verilerinin dağılımının pasta grafiğine dökülmesi 


fig, ax = plt.subplots()
ax.pie(cancer_patient["Level"].value_counts() ,labels=["Low","Medium","High"],
       autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')
plt.show()


#korelasyon çizdirilmiştir


correlation_matrix = cancer_patient.corr().round(2)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(data=correlation_matrix, annot=True)


#Sütun Grafiği Çizdirilmiştir


box=["Age","Gender","Air Pollution","Alcohol use","Dust Allergy","OccuPational Hazards"
     ,"Genetic Risk","chronic Lung Disease","Balanced Diet",
    "Obesity","Smoking","Passive Smoker",]
cancer_patient[box].plot(kind="box", subplots="True" , layout=(4, 3) ,figsize=(8,8))




#Veri Seti test ve eğitim için ayırma işlemi yapılmıştır

X=cancer_patient.iloc[:,1:24].values
y=cancer_patient.iloc[:,24].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)





#SVM modeli uygulanmıştır.

for kernel in ['linear','poly','sigmoid','rbf']:
    model = svm.SVC(kernel=kernel).fit(X_train, y_train)
    pred=model.predict(X_test)
    print('The classification report of svm of kernel =',kernel,'is :\n')
    print(classification_report(pred,y_test),'\n')
    print("Avg F1-score: %.4f" % f1_score(y_test, pred, average='weighted',zero_division="warn"))
    print("Jaccard score: %.4f" % jaccard_score(y_test, pred, average='micro'),'\n\n')
    


## Linear classification modeli

x=cancer_patient[["Air Pollution","Alcohol use","Dust Allergy","OccuPational Hazards","Genetic Risk","Balanced Diet","Coughing of Blood",
    "Obesity","Smoking","Passive Smoker","Fatigue","Clubbing of Finger Nails","Dry Cough"
       ,"Snoring","Age","Wheezing","Swallowing Difficulty"]]
y=cancer_patient[['Level']]


#Veri Seti test ve eğitim olarak ayırma işlemi
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


classifier = Perceptron(random_state=10)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print('Linear classification accuracy_score : ',accuracy_score(y_test, y_pred))






##  logistic classification modeli 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)
model2 = linear_model.LogisticRegression().fit(x_train, y_train)


# lojistik regresyon modeli
 
y_pred = model2.predict(x_test)
print('logistic classification accuracy_score : ',metrics.accuracy_score(y_test, y_pred))




#Regression Logistıque score

cancer_patient.drop(['Patient Id'], axis=1, inplace=True)
cancer_patient.head()

data_train = cancer_patient.sample(frac = 0.8, random_state=1)
data_test = cancer_patient.drop(data_train.index)

X_train = data_train.drop(['Level'], axis=1)
Y_train = data_train['Level']
X_test = data_test.drop(['Level'], axis=1)
Y_test = data_test['Level']

scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, Y_train) 
Y_lr = lr.predict(X_test)

score_LR = accuracy_score(Y_test, Y_lr)
print("Régression Logistique score : ",score_LR)








#Optimizasyon

model = Sequential()
model.add(Dense(10, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(3, activation = 'sigmoid')) #softmax seulement à la dernière couche


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train , Y_train , validation_data=(X_test,Y_test), epochs=50, verbose=1)


















