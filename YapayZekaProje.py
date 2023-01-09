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


#korelasyon çizdir (tüm değerlerin birbiri üzerindeki  etkisi)


correlation_matrix = cancer_patient.corr().round(2)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(data=correlation_matrix, annot=True)


#Sütun Grafiği Çizer Araştır yorumlanmasını 


box=["Age","Gender","Air Pollution","Alcohol use","Dust Allergy","OccuPational Hazards"
     ,"Genetic Risk","chronic Lung Disease","Balanced Diet",
    "Obesity","Smoking","Passive Smoker",]
cancer_patient[box].plot(kind="box", subplots="True" , layout=(4, 3) ,figsize=(8,8))




##Veri Setinin ilk 24 sütunu alınıp X değişkenine,Son sütununu Y değişkenine atar
#Veri Setini %20 tes %80 eğitim olarak ayırır. random_state değeri belirlenerek rastgele ayrışma engellenmiştir

X=cancer_patient.iloc[:,1:24].values
y=cancer_patient.iloc[:,24].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)





#SVM modeli uygulanmıştır.

# X_test verisi için tahminler yapar. Tahminler pred değişkenine aktarılır
# Verilen "X_train" ve "y_train" veri öğelerine dayalı olarak 4 farklı kernel (çekirdek) türünü kullanan
# 4 adet destek vektör makinesi(SVM) modeli oluşturur. Bu modeller, verilen "X_test" veri öğesiyle tahminler yapar ve
# bu tahminlerin doğruluğu değerlendirilir.
# SVM modeli oluşturulur ve "kernel" parametresine göre kernel seçilir.
# "classification_report" fonksiyonu kullanılarak modelin tahminleriyle
# gerçek sınıf etiketleri arasındaki doğruluk oranı ölçülür ve ekrana yazdırılır
# "f1_score" fonksiyonu kullanılarak modelin tahminlerinin F1 skoru hesaplanır ve ekrana yazdırılır.
# "jaccard_score" fonksiyonu kullanılarak modelin tahminlerinin Jaccard skoru hesaplanır ve ekrana yazdırılır


for kernel in ['linear','poly','sigmoid','rbf']:
    model = svm.SVC(kernel=kernel).fit(X_train, y_train)
    pred=model.predict(X_test)
    print('The classification report of svm of kernel =',kernel,'is :\n')
    print(classification_report(pred,y_test),'\n')
    print("Avg F1-score: %.4f" % f1_score(y_test, pred, average='weighted',zero_division="warn"))
    print("Jaccard score: %.4f" % jaccard_score(y_test, pred, average='micro'),'\n\n')
    


## Linear classification modeli




# Bu kod parçacığı, veri setindeki,(Hava Kirliliği),(Alkol Kullanımı),(Toz Alerjisi),(Mesleki Tehlikeler), (Genetik Risk),
# (Dengeli Beslenme), (Kan İçeren Öksürük), (Şişmanlık),  (Sigara İçme),(Pasif Sigara İçici),  (Yorgunluk),
# (Tırnakların Kümelenmesi),(Kuru Öksürük), (Horlamak),  (Yaş),(Nefes Darlığı), ve (Yutma Güçlüğü) özniteliklerini içeren
# bir veri çerçevesi oluşturur ve bu çerçevenin hedef değişkenini "Level" (Seviye) olarak ayarlar.
# bu değişkenler kullanılarak bir kanser risk modeli oluşturuluyor

x=cancer_patient[["Air Pollution","Alcohol use","Dust Allergy","OccuPational Hazards","Genetic Risk","Balanced Diet","Coughing of Blood",
    "Obesity","Smoking","Passive Smoker","Fatigue","Clubbing of Finger Nails","Dry Cough"
       ,"Snoring","Age","Wheezing","Swallowing Difficulty"]]
y=cancer_patient[['Level']]


#Veri Seti test ve eğitim olarak ayırma işlemi
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# Bu kod parçacığı, verilen eğitim verisi x_train ve hedef değişkenleri y_train kullanarak bir "Perceptron" sınıflandırıcısını
# eğitir. Perceptron, bir sinir ağı modeli olarak düşünülebilir ve lineer olarak ayrıştırılabilen sınıflar için kullanılır.
# Daha sonra, eğitilmiş sınıflandırıcıyı test verisi x_test üzerinde kullanarak tahminler yapar ve bu tahminlerin
# doğruluk oranını accuracy_score() fonksiyonu ile ölçer.

classifier = Perceptron(random_state=10)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print('Linear classification accuracy_score : ',accuracy_score(y_test, y_pred))






##  logistic classification modeli 

# Veri seti test ve eğitim olarak ayrılır
# verilen eğitim verisi x_train ve hedef değişkenleri y_train kullanarak bir "Logistic Regression" modeli oluşturur
# ve bu modeli eğitir. Lojistik Regresyon, bir sınıflandırma modelidir bir veri noktasının hangi sınıfa ait olduğunu tahmin eder

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)
model2 = linear_model.LogisticRegression().fit(x_train, y_train)


# lojistik regresyon modelini (model2) test verisi olarak x_test üzerinde kullanarak tahminler yapar.
# bu tahminlerin doğruluk oranını gerçek değerleri y_test ile karşılaştırarak ölçer.
# Bu doğruluk oranını accuracy_score() fonksiyonu ile hesaplar ve sonucu ekrana yazdırır.
 
y_pred = model2.predict(x_test)
print('logistic classification accuracy_score : ',metrics.accuracy_score(y_test, y_pred))




#Regression Logistıque score


#veri setinde bulunan 'Patient Id' sütununu kaldırır. inplace parametresi True olduğu için, bu değişiklikler
#orijinal veri çerçevesine yapılır ve yeni bir veri çerçevesi oluşturulmaz.
#axis=1 parametresi, sütunları (yani axis=1) etkileyeceğini belirtir. 
cancer_patient.drop(['Patient Id'], axis=1, inplace=True)
cancer_patient.head()




# Öncelikle, sample() fonksiyonu ile veri setinden rasgele seçilen bir örnekleme yapılır ve bu örneklemenin yüzde kaçının 
# eğitim verisi olarak kullanılacağı frac parametresi ile belirtilir.frac=0.8 demek, veri setinden
# yüzde 80'inin eğitim verisi olarak kullanılacağı anlamına gelir.
# drop() fonksiyonu ile eğitim verisi için seçilen satırlar data_test veri çerçevesinden çıkarılır ve geriye kalan satırlar
# data_test veri çerçevesine atanır. 


data_train = cancer_patient.sample(frac = 0.8, random_state=1)
data_test = cancer_patient.drop(data_train.index)



# eğitim ve test verilerini ayrıştırmak için kullanılır. X_train ve X_test değişkenleri, hedef değişkenleri olmayan öznitelikleri
# içerir. Bu değişkenler, modelin öğrenme sürecinde kullanılacağı verilerdir. Y_train ve Y_test değişkenleri ise
# hedef değişkenleri içerir ve bu değişkenler, modelin tahminlerinin doğruluğunu değerlendirirken kullanılacak verilerdir.
# Bu kod parçacığının etkisi, veri setinin eğitim ve test verilerine ayrıştırılmasıdır.
# Eğitim verileri X_train ve Y_train değişkenlerine atanır ve test verileri X_test ve Y_test değişkenlerine atanır.
X_train = data_train.drop(['Level'], axis=1)
Y_train = data_train['Level']
X_test = data_test.drop(['Level'], axis=1)
Y_test = data_test['Level']



# verilen öznitelikleri (X_train ve X_test) normalize etmek için kullanılır. Bu işlemin amacı, öznitelikler arasında
# değerler aralığının aynı olmasını sağlamaktır. Bu sayede, modelin öğrenme süreci daha doğru bir şekilde gerçekleşebilir.
# Öznitelikler normalize edilirken, MinMaxScaler sınıfı kullanılır ve bu sınıf, özniteliklerin değerlerini 0 ile 1 arasına
# normalize eder. feature_range parametresi, normalize edilen özniteliklerin aralığını belirtir.
# Bu kod parçacığının etkisi, X_train ve X_test özniteliklerinin normalize edilmesidir.

scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# verilen eğitim verisi X_train ve hedef değişkenleri Y_train kullanarak bir lojistik regresyon modeli oluşturur ve bu modeli
# eğitir. Bu model, verilen eğitim verisiyle çalışır ve modele ait parametreleri öğrenir
# Bu parametreler, daha sonra test verisi üzerinde kullanılarak tahminler yapmaya yardımcı olur.
# Daha sonra, eğitilmiş model X_test verisi üzerinde kullanılarak tahminler yapılır ve bu tahminler Y_lr değişkenine atanır.

lr = LogisticRegression()
lr.fit(X_train, Y_train) 
Y_lr = lr.predict(X_test)



# lojistik regresyon modelinin (Y_lr) doğruluk oranını ölçer. 
# doğruluk oranını accuracy_score() fonksiyonu ile hesaplar ve sonucu ekrana yazdırır.

score_LR = accuracy_score(Y_test, Y_lr)
print("Régression Logistique score : ",score_LR)








#Optimizasyon

# Bu kod, Keras'ta Sekansiyel model oluşturur. Sekansiyel model, katmanların doğrusal bir yığınıdır.
# Model üç katmana sahiptir:
# İlk katman, 10 birim ve ReLU aktivasyon fonksiyonunu kullanır.
# İkinci katman, 5 birim ve ayrıca ReLU aktivasyon fonksiyonunu kullanır.
# Üçüncü katman, 3 birim ve sigmoid aktivasyon fonksiyonunu kullanır.
# Aktivasyon fonksiyonu, katmanın çıkışının doğrusal dönüşümünün çıktısına uygulanan bir nonlinear fonksiyondur ve
# katmanın çıkışını elde etmek için kullanılır. ReLU (Düzeltilmiş Lineer Birim) aktivasyon fonksiyonu 
# f(x) = max(0, x) şeklinde tanımlanır. Sigmoid aktivasyon fonksiyonu ise f(x) = 1/(1 + exp(-x)) şeklinde tanımlanır.

model = Sequential()
model.add(Dense(10, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(3, activation = 'sigmoid')) #softmax seulement à la dernière couche


# Bu kod parçacığı, modelin eğitim sürecini yapılandırır ve eğitim işlemini başlatır.
# "compile()" metodu, modelin derlenmesini sağlar ve modeli derleme işlemini yapılandırır. Bu metodun iki parametresi vardır:
# "loss" parametresi, modelin eğitim sırasında kullanılacak kayıp fonksiyonunu belirtir.
# "sparse_categorical_crossentropy" kayıp fonksiyonu, ikili (binary) veya çok sınıflı (multi-class) sınıflandırma
# problemleri için kullanılır. Bu fonksiyon, tahmin edilen ve gerçek sınıf değerleri arasındaki farklılığı ölçer.

# "optimizer" parametresi, modelin ağırlıklarını güncelleme sırasında kullanılacak optimizasyon fonksiyonunu belirtir.
# "Adam" optimizasyon fonksiyonu, genellikle iyi bir performans gösterir ve yaygın olarak kullanılır.

# "metrics" parametresi, modelin performansını ölçmek için kullanılacak metrikleri belirtir.
# Bu kod parçacığında, "accuracy" metriği kullanılmıştır. Bu metrik, tahminlerin doğru sınıflara ait olma yüzdesini ölçer.

# "fit()" metodu, modelin eğitim işlemini başlatır. Bu metodun birçok parametresi vardır:
# "X_train" ve "Y_train" parametreleri, modelin eğitim verilerini ve eğitim hedeflerini temsil eder.

# "validation_data" parametresi, modelin performansının test edileceği veri kümesini ve hedeflerini belirtir.

# "epochs" parametresi, modelin kaç kez eğitim verisi üzerinden geçeceğini belirtir. Her bir epoch, tüm eğitim verisi
# üzerinden geçiş anlamına gelir.
# "verbose" parametresi, eğitim sırasında hangi bilgilerin ekrana yazdırılacağını belirtir. Bu kod parçacığında,
# "verbose=1" olduğundan eğitim sırasında ilerleme bilgisi ekrana yazdırılacaktır.

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train , Y_train , validation_data=(X_test,Y_test), epochs=50, verbose=1)


















