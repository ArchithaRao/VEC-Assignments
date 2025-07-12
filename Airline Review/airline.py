import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,roc_auc_score,auc
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
ar=pd.read_csv(r"C:\Users\Architha Rao\Downloads\archive\Airline_Reviews.csv")
ar.head(5)
ar.shape
ar.info()
nar = ar.drop(['Inflight Entertainment', 'Wifi & Connectivity', 'Aircraft', 'Value For Money', 'Cabin Staff Service', 'Unnamed: 0','Review Date', 'Review_Title', 'Review'], axis=1)
nar['Overall_Rating']=nar['Overall_Rating'].replace(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'n'],['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
nar['Type Of Traveller']=nar['Type Of Traveller'].fillna(nar['Type Of Traveller'].mode()[0])
nar['Seat Type']=nar['Seat Type'].fillna(nar['Seat Type'].mode()[0])
nar['Seat Comfort']=nar['Seat Comfort'].fillna(nar['Seat Comfort'].mode()[0])
nar['Route']=nar['Route'].fillna(nar['Route'].mode()[0])
nar['Date Flown']=nar['Date Flown'].fillna(nar['Date Flown'].mode()[0])
nar['Food & Beverages']=nar['Food & Beverages'].fillna(nar['Food & Beverages'].mode()[0])
nar['Ground Service']=nar['Ground Service'].fillna(nar['Ground Service'].mode()[0])
# For the above columns we are using mode instead median even though numerical values are present
# Because the cloumn consists of categories (0 to 5). So its considered as categorial data
nar[['Month Flown', 'Year Flown']]=nar['Date Flown'].str.split(expand=True)
nar['Origin']=nar.Route.str.split('to',expand=True)[0]
nar['Destination']=nar.Route.str.split('to',expand=True)[1]
# Route column has 3 values i.e., Place A to Place B via Place C, so inorder to chose, we gave indices for Moroni as 0 & Moheli as 1,then run the split function again to remove 'via'
nar['Destination']=nar.Destination.str.split('via',expand=True)[0]
del nar['Route']
del nar['Date Flown']
nar['Origin']=nar['Origin'].replace(['Tel Avivito Malta (MLA)','Bangalore toChennai','JFK toTLV via Baku','Krabi toBangkok','Hong Kong To Shanghai',
                                     'Edinburgh To Fuerteventura','Nuremburg toHamburg','Mumbai toJaipur','Sydney to- New York via Soul',
                                     'London Gatwick - Bangkok','SIN toi MFM','Jakartato Yogyakarta','Cardiff-Malta return','KIV-LIS',
                                     'GRR-ORD','LCY-FRA','NAP-RMF return','LEB-BOS','Bucharest-Brussels','Da Nang - Hong Kong','New-York',
                                     'LHR-DXB','Dublin - Charlotte','Kansas City via Dallas Ft Worth','Sydney via Singapore',
                                     'Geneva via Brussels','Nursultan via Dubai','Denpasar Medan via Jakarta',
                                     'Auckland Denpasar via Sydney / Melborne','Lima via Santiago','Manila via Los Angeles', # removed extra '
                                     'Dar es Salaam via Kigali','Singapore via Sydney','Grand Rapidsvto Orlando via Chicago',
                                     'Toronto via Varadero','Bangkok via Mumbai','A Coruna via Bilbao','LHR-DXB',
                                     'Paris Orly Las Angeles','Newark Los Angeles','Honolulu Seattle','San Paulo'],
                                    ['Tel Aviv(MLA)','Bangalore','JFK','Krabi','Hong kong','Edinburgh','Nuremburg','Mumbai',
                                     'Sydney','London Gatwick','SIN','Jakarta','Cardiff','KIV','GRR','LCY','NAP','LEB','Bucharest',
                                     'Da Nang','New York','LHR','Dublin','Kansas City','Sydney','Geneva','Nursultan','Denpasar Medan',
                                     'Auckland Denpasar','Lima','Manila','Dar es Salaam','Singapore','Grand Rapidsvto Orlando',
                                     'Toronto','Bangkok','A Coruna','LHR','Paris Orly','newark','Honolulu','San Paulo'])
# Destination reconnections
j=0
row_num=[2172,3788,5112,5368,7000,8314,10589,12993,17759,20572,
         20930,2225,2380,4339,5182,5758,6382,10991,12573,17051,21497,
         4293,6215,9787,10207,12372,13556,16022,17217,17732,18774,
         19462,20112,22449,11584,10001,12258,10886]
new_des=['Malta','Chennai','TLV','Bangkok','Shanghai','Fuerteventura','Hamburg',
         'Jaipur','New York','Bangkok','MFM','Yogyakarta','Malta','LIS','ORD','FRA',
         'RMF','BOS','Brussels','Hong Kong','DXB','Charlotte','Dallas Ft Worth',
         'Brussels','Dubai','Jakarta','Sydney / Melbourne','Santiago','Los Angeles','Kigali',
         'Sydney','Chicago','Varadero','Mumbai','Bilbao','Dallas','Los Angeles','Los Angeles','Seattle ']
new_column_order=['Airline Name','Seat Type','Type Of Traveller','Origin',
                  'Destination','Month Flown','Year Flown','Verified','Seat Comfort',
                  'Food & Beverages','Ground Service','Overall_Rating','Recommended']
# Reordering the columns of given data to our desired manner
nar=nar.reindex(columns=new_column_order)
nar.head()
from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
le2=LabelEncoder()
le3=LabelEncoder()
le4=LabelEncoder()
le5=LabelEncoder()
le6=LabelEncoder()
le7=LabelEncoder()
le8=LabelEncoder()
le9=LabelEncoder()
le10=LabelEncoder()
nar['Airline Name']=le1.fit_transform(nar['Airline Name'])
nar['Seat Type']=le2.fit_transform(nar['Seat Type'])
nar['Type Of Traveller']=le3.fit_transform(nar['Type Of Traveller'])
nar['Origin']=le4.fit_transform(nar['Origin'])
nar['Destination']=le5.fit_transform(nar['Destination'])
nar['Month Flown']=le6.fit_transform(nar['Month Flown'])
nar['Year Flown']=le7.fit_transform(nar['Year Flown'])
nar['Verified']=le8.fit_transform(nar['Verified'])
nar['Overall_Rating']=le9.fit_transform(nar['Overall_Rating'])
nar['Recommended']=le10.fit_transform(nar['Recommended'])
nar.head()
nar.describe()
# Line plot in seaborn
#line plot in seaborn
sns.set(rc={'figure.figsize': [15,15]})
sns.set(font_scale=1.5)
fig = sns.lineplot(x=nar.index, y=nar['Seat Type'], markevery=1, marker='d',
                    hue=nar['Seat Comfort'])
fig.set(xlabel='index')
plt.figure(figsize=(5,5))
plt.pie(nar['Seat Type'].value_counts(), startangle=90, autopct='%.3f', labels=['E_c', 'B_c', 'P_e', 'F_c'], shadow=True)
sns.barplot(data=nar, x="Type Of Traveller", y="Overall_Rating")
plt.subplots(figsize=(12,12))
sns.heatmap(nar.corr(),annot=True)
sns.jointplot(nar)
X=nar.iloc[:,0:12].values
y=nar.iloc[:, 12:13].values
X
y
nar. Recommended.value_counts()
# As the values are over_sampling we need to use smote technique
from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy='auto', random_state=50)
X,y=smote.fit_resample(X,y)
np.count_nonzero(y==1)
np.count_nonzero(y==0)
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
import pickle
pickle.dump(ss,open('ar_ss.pkl', 'wb'))
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy', random_state=50)
dtc.fit(X_train,y_train)
pred_dt=dtc.predict(X_test)
pred_dt
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, roc_curve, auc
fpr_dt, tpr_dt, threshold_dt=roc_curve(y_test, pred_dt)
print(classification_report(y_test, pred_dt))
roc_auc_dt=auc(fpr_dt, tpr_dt)
print("roc_auc_dt", roc_auc_dt)
cm_dt=confusion_matrix(y_test,pred_dt)
print("cm_dt:",cm_dt)
as_dt=accuracy_score(y_test, pred_dt)
print("as_dt:", as_dt)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
pred_knn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

fpr_knn, tpr_knn, threshold_knn = roc_curve(y_test, pred_knn)
print(classification_report(y_test, pred_knn))
roc_auc_knn = auc(fpr_knn, tpr_knn)
print("roc_auc_knn:", roc_auc_knn)
cm_knn=confusion_matrix(y_test, pred_knn)
print("cm_knn:",cm_knn)
as_knn=accuracy_score(y_test, pred_knn)
print("as_knn:",as_knn)
from sklearn.linear_model import LogisticRegression
Lr=LogisticRegression()
Lr.fit(X_train,y_train)
pred_Lr=Lr.predict(X_test)
pred_Lr
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
fpr_Lr, tpr_Lr, threshold_Lr=roc_curve(y_test, pred_Lr)
print(classification_report(y_test, pred_Lr))
roc_auc_Lr=auc(fpr_Lr, tpr_Lr)
print("roc_auc_Lr:", roc_auc_Lr)
cm_Lr=confusion_matrix(y_test, pred_Lr)
print("cm_Lr:",cm_Lr)
as_Lr=accuracy_score(y_test, pred_Lr)
print("as_Lr:",as_Lr)
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_nb=gnb.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_curve, auc
fpr_gnb, tpr_gnb, threshold_gnb=roc_curve(y_test, pred_Lr)
print(classification_report(y_test, pred_Lr))
roc_auc_nb=auc(fpr_gnb, tpr_gnb)
print("roc_auc_nb: ",roc_auc_nb)
cm_nb=confusion_matrix(y_test,pred_nb)
print("cm_nb:",cm_nb)
as_nb=accuracy_score(y_test, pred_nb)
print("as_nb:",as_nb)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=2)
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)
pred_rfc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
fpr_rfc, tpr_rfc, threshold_rfc=roc_curve(y_test,pred_rfc)
print(classification_report(y_test, pred_rfc))
roc_auc_rfc=auc(fpr_rfc, tpr_rfc)
print("roc_auc_rfc", roc_auc_rfc)
cm_rfc=confusion_matrix(y_test, pred_rfc)
print("cm_rfc:",cm_rfc)
as_rfc=accuracy_score(y_test, pred_rfc)
print("as_rfc:",as_rfc)
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
pred_svc=svc.predict(X_test)
pred_svc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_curve, auc
fpr_svc,tpr_svc,threshold_svc=roc_curve(y_test, pred_svc)
print(classification_report(y_test, pred_svc))
roc_auc_svc=auc(fpr_svc, tpr_svc)
print("roc_auc_svc:", roc_auc_svc)
cm_svc=confusion_matrix(y_test, pred_svc)
print("cm_svc:",cm_svc)
as_svc=accuracy_score (y_test, pred_svc)
print("as_svc:",as_svc)
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(X_train,y_train)
pred_xgb=xgb.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
fpr_xgb, tpr_xgb, threshold_xgb=roc_curve (y_test, pred_xgb)
print(classification_report(y_test, pred_xgb))
roc_auc_xgb=auc(fpr_xgb, tpr_xgb)
print("roc_auc_xgb:", roc_auc_xgb)
cm_xgb=confusion_matrix(y_test, pred_xgb)
print("cm_xgb:",cm_xgb)
as_xgb=accuracy_score(y_test, pred_xgb)
print("as_xgb:",as_xgb)
pred_xgb1=xgb.predict(X_train)
print(classification_report(y_train, pred_xgb1))
xgb.predict([[4,71,1,3,900,1133,1,6,1,5,5,5]])
#As there is a very less difference between accuracies of training and testing models, there is no issue of overfitting
com=pd.DataFrame({'Model':['DecissionTree Classification','K-Nearest Neighbours',
                    'Logistic Regression','Naive Bayes Classification',
                    'Random Classification','Support Vector Machine','XGBClassifier'],
           'roc_auc':[roc_auc_dt,roc_auc_knn,roc_auc_Lr,roc_auc_nb,roc_auc_rfc,roc_auc_svc,roc_auc_xgb],
           'accuracy':[as_dt,as_knn,as_Lr,as_nb,as_rfc,as_svc,as_xgb]})
com
maxi=0
for i in range(len(com['Model'])):
    if com.iloc[i:i+1,1:2].values>maxi:
      maxi=com.iloc[i:i+1,1:2].values
      model=com.iloc[i:i+1,0:1].values
    else:
        pass
print('Best accuracy score is:', maxi, 'by', model)
maxi=0
for i in range(len(com['Model'])):
    if com.iloc[i:i+1,2:3].values>maxi:
      maxi-com.iloc[i:i+1,2:3].values
      model=com.iloc[i:i+1,0:1].values
    else:
        pass
print('Best roc_auc is:',maxi, 'by', model)
import pickle
pickle.dump(xgb,open('ar_xgb.pkl', 'wb'))
