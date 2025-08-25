##This Algorithm is to model a Logistic Regressor 
# to predict Customer churning status from an MTN nigeria data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
#loading the dataset
data = pd.read_csv(r'C:\Users\eigenX\Desktop\BACKUP\Data_Analytics\mtn_Customer_churn_Algorithm\mtn_customer_churn.csv')
print('Data Loaded')
##DATA CLEANING
#truncating the unnecessary columns in the data.
data = data.drop(['Customer ID','Full Name','Reasons for Churn'], axis=1)
#Encoding Categorical Variables chronologically
#Gender encoding ....Male=1,Female = 0 u gerit????
data['Gender'] = data['Gender'].map({'Male': 1,'Female': 0})
#customer review encoding from 1 to 5
review_mapping = {'Poor': 1, 'Fair': 2, 'Good' : 3, 'Very Good' : 4,'Excellent' : 5}
data['Customer Review'] = data['Customer Review'].map(review_mapping)
#encoding the date
data['Date of Purchase']=pd.to_datetime(data['Date of Purchase'], format='%b-%y')
data['Date of Purchase']= data['Date of Purchase'].dt.month
data = data.drop('Date of Purchase', axis=1)
#the subscription plan is encoded based on the price
#the unit price in this dataset,we donot know if its the price of the device or the price of the bundle
plan_price_mapping = data.groupby('Subscription Plan')['Unit Price'].mean().to_dict()
data['Subscription Plan Encoded'] = data['Subscription Plan'].map(plan_price_mapping)
#after encoding this data,we take down the Subcription plan
data = data.drop(['Subscription Plan'],axis = 1)
#encoding the state and the device
data = pd.get_dummies(data, columns = ['State','MTN Device'], drop_first= True)
#converting the churn status to binary...churned = 1 , non churn = 0
data['Customer Churn Status'] = data['Customer Churn Status'].map({'Yes': 1 , 'No' : 0})
#quick check for missing values if it was not checked and handled in excel
#print(data.isnull().sum())
#data = data.fillna(method='ffill')
print('Data Prepared for Machine Learning Algorithm: Logisic Regression')
print(data.head())
#Spliting data into target and features.
X = data.drop('Customer Churn Status', axis=1)
y = data['Customer Churn Status']
#spliting to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
#Scaling Features: TO BE UNDERSTOOD(pre modeling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
#design the model proper
logreg = LogisticRegression(class_weight='balanced',max_iter= 1000,random_state= 42)
logreg.fit(X_train_scaled,y_train)
#model evaluation
y_pred = logreg.predict(X_test_scaled)
print("\n====MODEL===EVALUATION====")
print(f"Accuracy: {accuracy_score(y_test,y_pred):.2f}")
print(classification_report(y_test,y_pred))
print("\nConfusion Matrix: ")
conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
#plotting the confusion matrix
print("\nClassification Report: ")
print(classification_report(y_test,y_pred))
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap='viridis')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()