import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('Churn_Modelling.csv')

print("First 5 rows of the dataset:")
print(df.head())

print("\n Dataset information:")
print(df.info())

print("Missing values in the dataset:")
print(df.isnull().sum())

print("Duplicate rows in the dataset:")
print(df[df.duplicated()])

#one-hot encoding for categorical variables and label encoding for the 'Gender' column

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)


print("First 5 rows of the dataset after preprocessing:")
print(df.head())

# Define features and target variable for model training and then split the data into training and testing sets
features =['CreditScore', 'Gender','Age', 'Tenure','Balance','NumOfProducts','HasCrCard', 'IsActiveMember', 'EstimatedSalary','Geography_Germany', 'Geography_Spain']
X = df[features]
y = df['Exited']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler to improve the performance of the model
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("First 5 rows of the scaled test set:")
print(x_test[:5])
print("First 5 rows of the scaled training set:")
print(x_train[:5])


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

print(" Classification Report, Confusion Matrix, and Accuracy Score:")
conf_matrix=confusion_matrix(y_test, y_pred)
class_report=classification_report(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)

print(conf_matrix)
print(class_report)
print(f"Accuracy: {accuracy:.2f}")

# Plotting feature importance to visualize which features are most influential in predicting customer churn

importances =model.feature_importances_
indices =np.argsort(importances)[::-1] #    sort the feature importances in descending order
names = [features[i] for i in indices] # get the feature names based on the sorted indices


plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.barh(range(X.shape[1]), importances[indices]) # create a horizontal bar plot of feature importances and set the y-ticks to the feature names
plt.yticks(range(X.shape[1]),names) # set the y-ticks to the feature names
#plt.show()

# comparing the performance of the Random Forest model with a Logistic Regression model to see if there is any improvement in accuracy
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_train, y_train) # fit the logistic regression model to the training data
y_pred_log = log_reg.predict(x_test) # predict the target variable for the test set using the logistic regression model

conf_matrix_log_reg= confusion_matrix(y_test,y_pred_log)
class_report_log_reg=classification_report(y_test,y_pred_log)
accuracy_log_reg=accuracy_score(y_test,y_pred_log)


print(conf_matrix_log_reg,class_report_log_reg,accuracy_log_reg)

from sklearn.svm import SVC


svm_model= SVC(kernel='linear',random_state=42)
svm_model.fit(x_train,y_train)
y_pred_svm=svm_model.predict(x_test)

conf_matrix_svm= confusion_matrix(y_test,y_pred_svm)
class_report_svm=classification_report(y_test,y_pred_svm)
accuracy_svm=accuracy_score(y_test,y_pred_svm)

print(conf_matrix_svm,class_report_svm,accuracy_svm)

from sklearn.metrics import precision_score


#fixing the zero division error by setting zero_division=0 in the precision_score function for the SVM model. This works 
#by assigning a precision score of 0 to any class that has no predicted samples, which prevents the error from occurring and allows us to evaluate the performance of the SVM model without interruption.
precision_svm = precision_score (y_test, y_pred_svm, average='macro',zero_division=0)
print(f"Precision score for SVM: {precision_svm:.2f}")
 
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
y_pred_knn = knn_model.predict(x_test)

conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
class_report_knn = classification_report(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)   


print("K-Nearest Neighbors Model Performance:")
print(conf_matrix_knn, class_report_knn, accuracy_knn)

#Finding metrics for GBM - Gradient Boosting Machine model to see if it performs better than the previous models in predicting customer churn

from sklearn.ensemble import GradientBoostingClassifier
gbm_model=GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(x_train, y_train)

y_pred_gbm=gbm_model.predict(x_test)

conf_matrix_gbm=confusion_matrix(y_test, y_pred_gbm)
class_report_gbm=classification_report(y_test, y_pred_gbm)
accuracy_gbm=accuracy_score(y_test, y_pred_gbm)

print("Gradient Boosting Machine Model Performance:")
print(conf_matrix_gbm, class_report_gbm, accuracy_gbm)

#feature engineering by creating new features based on existing ones to see if it improves the performance of the model in predicting customer churn

df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1) # create a new feature that represents the ratio of balance to estimated salary

df['AgeGroup']=pd.cut(df['Age'], bins=[18,25,35,45,55,65,75,85,95], labels=['18-25','26-35','36-45','46-55','56-65','66-75','76-85','86-95']) # create a new feature that categorizes customers into age groups based on their age

df['BalanceZero'] = (df['Balance'] == 0).astype(int) # create a new binary feature that indicates whether a customer has a zero balance or not

df['ProductUsage'] = df['NumOfProducts'] * df['IsActiveMember'] # create a new feature that represents the total product usage by multiplying the number of products with the active member status

#tenure grouping
#this works by categorizing customers into different tenure groups based on their tenure with the bank. The pd.cut function is used to create these groups by specifying the bins and labels for each group. This new feature can help the model capture any patterns or trends related to customer tenure that may be relevant for predicting churn.
df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0,2,4,6,8,10], labels=['0-2','3-4','5-6','7-8','9-10']) # create a new feature that categorizes customers into tenure groups based on their tenure with the bank 


# label_encoder = LabelEncoder()
# df['Gender'] = label_encoder.fit_transform(df['Gender'])
# df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
df['Male_Germany'] = df['Gender'] * df['Geography_Germany']
df['Male_Spain'] = df['Gender'] * df['Geography_Spain']

df = pd.get_dummies(df, columns=['AgeGroup', 'TenureGroup'], drop_first=True)

#defines new features through engineering. if it starts with 'AgeGroup_' or 'TenureGroup_', it includes those features in the list of features to be used for model training. This allows the model to capture any patterns or trends related to age and tenure groups that may be relevant for predicting customer churn.
features =['CreditScore', 'Balance', 'EstimatedSalary', 'NumOfProducts', 'IsActiveMember', 'BalanceSalaryRatio', 'BalanceZero', 'ProductUsage', 'Male_Germany',
            'Male_Spain'] + [col for col in df.columns if col.startswith('AgeGroup_')] + [col for col in df.columns if col.startswith('TenureGroup_')]


X = df[features]
y = df['Exited']

print(df.head())

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

print(" Classification Report, Confusion Matrix, and Accuracy Score:")
conf_matrix=confusion_matrix(y_test, y_pred)
class_report=classification_report(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)

print(conf_matrix)
print(class_report)
print(f"Accuracy: {accuracy:.2f}")

import pickle #importing the pickle library to save the trained model to a file for later use. This allows us to reuse the model without having to retrain it every time we want to make predictions on new data.

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
