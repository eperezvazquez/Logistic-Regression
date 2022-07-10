import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn import linear_model
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv',sep=';')
data.info()
data.sample(10)
round(df.head())
data.describe()
len(data)
dummies = pd.get_dummies(data['y'], drop_first = True)
dummies.tail()
data['y'].hist()
fig = px.box(data, y="age")
fig.show()
# scatter plot y según edad
ax1 = data[data['y'] == 'no'].plot(kind='scatter', x='age', y='y', color='green', alpha=0.5, figsize=(8,6))
plt.legend(labels=['no', 'yes'])
plt.title('Relationship between Age and y', size=18)
plt.xlabel('Age', size=12)
plt.ylabel('y', size=12);
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True,cmap='viridis', vmax=1, vmin=-1, center=0)
# scatter plot y según pervious y emp.var.rate
ax1 = data.plot(kind='scatter', x='previous', y='previous', color='green', alpha=0.5, figsize=(8,6))
#plt.legend(labels=['no', 'yes'])
plt.title('Relationship between previous and emp.var.rate', size=18)
plt.xlabel('previous', size=12)
plt.ylabel('emp.var.rate', size=12);
# scatter plot y según pdays y campaign
ax1 = data.plot(kind='scatter', x='pdays', y='campaign', color='green', alpha=0.5, figsize=(8,6))
#plt.legend(labels=['no', 'yes'])
plt.title('Relationship between pdays and campaign', size=18)
plt.xlabel('pdays', size=12)
plt.ylabel('campaign', size=12);
# scatter plot y según y y campaign
ax1 = data.plot(kind='scatter', x='campaign', y='y', color='green', alpha=0.5, figsize=(8,6))
#plt.legend(labels=['no', 'yes'])
plt.title('Relationship between y and campaign', size=18)
plt.xlabel('campaign', size=12)
plt.ylabel('y', size=12);
data['education'].unique()
data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])
data['education'].unique()
data['y'].value_counts()
sns.countplot(x='y',data=data,palette='hls')
plt.show()
plt.savefig('count_plot')
data.groupby('y').mean()
data.groupby('job').mean()
data.groupby('marital').mean()
data.groupby('education').mean()
%matplotlib inline
pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')
table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')
table=pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('edu_vs_pur_stack')
pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')
pd.crosstab(data.month,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_month_bar')
data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')
pd.crosstab(data.poutcome,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_pout_bar')
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]
data_final.columns.values
X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
