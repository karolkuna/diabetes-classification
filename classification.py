
# coding: utf-8

# In[1]:

import numpy as np
import sklearn
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:

#data = pandas.read_csv('dataset_diabetes/diabetic_data.csv', ',')


# In[3]:

df = pd.DataFrame.from_csv('dataset_diabetes/diabetic_data.csv')


# In[4]:

df.head()


# # Preprocessing

# In[5]:

df.readmitted.unique()


# In[6]:

# transform readmitted categories into numerical values
#readmittedEncoder = preprocessing.LabelEncoder()
#readmittedEncoder.fit(df.readmitted)
#df.readmitted = readmittedEncoder.transform(df.readmitted)


# In[7]:

df.gender.unique()


# In[8]:

# transform genders into numerical values (female = 0, male = 1, unknown = 2)
genderEncoder = preprocessing.LabelEncoder()
genderEncoder.fit(df.gender)
df.gender = genderEncoder.transform(df.gender)


# In[9]:

df.age.unique()


# In[10]:

# transforms range string into a numerical value in the middle of the range
def rangeStringToMiddle(rangeString):
    rangeString = rangeString[1:-1] #remove brackets
    rangeString = str.split(rangeString, '-')
    return (int(rangeString[0]) + int(rangeString[1])) / 2


# In[11]:

rangeStringToMiddle('[70-80)')


# In[12]:

# transform age range into one numerical value
df.age = df.age.apply(rangeStringToMiddle)


# In[13]:

df.weight.unique()


# In[14]:

# transforms age range string into an age in the middle of the range
def weightRangeToWeight(weightRange):
    if weightRange == '?':
        return 0
    if weightRange == '>200':
        return 200
    return rangeStringToMiddle(weightRange)


# In[15]:

# transform weight range into one numerical value
df.weight = df.weight.apply(weightRangeToWeight)


# In[16]:

df.weight.unique()


# In[17]:

df.race.unique()


# In[18]:

df.admission_type_id.unique()


# In[19]:

df.admission_source_id.unique()


# In[20]:

df.payer_code.unique()


# In[21]:

df.discharge_disposition_id.unique()


# In[22]:

df.medical_specialty.unique()


# In[23]:

df.diag_1 = preprocessing.LabelEncoder().fit_transform(df.diag_1)
df.diag_2 = preprocessing.LabelEncoder().fit_transform(df.diag_2)
df.diag_3 = preprocessing.LabelEncoder().fit_transform(df.diag_3)


# In[24]:

# columns containing nominal values that will be split into categorical columns (one-hot)
categoricalColumns = ['readmitted', 'race', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'payer_code', 'medical_specialty']


# In[25]:

testResultsAndMedications = ['max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed'] # + ['diag_1', 'diag_2', 'diag_3']


# In[26]:

# encode binary values into 0s and 1s, and multi-values into categorical columns
binaryTMColumns = []
nominalTMColumns = []
for colName in testResultsAndMedications:
    if len(df[colName].unique()) <= 2:
        binaryTMColumns.append(colName)
        df[colName] = preprocessing.LabelEncoder().fit_transform(df[colName])
    else:
        nominalTMColumns.append(colName)
        categoricalColumns.append(colName)


# In[27]:

# create separate categorical column for each unique value
for columnName in categoricalColumns:
    oneHotColumns = pd.get_dummies(df[columnName], prefix=(columnName + '_'), prefix_sep='')
    df = df.join(oneHotColumns)


# In[28]:

df.head()


# In[29]:

for col in df.columns:
    print(col)


# # Training

# In[31]:

from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[49]:

featureColumns = binaryTMColumns + ['diag_1', 'diag_2', 'diag_3', 'age', 'weight', 'gender', 'time_in_hospital', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient']
nominalColumns = ['race', 'admission_type_id', 'change'] + nominalTMColumns

for col in df.columns:
    for colPrefix in nominalColumns:
        if col.find(colPrefix + '_') == 0:
            featureColumns.append(col)
            
labelColumns = ['readmitted_<30', 'readmitted_>30', 'readmitted_NO']
features = df[list(featureColumns)].values
#labels = df[list(labelColumns)].values
labels = df['readmitted_<30'].values


# In[33]:

df.isnull().values.any()


# In[34]:

featureColumns


# In[205]:

features


# In[50]:

from sklearn.preprocessing import StandardScaler
features = StandardScaler().fit_transform(features)


# In[35]:

#from sklearn.decomposition import PCA
#pca = PCA(whiten=True)


# In[36]:

#pca.fit(100,features)
#pcaFeatures = pca.transform(features)


# In[129]:

clf = LinearDiscriminantAnalysis()
#clf = QuadraticDiscriminantAnalysis()
#clf = OneVsRestClassifier(LinearDiscriminantAnalysis())
#clf = OneVsRestClassifier(QuadraticDiscriminantAnalysis())


# In[80]:

#originalColumns = ['race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed']
#features = df[list(originalColumns)].values
clf = RandomForestClassifier(10, class_weight='balanced')


# In[53]:

from sklearn.neural_network import MLPClassifier
clf =  MLPClassifier(verbose=True, hidden_layer_sizes=(150, 100), tol=-0.1, learning_rate_init=0.001, max_iter=1000)


# In[39]:

#clf.fit(pcaFeatures,labels)


# In[40]:

#clf.coef_


# In[41]:

#cdf = pd.DataFrame({'feature' : featureColumns, 'coef' : clf.coef_[0]})
#cdf['coef_abs'] = cdf.coef.abs()
#cdf.sort_values('coef_abs')
#cdf


# In[ ]:

cv = StratifiedKFold(5, shuffle=True)
cross_val_score(clf, features, labels, scoring='accuracy',  n_jobs=-1, verbose=2, cv=cv)


# In[91]:

from sklearn.svm import SVC
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.
lda = LinearDiscriminantAnalysis()
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=lda, step=4, cv=StratifiedKFold(4),
              scoring='accuracy', verbose=2)


# In[ ]:

#svc.fit(features[1:10000],labels[1:10000])


# In[51]:

#rfecv.fit(features, labels)


# In[49]:

#print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")
#plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#plt.show()


# In[146]:

labels[400:700]


# In[147]:

clf.predict(features[400:700])


# In[430]:

np.unique(clf.predict(features))


# In[162]:

g = sns.factorplot(x='acetohexamide', data=df, hue='readmitted', kind='count', size=8)
g.set_axis_labels('x', 'y')


# In[ ]:



