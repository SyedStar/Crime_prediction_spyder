import pandas as pd # panda's nickname is pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from  sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plot
from sklearn import linear_model
from IPython.display import Image 
from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
import sklearn.preprocessing as prep
import pydotplus 
import math
from sklearn.linear_model import RidgeCV
import sklearn.linear_model as linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import data
df_d = pd.read_csv('communities-crime-full.csv')
df_d.head()


def setHighCrime(df):
    '''Function to set value of highCrime depending on ViolentCrimesPerPop'''
    if df['ViolentCrimesPerPop'] > 0.1:
        return True
    else:
        return False
    

# Making data clean 

df_d_clean = df_d.replace('?', np.nan)
df_d_clean = df_d_clean.drop('community', axis=1).drop('state', axis=1).drop('communityname', axis=1).drop('fold', axis=1).drop('county', axis=1)
cols_numeric = df_d_clean.columns
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
Y = imp.fit_transform(df_d_clean)
df_d_clean = pd.DataFrame(Y)
df_d_clean.columns = cols_numeric
df_d_clean['highCrime'] = df_d_clean.apply(setHighCrime, axis=1)

percentage_intances = df_d_clean.groupby('highCrime').size() * 100 / len(df_d_clean)
print(percentage_intances)
print("------------------")
print("Percentage Positive Instance = {}\nPercentage Negative Instance = {} ".format(percentage_intances[1],percentage_intances[0]))

y = df_d_clean["highCrime"]
X = df_d_clean.drop('highCrime', axis=1).drop('ViolentCrimesPerPop', axis=1)

X_d = X
y_d = y
x_train, x_test, y_train, y_test = train_test_split(X_d, y_d,test_size=0.2,random_state=0)


clf = RandomForestClassifier(random_state=100, max_depth=3)
clf.fit(x_train,y_train)
rf_accuracy = cross_val_score(clf, X_d, y_d, cv=10, scoring='accuracy').mean()
rf_precision = cross_val_score(clf, X_d, y_d, cv=10, scoring='precision').mean()
rf_recall = cross_val_score(clf, X_d, y_d, cv=10, scoring='recall').mean()

#Predicting 
print ('Accuracy for RandomForestClassifier is', rf_accuracy)
print ('Precision for RandomForestClassifier is', rf_precision)
print ('Recall for RandomForestClassifier is', rf_recall)

#population	householdsize	racepctblack	racePctWhite	racePctAsian	racePctHisp	agePct12t21	agePct12t29	agePct16t24	agePct65up	numbUrban	pctUrban	medIncome	pctWWage	pctWFarmSelf	pctWInvInc	pctWSocSec	pctWPubAsst	pctWRetire	medFamInc	perCapInc	whitePerCap	blackPerCap	indianPerCap	AsianPerCap	OtherPerCap	HispPerCap	NumUnderPov	PctPopUnderPov	PctLess9thGrade	PctNotHSGrad	PctBSorMore	PctUnemployed	PctEmploy	PctEmplManu	PctEmplProfServ	PctOccupManu	PctOccupMgmtProf	MalePctDivorce	MalePctNevMarr	FemalePctDiv	TotalPctDiv	PersPerFam	PctFam2Par	PctKids2Par	PctYoungKids2Par	PctTeen2Par	PctWorkMomYoungKids	PctWorkMom	NumIlleg	PctIlleg	NumImmig	PctImmigRecent	PctImmigRec5	PctImmigRec8	PctImmigRec10	PctRecentImmig	PctRecImmig5	PctRecImmig8	PctRecImmig10	PctSpeakEnglOnly	PctNotSpeakEnglWell	PctLargHouseFam	PctLargHouseOccup	PersPerOccupHous	PersPerOwnOccHous	PersPerRentOccHous	PctPersOwnOccup	PctPersDenseHous	PctHousLess3BR	MedNumBR	HousVacant	PctHousOccup	PctHousOwnOcc	PctVacantBoarded	PctVacMore6Mos	MedYrHousBuilt	PctHousNoPhone	PctWOFullPlumb	OwnOccLowQuart	OwnOccMedVal	OwnOccHiQuart	RentLowQ	RentMedian	RentHighQ	MedRent	MedRentPctHousInc	MedOwnCostPctInc	MedOwnCostPctIncNoMtg	NumInShelters	NumStreet	PctForeignBorn	PctBornSameState	PctSameHouse85	PctSameCity85	PctSameState85	LemasSwornFT	LemasSwFTPerPop	LemasSwFTFieldOps	LemasSwFTFieldPerPop	LemasTotalReq	LemasTotReqPerPop	PolicReqPerOffic	PolicPerPop	RacialMatchCommPol	PctPolicWhite	PctPolicBlack	PctPolicHisp	PctPolicAsian	PctPolicMinor	OfficAssgnDrugUnits	NumKindsDrugsSeiz	PolicAveOTWorked	LandArea	PopDens	PctUsePubTrans	PolicCars	PolicOperBudg	LemasPctPolicOnPatr	LemasGangUnitDeploy	LemasPctOfficDrugUn	PolicBudgPerPop
a =	np.array([[0.01,0.55,0.02,0.95,0.09,0.05,0.38,0.38,0.23,0.36,0.02,0.9,0.5,0.72,0.16,0.68,0.44,0.11,0.71,0.46,0.43,0.41,0.28,0.0,0.74,0.51,0.48,0.0,0.06,0.25,0.3,0.33,0.12,0.65,0.67,0.38,0.42,0.46,0.22,0.27,0.2,0.21,0.51,0.91,0.91,0.89,0.85,0.4,0.6,0.0,0.06,0.0,0.03,0.07,0.2,0.27,0.01,0.02,0.04,0.05,0.88,0.05,0.16,0.19,0.59,0.6,0.37,0.89,0.02,0.19,0.5,0.01,0.89,0.87,0.04,0.55,0.73,0.05,0.14,0.31,0.31,0.3,0.4,0.36,0.38,0.38,0.22,0.51,0.21,0.0,0.0,0.11,0.72,0.64,0.61,0.53,0.06965517241379311,0.21746081504702197,0.9247335423197492,0.2463322884012539,0.09799373040752352,0.21520376175548586,0.34363636363636363,0.21749216300940438,0.6894043887147336,0.7269592476489029,0.22047021943573672,0.13485893416927902,0.114858934169279,0.25918495297805644,0.0755485893416928,0.556050156739812,0.305987460815047,0.04,0.09,0.02,0.16310344827586207,0.07670846394984326,0.69858934169279,0.44043887147335425,0.0,0.1950783699059561]])

pred_dt1= clf.predict(a)

print("Prediction: {}".format(pred_dt1))
