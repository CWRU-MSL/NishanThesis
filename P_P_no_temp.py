# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 13:19:08 2022

@author: 14198
"""

"""
Created on Wed Aug 17 17:13:52 2022

@author: 14198
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

Tesnsile_7FA = pd.read_csv("Data/7FA-Tensile.csv")
Tesnsile_9FA = pd.read_csv("Data/9FA-Tensile.csv")   
Final_Tensile = Tesnsile_7FA.append(Tesnsile_9FA )

#Data Preprocessing


strain = pd.read_csv("Data/Final_strain.csv")
grain_size= pd.read_csv("Data/Final_Micro.csv")

grain_size = grain_size.groupby(['Forging', 'Radial Location']).mean().reset_index()


Final = pd.merge(Final_Tensile, strain, how="outer", on=["Forging", "Radial Location"])

Final = pd.merge(Final, grain_size, how="outer", on=["Forging", "Radial Location"])

Final= Final.drop(['Forging', 'Serial Number', 'Forge Year','Age','Stock'], axis=1)


#Final_750 = Final[Final.Temperature!=73]
#Final_750.to_csv('Final_Process_750.csv')
#Final_73 = Final[Final.Temperature==73]

Final=pd.get_dummies(Final,columns=["Quench"])
Final= pd.get_dummies(Final, columns=["Chemistry"])
Final = pd.get_dummies(Final, columns=["Heat Treat Cycle"])
Final= pd.get_dummies(Final, columns=["Radial Location"])
Final = pd.get_dummies(Final, columns=["Orientation"])
Final = pd.get_dummies(Final, columns=["Finish Forge Heat Up"])
Final = pd.get_dummies(Final, columns=["Type"])
Final = pd.get_dummies(Final, columns=["Configuration"])
#fill nan values in agular location with mean value
Final['Angular Location'].fillna((Final['Angular Location'].mean()), inplace=True)

Final = Final.dropna(subset=['RA'])


Final_process_73= Final[Final.Temperature==73]


Final_Tensile= Final_process_73[['TS', 'Elon','RA','YS']]


Final_process_73= Final_process_73.drop(['Temperature','TS', 'Elon','RA','YS','Strain', 'Mean Radius Prime', 'Mean Radius Double Prime',
'Volume Fraction Prime', 'Volume Fraction Double Prime', 'Avg GS','Large GS'], axis=1)


#processing vs Property

name='YS'
y= Final_Tensile[name]*6.89 

X= Final_process_73
X_norm= ((X-X.min())/(X.max()-X.min()))
y_norm= ((y-y.min())/(y.max()-y.min()))
data_dmatrix = xgb.DMatrix(data=X_norm,label=y)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=402)
xg_reg = xgb.XGBRegressor()
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
preds_all=xg_reg.predict(X_norm)
rmse = np.sqrt(mean_squared_error(y_test, preds))
corr= np.corrcoef(y, preds_all)
corr=corr[0,1]
r_square=corr**2
print("RMSE: %f" % (rmse))
print("R2: %f" % (r_square))


feature_important = xg_reg.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())
data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features
data.reset_index(inplace=True)


#Removing outlier for YS

preds_all = np.delete(preds_all, np.argmin(preds_all))
np.argmin(y)
y=y.drop(labels=769)


font = {'family' : 'normal',
        'weight' : 'bold',
        'size': 12
        }



plt.figure(figsize=(6,6))
plt.plot(y,preds_all, 'ro', alpha=0.5, color="black")
plt.xlabel('Actual Yeild Strength (MPa)')
plt.ylabel('Predicted Yeild Strength (MPa)')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
fmin = min(xmin, ymin)
fmax = max(xmax, ymax)
plt.xlim(fmin, fmax)
plt.ylim(fmin, fmax)
y_lim = plt.ylim()
x_lim = plt.xlim()
plt.rc('font', **font)
plt.rc('axes', labelsize=18)
plt.plot(x_lim, y_lim, 'k-', color = 'b')
plt.savefig(name+"_no_temp_73_pp.png", dpi=300, bbox_inches = 'tight')
plt.show()





