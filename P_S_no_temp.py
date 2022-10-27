# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:32:07 2022

@author: 14198
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import nan
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

Tesnsile_7FA = pd.read_csv("Data/7FA-Tensile.csv")
Tesnsile_9FA = pd.read_csv("Data/9FA-Tensile.csv")   
Final_Tensile = Tesnsile_7FA.append(Tesnsile_9FA )

#Data Preprocessing


strain = pd.read_csv("Data/Thermo_calc_assumption-1.csv")
grain_size= pd.read_csv("Data/Final_Micro.csv")

grain_size = grain_size.groupby(['Forging', 'Radial Location']).mean().reset_index()


Final = pd.merge(Final_Tensile, strain, how="outer", on=["Forging", "Radial Location"])

Final = pd.merge(Final, grain_size, how="outer", on=["Forging", "Radial Location"])
#plot yield strength vs grian size





#processing vs microstructure

Final_process= Final.drop(['TS', 'Elon','RA','YS','Strain', 'Mean Radius Prime', 'Mean Radius Double Prime',
'Volume Fraction Prime', 'Volume Fraction Double Prime', 'Avg GS','Large GS'], axis=1)


Final_microstructure= Final[['Strain', 'Mean Radius Prime', 'Mean Radius Double Prime',
'Volume Fraction Prime', 'Volume Fraction Double Prime', 'Avg GS','Large GS', 'Temperature']]


gs_strain= pd.merge(grain_size, strain, how="outer", on=["Forging", "Radial Location"])

Final_tensile_tan= Final_Tensile[Final_Tensile.Orientation=="Tangential"]

Final_tensile_tan_0= Final_tensile_tan[Final_tensile_tan['Angular Location'] !=180]

Final_tensile_tan_0= Final_tensile_tan_0.dropna(subset=['RA'])

Final_micro_tensile = pd.merge(Final_tensile_tan_0, gs_strain, how="inner", on=["Forging", "Radial Location"])




Final_micro_tensile_73= Final_micro_tensile[Final_micro_tensile.Temperature!=73]



micro = Final_micro_tensile_73[['Strain', 'Mean Radius Prime', 'Mean Radius Double Prime',
'Volume Fraction Prime', 'Volume Fraction Double Prime', 'Avg GS','Large GS']]

micro =micro.iloc[0:234]
Final_micro_tensile_73=Final_micro_tensile_73[0:234]

name='Elon'
y=  Final_micro_tensile_73[name] #*6.89



X= micro
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
r_square=corr
print("RMSE: %f" % (rmse))
print("R2: %f" % (r_square))


feature_important = xg_reg.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())
data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features
data.reset_index(inplace=True)



font = {'family' : 'normal',
        'weight' : 'bold',
        'size': 12
        }

plt.figure(figsize=(6,6))
plt.plot(y,preds_all, 'ro', alpha=0.5, color="#ff6007")
plt.xlabel('Actual Elongation (%)')
plt.ylabel('Predicted Elongation (%)')
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
plt.savefig(name+"_no_temp_750_sp.png", dpi=300, bbox_inches = 'tight')
plt.show()

