# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 21:28:38 2022

@author: 14198
"""

#microstructure vs Property

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


Tesnsile_7FA = pd.read_csv("Data/7FA-Tensile.csv")
Tesnsile_9FA = pd.read_csv("Data/9FA-Tensile.csv")   
Final_Tensile = Tesnsile_7FA.append(Tesnsile_9FA )

#Data Preprocessing


strain = pd.read_csv("Data/Final_strain.csv")
grain_size= pd.read_csv("Data/Final_Micro.csv")

grain_size = grain_size.groupby(['Forging', 'Radial Location']).mean().reset_index()


Final = pd.merge(Final_Tensile, strain, how="outer", on=["Forging", "Radial Location"])

Final = pd.merge(Final, grain_size, how="outer", on=["Forging", "Radial Location"])
#plot yield strength vs grian size



gs_strain= pd.merge(grain_size, strain, how="outer", on=["Forging", "Radial Location"])

Final_tensile_tan= Final_Tensile[Final_Tensile.Orientation=="Tangential"]

Final_tensile_tan_0= Final_tensile_tan[Final_tensile_tan['Angular Location'] !=180]

Final_tensile_tan_0= Final_tensile_tan_0.dropna(subset=['RA'])

Final_micro_tensile = pd.merge(Final_tensile_tan_0, gs_strain, how="inner", on=["Forging", "Radial Location"])


Final_micro_tensile_73= Final_micro_tensile[Final_micro_tensile.Temperature==73]
Final_micro_tensile_73.to_csv('Carter/Final_S_P_73.csv')


micro = Final_micro_tensile[['Strain', 'Mean Radius Prime', 'Mean Radius Double Prime',
'Volume Fraction Prime', 'Volume Fraction Double Prime', 'Avg GS','Large GS', 'Temperature']]

micro =micro.iloc[0:373]
Final_micro_tensile=Final_micro_tensile[0:373]

name='YS'
y= Final_micro_tensile[name] *6.89



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
corr= np.corrcoef(y_test, preds)
corr=corr[0,1]
r_square=corr**2
print("RMSE: %f" % (rmse))
print("R2: %f" % (r_square))

#Feature Importance

feature_important = xg_reg.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())
data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features
data.reset_index(inplace=True)



#temperature Effect


temp=  X['Temperature']
temp= temp.reset_index()
temp= temp.drop('index', axis=1)  

y= y.reset_index()
y= y.drop('index', axis=1)  


preds_all=pd.DataFrame(preds_all)

final_res = pd.concat([y,preds_all,temp], axis=1)
final_res.rename(columns = {0:'pred'}, inplace = True)

final_res_73= final_res[final_res.Temperature==73]
final_res_900=final_res[final_res.Temperature!=73] 



rmse_73 = np.sqrt(mean_squared_error(final_res_73[name], final_res_73['pred']))
corr_73= np.corrcoef(final_res_73[name], final_res_73['pred'])
corr_73=corr_73[0,1]
r_square=corr_73**2
print("RMSE: %f" % (rmse_73))
print("R2: %f" % (r_square))


rmse_900 = np.sqrt(mean_squared_error(final_res_900[name], final_res_900['pred']))
corr_900= np.corrcoef(final_res_900[name], final_res_900['pred'])
corr_900=corr_900[0,1]
r_square=corr_900**2
print("RMSE: %f" % (rmse_900))
print("R2: %f" % (r_square))


font = {'family' : 'normal',
        'weight' : 'bold',
        }

plt.figure(figsize=(7,6))
plt.plot(final_res[name],final_res['pred'], 'ro',alpha=0.5)
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
plt.savefig(name+"sp_all.png", dpi=300, bbox_inches = 'tight')
