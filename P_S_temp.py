# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:34:49 2022

@author: 14198
"""

import pandas as pd
import matplotlib.pyplot as plt
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

micro = Final_micro_tensile[['Strain', 'Mean Radius Prime', 'Mean Radius Double Prime',
'Volume Fraction Prime', 'Volume Fraction Double Prime', 'Avg GS','Large GS', 'Temperature']]

# Final_micro_tensile_73= Final_micro_tensile[Final_micro_tensile.Temperature==73]

# Final_micro_tensile_73.to_csv("Final_micro_tensile_73.csv")

# Final_micro_tensile_750= Final_micro_tensile[Final_micro_tensile.Temperature!=73]

# Final_micro_tensile_750.to_csv("Final_micro_tensile_750.csv")

micro =micro.iloc[0:373]
Final_micro_tensile=Final_micro_tensile[0:373]

name='Elon'
y= Final_micro_tensile[name] #* 6.89



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
r_square=corr
print("RMSE: %f" % (rmse))
print("R2: %f" % (r_square))

import matplotlib.colors as colors
import matplotlib.cm as cmx

temp= pd.DataFrame(micro['Temperature'])
temp.loc[temp['Temperature'] == 900, 'Temperature'] = 750

temp.loc[temp['Temperature'] == 750, 'Temperature'] = '750F'
temp.loc[temp['Temperature'] == 73, 'Temperature'] = '73F'


uniq = list(set(temp['Temperature']))

z = range(1,len(uniq))
hot = plt.get_cmap('hot')
cNorm  = colors.Normalize(vmin=0, vmax=len(uniq))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)




font = {'family' : 'normal',
        'weight' : 'bold',
        'size': 12
        }
    
plt.figure(figsize=(6,6))
for i in range(len(uniq)):
    indx = temp['Temperature'] == uniq[i]
    plt.plot(y[indx],preds_all[indx], 'ro', alpha=0.5, color=scalarMap.to_rgba(i), label=uniq[i])
    #plt.scatter(y[indx],preds_all[indx], color=scalarMap.to_rgba(i), label=uniq[i])

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
plt.legend(loc='upper left')
plt.rc('font', **font)
plt.rc('axes', labelsize=18)
plt.plot(x_lim, y_lim, 'k-', color = 'b')
plt.savefig(name+"sp_temp_colors.png", dpi=300, bbox_inches = 'tight')
plt.show()

#Feature Importance

feature_important = xg_reg.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())
data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features
data.reset_index(inplace=True)


#Temperature Effect

name='RA'
y= Final_micro_tensile[name]

X= micro
X_norm= ((X-X.min())/(X.max()-X.min()))
y_norm= ((y-y.min())/(y.max()-y.min()))
data_dmatrix = xgb.DMatrix(data=X_norm,label=y)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=402)
xg_reg = xgb.XGBRegressor()
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
preds_all=xg_reg.predict(X_norm)



temp=  X['Temperature']
temp= temp.reset_index()
temp= temp.drop('index', axis=1)  
y= y.reset_index()
y= y.drop('index', axis=1)  
preds_all=pd.DataFrame(preds_all)



final_res = pd.concat([y,preds_all,temp], axis=1)
final_res.rename(columns = {0:'pred'}, inplace = True)

final_res_73= final_res[final_res.Temperature==73]
final_res_73.to_csv("final_res_73"+name+".csv")

final_res_750=final_res[final_res.Temperature!=73] 
final_res_750.to_csv("final_res_750"+name+".csv")


rmse_73 = np.sqrt(mean_squared_error(final_res_73[name], final_res_73['pred']))
corr_73= np.corrcoef(final_res_73[name], final_res_73['pred'])
corr_73=corr_73[0,1]
r_square=corr_73
print("RMSE: %f" % (rmse_73))
print("R2: %f" % (r_square))



font = {'family' : 'normal',
        'weight' : 'bold',
        'size': 12
        }

plt.figure(figsize=(6,6))
plt.plot(final_res_73[name],final_res_73['pred'], 'ro', alpha=0.5)
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
plt.savefig(name+"_temp_73_sp.png", dpi=300, bbox_inches = 'tight')
plt.show()







