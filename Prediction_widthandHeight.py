import tabula
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from numpy import sqrt
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb


filename = 'Xiong2014_Article_BeadGeometryPredictionForRobot.pdf'
table = tabula.read_pdf(filename, pages=[4,5], multiple_tables=False)
table[0]
# Extracting the F, S, V
got_values = table[0].values[2:40]
FSV = got_values[:,0]

S = []
V = []
D = []
W = []
H = []
F = []


index = [11,13,15,17,19,21,23]
FSV=np.delete(FSV,index)

for i in range(len(FSV)):
    F.append(float(FSV[i].split()[1]))
    S.append(float(FSV[i].split()[2]))
    V.append(float(FSV[i].split()[3]))
    
# Extracting the D
D = got_values[:,1]
index = [11,13,15,17,19,21,23]

D = np.delete(D,index)
D_n = []
for i in range(D.shape[0]):
    D_n.append(float(D[i]))
    
# Extracting W and H
WH = got_values[:,2]
index = [11,13,15,17,19,21,23]

WH = np.delete(WH,index)
W = []
H = []

for i in range(WH.shape[0]):
    W.append(float(WH[i].split()[0]))
    H.append(float(WH[i].split()[1]))
features = []
targets = []
for i in range(len(H)):
    features.append([F[i],S[i],V[i],D_n[i]])
    targets.append([W[i],H[i]])
    

scaler = MinMaxScaler()
scaler.fit(features)
features_scaled = scaler.transform(features)

features_scaled = np.array(features_scaled)
targets_0 = np.array(targets)[:,0]
targets_1 = np.array(targets)[:,1]



test_fea = [[4,21,17,12],[4,27,17,12],[4,30,17,12],[4,36,17,12],
            [4,39,17,12],[5.2,27,18.9,12],[5.2,30,18.9,12],
            [6,27,20.3,12],[5.2,22.5,19,12],[5.2,37.5,19,12],
            [4.4,37.5,17.5,12],[6,37.5,20.5,12]]
test_tar = [[8.798,3.346],[7.899,2.961],[7.954,2.854],[7.249,2.662],
            [7.193, 2.533], [10.002,3.218], [9.116,3.047],
            [11.233,3.304],[10.610,3.411],[8.494,2.790],[7.788,2.811],
            [9.849,2.876]]

# Width prediction
regressor_width = xgb.XGBRegressor(n_estimators = 200, reg_lambda=.5, gamma = 0.0003
                             , max_depth = 5,verbosity=2)

regressor_width.fit(features_scaled,targets_0)
# Height prediction
regressor_height = xgb.XGBRegressor(n_estimators = 100, reg_lambda=.5, gamma = 0.0003
                             , max_depth = 10,verbosity=2)

regressor_height.fit(features_scaled,targets_1)



test_fea_norm = scaler.transform(test_fea)

W_tar_hat = regressor_width.predict(test_fea_norm)
H_tar_hat = regressor_height.predict(test_fea_norm)

W_arr = []
H_arr = []
for i in range(len(test_tar)):
    W_arr.append(test_tar[i][0])
    H_arr.append(test_tar[i][1])
    
W_arr = np.array(W_arr)    

W_acc = abs((W_tar_hat-W_arr)/W_arr*100)
W_acc

H_arr = np.array(W_arr)    

H_acc = abs((H_tar_hat-H_arr)/H_arr*100)
H_acc
    

sqrt(mean_squared_error(W_tar_hat,W_arr))
sqrt(mean_squared_error(H_tar_hat,H_arr))

