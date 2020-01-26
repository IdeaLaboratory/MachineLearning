# time series prediction
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import style
from statsmodels.tsa.arima_model import ARIMA


def readFile(filename):
    return pd.read_csv(filename, header=0, parse_dates=[0], index_col=0)

'''
p: The number of lag observations included in the model, also called the lag order.
d: The number of times that the raw observations are differenced, also called the degree of differencing.
q: The size of the moving average window, also called the order of moving average.
'''
def ArimaForecasting(data, p_autoRegration, d_degreeOfDifferencing, q_movingAverage):
    model = ARIMA(data, order=(
        p_autoRegration, d_degreeOfDifferencing, q_movingAverage))
    trainedModel = model.fit(disp=0)
    prediction = trainedModel.forecast()
    prediction = prediction[0]
    # print(prediction)
    return prediction


data = readFile('exchange.csv')
# dummyList = [[1.5], [3.6], [6.6], [2.1], [1.0]]
# forcast = ArimaForecasting(dummyList,0,1,0)
# print(forcast)


for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            try:
                vals = ArimaForecasting(data, a, b, c)
                print('prediction at => {0}, {1}, {2} is: '.format(a, b, c))
                # print(a, b, c )
                # print('are : ')
                print(vals)
            except:
                continue

data = data['GBP/USD Close']
lenth = len(data)
trainigSet_len = int(lenth*0.7)

trainingSet1 = data[0:trainigSet_len]
testingSet1 = data[trainigSet_len:]

actaual = [x for x in trainingSet1]
prediction1 = list() # or []

for timePoint in range(len(testingSet1)):
    actaualVal = testingSet1[timePoint]
    pred = ArimaForecasting(actaual, 3,1,0)
    prediction1.append(pred)
    actaual.append(actaualVal)

from sklearn.metrics import mean_squared_error as mse
error = mse(testingSet1, prediction1)

print(error)
plt.plot(testingSet1, color="green")
plt.plot(prediction1, color="blue")

plt.show()
#print(data.describe())
# data.plot()

# plt.plot(data.head(),data.tail())
# plt.show()
print('done')