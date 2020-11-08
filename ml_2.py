
import pandas as pd
from sklearn.model_selection import train_test_split

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

#print(nyc.head(3))
#print(nyc.Date.values)
#print(nyc.Date.values.reshape(-1,1))

x_train, x_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1,1), nyc.Temperature.values, random_state=11)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X=x_train, y=y_train)

#print(lr.coef_)
#print(lr.intercept_)

predicted = lr.predict(x_test)
expected = y_test

for p,e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

predict = (lambda x: lr.coef_ * x + lr.intercept_)

#print(predict(2020))
#print(predict(1890))
#print(predict(2021))

import seaborn as sns

axes = sns.scatterplot(data=nyc,
                       x='Date',
                       y='Temperature',
                       hue='Temperature',
                       palette='winter',
                       legend=False)
axes.set_ylim(10,70)

import numpy as np

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])

#print(x)

y = predict(x)

#print(y)

import matplotlib.pyplot as plt

line = plt.plot(x,y)
f1 = plt.figure(1)

####
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

n = csv.reader(open("ave_yearly_temp_nyc_1895-2017.csv"))
next(n)

lines = list(n)
#print(lines)

for line in lines:
    date = line[0]
    new_date = date[:-2]
    line[0] = new_date
###
print(lines)

fields = ['Date','Value','Anomaly']

with open('ave_yearly_fixed_nyc_1895-2017.csv','w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(lines)


nyc_yearly = pd.read_csv("ave_yearly_fixed_nyc_1895-2017.csv")

x_train_yr, x_test_yr, y_train_yr, y_test_yr = train_test_split(nyc_yearly.Date.values.reshape(-1,1), nyc_yearly.Value.values, random_state=11)

from sklearn.linear_model import LinearRegression

lr_yrly = LinearRegression()
lr_yrly.fit(X=x_train_yr, y=y_train_yr)

#print(lr_yrly.coef_)
#print(lr_yrly.intercept_)

predicted_yr = lr_yrly.predict(x_test_yr)
expected_yr = y_test_yr

for p,e in zip(predicted_yr[::5], expected_yr[::5]):
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

predict_yr = (lambda x: lr_yrly.coef_ * x + lr_yrly.intercept_)

#print(predict_yr(2020))
#print(predict_yr(1890))
#print(predict_yr(2021))

import seaborn as sns

axes = sns.scatterplot(data=nyc_yearly,x='Date',y='Value',hue='Value',palette='winter',legend=False)
axes.set_ylim(10,70)

import numpy as np

x_yrly = np.array([min(nyc_yearly.Date.values), max(nyc_yearly.Date.values)])

#print(x_yrly)

y_yrly = predict_yr(x_yrly)

#print(y_yrly)

import matplotlib.pyplot as plt

line_yrly = plt.plot(x_yrly,y_yrly)
#f2 = plt.figure(2)

plt.show()
