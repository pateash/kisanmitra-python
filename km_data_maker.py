

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pp
from  sklearn import preprocessing



crop_name='banana'
state='mp'
region='bhopal'

data=pd.read_excel('data/'+state+'/'+region+'_data.xls')# eg. mp/bhopal_data.xls
crops=data['Crop'].unique().tolist()


for crop_name in crops:
    datat = data.loc[data['Crop'] == crop_name]
    datat = datat.sort_values(by='year', ascending=True).drop('Crop', axis=1).drop('Production Quantity - tonnes',
                                                                                 axis=1).drop('year', axis=1)
    model = LinearRegression()
    X, y = datat.iloc[:, 0:-1], datat.iloc[:, -1:]
    model.fit(X, y)

    y_temp = pd.DataFrame()
    y_temp['yeild'] = y.iloc[:-5]['Yield - Hg/Ha']
    y_temp['year'] = np.arange(1961, 2003)
    y_temp.set_index('year',inplace=True)
    y_temp.to_json(region + "_" + crop_name + "_training.json")

    y_temp = pd.DataFrame()
    y_temp['year'] = np.arange(2003, 2008)
    y_temp.set_index('year',inplace=True)
    y_temp['yeild'] = model.predict(X.iloc[-5:])
    y_temp.to_json(region + "_" + crop_name + "_predicated.json")



