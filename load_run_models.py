import numpy as np
import pandas as pd

import datetime
import os

from keras.models import load_model
from keras.utils import pad_sequences

import eikon as ek
import Claves


def read_reshape_txt(file):
    with open(file, "r") as f:    
        task = [np.array(x.split(";"), dtype="float64") for x in f.readlines()]
        task = np.array([x[1:] for x in task])
        task = task[:,]
    max_length = 20
    task_HomLen = pad_sequences(task, maxlen=max_length, dtype="float64")
    return task_HomLen.reshape((-1, max_length, 1)) # reshape all to the same size of max_lenght

def predict(model,df,dates):
    df['avg_exp'] = ""

    for i,RIC in enumerate(df['Identifier']):
        RIC_dates = RIC + '-' + dates
        name_file_txt = os.path.join('data' , RIC_dates + '.txt')
        
        if  RIC_dates + '.txt' in os.listdir('data'):
            print(RIC)
            
            task_HomLenRes = read_reshape_txt(name_file_txt)
            
            # Clasified the data
            exps = model.predict(task_HomLenRes)
            df.iloc[i,-1] = np.mean(exps)
    return df

def download_ATM_vol(df,start_date,end_date,excel_data):
    
    ek.set_app_key(Claves.API_key)
    sufix_implied_vol = 'ATMIV'
    df['Hist_Vol'] = ""
    df['Impl_Vol'] = ""

    dates_str = '-' + str(start_date.date()) + '-' + str(end_date.date())    

    for i,RIC in enumerate(df['Identifier'][::-1]):
        RIC_imp_vol = RIC.split('.')[0]+sufix_implied_vol+'.U'
        fields = ['TR.30DAYATTHEMONEYIMPLIEDVOLATILITYINDEXFORPUTOPTIONS.Date','TR.30DAYATTHEMONEYIMPLIEDVOLATILITYINDEXFORPUTOPTIONS']
        start_date = end_date-datetime.timedelta(1)
        
        dd,e=ek.get_data(RIC_imp_vol,
                        fields,
                        {'SDate':start_date.date().strftime("%Y%m%d"),
                        'EDate':end_date.date().strftime("%Y%m%d"),'Frq':'D'})
        
        if e==None:
            df['Impl_Vol'][i] = dd.iloc[1,2]
        else:
            df['Impl_Vol'][i] = np.nan
        RIC_dates = RIC + dates_str
        path_excel = os.path.join('data' , RIC_dates + '.xlsx')
        df_hist = pd.read_excel(path_excel)
        if df_hist['CLOSE'].isna().sum() > 10:
            print("Problems in ",RIC)
        df_hist.bfill(inplace=True)
        df_hist['Retuns'] = df_hist['CLOSE'].pct_change()*100
        #np.std(df_hist['Retuns'] ) # daily volatility
        #np.std(df_hist['Retuns'] )*np.sqrt(252) # anualy volatility
        df['Hist_Vol'][i] = np.std(df_hist['Retuns']/100 )*np.sqrt(20)*100 # monthly volatility
        
        df.to_excel(excel_data)
        print(i)
        
    return df


if __name__ == '__main__':
    # Load the model
    model = load_model('ANDI_Challenge/models/task1/task1_len_10_20.h5')

    # Load the data and prepare
    start_date = datetime.datetime(year=2020,month=7,day=1)
    end_date = datetime.datetime(year=2022,month=7,day=1)
    excel_data = 'analisis/S&P_500.xlsx'

    df = pd.read_excel(excel_data)
    dates_str = str(start_date.date()) + '-' + str(end_date.date())
    #df = predict(df,dates_str)
    #df.to_excel(excel_data)
    df = download_ATM_vol(df,start_date ,end_date,excel_data.replace('500','500_analisis'))



