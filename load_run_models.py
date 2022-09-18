import numpy as np
import pandas as pd

import datetime
import os

from keras.models import load_model
from keras.utils import pad_sequences

import eikon as ek
import claves
from create_datasets import log

def read_reshape_txt(file,max_length):
    with open(file, "r") as f:    
        task = [np.array(x.split(";"), dtype="float32") for x in f.readlines()]
        task = np.array([x[1:] for x in task])
        
        """import andi
        AD = andi.andi_datasets()
        dataset = AD.andi_dataset(N = 1_000, tasks = 2, dimensions = 1)
        task = np.array([x[1:] for x in [np.array(dataset[2][0][0]),np.array(dataset[2][0][1])]])
        """
        task = task[:,]
    
    task_HomLen = pad_sequences(task, maxlen=max_length, dtype="float32")
    return task_HomLen.reshape((-1, max_length, 1)) # reshape all to the same size of max_lenght

def predict_task1(model,df,start_date, end_date,task,len_min,len_max,now):
    returns_str = f"{start_date.date()}-{end_date.date()}-{len_min}-{len_max}"

    df[f'task{task}_len_{len_min}_{len_max}'] = ""

    for i,RIC in enumerate(df['Identifier']):
        RIC_dates = RIC + '-' + returns_str + '.txt'
        name_file_txt = os.path.join('data' , RIC_dates)
        
        if  RIC_dates in os.listdir('data'):
            log(RIC+' read',now)
            
            task_HomLenRes = read_reshape_txt(name_file_txt,len_max)
            
            # Clasified the data
            exps = model.predict(task_HomLenRes)
            # Averaged all the exponents in the trayectorias
            df.iloc[i,-1] = np.mean(exps)
            log(RIC+' predicted and save',now)
    return df

def predict_task2(model,df,start_date, end_date,task,len_min,len_max,now):
    returns_str = f"{start_date.date()}-{end_date.date()}-{len_min}-{len_max}"

    df[f'task{task}_len_{len_min}_{len_max}'] = ""

    for i,RIC in enumerate(df['Identifier']):
        RIC_dates = RIC + '-' + returns_str + '.txt'
        name_file_txt = os.path.join('data' , RIC_dates)
        
        if  RIC_dates in os.listdir('data'):
            log(RIC+' read',now)
            
            task_HomLenRes = read_reshape_txt(name_file_txt,len_max)
            
            # Clasified the data
            model_predicted = model.predict(task_HomLenRes) # XXXX aqui estan los NA
            
            df.iloc[i,-1] = model_predicted
            log(RIC+' predicted and save',now)
    return df

def download_ATM_vol(df,start_date,end_date,excel_data):
    
    ek.set_app_key(claves.API_key)
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

    # Load the data and prepare
    start_date = datetime.datetime(year=2012,month=7,day=1)
    end_date = datetime.datetime(year=2022,month=7,day=1)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    task = 2
    path = 'ANDI_Challenge/models/task'+str(task)
    data = 'analisis/S&P_500_analisis.xlsx'

    
    """for len_min,len_max in zip([10,100,400,800],[20,200,500,1_000]):
        name_model = f'task{task}_len_{len_min}_{len_max}.h5'
        df = pd.read_excel(data,index_col=0)
        # Load the data and prepare
        model = load_model(os.path.join(path,name_model))

        # Make predictions
        df = predict(model,df,start_date, end_date,task,len_min,len_max,now)
        df.to_excel(data)
        print(len_min,'-',len_max)
    """
    len_min,len_max = 800,1_000
    name_model = f'task{task}_dim{1}.h5'
    df = pd.read_excel(data,index_col=0)
        
    # Load the data and prepare
    model = load_model(os.path.join(path,name_model))

    # Make predictions
    df = predict_task2(model,df,start_date, end_date,task,len_min,len_max,now)
    df.to_excel(data)
    print(len_min,'-',len_max)
        
    #df = download_ATM_vol(df,start_date ,end_date,excel_data.replace('500','500_analisis'))



