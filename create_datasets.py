from unicodedata import name
import pandas as pd
import numpy as np

import eikon as ek
import claves

import datetime
import os

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

ek.set_app_key(claves.API_key)


def download_hist(RIC,start_date,end_date, name_file,RIC_dates,now,interval='daily'):
    """
    If the data is not already downloaded, it downloads the data and save it as xlsx
    parameters:
    fields = # 'VALUE', 'VOLUME', 'HIGH', 'LOW', 'OPEN', 'CLOSE', 'COUNT' By default all fields are returned.
    interval= 'tick', 'minute', 'hour', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly' (Default 'daily') Default: 'daily'
    """
    if os.path.exists(name_file):
        log("Datos ya existentes:"+RIC_dates,now)
        return pd.read_csv(name_file)
    else:
        try:
            df = ek.get_timeseries(rics=RIC,
                                    fields='*',
                                    start_date=str(start_date),
                                    end_date=str(end_date),
                                    interval=interval) 
            df.to_csv(name_file)
            log(f"Downloaded and save:{RIC_dates}\nSize:{df.shape[0]}",now)
            return df
        except Exception as e:
            print(e)
            log(str(e),now)
            return None

def save_hist(df,name_file, now, min_T = 10,max_T = 20):
    """
    Clean the df from NA and compute the log returns
    """
    size_df = len(df)
    df.dropna(subset=['CLOSE'], inplace=True)
    
    if size_df != len(df):
        log(str(size_df-len(df))+' rows with NA',now)
    
    
    # Define the random size of the vector
    len_route = np.random.randint(min_T,max_T+1,int(df.shape[0]/min_T))
    # Start with a 0 and create a cum sum vector to obtain the postions
    pos_df = np.insert(len_route, 0, 0).cumsum()
    # Only use the postions lower than the size
    pos_df = pos_df[pos_df<df.shape[0]-np.random.randint(min_T,max_T+1)]
    # The last postion will the the size of the vector
    pos_df = np.append(pos_df, [df.shape[0]-1])
    # Chop the closing price with the random positions
    trajectories = np.array([np.array(df['CLOSE'][pos_df[i]:pos_df[i+1]]) for i in range(len(pos_df)-1)])
    # Compute the returns compare with the x_0
    norm_trajectories = []
    for x in trajectories:
        norm_trajectories.append([np.log(_/x[0])*100 for _ in x])
    #np.savetxt('text.txt',norm_trajectories,fmt='%1.5f',delimiter=';')

    if not all(isinstance(_,float) for _ in sum(norm_trajectories,[])):
        log("Error in the returns of:"+name_file, now)
    
    # Save the retuns in a txt
    with open(name_file,'w') as f:
        for trajectory in norm_trajectories:
            f.write('1.0;'+';'.join([str(x) for x in trajectory])+'\n')
    
    log("Returns compute",now)

def log(txt,now):
    f = open(__file__[:-3]+'_'+now+'.log', 'a')
    f.write(txt + '\r')
    f.close()

if __name__ == '__main__':

    start_date = datetime.datetime(year=2012,month=7,day=1)
    end_date = datetime.datetime(year=2022,month=7,day=1)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    data = 'analisis/S&P_500.xlsx'
    
    df = pd.read_excel('analisis/S&P_500.xlsx')
    len_min = 10
    len_max = 20

    df = pd.read_excel(data)
    RICs = df['Identifier'].to_list()
    
    for RIC in RICs:    
        log(RIC,now)
        RIC_dates = RIC + '-' + str(start_date.date()) + '-' + str(end_date.date())
        name_file = os.path.join('data' , RIC_dates + '.csv')
        df = download_hist(RIC,start_date,end_date, name_file,RIC_dates)
            
        if df is not None:    
            name_file_txt = os.path.join('data' , f"{RIC_dates}-{len_min}-{len_max}.txt")
            save_hist(df,name_file_txt, min_T = len_min,max_T = len_max)
            log('Done',now)
