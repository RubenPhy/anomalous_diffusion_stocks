from unicodedata import name
import pandas as pd
import numpy as np

import eikon as ek
import Claves

import datetime
import os

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

ek.set_app_key(Claves.API_key)


def download_hist(name_file,RIC_dates):
    """
    If the data is not already downloaded, it downloads the data and save it as csv
    """
    if os.path.exists(name_file):
        df = pd.read_excel(name_file)
        print("Datos ya existentes:", RIC_dates)
        return df
    else:
        try:
            df = ek.get_timeseries(rics=RIC,
                                    fields='*',  # 'VALUE', 'VOLUME', 'HIGH', 'LOW', 'OPEN', 'CLOSE', 'COUNT' By default all fields are returned.
                                    start_date=str(start_date),
                                    end_date=str(end_date))
                                    #interval='daily')  # 'tick', 'minute', 'hour', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly' (Default 'daily') Default: 'daily'
            if df.shape[0]<400:
                return None
            df.to_excel(name_file)
            print(f"Size:{df.shape[0]} --> Download:", RIC_dates)
            return df
        except Exception as e:
            print(e)
            return None

def save_hist(df,name_file, min_T = 10,max_T = 20):
    if not os.path.exists(name_file):
        # Random size of the vector
        len_route = np.random.randint(min_T,max_T+1,int(df.shape[0]/min_T))
        # Start with a 0 and create a cum sum vector to obtain the postions
        pos_df = np.insert(len_route, 0, 0).cumsum()
        # Only use the postions lower than the size
        pos_df = pos_df[pos_df<df.shape[0]-np.random.randint(min_T,max_T+1)]
        # The last postion will the the size of the vector
        pos_df = np.append(pos_df, [df.shape[0]-1])
        # Chop the closing price with the previous random positions
        trajectories = np.array([np.array(df['CLOSE'][pos_df[i]:pos_df[i+1]]) for i in range(len(pos_df)-1)])
        # Compute the returns with the x0 as the base
        norm_trajectories = [np.array((x/x[0]-1)*100).tolist() for x in trajectories]
        #np.savetxt('text.txt',norm_trajectories,fmt='%1.5f',delimiter=';')

        # Save it a txt
        with open(name_file,'w') as f:
            for trajectory in norm_trajectories:
                f.write('1.0;'+';'.join([str(x) for x in trajectory])+'\n')

if __name__ == '__main__':

    start_date = datetime.datetime(year=2020,month=7,day=1)
    end_date = datetime.datetime(year=2022,month=7,day=1)

    df = pd.read_excel('analisis/S&P_500.xlsx')
    RICs = df['Identifier'].to_list()
    for RIC in RICs:    
        if RIC not in ['ACLT.DH','ENER.DH','NABV.NS','QUER.MOT','MLECO.EUA','ABENU.PK','EJGJ.SJ']:
            print(RIC)
            RIC_dates = RIC + '-' + str(start_date.date()) + '-' + str(end_date.date())
        
            name_file_xlsx = os.path.join('data' , RIC_dates + '.xlsx')
            df = download_hist(name_file_xlsx,RIC_dates)

            if df is not None and np.sum(np.isnan(df['CLOSE'])) == 0:
                
                name_file_txt = os.path.join('data' , RIC_dates + '.txt')
                save_hist(df,name_file_txt)
                print(f'Done')

