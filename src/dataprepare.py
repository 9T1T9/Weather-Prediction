import pandas as pd
import glob
import numpy as np

# get locations of stations
sm=pd.read_csv('location of meo.csv',index_col=0)
sm=sm.set_index('station_id').T.to_dict('list')

# locations of global weather stations
gwstations=['beijing_grid_000','beijing_grid_210','beijing_grid_420','beijing_grid_630',\
            'beijing_grid_637','beijing_grid_643','beijing_grid_007','beijing_grid_013',\
            'beijing_grid_020','beijing_grid_230','beijing_grid_440','beijing_grid_650']

"""Split raw data with different stations"""
def SplitByStation(station_id,df):
    df=df.copy()
    df.set_index(keys=['stationId'], drop=False,inplace=True)
    df=df.loc[station_id]
    """Remove duplicates in the dataframe"""
    df.drop_duplicates(subset=['utc_time'], keep='first', inplace=True)
    return df 

"""Split raw data with different stations"""
def SplitBymaeStation(station_id,df):
    df=df.copy()
    df.set_index(keys=['station_id'], drop=False,inplace=True)
    df=df.loc[station_id]
    """Remove duplicates in the dataframe"""
    df.drop_duplicates(subset=['time'], keep='first', inplace=True)
    return df

"""Generate a schedule from 2017.1 to 2018.4 of records with the columns be information about 
air quality and weather in a specific air quality station."""
def generateSchedule(weather,aq,station):
    time=pd.date_range(start='2017-01-01 00:00:00',end='2018-04-30 23:00:00',freq='H')
    weather['time']=pd.to_datetime(weather['time'])
    aq['time']=pd.to_datetime(aq['time'])
    df1=weather.set_index('time').reindex(time)
    df2=aq.set_index('time').reindex(time)
    df1['time']=df1.index
    df2['time']=df2.index
    df1.reset_index(drop=True)
    df2.reset_index(drop=True)
    sche=pd.merge(df1,df2,on='time')
    sche['stationId']=station
    return sche

"""Add weather information given station_id"""
def addW(df,mstation,k):
    wea=weadf.loc[weadf['station_id']==mstation]
    wea.drop(['station_id'],axis=1,inplace=True)
    wea.drop_duplicates('time',inplace=True)
    wea.columns=[i+str(k) for i in wea.columns]
    time=pd.date_range(start='2017-01-01 00:00:00',end='2018-04-30 23:00:00',freq='H')
    hh=pd.DataFrame(time,columns=['time'+str(k)])
    wea['time'+str(k)]=pd.to_datetime(wea['time'+str(k)])
    hh=pd.merge(hh,wea,how='outer',on='time'+str(k))
    hh.drop(['time'+str(k)],inplace=True,axis=1)
    new_df=pd.concat([df,hh],axis=1)
    return new_df

"""Add weather information from other stations."""
def addWeather(df,station):
    longitude=round(sm[station][0],1)
    latitude=round(sm[station][1],1)
    k=1
    location=pd.read_csv('Beijing_grid_weather_station.csv',names=['station','latitude','longitude'])
    for i in np.arange(round(longitude-0.1,1),round(longitude+0.1,1),0.1):
        for j in np.arange(round(latitude-0.1,1),round(latitude+0.1,1),0.1):
            i=round(i,1)
            j=round(j,1)
            if not (i==longitude and j == latitude):
                mstation=location.loc[(location['longitude']==i) & (location['latitude']==j)]['station'].values[0]
                df=addW(df,mstation,k)
                k+=1
    return df

"""Add global weather features."""
def addGweather(df):
    k=9
    for i in gwstations:
        df=addW(df,i,k)
        k+=1
    return df

"""Extract all information of a specific station from raw data."""
def extract(station_id,aqdf,weadf):
    stationloc=pd.read_csv('Beijing_grid_weather_station.csv',names=['station_id','latitude','longitude'])
    meoloc=pd.read_csv('location of meo.csv')
    meoloc.set_index(keys=['station_id'],drop=False,inplace=True)
    longitude=round(meoloc.loc[station_id]['longitude'],1)
    latitude=round(meoloc.loc[station_id]['latitude'],1)
    station=stationloc.loc[(stationloc['longitude']==longitude)&(stationloc['latitude']==latitude)]['station_id']
    aq=SplitByStation(station_id,aqdf)
    weather=SplitBymaeStation(station.values[0],weadf)
    aq.rename(index=str,columns={'utc_time':'time'},inplace=True)
    schedule=generateSchedule(weather,aq,station_id)
    return schedule

if __name__ =='__main__':
    weapath='gridweather'
    weaFiles = glob.glob(weapath + "/*.csv")
    
    """Read weather information from csv files. The result dataframe is named weadf."""
    wea=[]
    for file in weaFiles:    
        df=pd.read_csv(file)
        if file =='gridweather/gridWeather_201701-201803.csv':
            df.rename(index=str,columns={'stationName':'station_id','utc_time':'time','wind_speed/kph':'wind_speed'},inplace=True)
            df.drop(['longitude','latitude'],axis=1,inplace=True)
            df['weather']=np.nan
        else:
            df.drop(['id'],axis=1,inplace=True)
        wea.append(df)
        
    weadf=pd.concat(wea,axis=0,sort=True)
    """Read air quality information from csv files. The result dataframe is named aqdf"""
    aq=[]
    aq.append(pd.read_csv('aq/airQuality_201701-201801.csv'))
    aq.append(pd.read_csv('aq/airQuality_201802-201803.csv'))
    df=pd.read_csv('aq/aiqQuality_201804.csv')
    df.rename(index=str,columns={'station_id':'stationId','time':'utc_time','PM25_Concentration':'PM2.5','PM10_Concentration':'PM10','NO2_Concentration':'NO2','CO_Concentration':'CO','O3_Concentration':'O3','SO2_Concentration':'SO2'},inplace=True)
    df.drop(['id'],axis=1,inplace=True)
    aq.append(df)    
    aqdf=pd.concat(aq, axis=0,sort=True)
    
    """Merge the air quality and weather information together and extract information from a specific station."""
    #listoftestindex=list(pd.date_range(start='2017-05-01 00:00:00',end='2017-05-2 24:00:00',freq='H'))
    for k in set(aqdf['stationId']):
        print(k)
        sche=extract(k,aqdf,weadf)
        loc=pd.read_csv('location of meo.csv')
        latitude=loc.loc[loc['station_id']==k]['latitude']
        longitude=loc.loc[loc['station_id']==k]['longitude']
        sche['latitude']=float(latitude)
        sche['longitude']=float(longitude)
        """Add weather information from other grids."""
        sche=addWeather(sche,k)
        sche=addGweather(sche)
        """Save the sche as csv file."""
        sche.to_csv(k+'.csv',index=0)
    
        





