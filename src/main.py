from keras import Sequential,backend as K
from keras.layers import LSTM,Dense
import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import datetime
import matplotlib.pyplot as plt
from scipy import signal

# get locations of stations
sm=pd.read_csv('location of meo.csv',index_col=0)
sm=sm.set_index('station_id').T.to_dict('list')

# locations of global weather stations
gwstations=['beijing_grid_000','beijing_grid_210','beijing_grid_420','beijing_grid_630',\
            'beijing_grid_637','beijing_grid_643','beijing_grid_007','beijing_grid_013',\
            'beijing_grid_020','beijing_grid_230','beijing_grid_440','beijing_grid_650']

"""loss funciton"""
def smape(predict,actual):
    return K.mean(K.abs(predict-actual)/K.maximum((K.abs(actual)/2+K.abs(predict)/2),0.6))

def smapenp(predict,actual):
    return np.mean(np.abs(predict-actual)/(np.abs(actual)/2+np.abs(predict)/2))

"""Add time features."""
def addTime(df,offset):
    df=df.copy()
    time=list(df['time'])
    date=[[int(t[0:4]),int(t[5:7]),int(t[8:10]),int(t[11:13])] for t in time]
    day=[datetime.date(d[0],d[1],d[2]).weekday() for d in date]
    hour=[float(t[11:13]) for t in time]
    month=[float(t[5:7]) for t in time]
    phour=[(h+offset)%24 for h in hour]
    pday=[(datetime.datetime(d[0],d[1],d[2],d[3])+datetime.timedelta(hours=-offset)).weekday() for d in date]
    pmonth=[(datetime.datetime(d[0],d[1],d[2],d[3])+datetime.timedelta(hours=-offset)).month for d in date]
    df['month']=month
    df['day']=day
    df['hour']=hour
    df['phour']=phour
    df['pday']=pday
    df['pmonth']=pmonth
    return df

"""Add statistics features of wind speed of k hours."""
def addwind(df,k):
    s=df.index[0]
    e=s+k-1
    smean=[]
    while e<df.index[-1]:
        smean+=[df.loc[s:e]['wind_speed'].mean() for ke in range(s,e+1)]
        s=e+1
        e=s+k-1
    e=df.index[-1]
    smean+=[df.loc[s:e]['wind_speed'].mean() for ke in range(s,e+1)]
    df['wmean'+str(k)]=smean
    return df

"""Add statistics features of air concentration of the same hour in k days."""
def addair(df,air,k):
    df['amean'+str(k)]=np.zeros((len(df),1))
    df['amedian'+str(k)]=np.zeros((len(df),1))
    df['amin'+str(k)]=np.zeros((len(df),1))
    df['amax'+str(k)]=np.zeros((len(df),1))
    for i in range(0,24):
        s=0
        e=s+k-1
        while e<len(df.loc[df['hour']==i]):
            iindex=df['amean'+str(k)].loc[df['hour']==i].iloc[s:e+1].index
            df['amean'+str(k)].loc[iindex]=df.loc[df['hour']==i][air].iloc[s:e+1].mean()
            df['amedian'+str(k)].loc[iindex]=df.loc[df['hour']==i][air].iloc[s:e+1].median()
            df['amin'+str(k)].loc[iindex]=df.loc[df['hour']==i][air].iloc[s:e+1].min()
            df['amax'+str(k)].loc[iindex]=df.loc[df['hour']==i][air].iloc[s:e+1].max()
            s=e+1
            e=s+k-1
        e=len(df.loc[df['hour']==i])-1
        iindex=df['amean'+str(k)].loc[df['hour']==i].iloc[s:e+1].index
        df['amean'+str(k)].loc[iindex]=df.loc[df['hour']==i][air].iloc[s:e+1].mean()
        df['amedian'+str(k)].loc[iindex]=df.loc[df['hour']==i][air].iloc[s:e+1].median()
        df['amin'+str(k)].loc[iindex]=df.loc[df['hour']==i][air].iloc[s:e+1].min()
        df['amax'+str(k)].loc[iindex]=df.loc[df['hour']==i][air].iloc[s:e+1].max()
    return df

"""Add rate features"""
def addrate(df,air):
    rate=[]
    index=df.index.drop(df.index[-1])
    for i in index:
        t1=df.loc[i+1,'time']
        et=datetime.datetime(int(t1[0:4]),int(t1[5:7]),int(t1[8:10]),int(t1[11:13]))
        t2=df.loc[i,'time']
        st=datetime.datetime(int(t2[0:4]),int(t2[5:7]),int(t2[8:10]),int(t2[11:13]))
        hours=(et-st).seconds/3600
        rate.append((df.loc[i+1,air]-df.loc[i,air])/hours)
    rate.append(np.nan)
    df['rate']=rate
    return df

"""Construct data for training from schedule of a specific station by a fixed offset
,station and air type."""
def generateTrain(offset,air,df):
    if offset > -48:
        sys.exit('OFFSET SHOULD BE LESS OR EQUAL TO -48.')
    # drop time, station and mae station columns which are useless
    df.drop(['time','station_id','stationId'],axis=1,inplace=True)
    df.drop(['latitude','longitude'],axis=1,inplace=True)
    # count null values proportion for each column
    nancnt=[df[col].isna().sum()/len(df[col]) for col in df.columns]
    # record indices of columns where null values counts larger than 0.5
    delnan=[i for i in range(0,len(nancnt)) if nancnt[i]>0.2]
    # drop columns where null values proportion larger than 0.5
    df.drop([df.columns[i] for i in delnan],axis=1,inplace=True)
    # shift concentration of air to get labels
    df['result']=df[air].shift(offset)
#    if(len(df)<4000):
#        sys.exit('Not enough training samples.')
    return df

"""Preprocess data including processing null values and normalization."""
def preprocessTrain(df):
    columns2=df.columns
    # drop rows with null values
    new_df=df.dropna()
    # fill null values
    while len(new_df)<700:
        df.fillna(method='pad',limit=1,inplace=True)
        df.fillna(method='bfill',limit=1,inplace=True)
        new_df=df.dropna()
    # remove noise
#    if 'weather' not in new_df.columns:
#        new_df=denoise(new_df)
#    else:
#        num_cols = new_df.columns[new_df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
#        new_df.loc[:,num_cols]=denoise(new_df.loc[:,num_cols])
    # log result to avoid negative outputs    
    new_df.loc[:,'result']=np.log(new_df['result'])
    # correct values
    new_df.loc[new_df['wind_direction']==360]=0
    # drop useless columns
#    corr=new_df.corr()['result']
#    print(corr)
#    columns=corr.loc[abs(corr)>=0.1].index
#    new_df=new_df[columns]
    columns=new_df.columns
    # normalize the data
    scaler=MinMaxScaler(feature_range=(0,1))
    if 'weather' not in new_df.columns:
        new_df=scaler.fit_transform(new_df)
    else:
        num_cols = new_df.columns[new_df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
        new_df.loc[:,num_cols]=scaler.fit_transform(new_df.loc[:,num_cols])
    standard=StandardScaler()
#    new_df=standard.fit_transform(new_df)
    new_df=pd.DataFrame(data=new_df,columns=columns)
    new_df=pd.get_dummies(new_df)
    t=new_df.copy()
    new_df.drop(['result'],axis=1,inplace=True)
    new_df['result']=t['result']
    columns=new_df.columns
    return new_df,scaler,columns,standard,columns2

"""Denoise the data"""
def denoise(df):
    for column in df.columns.drop(['O3','PM2.5','PM10']):
        s=0
        nindex=df[column].index[df[column].isna()]
        if not len(nindex):
            n=round(len(df)/100)
#            plt.figure()
#            plt.plot(np.linspace(0,100,101),df[column][0:101],color='green')
#            plt.show()
            df.loc[:,column]=signal.filtfilt([1.0/n]*n,1,df.loc[:,column])
#            plt.figure()
#            plt.plot(np.linspace(0,100,101),df[column][0:101],color='green')
#            plt.show()
            continue
        for i in nindex:
            if s not in nindex and i-s>50:
                n=round((i-s)/20)
#                plt.figure()
#                plt.plot(np.linspace(s,i-1,i-s),df.loc[s:i-1,column],color='green')
#                plt.show()
                df.loc[s:i-1,column]=signal.filtfilt([1.0/n]*n,1,df.loc[s:i-1,column])
#                plt.figure()
#                plt.plot(np.linspace(s,i-1,i-s),df.loc[s:i-1,column],color='green')
#                plt.show()
            s=i+1
        if len(df)-nindex[-1]>25:
            n=round((len(df)-nindex[-1]-1)/20)
            df.loc[nindex[-1]+1:-1,column]=signal.filtfilt([1.0/int(n)]*int(n),1,df.loc[nindex[-1]+1:-1,column])
    return df

"""Get k nearest stations as fillna values."""
def getknn(station,k):
    o=sm[station]
    dis={}
    for s in sm.keys():
        dis[s]=np.linalg.norm(np.array(sm[s])-np.array(o))
    stations = sorted(dis,key=dis.get)
    return stations[1:k+1]

"""Compute the mean of data of stations."""
def getmeandf(stations):
    dfs=[pd.read_csv(s+'.csv',usecols=range(0,15)) for s in stations]
    ndfs=pd.concat(dfs)
    df=ndfs.groupby(ndfs.index).mean()
    return df

"""Fillna with knn method."""
def knnfillna(station,df,k):
    df=df.copy()
    stations=getknn(station,k)
    meandf=getmeandf(stations)
    for column in meandf.columns:
        index = df[column].index[df[column].isna()]
        df.loc[index,column]=meandf[column][index]
    return df

"""Build LSTM model for each combination of predicted air and station."""
def model(df,station,air,offset):
    df=df.copy()
    # Generate samples by shift operantion
    df=generateTrain(offset,air,df)
    # Preprocess data frames
    df,scaler,columns,standard,columns2=preprocessTrain(df)
    # Get training data and labels
    x=np.array(df.iloc[:,0:-1])
    y=np.array(df.iloc[:,-1])
    # reshape input to be 3D [samples, timesteps, features]
    x=x.reshape((x.shape[0],1,x.shape[1]))
    trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.2, random_state=42)
    model = Sequential()
    model.add(LSTM(50,input_shape=(trainx.shape[1],trainx.shape[2])))
    model.add(Dense(1))
    model.compile(loss=smape,optimizer='adam')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    history=model.fit(trainx, trainy, epochs=50, batch_size=512, verbose=False,\
                      callbacks=[early_stop],validation_data=(testx, testy),shuffle=True)
    
    print(history.history['val_loss'])
    return model,scaler,standard,columns.drop(['result']),columns2.drop(['result']),x,y

#"""Model Aggregation."""
#def ModelAggregation(station,air,offset):
#    df=pd.read_csv(station+'.csv')
#    """Construct the model with weather feature."""
#    df=df.iloc[10920:-1]
#    wmodel,scaler,standard,columns,testx,testy=model(df,station,air,offset)
    
"""Produce features for prediction. The dimension of the dataframe 
must be the same as training samples."""
def generateTest(offset,df):
    # get start and end indices of features for prediction
    start=len(df)+offset
    end=start+48
    df=df.iloc[start:end,:]
    return df

"""Preprocess test data. The dimension of the dataframe must 
be the same as training samples."""
def preprocessTest(df,columns,columns2,scaler,standard):
    df=df.copy()
    df=df[columns2]
    df.loc[df['wind_direction']==360]=0
    # exit when containing a large quantity null values
    if df.isna().sum().sum()/df.size>0.1:
        sys.exit("DATAFRAME CONTAINS TOO MANY NULL VALUES")
    #print(df.isna().sum().sum()/df.size)
    # fill null values
    df.fillna(method='pad',inplace=True)
    df.fillna(method='bfill',inplace=True)
    if df.isna().sum().sum()>0:
        sys.exit("DATAFRAME CONTAINS NULL VALUES.")
    # noise reduction
#    if 'weather' not in df.columns:
#        df=denoise(df)
#    else:
#        num_cols = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
#        df.loc[:,num_cols]=denoise(df.loc[:,num_cols])  
    # make the same dimension as scaler
    df['result']=np.linspace(1,10000,len(df))
    ncolumns=df.columns
    # normalization
    if 'weather' not in ncolumns:
        df=scaler.transform(df)
    else:
        num_cols = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
        df.loc[:,num_cols]=scaler.transform(df.loc[:,num_cols])
    df=pd.DataFrame(data=df,columns=ncolumns)
    df=pd.get_dummies(df)
    ccolumns=[i for i in columns if i not in df.columns]
    for i in ccolumns:
        df[i]=0
    df=df[columns]
    df=df.values
#    df=standard.transform(df)
    return df

"""Predict the air quality by one by one prediction when the offset\
is larger than or equal to 48."""
def onebyonepredict(df,offset,air,model,station,scaler,standard,columns,columns2):
    df=df.copy()
    x=generateTest(offset,df)
    x=preprocessTest(x,columns,columns2,scaler,standard)
    x=x.reshape((x.shape[0],1,x.shape[1]))
    y = np.array(model.predict(x))
    y=y.reshape((len(y),1))
    x=x.reshape((x.shape[0],x.shape[2]))
    x_y=np.concatenate((np.zeros((len(y),len(scaler.data_range_)-1)),y),axis=1)
#    x_y=standard.inverse_transform(x_y)
    x_y=scaler.inverse_transform(x_y)
    return np.exp(x_y[:,-1])

def evaluate(model,testx,testy,scaler,standard):
    #print(testx.shape)
    prediction=np.array(model.predict(testx))
#    ntestx=testx.reshape((testx.shape[0],testx.shape[2]))
    prediction=prediction.reshape((len(prediction),1))
    x_y=np.concatenate((np.zeros((len(prediction),len(scaler.data_range_)-1)),prediction),axis=1)
#    x_y=standard.inverse_transform(x_y)
    x_y=scaler.inverse_transform(x_y)
    prediction=np.exp(x_y[:,-1])
    testy=testy.reshape((len(testy),1))
    x_y2=np.concatenate((np.zeros((len(prediction),len(scaler.data_range_)-1)),testy),axis=1)
#    x_y2=standard.inverse_transform(x_y2)
    x_y2=scaler.inverse_transform(x_y2)
    testy=np.exp(x_y2[:,-1])
    plt.figure()
    plt.plot(np.linspace(0,100,len(testy[-148:-48])),testy[-148:-48],color='green')
    plt.plot(np.linspace(0,100,len(prediction[-148:-48])),prediction[-148:-48],color='blue')
    plt.show()
    print(((np.array(prediction[-148:-48])-np.array(testy[-148:-48]))**2).mean())
    print(smapenp(prediction[-148:-48],testy[-148:-48]))
    return smapenp(prediction[-148:-48],testy[-148:-48])
    
"""Predict air concentration of next 48 hours within stations list and airtype."""
def predict(result,offset,airtype,stations):
    for air in airtype:
        result[air]=[]
        for station in stations:
            # create model
            df=pd.read_csv(station+'.csv')
            # fill null values
            df.iloc[:,0:15]=knnfillna(station,df.iloc[:,0:15],3)
            df=df[10920:]
            # Add rate
            df=addrate(df,air)
            # add time features
            df=addTime(df,offset)
#            # Add statistics of air concentration
#            df=addair(df,air,3)
#            df=addair(df,air,7)
            # Add statistics of wind speed
#            df=addwind(df,3)
#            df=addwind(df,24)
#            df=addwind(df,48)
            nnmodel,scaler,standard,columns,columns2,testx,testy=model(df,station,air,offset)
            #while evaluate(nnmodel,testx,testy,scaler,standard)>1:
            #nnmodel,scaler,standard,columns,testx,testy=model(df,station,air,offset)
            prediction=onebyonepredict(df,offset,air,nnmodel,station,scaler,standard,columns,columns2)
            print(air,station)
            print(prediction)
            result[air]+=list(prediction)
    return result
    
if __name__ =='__main__':
    airtype=['PM2.5','PM10','O3']
    hehe=pd.read_csv('aq/aiqQuality_201804.csv')
    # get 35 stations
    stations=sorted(set(list(hehe['station_id'])), key=list(hehe['station_id']).index)
    # read sample submission's test_id column
    haha=pd.read_csv('sample_submission.csv')
    testid=haha['test_id']
    result={} # PM2.5 
    result['test_id']=list(testid)
    # predict the result and fit into the sample submission.
    result=predict(result,-48,airtype,stations)
    resultdf=pd.DataFrame(result)
    resultdf.to_csv('submission.csv',index=False)
    