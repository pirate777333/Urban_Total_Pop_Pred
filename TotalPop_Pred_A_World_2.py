import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import boxcox
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('D:/rektorov_rad/programirano/Prediction_Arima/a521a4cf-1ab8-40c7-8a9b-a63b7b910f3f_Data.csv')

#print(df.iloc[:-5,2:-1])
df=df.iloc[:-5,2:-1]

stupci=['Country Name','Country Code']
for i in df.columns[2:]:
    stupac=i.split(' ')[0]
    stupci.append(stupac)

#print(stupci)
df.columns=stupci

def stringops(text):
    if text=='..':
        return None
    else:
        return text

for i in df.columns:
    df[i]=df[i].apply(stringops)
#print(df.isnull().sum())
df=df.dropna()

for i in df.columns[2:]:
    df[i]=pd.to_numeric(df[i])

#print(df.isnull().sum())
#print(df.dtypes)
#print(df.shape)

croatia=df[df['Country Name']=='World'] # World or Croatia
croatia=croatia.T
croatia=croatia.iloc[2:,:]

#print(croatia)
#print(croatia.shape)
year=croatia.index.tolist()
data=croatia.iloc[:,0].tolist()
##fig=px.line(y=data,x=year,
##       template='plotly_dark',title='Total population',
##       labels=dict(x="Years", y="Total population"))
##fig.show()

croatia.index = pd.to_datetime(croatia.index)
croatia.iloc[:,0]=pd.to_numeric(croatia.iloc[:,0],downcast='float')
croatia.columns=['total_pop']

##izvor=croatia.copy()
##train2=croatia.iloc[:48,:]

world=croatia.copy()
##
##decomposition=sm.tsa.seasonal_decompose(croatia.iloc[:,0],model='additive')
##decomposition.plot()
##plt.show()
##
##length_train=48
##train=croatia.iloc[:length_train,:]
##test=croatia.iloc[length_train:,:]
###print(train.shape)
###print(test.shape)
###print(train)
##train.index = pd.to_datetime(train.index)
##train.iloc[:,0]=pd.to_numeric(train.iloc[:,0],downcast='float')
##train.columns=['urban_pop']
###print(test)
##test.index = pd.to_datetime(test.index)
##test.iloc[:,0]=pd.to_numeric(test.iloc[:,0],downcast='float')
##test.columns=['urban_pop']
#print(train.dtypes)
#print(test.dtypes)



# MANJE OD 0.05 JE STATIONARY
result=adfuller(croatia.iloc[:,0],autolag='AIC')
print(result[1])

croatia=pd.DataFrame(boxcox(croatia.iloc[:,0],lmbda=0),index=croatia.index)
izvor2=croatia.copy()
result=adfuller(croatia.iloc[:,0],autolag='AIC')
print(result[1])
year=croatia.index.tolist()
data=croatia.iloc[:,0].tolist()
##fig=px.line(y=data,x=year,
##       template='plotly_dark',title='Total population',
##       labels=dict(x="Years", y="Total population"))
##fig.show()
##
##croatia=pd.DataFrame(croatia-croatia.shift(periods=1),index=croatia.index)
##croatia.dropna(inplace=True)
###print(croatia)
##result=adfuller(croatia.iloc[:,0],autolag='AIC')
##print(result[1])
##year=croatia.index.tolist()
##data=croatia.iloc[:,0].tolist()
##fig=px.line(y=data,x=year,
##       template='plotly_dark',title='Urban population',
##       labels=dict(x="Years", y="Urban population"))
##fig.show()

##result2=kpss(croatia.iloc[:,0])
##print(result2[1])
##
croatia.iloc[:,0]=pd.to_numeric(croatia.iloc[:,0],downcast='float')
croatia.columns=['total_pop']

##plot_acf(croatia, ax=plt.gca(), lags=10)
##plt.show()
##
##plot_pacf(croatia, ax=plt.gca(), lags=20)
##plt.show()

length_train=48
train=croatia.iloc[:length_train,:]
test=croatia.iloc[length_train:,:]
#print(train.shape)
#print(test.shape)
#print(train)
train.index = pd.to_datetime(train.index)
train.iloc[:,0]=pd.to_numeric(train.iloc[:,0],downcast='float')
train.columns=['total_pop']
#print(test)
test.index = pd.to_datetime(test.index)
test.iloc[:,0]=pd.to_numeric(test.iloc[:,0],downcast='float')
test.columns=['total_pop']

stepwise_fit=auto_arima(croatia.iloc[:,0],trace=True,suppress_warnings=True,
                        start_p=0, d=None, start_q=0, max_p=5, max_d=5, max_q=5)
print(stepwise_fit.summary())



##model=ExponentialSmoothing(train.iloc[:,0], seasonal_periods=0, trend='additive')
##modelfit=model.fit(smoothing_level=0.2,smoothing_slope=0.04)
##yhw=test.copy()
##yhw['hwf']=modelfit.forecast(12)
##plt.plot(train,label='train')
##plt.plot(test,label='test')
##plt.plot(yhw['hwf'],label='pred')
##plt.legend()
##plt.show()
##rmse=sqrt(mean_squared_error(yhw['hwf'],test.iloc[:,0]))
##print(test.iloc[:,0].mean())
##print(rmse)



model=ARIMA(train.iloc[:,:],order=(1,2,0))
model=model.fit()
#print(model.summary())

start=len(train)
end=len(train)+len(test)-1
pred=model.predict(start=start,end=end,typ='levels')
#print(pred)

predvidanja=pred.to_frame()
indeksi=pred.index
#print(indeksi)
lista_pred=predvidanja.iloc[:,0].values
#print(lista_pred)

#pred.index=df.index[start:end+1]
plt.plot(train,label='train')
plt.plot(test,label='test')
plt.plot(pred,label='pred')
plt.legend()
plt.show()

rmse=sqrt(mean_squared_error(pred,test.iloc[:,0]))
print(test.iloc[:,0].mean())
print(rmse)

prave_vrijednosti1=[]
for i in lista_pred:
    prave_vrijednosti1.append(np.exp(i))

df2=pd.DataFrame({'totpop':prave_vrijednosti1})
df2.index=indeksi
print(df2)

plt.plot(world)
plt.plot(df2)
plt.show()

##
##b=izvor2.iloc[48,0]
##print(b)
####print(train2.iloc[-1,0])
##prave_vrijednosti0=[]
#####prave_vrijednosti0.append(b)
####
##for i in lista_pred:
##    b+=i
##    prave_vrijednosti0.append(b)
##
##print(prave_vrijednosti0)
##prave_vrijednosti1=[]
##
##for i in prave_vrijednosti0:
##    prave_vrijednosti1.append(np.exp(i))
##
##print(prave_vrijednosti1)
##
####print(len(prave_vrijednosti1))
####print(izvor.index[start:end+1])
##
##df2=pd.DataFrame({'totpop':prave_vrijednosti1})
##df2.index=indeksi
##print(df2)
##
##plt.plot(izvor)
##plt.plot(df2)
##plt.show()



model2=ARIMA(croatia.iloc[:,:],order=(1,2,0))
model2=model2.fit()
#print(model.summary())
#print(croatia.tail())

index_future_dates=pd.date_range(start='2019-01-01',end='2031-01-01',freq='YS')
#print(index_future_dates)
pred=model2.predict(start=len(croatia),end=len(croatia)+12,typ='levels').rename('ARIMA Preds')
pred.index=index_future_dates
print(pred)
indeksi=pred.index
predvidanja=pred.to_frame()
lista_pred=predvidanja.iloc[:,0].values
plt.plot(croatia,label='History')
plt.plot(pred,label='Future')
plt.legend()
plt.show()


prave_vrijednosti1=[]
for i in lista_pred:
    prave_vrijednosti1.append(np.exp(i))

df2=pd.DataFrame({'totpop':prave_vrijednosti1})
df2.index=indeksi
print(df2)

plt.plot(world)
plt.plot(df2)
plt.show()


##
##b=izvor2.iloc[-1,0]
##
##prave_vrijednosti0=[]
##
##for i in lista_pred:
##    b+=i
##    prave_vrijednosti0.append(b)
##
##prave_vrijednosti1=[]
##
##for i in prave_vrijednosti0:
##    prave_vrijednosti1.append(np.exp(i))
##
##df2=pd.DataFrame({'totpop':prave_vrijednosti1})
##df2.index=indeksi
##
##plt.plot(izvor)
##plt.plot(df2)
##plt.show()




###Function
##def invboxcox(y,ld):
##   if ld == 0:
##      return(np.exp(y))
##   else:
##      return(np.exp(np.log(ld*y+1)/ld))
##
### Test the code
##predictions_df=pred.to_frame()
##predictions_df=predictions_df.iloc[:,0]
###print(x)
##print(np.exp(predictions_df))
##predictions_df=(np.exp(predictions_df)).to_frame()
####ld = 0
####y = stats.boxcox(x,ld)
####print (invboxcox(y[0],ld))
##
##plt.plot(world,label='History')
##plt.plot(predictions_df,label='Future')
##plt.legend()
##plt.show()
##
##



fig = go.Figure()

fig.add_trace(go.Scatter(
    x=world.index,
    y=world.iloc[:,0],
    name = 'History',
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=df2.index,
    y=df2.iloc[:,0],
    name='Prediction',
    mode='lines'       
))
fig.update_layout(template='plotly_dark', title='Total Population')
fig.show()
