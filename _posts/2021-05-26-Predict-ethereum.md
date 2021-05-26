# LSTM을 이용한 이더리움 시세 예측

```
이 게시물은 한양대학교 AI+X 딥러닝 과제로 작성되었습니다.
```



## RNN(Recurrent Neural Network)

![RNN](C:\Users\KJH\Desktop\github.io\iproj222.github.io\_posts\2021-05-26-Predict-ethereum\RNN.png)

 **RNN**은 시간의 흐름에 따라 관찰되는 **시계열 데이터 또는 입력과 출력을 시퀀스 단위로 처리**하기 위해 고안된 모델입니다. 피드 포워드 신경망(Feed Forward Neural Network)은 은닉층에서 활성화 함수를 지난 결과값이 출력층 방향으로 향합니다. 그러나 RNN은 은닉층에서 나온 결과값을 출력층 방향으로 보내면서도 다시 다음 은닉층의 입력으로 보내는 특징이 있습니다.  RNN의 은닉층에서 결과를 내보내는 역할을 하는 노드를 **셀(Cell)**이라고 합니다. 셀은 이전의 맥락을 기억하려하는 메모리 역할을 합니다. 

## LSTM(Long Short-Term Memory)

![LSTM](C:\Users\KJH\Desktop\github.io\iproj222.github.io\_posts\2021-05-26-Predict-ethereum\LSTM.png)

RNN은 **앞부분의 맥락이 길어질수록 앞부분의 정보가 충분히 전달되지 못하는 현상**이 있습니다. 활성화 함수로 tanh(hyperbolic tangent, 하이퍼볼릭 탄젠트)를 사용하기 때문입니다. tanh를 통과한 값은 -1과 1 사이이기 때문에 은닉층을 통과할수록 기울기가 사라지는 **기울기 소실 문제(vanishing gradient problem)**가 발생합니다.

**LSTM**은 RNN의 단점을 보완한 모델입니다. LSTM은 기존 RNN의 Cell에서 게이트를 추가하여 불필요한 메모리는 잊어버리고 중요한 메모리는 기억합니다. 입력게이트, 망각게이트, 출력게이트가 추가되었으며 RNN과 비교하여 긴 시퀀스의 입력을 처리할 때 월등한 성능을 보입니다. LSTM에 대한 자세한 설명은 [ratsgo's blog](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/) 를 참고하시기 바랍니다.

## bithumb API를 이용하여 차트 데이터 가져오기

```python
import requests
import pandas as pd
import time

data=requests.get('https://api.bithumb.com/public/candlestick/ETH_KRW/1h')

data=data.json()
data=data.get("data")
df=pd.DataFrame(data)
```

bithumb api를 사용하여 이더리움의 1시간 KRW 차트를 불러옵니다.



## 데이터 전처리

```python
import time

df.rename(columns={0:'time',1:'open', 2:'high',3:'low',4: 'close',5: 'volume'},\ inplace=True)
df=df[['time','open', 'close', 'high','low','volume']].astype("float")
df.reset_index(drop=True, inplace=True)
df["date"]=df["time"].apply(lambda x:time.strftime('%Y-%m-%d %H:%M',\ time.localtime(x/1000)))
df = df.rename(index=df["date"])
```

column들을 살펴보면 open은 시가, hign는 고가, low는 저가, close는 종가, volume은 거래량입니다.
df를 확인해보면 다음과 같습니다.

![df](C:\Users\KJH\Desktop\github.io\iproj222.github.io\_posts\2021-05-26-Predict-ethereum\df.png)

총 4340개의 행이 있습니다. 1시간 단위의 차트 데이터이므로 약 180일간의 이더리움 차트 데이터 입니다.



## 데이터 시각화 

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 9))
sns.lineplot(y=df['close'], x=df['date'])
plt.xlabel('time')
plt.ylabel('price')
```

차트를 시각화해보면 다음과 같습니다.

![eth chart](C:\Users\KJH\Desktop\github.io\iproj222.github.io\_posts\2021-05-26-Predict-ethereum\eth chart.png)

## 데이터 정규화 및 데이터셋 분리

```python
X = df.drop(columns='volume')
X = X.drop(columns='time')
X = X.drop(columns='date')
y = df.iloc[:, 2:3]
```



```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

mm = MinMaxScaler()
ss = StandardScaler()

X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y) 

# Train Data
X_train = X_ss[:4040, :]
X_test = X_ss[4040:, :]

# Test Data 
y_train = y_mm[:4040, :]
y_test = y_mm[4040:, :] 
```



```python
import torch
import torch.nn as nn
from torch.autograd import Variable 

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1,\ X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1,\ X_test_tensors.shape[1])) 
```



## 모델

```python
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU() 

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0),\ self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0),\ self.hidden_size)).to(device) #internal state   
        # Propagate input through LSTM
		
        #lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) 

        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output

        return out 
```



## 하이퍼 파라미터 구성

```python
num_epochs = 50000 #1000 epochs
learning_rate = 0.00001 #0.001 lr

input_size = 4 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]).to(device)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)  # adam optimizer
```



## 학습

```python
for epoch in range(num_epochs):
    outputs = lstm1.forward(X_train_tensors_final.to(device)) #forward pass
    optimizer.zero_grad() #caluclate the gradient, manually setting to 0

    # obtain the loss function
    loss = criterion(outputs, y_train_tensors.to(device))

    loss.backward() #calculates the loss of the loss function

    optimizer.step() #improve from loss, i.e backprop
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
```



## 예측 및 시각화

```python
df_X_ss = ss.transform(X)
df_y_mm = mm.transform(df.iloc[:, 1:2])

df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))
#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))
train_predict = lstm1(df_X_ss.to(device))#forward pass
data_predict = train_predict.data.detach().cpu().numpy() #numpy conversion
dataY_plot = df_y_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict) #reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(16,9)) #plotting
plt.axvline(x=4140, c='r', linestyle='--') #size of the training set

plt.plot(dataY_plot, label='Actuall Data') #actual plot
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')

plt.xlabel('Date')
plt.ylabel('Price')

xpoint = np.arange(0,5000,1000)
ypoint = np.arange(500000,5500000,500000)

xidx = [X.index[0][0:10],X.index[1000][0:10],X.index[2000][0:10],X.index[3000][0:10],X.index[4000][0:10]]
yidx = np.arange(500000,5500000,500000)

plt.xticks(xpoint, xidx)
plt.yticks(ypoint, yidx)

plt.legend()
plt.show() 
```



![predict chart](C:\Users\KJH\Desktop\github.io\iproj222.github.io\_posts\2021-05-26-Predict-ethereum\predict chart.png)

## 레퍼런스

https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/

https://wegonnamakeit.tistory.com/52

https://coding-yoon.tistory.com/131
