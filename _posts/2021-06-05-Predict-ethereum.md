---
search: false
title: LSTM을 이용한 Ethereum 시세 예측
categories: 
  - RNN
tags:
  - LSTM
  - Ethereum
toc: true  
last_modified_at: 2021-06-05T08:06:00-05:00
---



```
 이 게시물은 한양대학교 AI+X 딥러닝 과제로 작성되었습니다.
```





# I. RNN과 LSTM

## RNN(Recurrent Neural Network)

![RNN](\assets\images\2021-06-05-Predict-ethereum\RNN.png)

 **RNN**은 시간의 흐름에 따라 관찰되는 **시계열 데이터 또는 입력과 출력을 시퀀스 단위로 처리**하기 위해 고안된 모델입니다. 피드 포워드 신경망(Feed Forward Neural Network)은 은닉층에서 활성화 함수를 지난 결과값이 출력층 방향으로 향합니다. 그러나 RNN은 은닉층에서 나온 결과값을 출력층 방향으로 보내면서도 다시 다음 은닉층의 입력으로 보내는 특징이 있습니다.  RNN의 은닉층에서 결과를 내보내는 역할을 하는 노드를 **셀(Cell)**이라고 합니다. 셀은 이전의 맥락을 기억하려하는 메모리 역할을 합니다. 

## LSTM(Long Short-Term Memory)

![LSTM](\assets\images\2021-06-05-Predict-ethereum\LSTM.png)

RNN은 **앞부분의 맥락이 길어질수록 앞부분의 정보가 충분히 전달되지 못하는 현상**이 있습니다. 활성화 함수로 tanh(hyperbolic tangent, 하이퍼볼릭 탄젠트)를 사용하기 때문입니다. tanh를 통과한 값은 -1과 1 사이이기 때문에 은닉층을 통과할수록 기울기가 사라지는 **기울기 소실 문제(vanishing gradient problem)**가 발생합니다.

**LSTM**은 RNN의 단점을 보완한 모델입니다. LSTM은 기존 RNN의 Cell에서 게이트를 추가하여 불필요한 메모리는 잊어버리고 중요한 메모리는 기억합니다. 입력게이트, 망각게이트, 출력게이트가 추가되었으며 RNN과 비교하여 긴 시퀀스의 입력을 처리할 때 월등한 성능을 보입니다. LSTM에 대한 자세한 설명은 [ratsgo's blog](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/) 를 참고하시기 바랍니다.

# II. 차트 데이터 분석 및 전처리

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

df.rename(columns={0:'time',1:'open', 2:'high',3:'low',4: 'close',5: 'volume'},\
          inplace=True)
df=df[['time','open', 'close', 'high','low','volume']].astype("float")
df.reset_index(drop=True, inplace=True)
df["date"]=df["time"].apply(lambda x:time.strftime('%Y-%m-%d %H:%M',\
                                                   time.localtime(x/1000)))
df = df.rename(index=df["date"])
```

column들을 살펴보면 open은 시가, hign는 고가, low는 저가, close는 종가, volume은 거래량입니다.
df를 확인해보면 다음과 같습니다.

![df](\assets\images\2021-06-05-Predict-ethereum\df.png)

총 5001개의 행이 있습니다. 1시간 단위의 차트 데이터이므로 약 208일간의 이더리움 차트 데이터 입니다.



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

![eth_chart](\assets\images\2021-06-05-Predict-ethereum\eth_chart.png)

## 데이터 정규화 및 데이터셋 분리

```python
X = df.drop(columns='time')
X = X.drop(columns='date')
y = df.iloc[:, 2:3]
```

예측에 사용하지 않을 feature들을 지웁니다. 시간, 날짜 행을 삭제합니다. 

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

mm = MinMaxScaler()
ss = StandardScaler()

X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y) 

# Train Data
X_train = X_ss[:4801, :]
X_test = X_ss[4801:, :]

# Test Data 
y_train = y_mm[:4801, :]
y_test = y_mm[4801:, :] 
```

사이킷런의 MinMaxScaler, StandardScaler를 사용하여 데이터를 정규화합니다.
총 5001의 행 중 4801개만 학습에 사용합니다. 나머지 200개의 행은 test 데이터로 분리하여 모델을 평가하는데에 사용합니다.

```python
import torch
import torch.nn as nn
from torch.autograd import Variable 

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

X_train_tensors_final = torch.reshape(\
    X_train_tensors,(X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(\
    X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 
```



# III. 모델 및 하이퍼파라미터 구성

## 모델

```python
class LSTMmodel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.leakyrelu = nn.LeakyReLU() 
    
    def forward(self, x, hidden):
        # Forward propagate /LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        output, hidden = self.lstm(x, hidden) 
        
        # Decode the hidden state of the last time step
        output = self.fc(output[:, -1, :])
        output = self.leakyrelu(output)
        
        return output, hidden
    
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                weight.new_zeros(self.num_layers, bsz, self.hidden_size))
```



## 하이퍼 파라미터 구성

```python
num_epochs = 10000 #10000 epochs
learning_rate = 0.00001

input_size = 5 #number of features
hidden_size = 5 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
batch_size = 24

num_classes = 5 #number of output classes 
model = LSTMmodel(num_classes, input_size, hidden_size, num_layers).to(device)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
```

Loss function은 MSE(Mean-Squared Error, 평균 제곱 오차)
Optimizer는 Adam을 사용합니다.



# IV. 학습 및 시세 예측

## 학습

```python
def batchify(data, bsz):
    nbatch = data.size(0) // bsz

    data = data.narrow(0, 0, nbatch * bsz)
    
    data = data.view(-1, bsz,data.shape[2]).contiguous()
    return data.to(device)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
```

데이터를 배치사이즈만큼 배치화하는 함수와 hidden layer를 repackaging하는 함수를 선언합니다.

```python
for epoch in range(num_epochs):
    hidden = model.init_hidden(batch_size)
    for i in range(batch_X.shape[0]):
        data = batch_X[i].reshape(batch_size,1,5)
        target = batch_t[i+1].reshape(batch_size,1,5).detach()    
            
        output, hidden = model.forward(data,hidden)
        
        if i == batch_X.shape[0]-1:
            lasthidden = hidden
        
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad() 

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()
        
    if epoch % 100 == 0:    
        print("Epoch: %d, loss: %1.7f" % (epoch, loss.item())) 
```

학습을 진행합니다. t-1셀의 hidden state가 t셀의 입력으로 들어가는 것이 특징입니다.

## 예측 및 시각화

```python
st = batch_t.shape[0]-1
hidden = lasthidden
target = batch_t[st].reshape(batch_size,1,4).detach()    

predict = []

for i in range(st):
    predict.append(batch_t[i].cpu().numpy())
    
with torch.no_grad():
    for i in range(st, batch_t.shape[0]+3):
        output, hidden = model.forward(target,hidden) #forward pass
        target = output.reshape(batch_size,1,5)
        predict.append(output.cpu().numpy())
        
predict = np.array(predict).reshape(-1,5)
```

학습한 모델을 이용하여 72시간 후의 이더리움 시세를 예측해봅시다.
2021.06.05 기준으로 2021.06.08일까지의 시세를 예측합니다.

```python
df_y_mm = mm.transform(df.iloc[:,2:3])

df_y_mm = Variable(torch.Tensor(df_y_mm))
#reshaping the dataset
dataY_plot = df_y_mm.data.numpy()

data_predict = ss.inverse_transform(predict) #reverse transformation
data_predict = data_predict[::,1]
dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(16,9)) #plotting
plt.axvline(x=5001, c='r', linestyle='--') #size of the training set

plt.xlabel('Date')
plt.ylabel('Price')

#plt.plot(dataY_plot, label='Actuall Data') #actual plot
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Ethereum Prediction')
plt.xticks(np.arange(0,6000,1000),[X.index[0][0:10],X.index[1000][0:10],X.index[2000][0:10], \
                                   X.index[3000][0:10],X.index[4000][0:10],X.index[5000][0:10]])
plt.yticks(np.arange(500000,5500000,500000),np.arange(500000,5500000,500000))
plt.legend()
plt.show() 
```

빨간 점선 이후의 그래프가 LSTM 모델이 예측한 결과값입니다.
생각보다 예측 성능이 좋지는 않습니다.
사실 주가나 코인 시세등은 랜덤성이 강하고 차트 데이터보다 다른 요소들(뉴스 등)이 영향을 끼치기 때문에 LSTM으로 예측하기 쉽지 않은 것 같습니다. 

![predict_chart](\assets\images\2021-06-05-Predict-ethereum\predict_chart.png)



# 래퍼런스

[1] [https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/)

[2] [https://wegonnamakeit.tistory.com/52](https://wegonnamakeit.tistory.com/52)

[3] [https://coding-yoon.tistory.com/131](https://coding-yoon.tistory.com/131)
