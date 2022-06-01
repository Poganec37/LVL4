import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy


#вход а 2 направления
INPUT_X=2
#переход t1 (W1,b1)
OUT_H1=5
#переход t2 (W2,b2)
OUT_H2=6
#светофор t3 (W3, b3)
OUT_t3=3

  
# весы
# x --> t1 Вход
W1=np.random.rand(INPUT_X,OUT_H1)  # (2x5)
b1=np.random.rand(1,OUT_H1) # (1x5)
# H1 -->t2 Множества
W2=np.random.rand(OUT_H1,OUT_H2) # (5x6)
b2=np.random.rand(1,OUT_H2) # (1x6)
# H2 --> t3 Правила
W3=np.random.rand(OUT_H2,OUT_t3) # (6x3)
b3=np.random.rand(1,OUT_t3) # (1x3)
# H2 --> t4  Критик
W4=np.random.rand(OUT_H2,OUT_t3) # (6x3)
b4=np.random.rand(1,OUT_t3) # (1x3)


W1=(W1-0.5)*2*np.sqrt(1/INPUT_X)
b1=(b1-0.5)*2*np.sqrt(1/INPUT_X)
W2=(W2-0.5)*2*np.sqrt(1/OUT_H2)
b2=(b2-0.5)*2*np.sqrt(1/OUT_H2)
W3=(W3-0.5)*2*np.sqrt(1/OUT_t3)
b3=(b3-0.5)*2*np.sqrt(1/OUT_t3)
W4=(W3-0.5)*2*np.sqrt(1/OUT_t3)
b4=(b3-0.5)*2*np.sqrt(1/OUT_t3)


def sigmoid(t):
    return 1/(1+np.exp(-t))

def sigmoid_deriv(t):
    return sigmoid(t)*(1-sigmoid(t))


def softmax(t):
    out=np.exp(t)
    return out/np.sum(out)

def to_full(y,num_classes):
    y_full=np.zeros((1,num_classes))
    y_full[0,y]=1
    return y_full

def sparse_cross_entropy(z,y):
    return -np.log(z[0,y])

def sparse_cross_entropy_batch(z,y):
    return -np.log(np.array([z[j,y[j]] for j in range(len(y))]))

def softmax_batch(t):
    out=np.exp(t)
    return out/np.sum(out,axis=1,keepdims=True)

loss_arr=[]
RAZ=150
NUM_EPOCHS=25
ALPHA= 0.053
BETA=0.06

ACCs=0

for i in range(0, NUM_EPOCHS):
    correct=0
    for j in range(0,RAZ):
        b= np.random.randint(15,20)
        a= np.random.randint(0,6)
        x= np.array([[a,b]])
        if(0<=a<7)and(0<=b<7): y=0
        if(6<a<15)and(0<=b<7): y=1
        if(6<b<15)and(0<=a<7): y=1
        if(6<b<10)and(6<a<10): y=1
        if(14<a<21)and(6<b<15): y=2
        if(14<b<21)and(6<a<15): y=2
        if(14<a<21)and(14<b<21): y=2
        if(14<a<21)and(0<=b<7): y=2
        if(14<b<21)and(0<=a<7): y=2 

        #Forward
        t1= x@W1 + b1
        h1= sigmoid(t1)
        t2= h1@W2 + b2
        h2= sigmoid(t2)
        t4= h2@W4 + b4
        h4=softmax(t4)
        y_krit=np.argmax(h4)
        t3= h2@W3 + b3
        z=softmax(t3)
        y_pred= np.argmax(z)
        if y_pred == y:
            correct +=1
        ACCs=correct/RAZ
        E= np.sum(sparse_cross_entropy(z,y))
        loss_arr.append(E)        

        #Backend
        y_full=to_full(y,OUT_t3)  
        dE_dt4 = h4-y_full
        dE_dW4 = h2.T@dE_dt4  
        dE_db4=np.sum(dE_dt4,axis=0, keepdims=True)        
        dE_dt3 = z-y_full
        dE_dW3 = h2.T@dE_dt3
        dE_db3=np.sum(dE_dt3,axis=0, keepdims=True)  
        dE_dh2=dE_dt3@W3.T
        dE_dt2=dE_dh2*sigmoid_deriv(t2)
        dE_db2=np.sum(dE_dt2,axis=0, keepdims=True)
        dE_dW2 = h1.T@dE_dt2
        dE_dh1=dE_dt2@W2.T
        dE_dt1=dE_dh1*sigmoid_deriv(t1)
        dE_db1=np.sum(dE_dt1,axis=0, keepdims=True)
        dE_dW1=x.T@dE_dt1

        if (ACCs>=0.85):
            print('Эпоха ',i)
            print('X= '); print(x)
            print('Y= ',y)   
            print('Критик ',y_krit)         
            print('Z= ',z)
            print('ВЕСА до Update')
            print('W1= '); print(W1)
            print('W2= '); print(W2)
            print('b1= '); print(b1)
            print('b2= '); print(b2)
            print('W3= '); print(W3)
            print('b3= '); print(b3)    
            print('W4= '); print(W4)
            print('b4= '); print(b4)            
            print('Оценка Эпохи ',i,ACCs)
        
        #Update
        W1=W1-ALPHA*dE_dW1
        b1=b1-ALPHA*dE_db1
        W2=W2-ALPHA*dE_dW2
        b2=b2-ALPHA*dE_db2
        W4=W4-ALPHA*dE_dW4
        b4=b4-ALPHA*dE_db4        
        W3=W3-ALPHA*dE_dW3+BETA*W4  
        b3=b3-ALPHA*dE_db3+BETA*b4


plt.title('ошибка системы')
plt.plot(loss_arr)
plt.show()
