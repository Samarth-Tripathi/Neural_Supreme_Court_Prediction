import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cpu,floatX=float32'
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from random import randint
randint(2,9)
import csv
import numpy as np
docket=[]
overturn_actual=[]
with open('case_outcome_data.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
         docket.append(row['docket'])
         if (row['overturn_actual']=='False'):
            overturn_actual.append(0)
         else:
            overturn_actual.append(1)
            

docketId=[]
overturn_actual2=[]
naturalCourt=[]
adminAction=[]
caseOrigin=[]
caseOriginState=[]
caseSource=[]
caseSourceState=[]
lawType=[]
lcDispositionDirection=[]
lcDisposition=[]
lcDisagreement=[]
issue=[]
issueArea=[]
jurisdiction=[]
petitioner=[]
respondent=[]
certReason=[]
with open("SCDB_2013_01_caseCentered_Citation.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for line in reader:
        dkid=line['docketId']
        if dkid in docket:
            ind=docket.index(dkid)
            docketId.append(dkid)
            overturn_actual2.append(overturn_actual[ind])
            if line['naturalCourt']=='':
                naturalCourt.append(0.0)
            else:
                naturalCourt.append(float(line['naturalCourt'])/1704.0)
                
            if line['adminAction']=='':
                adminAction.append(0.0)
            else:
                adminAction.append(float(line['adminAction'])/125.0)
                
            if line['caseOrigin']=='':
                caseOrigin.append(0.0)
            else:
                caseOrigin.append(float(line['caseOrigin'])/600.0)
                
            if line['caseOriginState']=='':
                caseOriginState.append(0.0)
            else:
                caseOriginState.append(float(line['caseOriginState'])/62.0)
                
            if line['caseSource']=='':
                caseSource.append(0.0)
            else:
                caseSource.append(float(line['caseSource'])/600.0)
                
            if line['caseSourceState']=='':
                caseSourceState.append(0.0)
            else:
                caseSourceState.append(float(line['caseSourceState'])/62.0)
                
            if line['lawType']=='':
                lawType.append(0.0)
            else:
                lawType.append(float(line['lawType'])/9.0)
                
            if line['lcDispositionDirection']=='':
                 lcDispositionDirection.append(0.0)
            else:
                 lcDispositionDirection.append(float(line['lcDispositionDirection'])/3.0)
                 
            if line['lcDisposition']=='':
                lcDisposition.append(0.0)
            else:
                lcDisposition.append(float(line['lcDisposition'])/12.0)
                
            if line['lcDisagreement']=='':
                lcDisagreement.append(0.0)
            else:
                lcDisagreement.append(float(line['lcDisagreement'])/1.0)
                
            if line['issue']=='':
                issue.append(0.0)
            else:
                issue.append(float(line['issue'])/140080.0)
                
            if line['issueArea']=='':
                issueArea.append(0.0)
            else:
                issueArea.append(float(line['issueArea'])/14.0)
                
            if line['jurisdiction']=='':
                 jurisdiction.append(0.0)
            else:
                 jurisdiction.append(float(line['jurisdiction'])/15.0)
                 
            if line['petitioner']=='':
                  petitioner.append(0.0)
            else:
                  petitioner.append(float(line['petitioner'])/600.0)
                  
            if line['respondent']=='':
                respondent.append(0.0)
            else:
                respondent.append(float(line['respondent'])/600.0)
                
            if line['certReason']=='':
                 certReason.append(0.0)
            else:
                 certReason.append(float(line['certReason'])/13.0)



print len(certReason)

count=len(certReason)

ntrX = np.zeros(shape=(count,16))
ntrY = np.zeros(shape=(count,2))

#training data
print "train data"
for i in xrange(0,count):
    a=naturalCourt[i]
    b=adminAction[i]
    c=caseOrigin[i]
    d=caseOriginState[i]
    e=caseSource[i]
    f=caseSourceState[i]
    g=lawType[i]
    h=lcDispositionDirection[i]
    ii=lcDisposition[i]
    j=lcDisagreement[i]
    k=issue[i]
    l=issueArea[i]
    m=jurisdiction[i]
    n=petitioner[i]
    o=respondent[i]
    p=certReason[i]
    con=np.concatenate([[a],[b],[c],[d],[e],[f],[g],[h],[ii],[j],[k],[l],[m],[n],[o],[p]])
    ntrX[i]=con
    #print overturn_actual2[i]
    if overturn_actual2[i]==1:
        ntrY[i]=np.array([0,1])
    else:
        ntrY[i]=np.array([1,0])

from sklearn.utils import shuffle


ntrX,ntrY=shuffle(ntrX,ntrY)

nntrX=ntrX[:7500]
nntrY=ntrY[:7500]
nnteX=ntrX[7500:]
nnteY=ntrY[7500:]

print nntrX[0]
print nntrY[0]
print nnteX[0]
print nnteY[0]


print nntrY


srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_h2,w_h3, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))
    
    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))


    h2 = dropout(h2, p_drop_hidden)
    h3 = rectify(T.dot(h2, w_h3))

    h3 = dropout(h3, p_drop_hidden)
    py_x = softmax(T.dot(h3, w_o))
    return h, h2,h3, py_x






X = T.fmatrix()
Y = T.fmatrix()
nodes=10
w_h = init_weights((16, nodes))
w_h2 = init_weights((nodes, nodes))
w_h3=init_weights((nodes,nodes))
w_o = init_weights((nodes, 2))

noise_h, noise_h2,noise_h3, noise_py_x = model(X, w_h, w_h2,w_h3, w_o, 0.2, 0.5)
#h, h2,h3, py_x = model(X, w_h, w_h2,w_h3, w_o, 0., 0.)
y_x = T.argmax(noise_py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2,w_h3, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
maxi=0.0
for i in range(2000):
    for start, end in zip(range(0, len(ntrX), 128), range(128, len(ntrX), 128)):
      cost = train(nntrX[start:end], nntrY[start:end])
    k=(np.mean(np.argmax(nnteY, axis=1) == predict(nnteX)))
    print k
    maxi=max(k,maxi)
print "hi"
print maxi