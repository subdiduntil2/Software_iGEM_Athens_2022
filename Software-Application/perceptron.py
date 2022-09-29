import sklearn
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
import pandas as pd
from sklearn import metrics 

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()


#functions
def transf_pos(x,K=2.0,y0=0.001,ym=1.5,n=2):
    result=ym*(np.power((x/K),n))/(1+np.power((x/K),n))+y0
    return result

def examine_output(w,P,phi_pos,phi_neg):
    y_n=np.zeros(len(P))
    w_ph_=np.zeros(len(P))

    for n in range(0,len(P)):
        pttn=np.array(P[n])
        
        phi_nw=get_phinw(w,pttn,phi_pos,phi_neg)
                
        w_ph_[n]=np.dot(w,phi_nw)/len(w)
        y_n[n]=activate(w_ph_[n])
        
    return y_n,w_ph_

def output_fcn(pttn,fcn):
    CHL_levels=np.zeros(len(pttn))
    for i in range(0,len(pttn)):
        if (abs(pttn[i])<0.5):
            CHL_levels[i]=fcn[i][0]
        elif(abs(pttn[i])>=0.5):
            CHL_levels[i]=fcn[i][1]
    return CHL_levels

def activate(x):
    result_out=transf_pos(x,K=50,y0=0.001,ym=1.0,n=4)
    return float(result_out)

# see eq 4 in Supp page 7
def get_phinw(w,pttn,phi_pos,phi_neg):
    # conitnuous means wether negative weights are allowed to vary
    phi_nw=output_fcn(pttn,phi_pos)
    return phi_nw

def train_step1(target,size,eta,epochs,phi_pos,phi_neg,pos=False,cont_=False):
    rndstate=RandomState(2)
    #eta is learning rate as in eq.8 in Supp page 7, explained on page 8
    #phi_pos and phi_neg are positive and negative basis functions, respectively
    
    # setting initial weights; by default, weights are initialized to negative values
    # initial values affect how the algorithm converges.
    if cont_==True or pos==True:
        h=rndstate.random_sample(size)
    else:
        h=rndstate.random_sample(size)*(-1)

    r=1

    # w_series for weight vector, y_series to store output during iterations, delta_ for errors 
    w_series=[]
    y_series=[]
    delta_series=[]
    print(patterns)

    # iterate for 300 times; this is an arbitrary setting. 
    for it in range(0,epochs):
        y_n=np.zeros(len(patterns))
        g_n=np.zeros(len(patterns))
        delta_err=np.zeros(size)
        #phi_n=[]
      
        w=h.copy()
    
        # set the negative to one value -735. This is because only one negative weight circuit has been built
        # 735 comes from Table 1, row S_plux_rep, column beta_m
        #if cont_==False:
         #   w[w<0]=-735
        
        # if only positive weights are allowed, set the negative values arbitrarily to 0.1
        if pos==True:
            w[w<0]=0.01
            
        # for each pattern, calculating errors, eq 7 in Supp on page 7
        for n in range(0,len(patterns)):
            # consider pttn as indexes in vector x, 0 means x[0], 1 means x[1]
            pttn=np.array(patterns[n])
            
            # eq4 in Supp on page 7
            phi_nw=get_phinw(w,pttn,phi_pos,phi_neg)
            # h is the hidden variable
            g_n[n]=np.dot(w,pttn)
           # eq 6 in Supp
            y_n[n]=activate(g_n[n])   
            # right hand side of eq7 inside the summation 
            tmp=(y_n[n]-target[n])
            delta_err+=tmp*pttn
        w_series.append(w)
        h=h-eta*delta_err 
        y_series.append(y_n)
        delta_series.append(delta_err)        
    return w,y_series,w_series,delta_series

if 1:
    breast_cancer = sklearn.datasets.load_breast_cancer()
    data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
    data_norm=data.apply(lambda x: x / x.max())
    patterns=data_norm.to_numpy()
    data["class"] = breast_cancer.target
    target_in=breast_cancer.target
    '''
    patterns=[[0,0,0],
    [0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    target_in=[0,0,0,0,1,1,1,1]
    '''
    print(patterns)
    print(target_in)


    target_L=0.0
    target_H=1
    X=[]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                X.append([i,j,k])
    '''
    target_in= [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] #labels array
    patterns = [[2.7810836,2.550537003,1.15],
	[1.465489372,2.362125076,0.05],
	[3.396561688,4.400293529,10],
	[1.38807019,1.850220317,0.05],
	[3.06407232,3.005305973,1],
	[7.627531214,2.759262235,8],
	[5.332441248,2.088626775,22],
	[6.922596716,1.77106367,1],
	[8.675418651,0.242068655,14],
	[7.673756466,3.508563011,13]]
    '''
    
    
    ATC_unit_level=100#uM 

size=np.shape(patterns[0])[0]
# initialze transfer function parameters
transf_p=np.zeros(2)
transf_n=np.zeros(2) #den paizei rolo to negative part afou den tha exoume ws<0

# parameter numbers are approximated from values in Table 1
# x=0 represents zero/low AHL input, x=AHL_unit_level for high AHL input
transf_p[0]=transf_pos(x=0,K=5,y0=0.001,ym=1,n=7)
transf_p[1]=transf_pos(x=ATC_unit_level,K=5,y0=0.001,ym=1,n=7)


phi_pos=np.array([transf_p]*size)
phi_neg=np.array([transf_n]*size)


epochs_=100
w0,y0_sc,w0_sc,d0_sc=train_step1(target_in,size=size,eta=0.05,epochs=epochs_,phi_pos=phi_pos,phi_neg=phi_neg,pos=False, cont_=True)
y0_per=[]
for i in range(len(y0_sc[epochs_-1])):
    if(y0_sc[epochs_-1][i]>0.5): y0_per.append(1)
    else:y0_per.append(0)
accuracy=np.sum(y0_per == target_in)
print(y0_sc[epochs_-1])
print(w0)
print(accuracy)
confusion_matrix = metrics.confusion_matrix(np.array(target_in), np.array(y0_per))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
cm_display.plot()
plt.show() 





