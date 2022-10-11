from asyncio.proactor_events import _ProactorBasePipeTransport
import streamlit as st
import pandas as pd
import matplotlib as plt
from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import seaborn as sns
from sklearn import metrics
import re
from sklearn.metrics import classification_report
from IPython import display
from PIL import Image
import csv
from sklearn.datasets import load_breast_cancer


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
    #print(patterns)

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

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def give_numpy(input_frame):
    target_in=input_frame.iloc[: , -1:]
    patterns=input_frame.iloc[:,:-1]
    patterns=patterns.apply(lambda x: x / x.max())
    patterns=patterns.to_numpy()
    target_in=target_in.to_numpy()
    target_in=np.squeeze(target_in)
    return target_in,patterns

def final_result(labels,w0,df_rbs):
    print(labels)
    output_df=pd.DataFrame(columns=['Weight_Class','Weight_Value','Sign','Corresponding Translation Rate[a.u.]','RBS_Name','RBS_Sequence'])
    for i in range(len(w0)):
        row=[]
        row.append(labels[i])
        row.append(abs(w0[i]))
        if (np.sign(w0[i]) > 0): 
            row.append("Positive")
        else:
            row.append("Negative")
        TIR_array=df_rbs['TIR_norm'].to_numpy()
        indx=np.abs(TIR_array - np.abs(w0[i])).argmin()
        print(df_rbs)
        from_rbs_df=df_rbs.iloc[indx,[2,0,1]].to_numpy()
        row_np=np.array(row)
        row_new=np.concatenate((row_np,from_rbs_df))
        output_df.loc[i]=row_new
    return output_df

def input_df(name,eta_f,epochs_f):
    temp_df=pd.DataFrame(columns=['Dataset Name','Learning Rate','Epochs'])
    substring=".csv"
    mod_name=re.sub(substring,"",name)
    temp_arr=np.array([mod_name,eta_f,epochs_f])
    temp_df.loc[0]=temp_arr
    return temp_df.reset_index(drop="True")

def perceptron_results(patterns,target_in,eta,epochs):
    size=np.shape(patterns)[1]
    ds_size=np.shape(patterns)[0]
    transf_p=np.zeros(2)
    transf_n=np.zeros(2) #den paizei rolo to negative part afou den tha exoume ws<0
    ATC_unit_level=100#uM
    # parameter numbers are approximated from values in Table 1
    # x=0 represents zero/low AHL input, x=AHL_unit_level for high AHL input
    transf_p[0]=transf_pos(x=0,K=5,y0=0.001,ym=1,n=7)
    transf_p[1]=transf_pos(x=ATC_unit_level,K=5,y0=0.001,ym=1,n=7)
    phi_pos=np.array([transf_p]*size)
    phi_neg=np.array([transf_n]*size)
    ATC_unit_level=100#uM
    w0,y0_sc,w0_sc,d0_sc=train_step1(target_in,size=size,eta=eta_,epochs=epochs,phi_pos=phi_pos,phi_neg=phi_neg,pos=False, cont_=True)
    y0_per=[]
    for i in range(len(y0_sc[epochs-1])):
        if(y0_sc[epochs-1][i]>0.5): y0_per.append([1])
        else:y0_per.append([0])
    accuracy=100*np.divide(np.squeeze(sum(x == y for x, y in zip(y0_per, target_in))),ds_size)
    y0_per=np.array(y0_per)
    y0_per=np.squeeze(y0_per)
    return w0,y0_sc[epochs-1],y0_per,accuracy

def process_input():
    df_rbs_bef=pd.read_csv("application_RBS_dataset.csv")
    df_rbs=df_rbs_bef.copy()
    TIR_MAX=(df_rbs['TIR'].max())/120
    df_rbs['TIR_norm']=((df_rbs['TIR'])/TIR_MAX).round(3)
    return df_rbs,df_rbs_bef

def download_dataset():
    sample_data={'X1':[0,0,0],'X2':[0.5,0.3,10],'X3':[0.2,0.4,0.6],'X4':[0.0,-4,5],'X5':[1,0,1],'Y':[0,0,1]}
    df_sample=pd.DataFrame(data=sample_data)
    csv = convert_df(df_sample)
    return csv

def download_seqs():
    sample_data_2={'RBS_Name':['rbs_sample_01','rbs_sample_02','rbs_sample_03'],'RBS_sequence':['ATTACC','CCACCA','GCGCAA'],'TIR':[300000,50000,1]}
    df_sample_2=pd.DataFrame(data=sample_data_2)
    csv_2 = convert_df(df_sample_2)
    return csv_2

def create_labels(input_df):
    labelss_=(input_df.columns.values).tolist()
    labels_=labelss_.remove('Y')
    return labels_

def import_breast_cancer(index):
    if(index==1):
        breast_cancer = sklearn.datasets.load_breast_cancer()
        data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
        data_norm=data.apply(lambda x: x / x.max())
        patterns=data_norm.to_numpy()
        data["class"] = breast_cancer.target
        target_in=breast_cancer.target
        return target_in,patterns

def import_rbs(uploaded_df_2,df_rbs_bef):
    df_rbs_bef=pd.concat([df_rbs_bef,uploaded_df_2])
    df_rbs_bef=df_rbs_bef.sort_values(by='TIR',ascending=False)
    df_rbs_bef.to_csv('application_RBS_dataset.csv', index=False)
    df_rbs=df_rbs_bef.copy()
    TIR_MAX=(df_rbs['TIR'].max())/120
    df_rbs['TIR_norm']=((df_rbs['TIR'])/TIR_MAX).round(3)
    return df_rbs_bef,df_rbs

def delete_rbs(text_input,df_rbs_bef,df_rbs):
    shape_in=np.shape(df_rbs_bef)[0]
    df_rbs_bef.drop(df_rbs_bef.loc[df_rbs_bef['RBS_Name']==text_input].index, inplace=True)
    shape_out=np.shape(df_rbs_bef)[0]
    df_rbs_bef.to_csv('application_RBS_dataset.csv', index=False)
    df_rbs.drop(df_rbs.loc[df_rbs['RBS_Name']==text_input].index, inplace=True)
    if(shape_in==shape_out): deleted=0
    else: deleted=1
    return df_rbs_bef,df_rbs,deleted

#main
#sidebar_1_2
st.sidebar.header("Give Perceptron Inputs")
uploaded_file = st.sidebar.file_uploader("Import Your Dataset")
eta_ = st.sidebar.slider('Learning Rate', 0.01, float(10))
epochs_=st.sidebar.slider('Epochs',1,1000)

st.sidebar.header("Modify RBS Library 👇")
with st.form("my-form", clear_on_submit=True):
    uploaded_file_2=st.sidebar.file_uploader("Import Your Sequences")

text_input=st.sidebar.text_input(
        label="Delete RBS Sequence, given its name",
        label_visibility="visible",
        disabled=False,
    )
st.sidebar.text("")

#center
st.header("PERspectives Design Application")
image = Image.open('header_image.jpg')
st.image(image)
st.text("")
st.text("")
st.markdown("""This application serves the purpose of fast and easy-to-execute implementation of the Biological Perceptron. Users should define:

""")
st.markdown("""
* **Dataset of interest:**. You can also download a sample dataset to test the application yourselves (link down below)
* **Learning Rate:** It should be inversely proportional to the Dataset samples for high accuracy scores
* **Number of epochs:** The higher they are the longer the training process

""")


st.markdown("""
After importing their dataset and the perceptron parameters, users can check accuracy metrics of the software model, the perceptron weights, as
well as corresponding RBS_Sequences for each class. Have fun!
""")

st.markdown("""
Furthermore, advanced users and designers are given the opportunity to import their own RBS values as an excel file in order to use the for future Perceptron training sessions.
A sample file is provided down below. In terms of strength,the desired value should be imported in Arbitrary Units, similar to 
the values given by DeNovo DNA Salis Lab.
"""
)

col1, col2 = st.columns([1,1])

with col1:
    st.download_button(
        label="Download sample dataset",
        data=download_dataset(),
        file_name='sample_dataset.csv',
        mime='text/csv',
    )
with col2:
    st.download_button(
        label="Download RBS Sequences",
        data=download_seqs(),
        file_name='sample_rbs_sequences.csv',
        mime='text/csv',
    )

st.write('---')
st.header("RBS Library")
placeholder = st.empty()
df_rbs,df_rbs_bef=process_input()
df_rbs_bef = df_rbs_bef.drop_duplicates()
placeholder.write(df_rbs_bef)
st.write('---')


if text_input is not None:
    if(text_input==""):
        st.sidebar.write("Null Input")
    else:
        uploaded_file_2=None
        uploaded_file=None
        st.sidebar.write("Requested Sequence Name to be deleted:",text_input)
        df_rbs_bef,df_rbs,deleted=delete_rbs(text_input,df_rbs_bef,df_rbs)
        if (deleted==0):
            st.sidebar.write("Requested Sequence not Found!")
            st.sidebar.text("")
        else:
            st.sidebar.write("Requested Sequence deleted successfully")
            st.sidebar.text("")
        df_rbs_bef = df_rbs_bef.drop_duplicates()
        placeholder.write(df_rbs_bef)
    text_input=None

if uploaded_file_2 is not None:
    uploaded_df_2= pd.read_csv(uploaded_file_2)
    df_rbs_bef,df_rbs=import_rbs(uploaded_df_2,df_rbs_bef)
    df_rbs_bef = df_rbs_bef.drop_duplicates()
    placeholder.write(df_rbs_bef)
    st.write('---')

if uploaded_file is not None:
    #computations
    inputs=input_df(uploaded_file.name,eta_,epochs_)
    uploaded_df= pd.read_csv(uploaded_file)
    labels_=uploaded_df.columns.values
    target_in,patterns=give_numpy(uploaded_df)

    w0,y0_an,y0_dig,accuracy=perceptron_results(patterns,target_in,eta_,epochs_)
    output_df=final_result(labels_,w0,df_rbs)
    conf_matrix = metrics.confusion_matrix(np.array(target_in), np.array(y0_dig))
    clf_report = classification_report(target_in, y0_dig, target_names=['No','Yes'],  output_dict=True)

    #SECTION_1 -> DISPLAY_INPUTS
    st.subheader('User Input features')
    st.write(inputs)
    st.write('---')

    #SECTION_2->ACCURACY METRICS
    #fig_1
    st.subheader('Accuracy Metrics')
    st.write('Overall Training Accuracy is:',accuracy)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax=sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['No', 'Yes']); ax.yaxis.set_ticklabels(['No', 'Yes'])
    st.pyplot(fig)
    
    #fig_2
    fig0, ax0 = plt.subplots(figsize=(5, 5))
    ax0=sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    ax0.set_title('Classification Report')
    st.pyplot(fig0)
    st.write('---')
    
    #SECTION_3 ->RBS_Converter
    st.subheader('RBS Weight Converter')
    st.write(output_df)
    
    #write_results
    #st.write(w0,y0_an,y0_dig,accuracy,output_df)





