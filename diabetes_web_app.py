# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 22:10:14 2020

@author: Lenovo
"""

# Pima Indian Diabetes 

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
import matplotlib.image as mpimg
import pickle

image1 = mpimg.imread('fsc1.png')
image2 = mpimg.imread('im1.png')
image3 = mpimg.imread('im2.png')
image4 = mpimg.imread('lr0.png')
image5 = mpimg.imread('lr2.png')
image6 = mpimg.imread('lr3.png')
image7 = mpimg.imread('lr4.png')    
image8 = mpimg.imread('lr5.png')    
image9 = mpimg.imread('lr6.png')    
image10 = mpimg.imread('lr7.png')    
image11 = mpimg.imread('lr8.png')    
image12 = mpimg.imread('lr9.png')    
image13 = mpimg.imread('lr10.png')    
image14 = mpimg.imread('lr11.png')    
image15 = mpimg.imread('lr12.png')    
image16 = mpimg.imread('lr13.png')    
image17 = mpimg.imread('lr14.png')    
image18 = mpimg.imread('lr15.png')    
image19 = mpimg.imread('lr16.png')    
image20 = mpimg.imread('lr17.png')    
image21 = mpimg.imread('lr18.png')    
image22 = mpimg.imread('lr19.png')
image23 = mpimg.imread('lr20.png')  
  
image24 = mpimg.imread('l1.png')     
image25 = mpimg.imread('l2.png')
image26 = mpimg.imread('l3.png')
image27 = mpimg.imread('l4.png')
image28 = mpimg.imread('l5.png')
image29 = mpimg.imread('l6.png')
image30 = mpimg.imread('l7.png')
image31 = mpimg.imread('l8.png')
image32 = mpimg.imread('l9.png')
image33 = mpimg.imread('l10.png')
image34 = mpimg.imread('l11.png')
image35 = mpimg.imread('l12.png')
image36 = mpimg.imread('l13.png')
image37 = mpimg.imread('l14.png')
image38 = mpimg.imread('l15.png')
image39 = mpimg.imread('l16.png')

image40 = mpimg.imread('corr1.png')     
image41 = mpimg.imread('p1.png')     
imag42 = mpimg.imread('p2.png')     
image43 = mpimg.imread('p3.png')     
image44 = mpimg.imread('p4.png')     

#import pickle
#filename = 'finalized_model.sav'
#loaded_model = pickle.load(open(filename, 'rb'))

st.set_page_config(page_title='Diabetes Detection Machine Learning App',layout='wide')
st.write("""
# Diabetes Detection Machine Learning App

In this implementation, various **Machine Learning** algorithms are used in this app for building a **Classification Model** to **Detect Diabetes in Women**.
""")

data = pd.read_csv('diabetes.csv')
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']
#st.markdown('**1.2. Data splits**')
#st.write('Training set')

#with st.sidebar.header('1. Upload your CSV data'):


Selection = st.sidebar.selectbox("Select Option", ("Diabetes Detection App","Logistic Regression Implementation", "Source Code"))

if Selection == "Diabetes Detection App":
    
    st.markdown('**This is the Implentaion of the App**')
    
    #st.markdown('The Diabetes dataset used for training the model is:')
    #st.write(data.head(5))
    
    st.write("'Pregnancies' - Number of times pregnant")
    st.write("'Glucose' - Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    st.write("'BloodPressure' - Diastolic blood pressure (mm Hg)")
    st.write("'SkinThickness' - Triceps skin fold thickness (mm)")
    st.write("'Insulin' - 2-Hour serum insulin (mu U/ml)")
    st.write("'BMI' - Body mass index (weight in kg/(height in m)^2)")
    st.write("'DiabetesPedigreeFunction' - Diabetes pedigree function")
    st.write("'Age' - Age (years) (21+)")
     
    
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(data.head(5))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variables')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(y.name)
    
    #st.sidebar.header('2. Set Parameters'):
    Pegnancies = st.slider('Pegnancies', 0, 17, 0, 1)
    Glucose = st.slider('Glucose', 44, 199, 80, 1)
    BloodPressure = st.slider('BloodPressure', 24, 122, 60, 1)
    SkinThickness = st.slider('SkinThickness', 7, 99, 60, 1)
    Insulin = st.slider('Insulin', 14, 846, 60, 3)
    BMI = st.slider('BMI', 18.2, 67.1, 30.0, 0.1)
    DiabetesPedigreeFunction = st.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.01, 0.01)
    Age= st.slider('Age', 21, 81, 25, 1)
    
    
    X_test_sc = [[Pegnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]

    #logregs = LogisticRegression()
    #logregs.fit(X_train, y_train)
    #y_pred_st = logregs.predict(X_test_sc)
    
     
    load_clf = pickle.load(open('diabetes_clf.pkl', 'rb'))

# Apply model to make predictions
    prediction = load_clf.predict(X_test_sc)
    prediction_proba = load_clf.predict_proba(X_test_sc)
    
    answer = prediction[0]
    
    
    if answer == 0:

        st.title("**The prediction is that you don't have Diabetes**")
   
    else:   
        st.title("**The prediction is that you have Diabetes**")
        
    st.write('Note: This prediction is based on the Machine Learning Algorithm, Logistic Regression.')


elif Selection == "Logistic Regression Implementation":
    
    st.write("'Pregnancies' - Number of times pregnant")
    st.write("'Glucose' - Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    st.write("'BloodPressure' - Diastolic blood pressure (mm Hg)")
    st.write("'SkinThickness' - Triceps skin fold thickness (mm)")
    st.write("'Insulin' - 2-Hour serum insulin (mu U/ml)")
    st.write("'BMI' - Body mass index (weight in kg/(height in m)^2)")
    st.write("'DiabetesPedigreeFunction' - Diabetes pedigree function")
    st.write("'Age' - Age (years) (21+)")
    
    st.subheader('1. Dataset')
    st.info('Awaiting for CSV file to be uploaded.')
    
    if st.button('Press to Display the Logistic Regression Implementation'):
        st.markdown('The Diabetes dataset used for training the model is:')
        st.write(data.head(5))
        
        st.write("[This is the link for the code for Logistic Regression Classifier](https://github.com/ashinde8/Predicting-Pima-Indians-Diabetes/blob/main/Predicting%20Pima%20Indian%20Diabetes%20using%20Logistic%20Regression.ipynb)")
    
        st.markdown('**Variables**')
        st.write('Independent Variable')
        st.info(X.columns)
    
        st.write('Target Variable')
        st.info("Outcome")
    
        st.write("** Data Preprocessing - Imputing Mean Values **")
        st.image(image2)
        st.image(image3)
    
        st.write("**Train Test Split**")
        st.write("Training Set")
        st.info("(614, 8)")   
        
        st.write("Testing Set")
        st.info("(154, 8)") 
        
        st.write("""**Feature Scaling**""")
        st.image(image1)

        st.write("Logistic Regression Baseline Model")
        st.image(image4)
        
        st.write("Logistic Regression with Cross Validation")
        st.image(image5)
        
        st.write("Logistic Regression with KFold Cross Validation")
        st.image(image6)
        
        # lr4....
        #st.write("Logistic Regression with Cross Validation")
        st.write("lr4")
        st.image(image7)
        st.write("lr5")
        st.image(image8)
        st.write("lr6")
        st.image(image9)
        st.write("lr7")
        st.image(image10)
        st.write("lr8")
        st.image(image11)
        st.write("lr9")
        st.image(image12)
        st.write("lr10")
        st.image(image13)
        st.write("lr11")
        st.image(image14)
        st.write("lr12")
        st.image(image15)
        st.write("lr13")
        st.image(image16)
        st.write("lr14")
        st.image(image17)
        st.write("lr15")
        st.image(image18)
        st.write("lr16")
        st.image(image19)
        st.write("lr17")
        st.image(image20)
        st.write("lr18")
        st.image(image21)
        st.write("lr19")
        st.image(image22)
        st.write("lr20")
        st.image(image23)
        
        st.image(image24)
        st.image(image25)
        st.image(image26)
        st.image(image27)
        st.image(image28)
        st.image(image29)
        st.image(image30)
        st.image(image31)
        st.image(image32)
        st.image(image33)
        st.image(image34)
        st.image(image35)
        st.image(image36)
        st.image(image37)
        st.image(image38)
        st.image(image39)
        
else:
    
    st.subheader("Source Code")
    
    code = """

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.image as mpimg

image1 = mpimg.imread('fsc1.png')
image2 = mpimg.imread('im1.png')
image3 = mpimg.imread('im2.png')
image4 = mpimg.imread('lr0.png')
image5 = mpimg.imread('lr2.png')
image6 = mpimg.imread('lr3.png')
image7 = mpimg.imread('lr4.png')    
image8 = mpimg.imread('lr5.png')    
image9 = mpimg.imread('lr6.png')    
image10 = mpimg.imread('lr7.png')    
image11 = mpimg.imread('lr8.png')    
image12 = mpimg.imread('lr9.png')    
image13 = mpimg.imread('lr10.png')    
image14 = mpimg.imread('lr11.png')    
image15 = mpimg.imread('lr12.png')    
image16 = mpimg.imread('lr13.png')    
image17 = mpimg.imread('lr14.png')    
image18 = mpimg.imread('lr15.png')    
image19 = mpimg.imread('lr16.png')    
image20 = mpimg.imread('lr17.png')    
image21 = mpimg.imread('lr18.png')    
image22 = mpimg.imread('lr19.png')
image23 = mpimg.imread('lr20.png')  
  
image24 = mpimg.imread('l1.png')     
image25 = mpimg.imread('l2.png')
image26 = mpimg.imread('l3.png')
image27 = mpimg.imread('l4.png')
image28 = mpimg.imread('l5.png')
image29 = mpimg.imread('l6.png')
image30 = mpimg.imread('l7.png')
image31 = mpimg.imread('l8.png')
image32 = mpimg.imread('l9.png')
image33 = mpimg.imread('l10.png')
image34 = mpimg.imread('l11.png')
image35 = mpimg.imread('l12.png')
image36 = mpimg.imread('l13.png')
image37 = mpimg.imread('l14.png')
image38 = mpimg.imread('l15.png')
image39 = mpimg.imread('l16.png')

image40 = mpimg.imread('corr1.png')     
image41 = mpimg.imread('p1.png')     
imag42 = mpimg.imread('p2.png')     
image43 = mpimg.imread('p3.png')     
image44 = mpimg.imread('p4.png')  
import pickle
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

st.set_page_config(page_title='Diabetes Detection Machine Learning App',layout='wide')
st.write(
Diabetes Detection Machine Learning App

In this implementation, various **Machine Learning** algorithms are used in this app for building a **Classification Model** to **Detect Diabetes in Women**.
)

data = pd.read_csv('diabetes.csv')
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']
#st.markdown('**1.2. Data splits**')
#st.write('Training set')

#with st.sidebar.header('1. Upload your CSV data'):


Selection = st.sidebar.selectbox("Select Option", ("Diabetes Detection App","Exploratory Data Analysis","Logistic Regression Implementation", "Source Code"))

if Selection == "Diabetes Detection App":
    
    st.markdown('**This is the Implentaion of the App**')
    
    #st.markdown('The Diabetes dataset used for training the model is:')
    #st.write(data.head(5))
    
    st.write("'Pregnancies' - Number of times pregnant")
    st.write("'Glucose' - Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    st.write("'BloodPressure' - Diastolic blood pressure (mm Hg)")
    st.write("'SkinThickness' - Triceps skin fold thickness (mm)")
    st.write("'Insulin' - 2-Hour serum insulin (mu U/ml)")
    st.write("'BMI' - Body mass index (weight in kg/(height in m)^2)")
    st.write("'DiabetesPedigreeFunction' - Diabetes pedigree function")
    st.write("'Age' - Age (years) (21+)")
     
    
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(data.head(5))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variables')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(y.name)
    
    #st.sidebar.header('2. Set Parameters'):
    Pegnancies = st.slider('Pegnancies', 0, 17, 0, 1)
    Glucose = st.slider('Glucose', 44, 199, 80, 1)
    BloodPressure = st.slider('BloodPressure', 24, 122, 60, 1)
    SkinThickness = st.slider('SkinThickness', 7, 99, 60, 1)
    Insulin = st.slider('Insulin', 14, 846, 60, 3)
    BMI = st.slider('BMI', 18.2, 67.1, 30.0, 0.1)
    DiabetesPedigreeFunction = st.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.01, 0.01)
    Age= st.slider('Age', 21, 81, 25, 1)
    
    
    X_test_sc = [[Pegnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]

    logregs = LogisticRegression()
    logregs.fit(X_train, y_train)
    y_pred_st = logregs.predict(X_test_sc)
    
    answer = y_pred_st[0]
        
    if answer == 0:

        st.title("**The prediction is that you don't have Diabetes**")
   
    else:   
        st.title("**The prediction is that you have Diabetes**")
        
    st.write('Note: This prediction is based on the Machine Learning Algorithm, Logistic Regression.')

elif Selection == "Exploratory Data Analysis":
    
    st.write("ffff")

elif Selection == "Logistic Regression Implementation":
    
    st.write("'Pregnancies' - Number of times pregnant")
    st.write("'Glucose' - Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    st.write("'BloodPressure' - Diastolic blood pressure (mm Hg)")
    st.write("'SkinThickness' - Triceps skin fold thickness (mm)")
    st.write("'Insulin' - 2-Hour serum insulin (mu U/ml)")
    st.write("'BMI' - Body mass index (weight in kg/(height in m)^2)")
    st.write("'DiabetesPedigreeFunction' - Diabetes pedigree function")
    st.write("'Age' - Age (years) (21+)")
    
    st.subheader('1. Dataset')
    st.info('Awaiting for CSV file to be uploaded.')
    
    if st.button('Press to Display the Logistic Regression Implementation'):
        st.markdown('The Diabetes dataset used for training the model is:')
        st.write(data.head(5))
        
        st.write("[This is the link for the code for Logistic Regression Classifier](https://github.com/ashinde8/Predicting-Pima-Indians-Diabetes/blob/main/Predicting%20Pima%20Indian%20Diabetes%20using%20Logistic%20Regression.ipynb)")
    
        st.markdown('**Variables**')
        st.write('Independent Variable')
        st.info(X.columns)
    
        st.write('Target Variable')
        st.info("Outcome")
    
        st.write("** Data Preprocessing - Imputing Mean Values **")
        st.image(image2)
        st.image(image3)
    
        st.write("**Train Test Split**")
        st.write("Training Set")
        st.info("(614, 8)")   
        
        st.write("Testing Set")
        st.info("(154, 8)") 
        
        st.write("**Feature Scaling**")
        st.image(image1)

        st.write("Logistic Regression Baseline Model")
        st.image(image4)
        
        st.write("Logistic Regression with Cross Validation")
        st.image(image5)
        
        st.write("Logistic Regression with KFold Cross Validation")
        st.image(image6)
        
        # lr4....
        #st.write("Logistic Regression with Cross Validation")
        st.write("lr4")
        st.image(image7)
        st.write("lr5")
        st.image(image8)
        st.write("lr6")
        st.image(image9)
        st.write("lr7")
        st.image(image10)
        st.write("lr8")
        st.image(image11)
        st.write("lr9")
        st.image(image12)
        st.write("lr10")
        st.image(image13)
        st.write("lr11")
        st.image(image14)
        st.write("lr12")
        st.image(image15)
        st.write("lr13")
        st.image(image16)
        st.write("lr14")
        st.image(image17)
        st.write("lr15")
        st.image(image18)
        st.write("lr16")
        st.image(image19)
        st.write("lr17")
        st.image(image20)
        st.write("lr18")
        st.image(image21)
        st.write("lr19")
        st.image(image22)
        st.write("lr20")
        st.image(image23)
        
        st.image(image24)
        st.image(image25)
        st.image(image26)
        st.image(image27)
        st.image(image28)
        st.image(image29)
        st.image(image30)
        st.image(image31)
        st.image(image32)
        st.image(image33)
        st.image(image34)
        st.image(image35)
        st.image(image36)
        st.image(image37)
        st.image(image38)
        st.image(image39)
        

    """
    st.code(code, language='python')

st.sidebar.title("Created By:")
st.sidebar.subheader("Ashutosh Shinde")
st.sidebar.subheader("[LinkedIn Profile](https://www.linkedin.com/in/ashinde8/)")
st.sidebar.subheader("[GitHub Repository](https://github.com/ashinde8/Predicting-Pima-Indians-Diabetes)")         
       
    
