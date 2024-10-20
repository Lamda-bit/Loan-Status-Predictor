import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_clean_data():
    data = pd.read_csv('data/loan.csv')
    unclean = pd.read_csv('data/loan.csv')
    pd.set_option('future.no_silent_downcasting', True)
    data.dropna(inplace=True)
    data['Loan_Status'] = data['Loan_Status'].map({ 'Y': 1, 'N': 0 })
    #data.replace({'Loan_Status':{'Y':1, 'N':0}}, inplace= True)
    data.replace(to_replace='3+', value=4, inplace=True, )
    data.replace({'Married':{'Yes':1, 'No':0}, 'Gender':{'Male':1, 'Female':0}, 'Self_Employed':{'Yes':1, 'No':0},
                      'Property_Area':{'Rural':0, 'Semiurban': 1, 'Urban':2}, 'Education': {'Graduate':1,'Not Graduate':0}}, inplace=True)
    #print(data)

    return data, unclean

username = 'Lamda'
url = "www.linkedin.com/in/tettey-collins-kwabena-10351332a"
#contact = "LinkedIn Profile:  [Link](%s)" % url
contact = "LinkedIn: [Link](https://www.linkedin.com/in/tettey-collins-kwabena-10351332a)"

imagepath = r'assets/images/croped.PNG'

def user_card(username, contact, imagepath):
    img_col, empt, contact_col = st.columns([0.5,0.1, 4])
    
    with img_col:
        st.image(imagepath, caption='Profile Picture')
    with contact_col:
        st.write('')
        st.write('Username: ', username)
        st.write(contact, unsafe_allow_html=True)
    

def table(data):
    columns = st.columns(5)
    
    dict={}
    with columns[0]:
        st.write('Gender')
        label = 'Gender'
        male = st.checkbox(label='Male')
        female = st.checkbox(label='Female')

        if male and female:
            st.warning(body='Check one box')
        elif male:
            
            dict[label]=1
        else:
            dict[label]=0

    with columns[1]:
  
        st.write('Marital Status')
        label='Married'
        maried = st.checkbox(label='Maried')
        single = st.checkbox(label='Single')

        if maried and single:
            st.warning(body='Check one box')
        elif maried:
            
            dict[label]=1
        else:
            dict[label]=0

    with columns[2]:
        st.write('Education')
        label = 'Education'
        graduate = st.checkbox(label='Graduate')
        not_graduate = st.checkbox(label='Not Graduate')

        if graduate and not_graduate:
                    st.warning(body='Check one box')

        elif graduate:      
            dict[label]=1

        else:
            dict[label]=0       
        
    
            

    with columns[3]:
        st.write('Self Employed')
        label = 'Self_Employed'
        positive = st.checkbox(label='Yes')
        negative = st.checkbox(label='No')

        if positive and negative:
            st.warning(body='Check one box')
        elif positive:
            
            dict[label]=1
        else:
            dict[label]=0

    with columns[4]:
        st.write('Credit History')
        label = 'Credit_History'
        positive = st.checkbox(label='1')
        negative = st.checkbox(label='0')

        if positive and negative:
            st.warning(body='Check one box')
        elif positive:
            
            dict[label]=1
        else:
            dict[label]=0


    st.write('---')

    row2 = st.columns(6)
    
    with row2[0]:
       label = 'ApplicantIncome'
       value = st.number_input(
           label='Applicant Income',
           min_value=float(0),
           value=float(),
           placeholder='Enter income',
           step=50.0

       )
    dict[label]=value


    with row2[1]:
        label = 'Dependents'
        value=st.number_input(
           label='Dependents',
           min_value=0,
           max_value=100,
           value=0,
           placeholder='Enter Dependents',
           step=2
       )
        if value > 2:
            value = 4
        
    dict[label]=value

    with row2[2]:
       label = 'CoapplicantIncome'
       value = st.number_input(
           label='Coapplicant Income',
           min_value=float(0),
           value=float(),
           placeholder='Enter co-applicant income',
           step=10.0

       )
    dict[label]=value

    with row2[3]:
       label = 'LoanAmount'
       value = st.number_input(
           label='Loan Amount',
           min_value=float(0),
           value=float(),
           placeholder='Enter loan amount',
           step=10.0

       )
    dict[label]=value   

    with row2[4]:
       label = 'Loan_Amount_Term'
       value = st.number_input(
           label='Loan Amount Term',
           min_value=float(0),
           value=float(),
           placeholder='Enter loan term',
           step=30.0

       )
    dict[label]=value 

    with row2[5]:
        Area = st.selectbox('Property Area',('Rural','Semiurban','Urban'),placeholder='Choose an area')
        label = 'Property_Area'

        if Area=='Rural' :
            dict[label]=0

        elif Area == 'Urban':
            dict[label]=2
        else:
            dict[label]=1

    

    return dict  


def frames(data, unclean):
    expander = st.expander("Data frame used for project", expanded=False)
    with expander:
        st.subheader('Cleaned dataset')
        st.write(data)
        columns = st.columns(2)
        with columns[0]:
            st.subheader('Statistics Summary')
            st.write(data.describe())
        with columns[1]:
            st.subheader('Summary Chart')
            st.text('Feel free to zoom in and out of chart')
            st.line_chart(data.describe())
        st.subheader('Raw dataset')
        st.write(unclean)
        

def chart(dict):
    st.line_chart(data=dict)


def predictor(sorted_dict):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    input = (1,1,1,1,0,4583,1508,128,360,1,0) 
    # Not Deserve 1,1,4,1,0,3036,2504,158,360,0,0
    input_array = np.asarray(list(sorted_dict.values())).reshape(1, -1)
    #input_array = np.asarray(input).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    #print(input_array_scaled)
    print(input_array[0])
    position = input_array[0]
    print(position[0])
    prediction = model.predict(input_array_scaled)

    if (position[5] < position[6]) and position[9]==1:

        st.write("Applicant is:")
        print('Does not deserve a loan')
        print('')
        st.write("<span class='b Not Qualified'>Not Qualified</span>", unsafe_allow_html=True)
        st.write("<span class='c'>Prediction by org. policy</span>", unsafe_allow_html=True)
    
    elif (position[5] < 4500) and position[9]==1 and position[6]==0 and position[10]==0:

        st.write("Applicant is:")
        print('Does not deserve a loan')
        print('')
        st.write("<span class='b Not Qualified'>Not Qualified</span>", unsafe_allow_html=True)
        st.write("<span class='c'>Prediction by org. policy</span>", unsafe_allow_html=True)

    elif position[6]==0 and position[9]==1 and position[7]>50 and position[7]<100 and position[10]==0:

        st.write("Applicant is:")
        print('Does not deserve a loan')
        print('')
        st.write("<span class='b Not Qualified'>Not Qualified</span>", unsafe_allow_html=True)
        st.write("<span class='c'>Prediction by org. policy</span>", unsafe_allow_html=True)

    elif position[6]==0 and position[9]==1 and position[7]>200 and position[10]==1:

        st.write("The prediction is:")
        print('Does not deserve a loan')
        print('')
        st.write("<span class='b Not Qualified'>Not Qualified</span>", unsafe_allow_html=True)
        st.write("<span class='c'>Prediction by org. policy</span>", unsafe_allow_html=True)
    

    elif position[6]==0 and position[9]==1 and position[3]==0:

        st.write("Applicant is:")
        print('Does not deserve a loan')
        print('')
        st.write("<span class='b Not Qualified'>Not Qualified</span>", unsafe_allow_html=True)
        st.write("<span class='c'>Prediction by org. policy</span>", unsafe_allow_html=True)

    elif position[0]==1 and position[7]< 1 and position[9]==1 and position[2]:

        st.write("Applicant is:")
        print('Does not deserve a loan')
        print('')
        st.write("<span class='b Not Qualified'>Not Qualified</span>", unsafe_allow_html=True)
        st.write("<span class='c'>Prediction by org. policy</span>", unsafe_allow_html=True)


    else:
        st.write("Applicant is:")

        if prediction[0] == 0:
            print('Does not deserve a loan')
            print('')
            st.write("<span class='b Not Qualified'>Not Qualified</span>", unsafe_allow_html=True)
            st.write("<span class='a'>Prediction by ML regression</span>", unsafe_allow_html=True)
        else:
            print('Deserves a Loan')
            print('')
            st.write("<span class='b Qualified'>Qualified</span>", unsafe_allow_html=True)
            st.write("<span class='a'>Prediction by ML regression</span>", unsafe_allow_html=True)
            
        
            st.write("Probability of not being qualified: ", model.predict_proba(input_array_scaled)[0][0])
            st.write("Probability of being qualified: ", model.predict_proba(input_array_scaled)[0][1])



def sort_dict(dict):
    sorted_dict = {
        'Gender': 0, 
        'Married': 0, 
        'Dependents': 0, 
        'Education': 0, 
        'Self_Employed': 0, 
        'ApplicantIncome': 0.0,  
        'CoapplicantIncome': 0.0,  
        'LoanAmount': 0.0, 
        'Loan_Amount_Term': 0.0, 
        'Credit_History': 0,  
        'Property_Area': 0
    }

    for key, value in dict.items():
        for sortedkey, sortedvalue in sorted_dict.items():
            if key == sortedkey:
                sorted_dict[sortedkey]=value
        
    return sorted_dict
        


def main():

    data, unclean = get_clean_data()
    
    st.set_page_config(
        page_title='Loan Eligibility Checker',
        page_icon='',
        layout='wide',
        initial_sidebar_state='collapsed'
    )
    
    with open('assets/style/style.css') as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    #st.write("<span class='h'>Loan Eligibility Checker</span>", unsafe_allow_html=True)
    st.header('Loan Eligibility Checker')
    
    user_card(username, contact, imagepath)
    
    dict = table(data)
    sorted_dict = sort_dict(dict)

    
    st.write('---')

    column1, column2 = st.columns([1.5,1])
    with column1:
        st.subheader('Loan Status Visualiser')
        chart(sorted_dict)

    with column2:
        st.subheader('Loan Status Predictor')
        predictor(sorted_dict)

    st.write('---')

    frames(data, unclean)
    #print(list(dict.values()))
    #print(dict)
    print('')
    d = list(dict.values())
    #print(d)
    #st.write(dict)
    #st.write(sorted_dict)
    #print(sorted_dict)
    print(list(sorted_dict.values()))


if __name__=='__main__':
    main()