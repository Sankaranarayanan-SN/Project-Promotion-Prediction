import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.title("Promotion Of Employee")
st.write("""client is a large MNC and they have 9 broad verticals across the organisation.
            One of the problem your client is facing is around identifying the right people 
            for promotion (only for manager position and below) and prepare them in time. 
            Currently the process, they are following is:
            
1.They first identify a set of employees based on recommendations/ past performance.
            
2.Selected employees go through the separate training and evaluation program for each vertical. 
  These programs are based on the required skill of each vertical.
            
3.At the end of the program, based on various factors such as training performance,KPI completion
  (only employees with KPIs completed greater than 60% are considered) etc., employee gets promotion.
            
For above mentioned process, the final promotions are only announced after the evaluation and this leads to 
delay in transition to their new roles. Hence, company needs your help in identifying the eligible candidates 
at a particular checkpoint so that they can expedite the entire promotion cycle.""")

DTC = pickle.load(open('DT.pkl', 'rb'))
LogReg = pickle.load(open('LogReg.pkl', 'rb'))
data = pd.read_csv('data//traindf.csv')

nav = st.sidebar.radio('Navigation', ['Home', 'Prediction', 'Insights'])
if nav == 'Home':
    st.title('ABC.INC')
    st.subheader('Promotion Prediction')
    st.image('data//img.jpg')
    if st.checkbox('Show Data'):
        st.dataframe(data)

if nav == 'Prediction':
    st.subheader('Please give the following information:')
    department = st.number_input('Department 0=Analytics 1=Finance 2=HR 3=Legal 4=Operations 5=Procurement 6=R&D '
                                 '7=Sales&marketing 8=Technology')
    education = st.number_input('Education  Bachelors:1,Masters & above:2,Below Secondary:3 ')
    gender = st.number_input('Gender 1 is Male 0 is Female')
    recruitment_channel = st.number_input('recruitment_channel other:1,sourcing:2,referred:3')
    no_of_trainings = st.number_input('no_of_trainings')
    age = st.number_input('age')
    previous_year_rating = st.number_input('previous_year_rating')
    length_of_service = st.number_input('length_of_service')
    KPIs_met = st.number_input('KPIs_met')
    Awards = st.number_input('awards, if yes then 1 no means 0')
    avg_training_score = st.number_input("avg_training_score")

    x = np.array([department, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating,
                  length_of_service, KPIs_met, Awards, avg_training_score])
    x = x.reshape(1, 11)

    st.header("Select The Classifier")
    CLS = st.sidebar.radio('Algorithm', ['DecisionTreeClassifier', 'LogisticsRegression', "ALL"])

    if CLS == "DecisionTreeClassifier":
        if st.button('Predict'):
            S = DTC.predict(x)
            if S == 0:
                st.header("Based on features gender is Female")
            else:
                st.header("Based on features gender is male")

    if CLS == "LogisticsRegression":
        if st.button('Predict'):
            S = LogReg.predict(x)
            if S == 0:
                st.header("Based on features gender is Female")
            else:
                st.header("Based on features gender is male")

    if CLS == "ALL":
        if st.button('Predict'):
            st.header("LogisticRegression")
            S = LogReg.predict(x)
            if S == 0:
                st.header("Based on features gender is Female")
            else:
                st.header("Based on features gender is male")

            st.header("DecisionTreeClassifier")
            C = DTC.predict(x)
            if C == 0:
                st.header("Based on features gender is Female")
            else:
                st.header("Based on features gender is male")

if nav == 'Insights':
    st.title('Insights from the dataset')
    st.image('data//correlation.jpg')

    st.image('data//agehistplot.jpg')
    st.write("This plot shows people age group of people")

    st.image('data//award.jpg')
    st.write(" Promoted Vs Awards Acquired")

    st.image("data//dept.jpg")
    st.write("DepartmentsWise employees Selected for promotion")

    st.image("data//department.jpg")
    st.write("Department vs promoted")

    st.image("data//education.jpg")
    st.write("Education Vs gender")

    st.image("data//g.jpg")
    st.write("Gender vs Promoted")

    st.image("data//gdept.jpg")
    st.write("Gender wise department count")

    st.image("data//previous.jpg")
    st.write("Previous year rating vs Promoted")

    st.image("data//edu.jpg")
    st.write("education vs promoted")

    st.image("data//trainingscore1.jpg")
    st.write("training score vs promoted")
