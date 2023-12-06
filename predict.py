import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_model.pkl' , 'rb') as file:
     data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le = data["le_education"]

def predict_page():
    st.title("Software Salary Prediction")
    
    st.header("We need some Information")
    
    countries = ("United States of America",
                "Germany",                                                  
                "United Kingdom of Great Britain and Northern Ireland",   
                "Canada",                                                   
                "India",                                                    
                "France",                                                   
                "Netherlands",                                              
                "Australia",                                                 
                "Brazil",                                                    
                "Spain",                                                     
                "Sweden",                                                    
                "Italy",                                                     
                "Poland"  ,                                                  
                "Switzerland"  ,                                            
                "Denmark",                                                    
                "Norway" ,                                                   
                "Israel" ,
    )

    education = (
       "Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
       "Less than Bachelors",
       "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
       "Post grad"
    )
    
    countries = st.selectbox("Country" , countries)
    education = st.selectbox("Education Level" , education)
    
    experience = st.slider("Years of Experience" , 0 , 50 , 3)
    
    Calculate = st.button("Predict Salary")
    if Calculate:
        X = np.array([[countries,education , experience ]])
        X[: , 0] = le_country.transform(X[: , 0])
        X[: , 1] = le.transform(X[: , 1])
        X = X.astype(float)
        
        salary = regressor.predict(X)
        st.subheader(f"Estimated Salary is ${salary[0]:.2f}")
        
        
    
