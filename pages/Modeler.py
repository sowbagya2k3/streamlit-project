import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pickle
from sklearn.metrics import accuracy_score
df=st.session_state.dfFiltered
tabs=st.tabs(["New","Saved Model"])
model=st.sidebar.selectbox("select Model",["Logistic Regression","Linear Regression","Random Forest Classifier","Naive Bayes Classifier","KNN","Decision Tree"])
outcome=st.sidebar.selectbox("Select outcome columns",df.columns)
str=st.sidebar.multiselect("select feauture",df.columns)

with tabs[0]:
    if outcome and  model and str:
        x=df.loc[:,df.columns!=outcome]
        y=df.loc[:,df.columns==outcome]
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
        if model=="Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model=LogisticRegression()
            model.fit(xtrain,ytrain)
            output=model.predict(xtest)
            accuracy=accuracy_score(output,ytest)
            st.metric("accuracy of trained model",accuracy)
            model_name=st.text_input("Enter the Model Name")
            with st.expander("Save Model"):
                save=st.button("save the Model")
                if save and model_name is not None:
                    pickle.dump(model, open(model_name+".sav", 'wb'))
            
                    x.loc[:1].to_pickle(model_name+".pkl")
            with st.expander("Predict"):
                mydict={}
                for col in x.columns:
                    mydict[col]=int(st.text_input("enter value for "+col))
                
                mydict=[mydict]
                dfp=pd.DataFrame.from_dict(mydict)
                
                st.metric("predicted value",model.predict(dfp[:1]))
        elif model=="Linear Regression":
            from sklearn.linear_model import LinearRegression
            model=LinearRegression()
            model.fit(xtrain,ytrain)
            output=model.predict(xtest)
            accuracy=accuracy_score(output,ytest)
            st.metric("accuracy of trained model",accuracy)
            model_name=st.text_input("Enter the Model Name")
            with st.expander("Save Model"):
                save=st.button("save the Model")
                if save and model_name is not None:
                    pickle.dump(model, open(model_name+".sav", 'wb'))
            
                    x.loc[:1].to_pickle(model_name+".pkl")
            with st.expander("Predict"):
                mydict={}
                for col in x.columns:
                    mydict[col]=int(st.text_input("enter value for "+col))
                
                mydict=[mydict]
                dfp=pd.DataFrame.from_dict(mydict)
                
                st.metric("predicted value",model.predict(dfp[:1]))
        elif model=="Random Forest Classifier":
            from sklearn.ensemble import RandomForestClassifier
            model=RandomForestClassifier()
            model.fit(xtrain,ytrain)
            output=model.predict(xtest)
            accuracy=accuracy_score(output,ytest)
            st.metric("accuracy of trained model",accuracy)
            model_name=st.text_input("Enter the Model Name")
            with st.expander("Save Model"):
                save=st.button("save the Model")
                if save and model_name is not None:
                    pickle.dump(model, open(model_name+".sav", 'wb'))
            
                    x.loc[:1].to_pickle(model_name+".pkl")
            with st.expander("Predict"):
                mydict={}
                for col in x.columns:
                    mydict[col]=int(st.text_input("enter value for "+col))
                
                mydict=[mydict]
                dfp=pd.DataFrame.from_dict(mydict)
                
                st.metric("predicted value",model.predict(dfp[:1]))
        elif model=="Naive Bayes Classifier":
            from sklearn.naive_bayes import GaussianNB
            model=GaussianNB()
            model.fit(xtrain,ytrain)
            output=model.predict(xtest)
            accuracy=accuracy_score(output,ytest)
            st.metric("accuracy of trained model",accuracy)
            model_name=st.text_input("Enter the Model Name")
            with st.expander("Save Model"):
                save=st.button("save the Model")
                if save and model_name is not None:
                    pickle.dump(model, open(model_name+".sav", 'wb'))
            
                    x.loc[:1].to_pickle(model_name+".pkl")
            with st.expander("Predict"):
                mydict={}
                for col in x.columns:
                    mydict[col]=int(st.text_input("enter value for "+col))
                
                mydict=[mydict]
                dfp=pd.DataFrame.from_dict(mydict)
                
                st.metric("predicted value",model.predict(dfp[:1]))
        elif model=="KNN":
            from sklearn.neighbors import KNeighborsClassifier
            model=KNeighborsClassifier()
            model.fit(xtrain,ytrain)
            output=model.predict(xtest)
            accuracy=accuracy_score(output,ytest)
            st.metric("accuracy of trained model",accuracy)
            model_name=st.text_input("Enter the Model Name")
            with st.expander("Save Model"):
                save=st.button("save the Model")
                if save and model_name is not None:
                    pickle.dump(model, open(model_name+".sav", 'wb'))
            
                    x.loc[:1].to_pickle(model_name+".pkl")
            with st.expander("Predict"):
                mydict={}
                for col in x.columns:
                    mydict[col]=int(st.text_input("enter value for "+col))
                
                mydict=[mydict]
                dfp=pd.DataFrame.from_dict(mydict)
                
                st.metric("predicted value",model.predict(dfp[:1]))
        else:
            from sklearn.tree import DecisionTreeRegressor 
            model=DecisionTreeRegressor()
            model.fit(xtrain,ytrain)
            output=model.predict(xtest)
            accuracy=accuracy_score(output,ytest)
            st.metric("accuracy of trained model",accuracy)
            model_name=st.text_input("Enter the Model Name")
            with st.expander("Save Model"):
                save=st.button("save the Model")
                if save and model_name is not None:
                    pickle.dump(model, open(model_name+".sav", 'wb'))
            
                    x.loc[:1].to_pickle(model_name+".pkl")
            with st.expander("Predict"):
                mydict={}
                for col in x.columns:
                    mydict[col]=int(st.text_input("enter value for "+col))
                
                mydict=[mydict]
                dfp=pd.DataFrame.from_dict(mydict)
                
                st.metric("predicted value",model.predict(dfp[:1]))
