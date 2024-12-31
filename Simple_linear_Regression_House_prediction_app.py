#importing the libraries
import streamlit as st
import pandas as pd              #this model needs improvement due to its very low accuracy mostly needs improvement in the step of preprocessing.import numpy as np
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings('ignore')
#load the dataset
data=pd.read_csv("D:/Machine Learning/datasets to work with/housing price dataset.csv")
print(data.columns)
total=0
for each in data.columns:
    if data[each].dtype=="object" or data[each].dtype=="category":
        print(each)
        total+=1
print(total)
def train_model():
    #deciding features and the predictable column
    X=data[["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition",'sqft_above','yr_built', 'yr_renovated']]
    y=data["price"]

    #preprocessing
    scaler=StandardScaler()
    cols_to_scale = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',"view", 'condition', 'sqft_above','yr_built', 'yr_renovated']

    X.loc[:,cols_to_scale]=scaler.fit_transform(X[cols_to_scale])  #.loc is used to avoid warning that appears in its absence

    #train test split
    X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=0.8, test_size=0.2, random_state=42)

    #call the model
    model=LinearRegression()

    #train the model
    model.fit(X_train,y_train)

#predict the model results
    y_pred=model.predict(X_test)

#evaluate the model errors'
    print("MSE= ",mean_squared_error(y_test,y_pred))
    print("RMSE= ",np.sqrt(mean_squared_error(y_test,y_pred)))
    print("r2 score= ",r2_score(y_test,y_pred))
    return model
def main():
    st.title("Simple Linear Regression House prediction App")
    st.write("Put the following information to predict the price of house")
    model=train_model()
    n_bedrooms=st.number_input("Enter the number of bedrooms in the house",1,20,key='bedrooms')
    n_bathrooms=st.number_input("Enter the number of bathrooms in the house",1,20,key='bathrooms')
    sqft_lot=st.number_input("Enter the total area size of the house in sqft",min_value=100,max_value=60000,key='sqft_lot')
    sqft_living=st.number_input("Enter the living area size of the house in sqft",min_value=100,max_value=sqft_lot,key='sqft_living')
    n_floors=st.number_input("Enter the number of floors in the house(Considering ground floor as 0 floor)",0,30,key='floors')
    water_front=st.selectbox('Does the house have water view in front?',options=[0,1])
    st.info("O means not waterfront and 1 means have waterfront")
    view=st.select_slider("Select the number of stars for the view of the house",[0,1,2,3,4],key="View")
    condition=st.select_slider("Select the number of stars for the condition of the house",[1,2,3,4,5],key="condition")
    sqft_above=st.number_input("Enter the area size of the house that is constructed above in sqft",min_value=0,max_value=sqft_living,key='sqft_above')
    year_built=st.number_input("Enter the year in which it was constructed",1900,2024,key='Year_built')
    year_renovted=st.number_input("Enter the year in which it was renovted(if no then write the year of construction)",min_value=year_built,max_value=2024,key="year_renovated")
    if st.button("Predict Price"):
    #X=data[["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition",'sqft_above', 'yr_built', 'yr_renovated',
    #    'long'] 0.6045886752294308]
        predicted_price=model.predict([[n_bedrooms,n_bathrooms,sqft_living,sqft_lot,n_floors,water_front,view,condition,sqft_above,year_built,year_renovted]])
        st.success(f"Estimated price:${predicted_price[0]:,.2f}")  

main()
      

