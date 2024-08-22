from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app=Flask(__name__)
data=pd.read_csv('final_dataset.csv')
#pipe=pickle.load(open("ridgemodel.pkl",'rb'))
pipe=pickle.load(open("randomForestmodel.pkl",'rb'))


data['beds'] = data['beds'].astype(str)
data['size_sqft'] = data['size_sqft'].astype(str)
data['bath'] = data['bath'].astype(str)
data['balcony'] = data['balcony'].astype(str)
data['location'] = data['location'].astype(str)

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    sizes = sorted(data['size_sqft'].unique())
    bathrooms = sorted(data['bath'].unique())
    balconies = sorted(data['balcony'].unique())
    locations = sorted(data['location'].unique())
    
    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, balconies=balconies, locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms= request.form.get('beds')
    sizes= request.form.get('size_sqft')
    bathrooms= request.form.get('bath')
    balconies= request.form.get('balcony')
    locations= request.form.get('location')

    # create a dataframe wiht the input data 
    input_data = pd.DataFrame([[bedrooms,bathrooms,sizes,balconies,locations]],columns=['beds','bath','size_sqft','balcony','location'])
    
    print("Input Data: ")
    print(input_data)
    
    #predicting price
    prediction = pipe.predict(input_data)[0]
    return str(prediction)

if __name__ =="__main__":
    app.run(debug=True, port=5000)