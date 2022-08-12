from flask import Flask, make_response, request
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle



app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return """
        <html>
            <body>
                <h1>Asteroid Diameter Prediction</h1>
                <br>
                <br>
                <p> Insert your CSV file and then download the Result
                <p> Make Sure that your dataset has atleast the following attributes
                <ul>
                <li>H: Absolute Magnitude Parameter{float value}</li>
                <li>n_obs_used: No of Radar Observations used{integer value}</li>
                <li>data_arc: Observation Arc in days{float value}</li>
                <li>albedo: Geometric Albedo{float value}</li>
                <li>a: Semi major Axis{float value}</li>
                <li>q: Perihelion Distance{float value}</li>
                <li>moid: Minimum Earth Orbit Intersection Distance{float value}</li>
                <li>neo: Near Earth Object{Categorical feature with either Yes or No}</li>
                <li>pha: Potentially Hazardous Asteroid{Categorical feature with either Yes or No}</li>
                </ul>
                <form action="/transform" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" class="btn btn-block"/>
                    </br>
                    </br>
                    <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
                </form>
            </body>
        </html>
    """
@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result))


    # load the model from disk
    model1 = pickle.load(open('xgboost_r2.pkl', 'rb'))
    model2 = pickle.load(open('xgboost_nmae.pkl', 'rb'))
    sd= pickle.load(open('Scalar.sav', 'rb'))
    x=df[['H','data_arc','n_obs_used','moid','q','a','albedo','neo','pha','diameter']]
    for i in df["a"]:
        if i<-32588.9430 or i>3043.14907e+03:
            raise ValueError
        
    for i in df["q"]:
        if i<.0705107320 or i>80.4241748:
            raise ValueError

    for i in df["n_obs_used"]:
        if i<2 or i>9325:
            raise ValueError

    for i in df["H"]:
        if i<-1.1 or i>33.2:
            raise ValueError

    for i in df["albedo"]:
        if i<0 or i>1:
            raise ValueError

    for i in df["moid"]:
        if i<3.437640e-07 or i>79.5:
            raise ValueError

    for i in df["data_arc"]:
        if i<0 or i>72684:
            raise ValueError

    x_norm=sd.transform(x[['a', 'q', 'data_arc', 'n_obs_used', 'H',
       'albedo', 'moid']])
    x_norm=pd.DataFrame(data=x_norm,columns=['a', 'q', 'data_arc', 'n_obs_used', 'H',
       'albedo', 'moid'])
    x_neo_encode=pd.get_dummies(x['neo'], drop_first=True)
    x_pha_encode=pd.get_dummies(x['pha'], drop_first=True)

    x=x_norm[['a','q','data_arc','n_obs_used','H','albedo','moid']]
    x['neo']=x_neo_encode.values
    x['pha']=x_pha_encode.values
    ypred = model1.predict(x)
    a=pd.DataFrame()
    a['Predicted Diameter with R Squared']=ypred
    ypred = model2.predict(x)
    a['Actual Diameter']=df['diameter']
    a['Predicted Diameter with NMAE']=ypred

    response = make_response(a.to_csv(index=False))
    response.headers["Content-Disposition"] = "attachment; filename=Predictions with R Squared and NMAE.csv"
    return response

if __name__ == "__main__":
    app.run(debug=False,port=5000)
