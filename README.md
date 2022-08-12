# Flask-Asteroid-Prediction

1. Run the app.py using Flask Framework.
2. Open the Flask App.
3. Upload the Asteroid Sample Dataset in a .csv format
4. Note that the Asteroid dataset should have atleast the following attributes.
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
5. Afterwards, A predicted Dataset in .csv format will be downloaded.

### Important Note: You should have xgboost module version 1.5.2 and scikit-learn version 1.0.2 installed in your Flask Virtual environment 
