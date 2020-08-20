import sklearn
from flask import Flask, request
import pickle
import numpy as np
import pandas as pd
import gunicorn
import os

app = Flask(__name__)


def load_model():
    file_name = "lr.p"
    with open(file_name, 'rb') as pickled:
        p_model = pickle.load(pickled)
    return p_model


model = load_model()


@app.route('/predict_single', methods=['GET'])
def predict_single():
    features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    x = np.array([float(request.args.get(key)) for key in features]).reshape(1,-1)

    prediction = round(model.predict(x)[0], 4)
    return str(prediction)


@app.route('/predict_all/', methods=['POST'])
def predict_all():
    data = request.get_json()
    df = pd.DataFrame(data)
    df['Prediction'] = model.predict(df)
    return df.to_json(orient='records')


if __name__ == '__main__':
    port = os.environ.get('PORT')

    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()
