from flask import Flask, request,jsonify
import sklearn
import pickle
import pandas as pd
import json
import numpy as np


with open('bestPredictor.pickle', 'rb') as f:
    pipeline = pickle.load(f)
readIn = np.genfromtxt('PCAcomponents.csv',delimiter=',',dtype='float64')
PCcomps = readIn[:8,:]
PCmean = readIn[-1,:]
app=Flask(__name__)
@app.route('/predict', methods=['GET','POST'])
def predict():
    send = request.get_json(force=True)
    predictors = pd.DataFrame.from_dict([send])
    predictors['Sex']='F'
    
    PCScores = np.asarray(pipeline.predict(predictors))
    
    points = np.dot(PCScores,PCcomps)+PCmean
    return {'vertices':points.tolist()}

if __name__=='__main__':
    app.run(host='127.0.0.1',debug=True)
