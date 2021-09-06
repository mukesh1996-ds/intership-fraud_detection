from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_featurs = [int(x) for x in request.form.values()]
    final = [np.array(int_featurs)]
    prediction = model.predict_proba(final)
    output = '{0:{1}f}'.format(prediction[0][1],2)

    if output >= str(0):
        return render_template('index.html',pred = 'It is a fraud transaction.\nProbability of fraud is {}'.format(output))
    else:
        return render_template('index.html', pred = 'It is not a fraud transaction.\nProbability of fraud occouring is {}'.format(output))

if __name__== '__main__':
    app.run()
