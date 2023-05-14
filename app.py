from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'file_cancer.pkl'
#model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    uniformity_cell_size = float(request.form['uniformity_cell_size'])
    uniformity_cell_shape = float(request.form['uniformity_cell_shape'])
    bare_nuclei = float(request.form['bare_nuclei'])
    bland_chromatin = float(request.form['bland_chromatin'])

    
      
    pred = model.predict(np.array([[uniformity_cell_size, uniformity_cell_shape, bare_nuclei, bland_chromatin ]]))
    print(pred)
    return render_template('index.html', predict=str(pred))


if __name__ == '__main__':
    app.run
