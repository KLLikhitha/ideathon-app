from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)

models = {}
models['C1'] = joblib.load('C1.pkl')
models['C2'] = joblib.load('C2.pkl')
models['C3'] = joblib.load('C3.pkl')
models['E1'] = joblib.load('E1.pkl')
models['E2'] = joblib.load('E2.pkl')
models['E3'] = joblib.load('E3.pkl')
models['F1'] = joblib.load('F1.pkl')
models['G1'] = joblib.load('G1.pkl')
models['G2'] = joblib.load('G2.pkl')
models['G3'] = joblib.load('G3.pkl')
models['I1'] = joblib.load('I1.pkl')
models['I2'] = joblib.load('I2.pkl')
models['I3'] = joblib.load('I3.pkl')
models['I4'] = joblib.load('I4.pkl')
models['I5'] = joblib.load('I5.pkl')
models['I6'] = joblib.load('I6.pkl')
models['L1'] = joblib.load('L1.pkl')
models['M1'] = joblib.load('M1.pkl')
models['M2'] = joblib.load('M2.pkl')
models['M3'] = joblib.load('M3.pkl')
models['R1'] = joblib.load('R1.pkl')
models['R2'] = joblib.load('R2.pkl')
models['R3'] = joblib.load('R3.pkl')

@app.route('/', methods=['GET'])
def predict():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        print(float(request.args.get('I')), float(request.args.get('T')))
        output = {}
        i = float(request.args.get('I'))
        t = float(request.args.get('T'))
        for key in models.keys():
            op = models[key].predict([[i, t]])
            output[key] = op[0]
    return output

if __name__ == '__main__':
    app.run(debug = False)