from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# 1. Front end and back end
# 2. routing
model = joblib.load('pipe.pkl')


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    test_df = pd.DataFrame([request.form])
    # print(test_df.shape)
    value = model.predict(test_df)[0]
    value = str(round(value, 2))
    return render_template("predict.html", price=value + " Lakhs")


if __name__ == '__main__':
    app.run(debug=True)
