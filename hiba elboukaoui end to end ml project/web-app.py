from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


credit_score_model = joblib.load(r'C:\Users\hiba\Desktop\hiba elboukaoui end to end ml project\best_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return 'Credit Score Prediction API: Predict your credit score instantly!'

@app.route('/predict', methods=['POST'])
def make_prediction():
    
    request_data = request.get_json()

    
    if not request_data:
        return jsonify({'error': 'No JSON payload provided'}), 400  

    
    input_df = pd.DataFrame(request_data, index=[0])

   
    try:
        predicted_score = credit_score_model.predict(input_df)
        return jsonify({'Predicted_Credit_Score': predicted_score[0]}), 200  
    except Exception as error:
        return jsonify({'error': str(error)}), 500 

if __name__ == '__main__':
    app.run(debug=True)
