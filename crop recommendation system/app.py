from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load('crop recommendation system\model.pkl')
label_encoder = joblib.load('crop recommendation system\label_encoder.pkl')

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Replace 'index.html' with your HTML file name

# Route for crop recommendation
@app.route('/recommend', methods=['POST'])
def recommend_crop():
    data = request.json
    input_features = [[
        data['nitrogen'], data['phosphorus'], data['potassium'],
        data['temperature'], data['humidity'], data['ph'], data['rainfall']
    ]]
    prediction = model.predict(input_features)
    crop = label_encoder.inverse_transform(prediction)[0]
    return jsonify({'recommendation': crop})

if __name__ == '__main__':
    app.run(debug=True)
