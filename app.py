from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('data/fetal_health.csv')
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return jsonify({"message": "Fetal Health Prediction System v1 - Powered by Random Forest"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    model_name = data.get('model', 'randomforest').lower()
    
    feature_names = [
        'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
        'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
        'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability',
        'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min',
        'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes',
        'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance',
        'histogram_tendency'
    ]
    
    try:
        input_features = np.array([[float(data[feat]) for feat in feature_names]])
    except KeyError as e:
        return jsonify({'error': f'Missing input feature: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': 'All inputs must be numeric'}), 400

    input_scaled = scaler.transform(input_features)

    if model_name == 'logistic':
        model = log_model
    elif model_name == 'knn':
        model = knn_model
    elif model_name == 'randomforest':
        model = rf_model
    else:
        return jsonify({'error': 'Invalid model name. Choose from logistic, knn, randomforest'}), 400

    prediction = model.predict(input_scaled)[0]
    health_map = {
        1.0: 'Normal',
        2.0: 'Suspect',
        3.0: 'Pathological'
    }

    return jsonify({
        'prediction': int(prediction),
        'health_status': health_map.get(prediction, 'Unknown'),
        'model_used': model_name
    })

if __name__ == '__main__':
    app.run(debug=True)