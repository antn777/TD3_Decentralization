from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib as jb
from consensus import WeightedConsensusPrediction
from stake_manager import StakeManager

class TitanicNet(nn.Module):
    def __init__(self):
        super(TitanicNet, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
model = TitanicNet()
model.load_state_dict(torch.load("titanic_model.pth", weights_only=True))
model.eval()
scaler = jb.load('scaler.pkl')

app = Flask(__name__)
consensus_model = WeightedConsensusPrediction()
stake_manager = StakeManager()

@app.route('/')
def home():
    return "ça fonctionne"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            feature_1 = float(request.args.get('feature_1', 0))
            feature_2 = float(request.args.get('feature_2', 0))
            feature_3 = float(request.args.get('feature_3', 0))
            feature_4 = float(request.args.get('feature_4', 0))
            feature_5 = float(request.args.get('feature_5', 0))
            feature_6 = float(request.args.get('feature_6', 0))
            features = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]])
        elif request.method == 'POST':
            data = request.get_json()
            features = np.array(data["features"]).reshape(1, -1)

        features = scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(features)
            _, predicted_class = torch.max(outputs, 1)

        response = {
            "status": "success",
            "prediction": int(predicted_class.item())
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/register', methods=['POST'])
def register_model():
    data = request.json
    model_id = data['model_id']
    endpoint = data['endpoint']
    consensus_model.add_model(model_id, endpoint)
    stake_manager.register_model(model_id)
    return jsonify({"status": "success", "message": "Model registered"})

@app.route('/consensus_predict', methods=['GET'])
def consensus_predict():
    features = request.args.get('features')
    prediction = consensus_model.predict(features)
    return jsonify({"status": "success", "prediction": prediction})

@app.route('/update_weights', methods=['POST'])
def update_weights():
    data = request.json
    model_id = data['model_id']
    accuracy = data['accuracy']
    consensus_model.update_weights(model_id, accuracy)
    if accuracy < 0.5:  # Simple slashing condition
        stake_manager.slash(model_id, (0.5 - accuracy) * 100)  # Slash proportional to inaccuracy
    stake_manager.save_balances()
    return jsonify({"status": "success", "message": "Weights updated"})

if __name__ == '__main__':
    stake_manager.load_balances()
    app.run(host="0.0.0.0", port=5000, debug=True)
