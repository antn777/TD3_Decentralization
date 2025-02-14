import requests

class ConsensusPrediction:
    def __init__(self):
        self.models = {}  # Dictionary to store model endpoints

    def add_model(self, model_id, endpoint):
        self.models[model_id] = endpoint

    def predict(self, features):
        predictions = []
        for model_id, endpoint in self.models.items():
            response = requests.get(f"{endpoint}/predict", params={"features": features})
            predictions.append(response.json()["prediction"])
        return sum(predictions) / len(predictions)  # Simple average

class WeightedConsensusPrediction(ConsensusPrediction):
    def __init__(self):
        super().__init__()
        self.weights = {}

    def add_model(self, model_id, endpoint, initial_weight=1.0):
        super().add_model(model_id, endpoint)
        self.weights[model_id] = initial_weight

    def predict(self, features):
        predictions = []
        total_weight = sum(self.weights.values())
        for model_id, endpoint in self.models.items():
            response = requests.get(f"{endpoint}/predict", params={"features": features})
            predictions.append(response.json()["prediction"] * self.weights[model_id])
        return sum(predictions) / total_weight

    def update_weights(self, model_id, accuracy):
        self.weights[model_id] = max(0, min(1, self.weights[model_id] + accuracy - 0.5))
