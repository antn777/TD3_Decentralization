import json

class StakeManager:
    def __init__(self, initial_deposit=1000):
        self.balances = {}
        self.initial_deposit = initial_deposit

    def register_model(self, model_id):
        if model_id not in self.balances:
            self.balances[model_id] = self.initial_deposit

    def slash(self, model_id, amount):
        if model_id in self.balances:
            self.balances[model_id] = max(0, self.balances[model_id] - amount)

    def save_balances(self, filename='balances.json'):
        with open(filename, 'w') as f:
            json.dump(self.balances, f)

    def load_balances(self, filename='balances.json'):
        try:
            with open(filename, 'r') as f:
                self.balances = json.load(f)
        except FileNotFoundError:
            pass
