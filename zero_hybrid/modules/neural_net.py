import numpy as np
import json
import os

FILE = "data/nn.json"

class TinyNN:
    def __init__(self):
        self.w = np.random.randn(4, 1)
        self.load()

    def forward(self, x):
        return float(np.dot(x, self.w))

    def train(self, x, target, lr=0.001):
        pred = self.forward(x)
        error = target - pred
        self.w += lr * error * x.reshape(-1,1)

    def save(self):
        os.makedirs("data", exist_ok=True)
        json.dump(self.w.tolist(), open(FILE,"w"))

    def load(self):
        if os.path.exists(FILE):
            try:
                self.w = np.array(json.load(open(FILE)))
            except:
                pass

nn = TinyNN()