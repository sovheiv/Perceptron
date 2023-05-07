import json
import os

import numpy as np


class TrainingData:
    def __init__(self, example_size, example_answer) -> None:
        self.example_size = example_size
        self.example_answer = example_answer

    def gen_example(self):
        data = np.random.randint(0, 2, self.example_size, dtype=np.int8)
        return {"data": data.tolist(), "answer": int(data[self.example_answer])}

    def gen_training_data(self, training_data_size):
        self.save_training_data([self.gen_example() for _ in range(training_data_size)])

    @staticmethod
    def save_training_data(data):
        os.makedirs("data", exist_ok=True)
        with open("data/training_data.json", "w") as f:
            json.dump(data, f)
