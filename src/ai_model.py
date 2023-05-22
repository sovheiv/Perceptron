import json

import numpy as np

from src.logger import logger
from .utils import timer


class AIModel:
    def __init__(self, input_neurons: int = 3, seed: int = None) -> None:
        if seed:
            np.random.seed(seed)

        self.weights = np.random.rand(input_neurons)

        with open("data/training_data.json", "r") as f:
            self.training_data = json.load(f)

        logger.info("AI inited")

    @timer
    def train_model(self, save_training_stat: bool = False):
        logger.info("Training started")

        if save_training_stat:
            training_stat = [{**self.get_errors_sum(), **{"weights": self.weights_s}}]

        for batch in self.training_data:
            self.back_propagation(batch["data"], batch["answer"])

            if save_training_stat:
                training_stat.append(
                    {**self.get_errors_sum(), **{"weights": self.weights_s}}
                )

        if save_training_stat:
            self.save_json(stats=training_stat, weights=self.weights.tolist())
            return training_stat[-1]
        self.save_json(weights=self.weights.tolist())

    def get_errors_sum(self):
        all_errors = [
            batch["answer"] - self.solve(batch["data"]) for batch in self.training_data
        ]
        errors_val = round(sum([abs(e) for e in all_errors]), 3)
        wrong_answers = sum([1 if abs(e) > 0.5 else 0 for e in all_errors])

        return {
            "errors_val": errors_val,
            "wrong_answers": wrong_answers,
            "max_error": max(all_errors, key=abs),
            "avg_error": errors_val / len(all_errors),
        }

    def back_propagation(self, input: list[int], real_answer):
        """1 iteration"""

        answer = self.solve(input)
        error = real_answer - answer
        adjustment = (error * 4) ** 3

        npinput = np.array(input)
        adjustments = npinput * adjustment
        self.weights += adjustments

        logger.debug(
            f"Error: {round(error,3)}\n"
            f"Adjusment: {round(adjustment,3)}\n"
            f"Input: {input} {real_answer}\n"
            f"Adjusments: {adjustments}\n"
            f"New weights: {self.weights}\n"
        )

    def solve(self, input: list[int]):
        return self.sigmoid(np.dot(self.weights, input))

    @property
    def weights_s(self):
        return [round(float(x), 3) for x in self.weights]

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def save_json(**kwargs):
        for key in kwargs:
            with open(f"data/{key}.json", "w") as f:
                json.dump({key: kwargs[key]}, f)

    def load_weights(self):
        with open("data/weights.json", "r") as f:
            self.weights = np.array(json.load(f)["weights"])
