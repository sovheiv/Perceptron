import json

import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, json_file="data/stats.json"):
        self.json_file = json_file
        with open(json_file, "r") as f:
            self.stats_data = json.load(f)["stats"]

    def plot_errors_val(self):
        errors_val = [stats_dict["errors_val"] for stats_dict in self.stats_data]
        plt.plot(errors_val)
        plt.title("errors_val by weight set")
        plt.xlabel("Weight set index")
        plt.ylabel("errors_val")
        plt.show()

    def plot_wrong_answers(self):
        wrong_answers = [stats_dict["wrong_answers"] for stats_dict in self.stats_data]
        plt.plot(wrong_answers)
        plt.title("wrong_answers by weight set")
        plt.xlabel("Weight set index")
        plt.ylabel("wrong_answers")
        plt.show()

    def plot_all(self):
        errors_val = [stats_dict["errors_val"] for stats_dict in self.stats_data]
        wrong_answers = [stats_dict["wrong_answers"] for stats_dict in self.stats_data]

        plt.plot(errors_val, label="errors_val")
        plt.plot(wrong_answers, label="wrong_answers")
        plt.title("errors_val and wrong_answers by weight set")
        plt.xlabel("Weight set index")
        plt.legend()
        plt.show()
