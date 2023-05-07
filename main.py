from src import AIModel
from src import TrainingData
from src import Visualizer
OPTIONS_NUM = 3

td = TrainingData(OPTIONS_NUM, 2)
td.gen_training_data(1000)

ai = AIModel(input_neurons=OPTIONS_NUM, seed=111)
ai.train_model(save_training_stat=True)

print(ai.get_errors_sum())

sv = Visualizer()
sv.plot_all()
