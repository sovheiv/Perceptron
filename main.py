from src import AIModel
from src import TrainingData
from src import Visualizer


OPTIONS_NUM = 20
DATASET_SIZE = 1000
TRAINING_VISUALISATION = True

td = TrainingData(OPTIONS_NUM, 3)
td.gen_training_data(DATASET_SIZE)

ai = AIModel(input_neurons=OPTIONS_NUM)
ai.train_model(save_training_stat=TRAINING_VISUALISATION)

# ai.load_weights()
print(ai.get_errors_sum())

if TRAINING_VISUALISATION:
    sv = Visualizer()
    sv.plot_all()
