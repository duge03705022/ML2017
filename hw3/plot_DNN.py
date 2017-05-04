import os
import sys
import argparse
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.callbacks import *
import pydot
import graphviz

model_name = sys.argv[1]

emotion_classifier = load_model("model/"+model_name)
emotion_classifier.summary()
plot_model(emotion_classifier,to_file="model_data/model_plot/DNN/"+model_name+".png")
