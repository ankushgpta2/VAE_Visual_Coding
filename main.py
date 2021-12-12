from artificial_data import *
from handle_inputs import *
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import make_scorer, mean_squared_error
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, LeakyReLU, Flatten, GaussianDropout, Lambda
import matplotlib.pyplot as plt

# get hyperparameters
general_parameters, vae_parameters, artificial_info = get_hyperparams()

# ----------------------------------------------- ARTIFICIAL DATA THROUGH VAE ------------------------------------------------
# whether or not to run on artificial data
if artificial_info['run_artificial'] is True:
    artificial_main(repeat=False)
else:
    pass
