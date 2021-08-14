import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm 
from scipy.stats import norm
import pylab as py 
import statistics
import math
import os
import keras
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD,Adadelta,Adagrad
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, LSTM, Softmax, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pylab import rcParams
%matplotlib inline


class activity_recognition(object):
    def __init__(self):
        self.path = "Activity_Data"
        self.training_data = pd.read_csv(self.path + "train.csv")
        self.test_data = pd.read_csv(self.path + "test.csv")
        self.Labels = None
        self.Label_keys 


    def label_data(self):
        self.Labels = self.training_data['activity']
        data_training_label = self.training_data.drop(['rn', 'activity'], axis = 1)
        Labels_keys = Labels.unique().tolist()
        Labels = np.array(Labels)









if if __name__ == "__main__":
    pass
