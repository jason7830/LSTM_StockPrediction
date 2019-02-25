from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/2330_indicators.csv")

trainset = df[40:5000]
testset = df[5000:]

sc = MinMaxScaler()
