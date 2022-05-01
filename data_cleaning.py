import pandas as pd
import numpy as np
import os
import math

script = os.path.realpath(f"Data\\train.csv")

def data(script):
    df = pd.read_csv(script)
    print(df.describe())
    print(df.info())
    return

data(script)