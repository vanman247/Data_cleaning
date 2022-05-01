import pandas as pd
import numpy as np
import os
import math

cwd = os.getcwd() + "\Data\train.csv"

def data(URL = cwd):
    df = pd.DataFrame(URL)
    print(df.head())
    return

data(URL=cwd)
