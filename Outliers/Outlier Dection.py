import pandas as pd
import numpy as np
import os
import math
from sklearn.preprocessing import LabelEncoder

script = os.path.realpath(f"Data\\train.csv")

pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 25)

## Write out the columns name should look like this: str("Feature_1")
column1_name = str("Id")
column2_name = str("LotArea")
column3_name = str("Street")
column4_name = str("Utilities")
column5_name = str("Condition1")
column6_name = str("BldgType")
column7_name = str("HouseStyle")
column8_name = str("OverallQual")
column9_name = str("OverallCond")
column10_name = str("YearBuilt")
column11_name = str("RoofStyle")
column12_name = str("RoofMatl")
column13_name = str("Exterior1st")
column14_name = str("SaleCondition")
column15_name = str("SalePrice")

## Write out the columns data types should look like this: str("int8")
column1_dtype = str("int16")
column2_dtype = str("int32")
column3_dtype = str("category")
column4_dtype = str("category")
column5_dtype = str("category")
column6_dtype = str("category")
column7_dtype = str("category")
column8_dtype = str("int8")
column9_dtype = str("int8")
column10_dtype = str("int16")
column11_dtype = str("category")
column12_dtype = str("category")
column13_dtype = str("category")
column14_dtype = str("category")
column15_dtype = str("int64")

def importing_correct_data_form(script, column1_dtype, column2_dtype, column3_dtype, column4_dtype, column5_dtype, column6_dtype, column7_dtype, column8_dtype, column9_dtype, column10_dtype, column11_dtype, column12_dtype, column13_dtype, column14_dtype, column15_dtype, column1_name, column2_name, column3_name, column4_name, column5_name, column6_name, column7_name, column8_name, column9_name, column10_name, column11_name, column12_name, column13_name, column14_name, column15_name):
    df = pd.read_csv(script, dtype={column1_name:column1_dtype,
                                    column2_name:column2_dtype,
                                    column3_name:column3_dtype,
                                    column4_name:column4_dtype,
                                    column5_name:column5_dtype,
                                    column6_name:column6_dtype,
                                    column7_name:column7_dtype,
                                    column8_name:column8_dtype,
                                    column9_name:column9_dtype,
                                    column10_name:column10_dtype,
                                    column11_name:column11_dtype,
                                    column12_name:column12_dtype,
                                    column13_name:column13_dtype,
                                    column14_name:column14_dtype,
                                    column15_name:column15_dtype
                                    })#.set_index(column1_name)
    #print(df.describe())
    print(df.info())
    return df

df = importing_correct_data_form(script,
                                 column1_dtype,
                                  column2_dtype,
                                   column3_dtype,
                                    column4_dtype,
                                     column5_dtype,
                                      column6_dtype,
                                       column7_dtype,
                                        column8_dtype,
                                         column9_dtype,
                                          column10_dtype,
                                           column11_dtype,
                                            column12_dtype,
                                             column13_dtype,
                                              column14_dtype,
                                               column15_dtype,
                                                column1_name,
                                                 column2_name,
                                                  column3_name,
                                                   column4_name,
                                                    column5_name,
                                                     column6_name,
                                                      column7_name,
                                                       column8_name,
                                                        column9_name,
                                                         column10_name,
                                                          column11_name,
                                                           column12_name,
                                                            column13_name,
                                                             column14_name,
                                                              column15_name)

label_encoder = LabelEncoder()
df["Street"] = label_encoder.fit_transform(df["Street"])
df["Utilities"] = label_encoder.fit_transform(df["Utilities"])
df["Condition1"] = label_encoder.fit_transform(df["Condition1"])
df["BldgType"] = label_encoder.fit_transform(df["BldgType"])
df["HouseStyle"] = label_encoder.fit_transform(df["HouseStyle"])
df["RoofStyle"] = label_encoder.fit_transform(df["RoofStyle"])
df["RoofMatl"] = label_encoder.fit_transform(df["RoofMatl"])
df["Exterior1st"] = label_encoder.fit_transform(df["Exterior1st"])
df["SaleCondition"] = label_encoder.fit_transform(df["SaleCondition"])

df = df[df["SalePrice"] < 354000]
df = df[df["YearBuilt"] > 1882]
df = df[df["LotArea"] < 18890]
df = df[df["OverallQual"] > 2]
df = df[df["OverallQual"] < 10]
df = df[df["OverallCond"] > 2]
df = df[df["OverallCond"] < 9]
df = df[df["Street"] > 0]
df = df[df["Utilities"] > 0]

outliers=[]
def detect_outliers(data):

    # This is the Standard deviations (3 for 99.7% of the data)
    threshold=3

    # This is the Neccessary information needed to calculate Z_score
    mean = np.mean(data)
    std = np.std(data)

    for i in data:
        z_score = (i - mean)/std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers

detect_outliers(df["SaleCondition"])
print(outliers)
print(np.min(outliers))
print(np.max(outliers))