import pandas as pd
import numpy as np
import os
import math

script = os.path.realpath(f"Data\\train.csv")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

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

def data(script, column1_dtype, column2_dtype, column3_dtype, column4_dtype, column5_dtype, column6_dtype, column7_dtype, column8_dtype, column9_dtype, column10_dtype, column11_dtype, column12_dtype, column13_dtype, column14_dtype, column15_dtype, column1_name, column2_name, column3_name, column4_name, column5_name, column6_name, column7_name, column8_name, column9_name, column10_name, column11_name, column12_name, column13_name, column14_name, column15_name):
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
                                    })
    print(df.describe())
    print(df.info())
    return

data(script, column1_dtype, column2_dtype, column3_dtype, column4_dtype, column5_dtype, column6_dtype, column7_dtype, column8_dtype, column9_dtype, column10_dtype, column11_dtype, column12_dtype, column13_dtype, column14_dtype, column15_dtype, column1_name, column2_name, column3_name, column4_name, column5_name, column6_name, column7_name, column8_name, column9_name, column10_name, column11_name, column12_name, column13_name, column14_name, column15_name)
