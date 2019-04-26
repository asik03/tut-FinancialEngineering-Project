# Name: Asier
# Surname: Alcaide Martinez
# Student number: 282679
# Student e-mail: asier.alcaidemartinez@tuni.fi
# Python version: 3.6

# Glossary
# -  Issue price: the issue price of shares is the price at which they are offered for sale when they first
#    become available to the public.
# -  P: payment at maturity
# -  Sm: minimum supplemental amount

# Load the required libraries.
import numpy as np
import pandas as pd
from functools import reduce

# - PARAMETERS - #
# SP500, Euro Stoxx 50 and Hang Seng Index indexes weight values in the basket.
SPX_weight = 0.3333
SX5E_weight = 0.3333
HSI_weight = 0.3334

# Path of the historical data of the three indexes.
hsi_path = "./data/HSI-index.csv"
spx_path = "./data/SPX-index.csv"
sx5e_path = "./data/SX5E-index.csv"

# Free risk interest rate
r = 0.05

# Volatility
volatility = 0.05
# Minimum supplemental amount
Sm = 70
# Basket initial value at t0
basket_initial_value = 100
# Index values. The array has two dimensions: time from 0 to 19, and the three indexes values depending on the datetime.
index_values = [[]]
# Basket values of the 20 epochs. Array if 1 dimension with the values of the basket according to the index prices.
basket_values = []
# multiplier
#m = (33.33/index_values[0][1], 33.3/index_values[0][2], 33.3/index_values[0][3])
# Basket closing value at t20
basket_closing_value = sum(basket_values)/20
# Average basket percent change
basket_change = (basket_closing_value - basket_initial_value)/basket_initial_value
# Supplemental amount
S = max(1000 * basket_change, Sm)
# Payment at maturity
P = 1000 + S

basket_weights = (SPX_weight, SX5E_weight, HSI_weight)


# - METHODS - #

# Function that calculates the estimated volatility of the historical data of the indexes.
def calculate_estimated_volatility(closing_price):
    # Calculate the sequential difference between each consecutive index, after the log calculation of every element
    # of the array.
    log_returns = np.diff(np.log(closing_price), axis=0)

    # Volatility calculation
    est_vol = (np.std(log_returns)) / np.sqrt(1/252)

    return est_vol


# Function to check if the total basket weights are correct.
def check_basket_weights(weights):
    if sum(weights) == 1.0:
        print("Basket ok")
    else:
        print("Basket weights are not correct")


def get_basket_value(time):
    return m[0]*index_values[time][0] + m[1]*index_values[time][1] + m[2]*index_values[time][2]


# Function that load the csv data and drops the columns that are not necessary to use (Open, High and Low).
def read_closed_values(path):
    data = pd.read_csv(path)
    data = data.drop(data.columns[[1, 2, 3]], axis=1)
    #close_values = data.iloc[:, [0, 4]].values
    return data


check_basket_weights(basket_weights)

hsi_close = read_closed_values(hsi_path)
spx_close = read_closed_values(spx_path)
sx5e_close = read_closed_values(sx5e_path)

print(spx_close)

print(str(hsi_close.size) + " close values available of index HSI.")
print(str(spx_close.size) + " close values available of index SPX.")
print(str(sx5e_close.size) + " close values available of index SX5E.")
print("---------------")

# As the total number of values are not equal, we have to take only the values that in the three indexes has the same
# day at the same time. The intersection f the three values are done as a inner join operation. Columns are renamed for
# better understanding.
intersect_aux_df = pd.merge(hsi_close, spx_close, on='Date', how='inner')
intersection_df = pd.merge(intersect_aux_df, sx5e_close, on='Date', how='inner')
intersection_df.columns = ['Date', 'HSI_Close', 'SPX_Close', 'SX5E_Close']
print(intersection_df)

hsi_close = intersection_df.iloc[:, [1]].values
spx_close = intersection_df.iloc[:, [2]].values
sx5e_close = intersection_df.iloc[:, [3]].values

#historical_data = intersection_df.values.transpose()
#print(historical_data.shape)


HSI_volatility = calculate_estimated_volatility(hsi_close)
SPX_volatility = calculate_estimated_volatility(spx_close)
SX5E_volatility = calculate_estimated_volatility(sx5e_close)

print("Volatility of historical data od index HSI: " + str(HSI_volatility))
print("Volatility of historical data od index SPX: " + str(SPX_volatility))
print("Volatility of historical data od index SX5E: " + str(SX5E_volatility))
