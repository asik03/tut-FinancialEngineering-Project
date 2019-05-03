# - STUDENT INFO - #
# Name: Asier
# Surname: Alcaide Martinez
# Student number: 282679
# Student e-mail: asier.alcaidemartinez@tuni.fi
# Python version: 3.6

# - GLOSSARY - #
# -  Issue price: the issue price of shares is the price at which they are offered for sale when they first
#    become available to the public.
# -  P: payment at maturity
# -  Sm: minimum supplemental amount

# - LIBRARIES - #
# Load the required libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# - PARAMETERS - #

# SP500, Euro Stoxx 50 and Hang Seng Index indexes weight values in the basket.
SPX_weight = 0.3333
SX5E_weight = 0.3333
HSI_weight = 0.3334

# Path of the historical data of the three indexes.
hsi_path = "./data/HSI-index.csv"
spx_path = "./data/SPX-index.csv"
sx5e_path = "./data/SX5E-index.csv"

# Number of simulations for the testing
n_simulations = 500000
# Free risk interest rate of
risk_rate = np.array([0.0225, 0.0225, 0.0225])
# Volatility of the three indexes
volatility = np.array([0.05, 0.05, 0.3])
# Use locked volatility if we want to specify the volatility by ourselves rather then obtaining from historical data.
locked_volatility = False
# Cap to set if we want to discard payoff values higher than 1300
set_cap = False
cap_value = 1300
# Minimum supplemental amount
Sm = 70
# Basket initial value at t0
basket_initial_value = 100
# Years from which historical data is collected
years_of_hist_data = 5


# Weights of the three indexes in the basket
basket_weights = (SPX_weight, SX5E_weight, HSI_weight)
# Time passed between the observation dates, used to calculate the end_prices of every step.
times_between_dates = [92, 90, 91, 92, 91, 91, 92, 92, 89, 92, 93, 91, 89, 92, 92, 94, 87, 92, 94, 91]


br = "---------------------------------------------------------------------------------"


# - METHODS - #
def check_basket_weights(weights):
    """Function to check if the total basket weights are correct."""
    if sum(weights) == 1.0:
        return
    else:
        raise Exception("Basket weights are not correct. All of them has to sum 1 in total.")


def check_valid_params(n_sims, sm):
    """Function to check if the parameters values are correct."""

    if n_sims > 0 and sm >= 0:
        return
    else:
        raise Exception("Values of the parameters are not correct.")


def read_closed_values(path):
    """Function that load the csv data and drops the columns that are not necessary to use (Open, High and Low)."""
    data = pd.read_csv(path)
    data = data.drop(data.columns[[1, 2, 3]], axis=1)
    return data


def calculate_estimated_volatility(closing_price):
    """Function that calculates the estimated volatility of the historical data of the indexes."""
    # Calculate the sequential difference between each consecutive index, after the log calculation of every element
    # of the array.
    log_returns = np.diff(np.log(closing_price), axis=0)
    # Volatility calculation
    est_vol = (np.std(log_returns)) / np.sqrt(1 / 252)

    return est_vol


def end_price(days_dif, curr_spot_price):
    """Function that returns the end prices between a phase or period."""

    # Generating three random draws under the correlation matrix for each time period. Vector of correlated random
    # numbers of the three indexes.
    epsilon = np.matmul(np.sqrt(R), np.random.randn(R.shape[0], 1))
    w = epsilon.T * np.sqrt(days_dif/365)
    return np.squeeze(curr_spot_price * np.exp((risk_rate - 0.5 * sig**2) * (days_dif/365) + sig * w))


def get_basket_value(time, ind_values):
    """Calculate the values of the basket on a specific time"""
    return m[0] * ind_values[time][0] + m[1] * ind_values[time][1] + m[2] * ind_values[time][2]


check_basket_weights(basket_weights)
check_valid_params(n_simulations, Sm)


# - Load and filter data - #
# Load with Pandas the required data from the historical prices. Date and closed values.
hsi_close = read_closed_values(hsi_path)
spx_close = read_closed_values(spx_path)
sx5e_close = read_closed_values(sx5e_path)

print(str(hsi_close.size) + " close values loaded of index HSI.")
print(str(spx_close.size) + " close values loaded of index SPX.")
print(str(sx5e_close.size) + " close values loaded of index SX5E.")


# As the total number of values are not equal, we have to take only the values that in the three indexes has the same
# day at the same time. The intersection f the three values are done as a inner join operation. Columns are renamed for
# better understanding.
print(br)
intersect_aux_df = pd.merge(hsi_close, spx_close, on='Date', how='inner')
intersection_df = pd.merge(intersect_aux_df, sx5e_close, on='Date', how='inner')
intersection_df.columns = ['Date', 'HSI_Close', 'SPX_Close', 'SX5E_Close']

print("Filtered historical data:")
print(intersection_df)

# Getting the historical data from 02/26/14 to 02/25/19
hsi_close_5y = intersection_df.iloc[:, [1]].values
spx_close_5y = intersection_df.iloc[:, [2]].values
sx5e_close_5y = intersection_df.iloc[:, [3]].values

# Getting the historical data from 02/26/16 to 02/25/19
hsi_close_3y = intersection_df.iloc[:-719, [1]].values
spx_close_3y = intersection_df.iloc[:-719, [2]].values
sx5e_close_3y = intersection_df.iloc[:-719, [3]].values

# Getting the historical data from 02/26/18 to 02/25/19
hsi_close_1y = intersection_df.iloc[:-238, [1]].values
spx_close_1y = intersection_df.iloc[:-238, [2]].values
sx5e_close_1y = intersection_df.iloc[:-238, [3]].values

# Total payoffs of the total number of simulations
payoffs = []

# Using different historical data depending on the number of years specified as a parameter. Available only for last 1,
# 3 or 5 years of data.
if years_of_hist_data == 5:
    hsi_close = hsi_close_5y
    spx_close = spx_close_5y
    sx5e_close = sx5e_close_5y
elif years_of_hist_data == 3:
    hsi_close = hsi_close_3y
    spx_close = spx_close_3y
    sx5e_close = sx5e_close_3y
elif years_of_hist_data == 1:
    hsi_close = hsi_close_1y
    spx_close = spx_close_1y
    sx5e_close = sx5e_close_1y
else:
    raise Exception("Wrong number of years of getting historical data.")

# Matrix of correlation of the log-returns of historical prices
close_matrix = np.concatenate(
    (np.diff(np.log(hsi_close), axis=0), np.diff(np.log(spx_close), axis=0), np.diff(np.log(sx5e_close), axis=0)),
    axis=1)
R = np.absolute(np.corrcoef(close_matrix.T))
print("Correlation matrix of shape " + str(R.shape) + ":")
print(R)
print(br)

# Starting the simulations.
for n in range(n_simulations):
    # Index values. The array has two dimensions: time from 0 to 20 (initial time included), and the three indexes
    # values depending on the datetime.
    index_values = []
    # Basket values of the 20 epochs. Array if 1 dimension with the values of the basket according to the index prices.
    basket_values = []

    # If we are in the first iteration:
    if n == 0:
        # Calculate the historical volatility if the flag is activated.
        if not locked_volatility:
            # Calculate the volatility of each index based on the filtered historical data.
            HSI_volatility = calculate_estimated_volatility(hsi_close)
            SPX_volatility = calculate_estimated_volatility(spx_close)
            SX5E_volatility = calculate_estimated_volatility(sx5e_close)
            # Array with the three volatilises of the index
            sig = np.array([HSI_volatility, SPX_volatility, SX5E_volatility])
        # Otherwise, the volatility is the one we specify in the parameters.
        else:
            sig = volatility
        print("Volatility of historical data od index HSI: " + str(sig[0]))
        print("Volatility of historical data od index SPX: " + str(sig[1]))
        print("Volatility of historical data od index SX5E: " + str(sig[2]))
        print(br)

    # Getting the initial prices
    index_values.append(np.concatenate(np.array([hsi_close[0], spx_close[0], sx5e_close[0]])))

    # multiplier
    m = [33.33/index_values[0][0], 33.3/index_values[0][1], 33.3/index_values[0][2]]

    # Calculation of the index values of the estimated times, and the basket value each time.
    for i in range(len(times_between_dates)):
        index_values.append(end_price(days_dif=times_between_dates[i], curr_spot_price=index_values[i]))
        basket_values.append(get_basket_value(i+1, ind_values=index_values))  # Add +1 to the i in order to ignore the T0

    # Basket closing value at t20
    basket_closing_value = sum(basket_values) / 20

    # Average basket percent change (% per 1)
    basket_change = (basket_closing_value - basket_initial_value) / basket_initial_value

    # Supplemental amount
    S = max(1000 * basket_change, Sm)

    # Payment at maturity
    P = 1000 + S

    # Setting a cap with values more than 1300
    if set_cap:
        if P > cap_value:
            P = cap_value

    #print("Payment at maturity: " + str(P))
    #print(br)

    payoffs = np.append(payoffs, P)

print("Simulation finished. Payoffs of the simulation:")
#print(payoffs)

# Calculate the payoffs mean in order to compare the results when changing the parameters.
payoffs_mean = np.mean(payoffs)
print("Average payoff of the simulation: ")
print(np.mean(payoffs_mean))

plt.plot(payoffs)
plt.show()

quantile_0025 = np.quantile(payoffs, 0.025)
quantile_05 = np.quantile(payoffs, 0.5)
quantile_095 = np.quantile(payoffs, 0.95)
print("Quantiles:")
print("Quantil 0.025: " + str(quantile_0025))
print("Quantil 0.5: " + str(quantile_05))
print("Quantil 0.95: " + str(quantile_095))


# - ANALYSIS AND RESULTS - #
# -- 1 --
# By having a 2.225% of risk interest rate and volatility of the historical data, 500000 simulations and 5 years
# of historical data, we get an average Payoff of $1206.913.
# If we change the total historical data (years_of_hist_data)
# collected to 3 years, the average Payoff of all simulations is 1178.
# If we change the total years to 1, we get an average Payoff of 1212.15.
# The number of years only determines the volatility in this case, so the volatility can vary and alter the expected
# values of the simulation. It depends on the success in the past, but the more data we collect, the more possibilities
# we have to reach a better estimation.

# -- 2 --
# Quantiles are specified in the last lines of code, printed by the simulation. Q0.025 and Q0.5 are both 1070 in the
# three cases of 1, 3 and 5 years of historical data. The Q0.95 vary between the years.

# -- 3 --
# SX5E is the index with the highest volatility (0.1840343888758252).
#
# CASE 1: we set to 0 the weight of SX5E index, and 50% the other two. The final payoff average we get is
# 1206.06. Compared with the average payoff of the previous weights, 1208.54, now the payoff is quite lower. This is
# caused because of the lower total volatility obtained, that generates less high and low values, in which the lower
# values are covered by the minimum supplemental amount. That means that only the higher values are counted in the
# averaging payoff.
#
# CASE 2: if we set to 1 the SPX weight, and 0 the rest of them, we get the averaging payoff as 1205.94. The volatility
# is 0.13, the lowest, also the lowest payoff as it is explained in the previous comment.

# -- 4 -- Free risk and volatility rates
# PART 1: Different free risk rates
# By changing the free risk rate from all 0.0225 to 0.005, 0.005, and 0.3 each, and resetting the weights to 33.33%
# each, we get an averaging payoff of 1566.44, much more bigger than the rest of the payoffs. That is caused because of
# high risk rate it exist in one of the three indexes, even if the both remaining 2 indexes have been decreased their
# values. A risk rate of 0.3 (30%) is hugh. Quantiles also has varied, 1438 and 2563 as the 0.5 and 0.95 ones, and 1070
# the 0.025.
#
# PART 2: Different volatility values
# Fist of all we set the flag "locked_volatility" to True to allow manually to set the volatilities and not get them
# from the historical data. Then we change the values in the "volatility" variable to  0.05, 0.05, and 0.3. We set the
# risk rates again to 0.0225. After running it, the result we get is an average payoff of 1195.75, the lowest value at
# the moment. Because of the lower values of the two first indexes, the average payoff tends to be lower as there are
# not as many high and low values than before, where the high values are only taken (values less than 1070 are not
# considered).

# -- 5 --
# To set a cap in the value of 1300, we just need to change the flag "set_cap" to True, and check if the cap_value is
# set to 1300. Then, all the expected payoffs that has a value more than 1300 are considered as 1300. Then, the average
# payoff of the simulation is, compared with the 1208.54 we obtained in default conditions (part 1), of 1146.05, much
# lower. The most different result is the 0.95 quantile, of 1300 as id the maximum value of the results. This means that
# more than 5% of the total values are 1300.

# -- 6 --
# To calculate the value of the contract on 25 of march, we need to use the volatilities of the historical data, as the
# initial day is before that date, and then we need to set as the time in the Black Scholes formula as the difference
# of day between 25 of February and 25 of march, which is 28 days. By loading codes from 1 to 188, and loading the code
# inside the simulation for loop as only the fist iteration, until line 220, and loading the end_price function with as
# index_values.append(end_price(days_dif=28, curr_spot_price=index_values[0])), and the line
# basket_values.append(get_basket_value(1, ind_values=index_values)), anf finally loading the code from lines 226 to
# 246, we get the value of the contract as 1070.

# --- 8: CONCLUSION ---
# Based on the results obtained along the project work, there are many options to choose depending on the different
# parameters that exists in the contract, firstly, the risk rate is one of the most relevant parameters we need to
# consider in order to get beneficial results in the contract. One low variation will get vary different reslults.
# It is also important to say that calculating an optimal
# parameter or minimum risk value rate would be better for helping to get a better decision. Volatilities are also
# important to consider, as the higher they are, the higher profits are gotten. Moreover, if the risk rates and
# volatilities are high, banks would not see a good opportunity in this contract, so they would not accept it. To avoid
# it, the selection of a specified cap in case the payoffs are higher that excepted would make banks more open to close
# the contract. Overall, the implementation of this code has given the opportunity to learn and understand the basics
# of the stock value forecasting and prediction, and the possible changes it can be produced when we have a lack of
# sufficient data (changing between 1, 3 and 5 years of historical data), changes in volatilities and changes in risk
# free interest rates.



