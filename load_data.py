# Name: Asier
# Surname: Alcaide Martinez
# Student number: 282679
# Student e-mail: asier.alcaidemartinez@tuni.fi
# Python version: 3.6

# Glossary
# -  Issue price: the issue price of shares is the price at which they are offered for sale when they first
#    become available to the public.
# -


# - PARAMETERS - #
# SP500, Euro Stoxx 50 and Hang Seng Index indexes weight values in the basket.
SPX_weight = 0.3333
SX5E_weight = 0.3333
HSI_weight = 0.3334

# Free risk interest rate
r = 0.05

# Volatility
volatility = 0.05

#


basket_weights = (SPX_weight, SX5E_weight, HSI_weight)


# Function to check if the total basket weights are correct.
def check_basket_weights(weights):
    if sum(weights) == 1.0:
        print("Basket ok")
    else:
        print("Basket weights are not correct")


check_basket_weights(basket_weights)
