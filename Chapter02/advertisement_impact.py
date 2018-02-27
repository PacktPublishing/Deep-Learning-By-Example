import pandas as pd
import matplotlib.pyplot as plt

# To use the formula notation below, we need to import the module like the following
import statsmodels.formula.api as smf

# read advertising data samples into a DataFrame
advertising_data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)


# DataFrame.head method shows the first n rows of the data where the
# default value of n is 5, DataFrame.head(n=5)
advertising_data.head()

print(list(advertising_data))

# print the shape of the DataFrame
print(advertising_data.shape)

fig, axs = plt.subplots(1, 3, sharey=True)

# Adding the scatterplots to the grid
advertising_data.plot(kind='scatter', x='TV', y='sales', ax=axs[0], figsize=(16, 8))
advertising_data.plot(kind='scatter', x='radio', y='sales', ax=axs[1])
advertising_data.plot(kind='scatter', x='newspaper', y='sales', ax=axs[2])

# create a fitted model in one line of code(which will represent the least squares line)
lm = smf.ols(formula='sales ~ TV', data=advertising_data).fit()
# show the trained model coefficients
lm.params

# creating a Pandas DataFrame to match Statsmodels interface expectations
new_TVAdSpending = pd.DataFrame({'TV': [50000]})
new_TVAdSpending.head()

# use the model to make predictions on a new value
preds = lm.predict(new_TVAdSpending)

# create a DataFrame with the minimum and maximum values of TV
X_min_max = pd.DataFrame({'TV': [advertising_data.TV.min(), advertising_data.TV.max()]})
print(X_min_max.head())


# predictions for X min and max values
predictions = lm.predict(X_min_max)
print('Predictions for the minmum and maximum value of TV..')
print(predictions)

# plotting the acutal observed data
advertising_data.plot(kind='scatter', x='TV', y='sales')
#plotting the least squares line
plt.plot(new_TVAdSpending, preds, c='red', linewidth=2)


