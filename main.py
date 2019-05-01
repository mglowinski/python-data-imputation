import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def writeInfoAboutDataFrame(df):
    print("std: \n" + str(df.std()) + "\n")
    print("mean: \n" + str(df.mean()) + "\n")
    print("quantile: \n" + (df.quantile([0.25, 0.5, 0.75]).to_string()))

df = pd.read_csv('auto-mpg.data')
columnCount = len(df.columns)
rowCount = df.shape[0]
expectedValuesCount = rowCount * columnCount
print("Expected values count =  " + str(expectedValuesCount))
filledValuesCount = df.count().sum()
print("Filled values count = " + str(filledValuesCount))
filledValuesPercentage = filledValuesCount / expectedValuesCount * 100
missingDataPercentage = float("{0:.2f}".format(100 - filledValuesPercentage))
print("There is " + str(missingDataPercentage) + "% missing data")

print("-----------------------------------------------------")

print("Missing data count for each column:")
print(df.isnull().sum())

print("-----------------------------------------------------")

# before mean imputation - drop NAN values
dfWithoutNA = df.copy().dropna()

print("Data characteristic before mean imputation:")
writeInfoAboutDataFrame(dfWithoutNA)

# after mean imputation - fill gaps with MEAN imputation - with average of each columns
dfFilled = df.copy()
dfFilled = dfFilled.fillna(dfFilled.mean())

print("-----------------------------------------------------")

print("Data characteristic filled by mean imputation:")
writeInfoAboutDataFrame(dfFilled)

# without NA - before imputation
# Get the linear models
lm_withoutNA = np.polyfit(dfWithoutNA.displacement, dfWithoutNA.weight, 1)
# calculate the x and y values based on the co-efficients from the model
r_x, r_y = zip(*((i, i * lm_withoutNA[0] + lm_withoutNA[1]) for i in dfWithoutNA.displacement))
df_withoutNA_co_efficients = pd.DataFrame({
    'displacement': r_x,
    'weight': r_y
})

# filled - after imputation
# Get the linear models - calculate a and b co-efficients - least squares
lm_filled = np.polyfit(dfFilled.displacement, dfFilled.weight, 1)
# calculate the x and y values based on the co-efficients from the model
r_x, r_y = zip(*((i, i * lm_filled[0] + lm_filled[1]) for i in dfFilled.displacement))
df_filled_co_efficients = pd.DataFrame({
    'displacement': r_x,
    'weight': r_y
})

fig, axes = plt.subplots(nrows=1, ncols=2)

dfWithoutNA.plot(kind='scatter', color='Blue', x='displacement', y='weight', ax=axes[0], title='Before')
df_withoutNA_co_efficients.plot(kind='line', color='Red', x='displacement', y='weight', ax=axes[0], label='reg line')

dfFilled.plot(kind='scatter', color='Blue', x='displacement', y='weight', ax=axes[1], title='After')
df_filled_co_efficients.plot(kind='line', color='Red', x='displacement', y='weight', ax=axes[1], label='reg line')

plt.show()