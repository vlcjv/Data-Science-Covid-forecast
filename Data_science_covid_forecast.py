#!/usr/bin/env python
# coding: utf-8

# # <span style="color:purple">December Covid Forecast

# ***Welcome to my first attempt of forecast in python!***
# 
# 
# The data provided in this project is taken from the set pointed
# by World Health Organisation as their source of analysis.
# [Source]: https://ourworldindata.org/covid-cases

# In[32]:


#libraries

import pandas as pd
import numpy as np
from sklearn import svm
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns


# In[2]:


# Data import
data_covid = pd.read_excel(r'C:\Users\alaad\Downloads\covid_data.xlsx')


# In[3]:


# Display of the data to check whether all is needed
data_covid


# In[4]:


data_covid.info()
data_covid.plot(kind ='line')


# In[5]:


# Display the data as plot
plt.show()


# In[6]:


#   Since the provided data refers to the whole world it needs to me narrowed down 
# to those rows relating to Poland specifically. To achieve that, the unwanted columns
# and rows will be deleted.

# Country: Poland
# Columns: date and the cases

column_name = 'location'
value_pl = 'Poland'
columns_keep = ['date', 'total_cases', 'new_cases', 'new_cases_smoothed', 'total_cases_per_million',
               'new_cases_per_million', 'new_cases_smoothed_per_million']

mask = data_covid[column_name] == value_pl

data_covid_filtered = data_covid[mask]

data_covid_filtered = data_covid_filtered[columns_keep]

data_covid_filtered.to_excel('covid_data_pl.xlsx', index= False)



# In[7]:


data_covid_pl = pd.read_excel('covid_data_pl.xlsx')


# In[8]:


data_covid_pl


# In[9]:


# Display the data as plot
data_covid_pl.plot
plt.show()


# In[10]:


# The plot wasn't possible, the data needs preparation

# Type of the data
print(type(data_covid_pl))

# Veryfication of the columns
print(data_covid_pl.columns)

# Sum of the missing values
print(data_covid_pl.isnull().sum())


# In[11]:


# Replacing missing values with zeroes
data_covid_pl.fillna(0, inplace = True)


# In[12]:


#Plotting the prepared data
data_covid_pl.plot(x='date', y=['total_cases', 'new_cases', 'new_cases_smoothed', 'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million'])
plt.show()


# In[13]:


# Plotting just the new cases
data_covid_pl.plot(x='date', y='new_cases')
plt.show()


# In[14]:


# Plotting only two last months2
data_covid_pl['date'] = pd.to_datetime(data_covid_pl['date'])
two_months_ago = pd.to_datetime('today') - pd.DateOffset(months=2)
filtered_data = data_covid_pl[data_covid_pl['date'] >= two_months_ago]

# Plot the filtered data
filtered_data.plot(x='date', y='new_cases')
plt.show()


# ***Forecasting: ARIMA MODEL***
# 
# 

# In[33]:


# Determing the values for the modeling
# Plot ACF and PACF

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(data_covid_pl['new_cases'], lags=40, ax=ax1)

# ywm method used for the warnings as it is for unadjusted Yule-Walker method
plot_pacf(data_covid_pl['new_cases'], lags=40, ax=ax2, method='ywm') 


# In[ ]:





# In[41]:


# Determining the values p, d, q

p, d, q = 3, 1, 1

# Creating the ARIMA model
model = ARIMA(data_covid_pl['new_cases'], order=(p, d, q))

start_params = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
results = model.fit(start_params=start_params)

# Forecasting
forecast_steps = 31  # as for December
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data_covid_pl['date'].max() + pd.DateOffset(days=1), periods=forecast_steps)

# Convert conf_int to DataFrame
conf_int_df = pd.DataFrame(forecast.conf_int(), index=forecast_index)

# Plotting original data and forecast
plt.figure(figsize=(10, 6))
plt.plot(data_covid_pl['date'], data_covid_pl['new_cases'], label='Original Data')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecasted Data', color='red')
plt.fill_between(conf_int_df.index, conf_int_df.iloc[:, 0], conf_int_df.iloc[:, 1], color='red', alpha=0.2)
plt.title('ARIMA Forecast for New Cases')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()

# Displaying total cases in December
total_cases_december = forecast.predicted_mean.loc[forecast_index.month == 12].sum()
plt.text(forecast_index.max(), forecast.predicted_mean.max(), f'Total Cases in December: {total_cases_december:.0f}', ha='right', va='center', color='purple')

plt.show()


# ***Thank you:)***
