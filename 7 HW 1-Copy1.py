#!/usr/bin/env python
# coding: utf-8

# Use "realtor_data.csv" to answer the following questions  
# Review the code we have learned in class when completing these questions

# In[52]:


import pandas as pd
import matplotlib.pyplot as plt


# In[54]:


url = 'https://drive.google.com/uc?id=1UZuosGdSc8CSVofDBNDx6Ju_YtkwTL56'

df = pd.read_csv(url)
df.head(30)

df.info()


# 12. (Open-Ended Analysis) As a data analyst helping a client select a property for Airbnb investment,   
# develop a data-driven search strategy. Use the available dataset to:  
# 
#     Analyze relationships between property characteristics and potential returns  
#     Provide recommendations backed by statistical analysis  
#     Include visualizations and statistical evidence (descriptive statistics / regression) to support your recommendations.  
#     At least one regression analysis is required.

# In[30]:


df.describe()


# In[31]:


top_10_cash = (
    df.dropna(subset=["Airbnb Cash on Cash"])
      .sort_values(by="Airbnb Cash on Cash", ascending=False)
      .head(10)
)

print(top_10_cash)


# In[35]:


df_clean = df.dropna(subset=["Airbnb Cash on Cash", "Listing Price"])

top_10_best_deals = (
    df_clean
    .sort_values(
        by=["Airbnb Cash on Cash", "Listing Price"],
        ascending=[False, True]   # High return, Low price
    )
    .head(10)
    [["Zip", "Beds", "SQFT", "Listing Price", "Airbnb Cash on Cash"]]
)

top_10_best_deals


# In[41]:


top_10_best_deals.plot(
    kind='scatter',
    x='Listing Price',
    y='Airbnb Cash on Cash'
)




# In[45]:


import matplotlib.pyplot as plt

df.plot(kind='scatter', x='Listing Price', y='Airbnb Cash on Cash')

plt.ticklabel_format(style='plain', axis='x')

plt.show()



# In[46]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(data=df, x='Listing Price', y='Airbnb Cash on Cash')

plt.ticklabel_format(style='plain', axis='x')

plt.show()




# In[47]:


import statsmodels.api as sm

df_reg = df.dropna(subset=["Listing Price", "Airbnb Cash on Cash"])

X = df_reg["Listing Price"]
X = sm.add_constant(X)

y = df_reg["Airbnb Cash on Cash"]

model = sm.OLS(y, X).fit()

print(model.summary())


# In[ ]:


# Conclusion from this regression: R2 value of .427 means that 42.7% of the variance in return is  
# based on listing price- listing prices Coefficient of -2.766e-05  means that cash on cash return 
# decreases by 2.77 for every $100,000 in listing price 


# In[49]:


import statsmodels.api as sm

df_reg = df.dropna(subset=[
    "Airbnb Cash on Cash",
    "Listing Price",
    "SQFT",
    "Beds",
    "Year Built"
])

X = df_reg[["Listing Price", "SQFT", "Beds", "Year Built"]]
X = sm.add_constant(X)

y = df_reg["Airbnb Cash on Cash"]

model_multi = sm.OLS(y, X).fit()

print(model_multi.summary())


# In[ ]:


#Multiple Regression Co-efficients and their translation: 

#Listing Price: Coefficient = −2.337e−05 (p < 0.001)- for every $100,000 increase in listing price, cash on cash return 
#decreases by 1.06 points, holding other factors constant

#Year Built Coefficient = −0.0263 (p = 0.008)- each additional year newer reduces expected cash on
#cash values by .026 points

#Beds- Coefficient = 1.055 (p = 0.026)- each additional  bedroom increases expected cash on
#cash return by approximately 1.06 points, holding other factors constant

# SQFT: Coefficient = −0.0008 (p = 0.153)- Not statistically significant- does not meaningfully
#effect return



# In[61]:


top_deals = (
    df.sort_values(
        by=["Airbnb Cash on Cash", "Listing Price"],
        ascending=[False, True]
    )
    .loc[:, ["Zip", "Beds", "Baths", "SQFT", "Listing Price", "Airbnb Cash on Cash"]]
    .head(10)
)

top_deals


# In[ ]:




