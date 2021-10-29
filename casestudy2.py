import pandas as pd

df = pd.read_csv('casestudy.csv')

years = sorted(list(set(df.loc[:, 'year'])))

net_rev_yr = {}

dict_df_yrs = {}

cust_yr = {}
new_cust_rev = {}
ex_cust_rev = {}
tot_cust_yr = {}
new_cust_yr = {}
lost_cust_yr = {}
ex_cust_growth = {}
rev_per_ex_cust = {}
rev_per_new_cust = {}
ex_cust_yr = {}

n = len(df)


def isin(el, set0):
    return el in set0


for yr in years:
    dict_df_yrs[yr] = df[df['year'] == yr]
    net_rev_yr[yr] = df[df['year'] == yr].loc[:, 'net_revenue'].sum()
    cust_yr[yr] = set(dict_df_yrs[yr]['customer_email'])
    tot_cust_yr[yr] = dict_df_yrs[yr].loc[:, 'customer_email'].nunique()
    if yr > min(years):
        ex_cust_yr[yr] = df[(df['year'] == yr) & (df['customer_email'].map(lambda x: x in cust_yr[yr - 1]))].loc[:, 'customer_email'].nunique()
        new_cust_yr[yr] = df[(df['year'] == yr) & (df['customer_email'].map(lambda x: x not in cust_yr[yr-1]))].loc[:, 'customer_email'].nunique()
        lost_cust_yr[yr] = df[(df['year'] == yr-1) & (df['customer_email'].map(lambda x: x not in cust_yr[yr]))].loc[:, 'customer_email'].nunique()
        new_cust_rev[yr] = df[(df['year'] == yr) & (df['customer_email'].map(lambda x: x not in cust_yr[yr-1]))].loc[:, 'net_revenue'].sum()
        ex_cust_rev[yr] = df[(df['year'] == yr) & (df['customer_email'].map(lambda x: x in cust_yr[yr-1]))].loc[:, 'net_revenue'].sum()
        if yr-1 > min(years):
            ex_cust_growth[yr] = ex_cust_rev[yr] - ex_cust_rev[yr-1]
        rev_per_ex_cust[yr] = ex_cust_rev[yr] / ex_cust_yr[yr]
        rev_per_new_cust[yr] = new_cust_rev[yr] / new_cust_yr[yr]


# One observation is that existing customer revenue dropped quite a lot
# The increase in sale that 2017 saw was supported by new customers

# What graphs I want to make: Bar graph with net revenue for each year,
# perhaps on that graph I want to show how much was existing customer
# revenue and how much was new customer revenue. Also include how many new
# customers and how many existing customers as well as the revenue per
# existing customer and revenue per new customer

# I also want to make a histogram of how much each customer generates
# in revenue, to get an idea of the distribution
"""
for i in range(n):
    entry = df.loc[i]
    if entry['year'] in net_rev_yr:
        net_rev_yr[entry['year']] += entry['net_revenue']
    else:
        net_rev_yr[entry['year']] = entry['net_revenue']
"""
