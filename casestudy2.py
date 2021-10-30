import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import os

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


def info_digest():
    print("Net Revenue per Year : "+str(net_rev_yr))
    print("Revenue from new customers : "+str(new_cust_rev))
    print("Revenue from existing customers : "+str(ex_cust_rev))
    print("Total customers per year : "+str(tot_cust_yr))
    print("How many new customers per year : "+str(new_cust_yr))
    print("How many existing customers per year : "+str(ex_cust_yr))
    print("Existing Customer Growth : "+str(ex_cust_growth))
    print("Revenue per new customer : "+str(rev_per_new_cust))
    print("Revenue per existing customer : "+str(rev_per_ex_cust))
    print("How many existing customers were lost each year : "+str(lost_cust_yr))
    rows = ["Revenue from new customers","Revenue from existing customers","Total customers per year","How many new customers per year",
            "How many existing customers per year","Existing Customer Growth","Revenue per new customer","Revenue per existing customer",
            "How many existing customers were lost each year"]
    columns = [2015, 2016, 2017]



n = len(df)

custs = df.loc[:, 'customer_email'].values


rev_per_cust_per_year = [{}] * len(years)

#for i in range(len(years)):
#    rev_per_cust_per_year[i] = df[(df.loc[:, 'year'] == yr) & (df.loc[:, 'customer_email'] == cust)].loc[:, 'net_revenue'].sum()

ex_cust_growth[2015]= 'NA'
ex_cust_growth[2016]= 'NA'
new_cust_yr[2015]= 'NA'
ex_cust_yr[2015]= 'NA'
lost_cust_yr[2015]= 'NA'
new_cust_rev[2015]= 'NA'
ex_cust_rev[2015]= 'NA'
rev_per_ex_cust[2015]= 'NA'
rev_per_new_cust[2015]= 'NA'

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

def make_rev_share_bar_graph():
    plt.clf()
    ax = plt.gca()
    #print(net_rev_yr)
    nry = [net_rev_yr[yr] for yr in years]
    plt.bar(years, nry, color='green',
                       edgecolor='black', linewidth=1, tick_label=years, )
    ax.yaxis.set_major_formatter(EngFormatter())
    plt.ylabel('Revenue (dollars)')
    plt.title('Net Revenue')
    plt.show()
    plt.clf()
    ecr = [ex_cust_rev[yr] for yr in years[1:]]
    ncr = [new_cust_rev[yr] for yr in years[1:]]
    ax = plt.gca()
    #print(ecr)
    #print(ncr)
    plt.bar([2016,2017], ecr, color='orange', edgecolor='black', linewidth=1)
    plt.bar([2016,2017], ncr, color='purple', bottom=ecr,
                       edgecolor='black', linewidth=1, tick_label=years[1:], )
    ax.yaxis.set_major_formatter(EngFormatter())
    plt.title('Net Revenue, New and Existing customer split')
    plt.ylabel('Revenue (dollars)')
    ax.legend(['Orange - Existing Customer Revenue', 'Purple - New Customer Revenue'], loc='upper left')
    plt.show()


def make_histograms():
    for yr in years:
        plt.clf()
        n, bins, patches = plt.hist(dict_df_yrs[yr].loc[:, 'net_revenue'], edgecolor='black', bins=100,
                                    color='rebeccapurple')
        plt.title('Net Revenue distribution of Customers in '+str(yr))
        plt.ylabel('Frequency')
        plt.xlabel('Net Revenue')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(EngFormatter())
        #ax.axes.yaxis.set_ticklabels([])
        filename = os.path.join(os.getcwd(), 'casestudy2_figures', 'histograms', '_' + str(yr) + '_net_revenue' + '__histogram.jpg')
        plt.show()
        #plt.savefig(filename, dpi=1000, )
        plt.clf()

"""
for i in range(n):
    entry = df.loc[i]
    if entry['year'] in net_rev_yr:
        net_rev_yr[entry['year']] += entry['net_revenue']
    else:
        net_rev_yr[entry['year']] = entry['net_revenue']
"""
