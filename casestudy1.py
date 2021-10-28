import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.feature_selection import VarianceThreshold
import matplotlib as plt
import jinja2
import seaborn as sns
import time
import random

pd.options.display.width = 170
pd.options.display.max_colwidth = 75
pd.options.display.max_columns = 75

df = pd.read_csv('loans_full_schema.csv')

#print(df.describe(include='all'))
# Helping me get a grasp on what this dataset looks like, other than just looking at the csv in excel


# This table of subgrades to the risk adjustment percentage rates was on lendingclub.com,
# and it seems like it'll be super useful

# Hmmm, these seem too well correlated. I'm not sure if I should be allowed to use these
# A .977098 correlation with the interest rate.
sub_grade_risk_rate_adj_table = {'A1': 3.41,'A2': 3.97,'A3': 4.51,'A4': 5.14,'A5': 5.14,
                                 'B1': 8.28,'B2': 8.97,'B3': 9.66,'B4':10.35,'B5':11.03,
                                 'C1':12.25,'C2':13.19,'C3':14.07,'C4':14.90,'C5':15.69,
                                 'D1':17.57,'D2':19.50,'D3':22.00,'D4':24.60,'D5':25.94,
                                 'E1':23.85,'E2':23.87,'E3':23.90,'E4':23.92,'E5':23.95,
                                 'F1':24.30,'F2':24.64,'F3':25.12,'F4':25.60,'F5':25.70,
                                 'G1':25.74,'G2':25.79,'G3':25.84,'G4':25.89,'G5':25.94}


def process_dataframe(df):
    df_dummies = pd.get_dummies(df.loc[:, ['state', 'homeownership', 'verified_income',
                                'verification_income_joint', 'loan_purpose', 'application_type', 'grade',
                                'sub_grade', 'issue_month', 'loan_status', 'initial_listing_status',
                                'disbursement_method']])
    # Non-numeric columns: [emp_title, state, homeownership, verified_income,
    #                       verification_income_joint, loan_purpose, application_type, grade, sub_grade,
    #                       issue_month, loan_status, initial_listing_status, disbursement_method]

    # emp_title has entirely too many classes to try to do anything with, without classifying into broader categories

    # many missing columns: annual_income_joint, months_since_90d_late, months_since_last_credit_inquiry,


    na_dict_dummies = {str(key) + "__NA": df.loc[:, key] for key in ['emp_length', 'emp_title',
                                                                     'months_since_last_delinq',
                                                                     'months_since_90d_late',
                                                                     'months_since_last_credit_inquiry',
                                                                     'annual_income_joint']}
    na_dummies = pd.DataFrame(na_dict_dummies).isna()

    # For annual income joint, I will try replacing values of annual income with annual_income_joint/2,
    # if that is the larger value. I'll see how it goes
    annual_income = df.loc[:, 'annual_income']
    mod_annual_income_joint = df.loc[:, 'annual_income_joint'].fillna(value=0)/2

    mod_ann_income = pd.Series([max(annual_income[i], mod_annual_income_joint[i]) for i in range(len(annual_income))],
                               name='modded_annual_income')
    avg_debt_to_income = df.loc[:, 'debt_to_income'].mean()
    debt_to_income = df.loc[:, 'debt_to_income'].fillna(value=avg_debt_to_income)  # There is enough data here to probably be safe
    # I have this here on two separate lines because I was thinking about using some other value, but, I would need
    # more information about the data collection process to know what would be preferable

    BIGNESS = df.loc[:, 'debt_to_income_joint'].max() * 100 # I want to not have this influence the minimum thing I'm
    # finding in the mod_debt_to_income. So I'm setting it to an absurdly large value
    mod_debt_to_income_joint = df.loc[:, 'debt_to_income_joint'].fillna(value=BIGNESS)
    mod_debt_to_income = pd.Series([min(debt_to_income[i], mod_debt_to_income_joint[i]) for i in range(len(debt_to_income))])



    # The columns that I want to get binary NA data for are :
    # emp_length, emp_title, months_since_last_delinq, months_since_90d_late, months_since_last_credit_inquiry,

    sub_grades = df.loc[:, 'sub_grade']
    sub_grade_rates = pd.Series([sub_grade_risk_rate_adj_table[sub_grades[i]] for i in range(len(sub_grades))],
                                name='sub_grade_rates')
    df.loc[:, 'emp_length'].fillna(value=df.loc[:, 'emp_length'].mean(), inplace=True) # filling with the average, enough samples, and hopefully not
    # too biased in terms of what is missing.

    df.loc[:, 'months_since_90d_late'].fillna(value=df.loc[:, 'months_since_90d_late'].max() * 1.1,
                                              inplace=True)
    # I'm assuming that having and NA value for months since 90d late means they have never been 90days late
    # so they should have perhaps an infinite value, but that won't work numerically, so I'm using 1.1 times max
    df.loc[:, 'months_since_last_credit_inquiry'].fillna(value=df.loc[:, 'months_since_last_credit_inquiry'].max()*1.1,
                                                         inplace=True)
    # Similar to that last one
    df.loc[:, 'months_since_last_delinq'].fillna(value=df.loc[:, 'months_since_last_delinq'].max()*1.1,
                                                         inplace=True)
    # Similar again, I can probably do all of these in one line, but that might be something for later
    df_with_dropped = df.drop(['grade', 'sub_grade', 'emp_title', 'state', 'homeownership', 'verified_income',
                                'verification_income_joint', 'loan_purpose', 'application_type', 'grade', 'sub_grade',
                               'issue_month', 'loan_status', 'initial_listing_status', 'disbursement_method',
                                'annual_income', 'annual_income_joint', 'num_accounts_120d_past_due', 'debt_to_income',
                               'debt_to_income_joint'], axis=1)

    df_processed = pd.concat([df_with_dropped, sub_grade_rates, na_dummies, mod_debt_to_income, mod_ann_income,
                              df_dummies], axis=1)
    df_processed.fillna(value=df_processed.mean(), inplace=True)
    df_processed.columns = df_processed.columns.astype(str)
    return df_processed


    # num_accounts_120d_past_due, probably fill the NAs with zeros, or just omit that column, it seems minimally useful


processed_data = process_dataframe(df)

y = processed_data.loc[:, 'interest_rate']
x = processed_data.loc[:, processed_data.columns != 'interest_rate']

#correlations = pd.concat([y, x], axis=1).corr()

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=None, train_size=None)

corr0 = df.corr()

#corr0.style.background_gradient(cmap='coolwarm').set_precision(2)

#sns.heatmap(corr0,
#            xticklabels=corr0.columns.values,
#            yticklabels=corr0.columns.values)

#mlp = make_pipeline(StandardScaler(),
#                    MLPRegressor(activation='relu', random_state=1337, alpha=1.5e-3,
#                                  hidden_layer_sizes=(4, 2), max_iter=1000, verbose=True,
#                                 tol=1e-6))
#mlp.fit(x_train, y_train)
#MLP_PREDS = mlp.predict(x_val)
#print("MLP R2_Score : "+str(r2_score(y_val, MLP_PREDS)))


def check_layer_sizes(layer_sizes, trials=1):
    # layer_sizes should be a list of tuples, each tuple is the range of sizes i want to check in that layer
    configs = 1
    lyrs = len(layer_sizes)
    sizing = [0]*lyrs
    for i in range(len(layer_sizes)):
        configs *= layer_sizes[i][1] - layer_sizes[i][0] + 1
    for conf in range(configs):
        C = conf
        for i in range(lyrs):
            sizing[i] = C % (layer_sizes[i][1] - layer_sizes[i][0] + 1) + layer_sizes[i][0]
            C = int(C/(layer_sizes[i][1] - layer_sizes[i][0] + 1))
        print(sizing)
        for _ in range(trials):
            mlp = make_pipeline(StandardScaler(),
                                MLPRegressor(activation='relu', random_state=random.randint(111, 999999), alpha=1.5e-4,
                                             hidden_layer_sizes=sizing, max_iter=100000, verbose=False,
                                             tol=1e-7, learning_rate='adaptive', learning_rate_init=.0005))
            mlp.fit(x_train, y_train)
            MLP_PREDS = mlp.predict(x_val)
            print("MLP" + str(sizing) + " R2_Score : " + str(r2_score(y_val, MLP_PREDS)))
    return


def check_linear_regression(trials=1):
    for _ in range(trials):
        linregr = make_pipeline(StandardScaler(),
                                LinearRegression(copy_X=True))
        linregr.fit(x_train, y_train)
        LINREGR_PREDS = linregr.predict(x_val)
        print("LINREGR R2_Score : " + str(r2_score(y_val, LINREGR_PREDS)))
        if trials == 1:
            return linregr
    return


def check_sgdregression(trials=1):
    for _ in range(trials):
        sgdregr = make_pipeline(StandardScaler(),
                                SGDRegressor(max_iter=100000, tol=1e-4, penalty='l2',
                                             learning_rate='adaptive', eta0=.004))
        sgdregr.fit(x_train, y_train)
        SGDREGR_PREDS = sgdregr.predict(x_val)
        print("SGDREGR R2_Score : " + str(r2_score(y_val, SGDREGR_PREDS)))
        if trials ==1 :
            return sgdregr
    return