import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
quantitative_features=[]
qualitative_features=[]
train=pd.DataFrame(pd.read_csv("/home/viki/kaggle/house_price_data/train.csv"))
test=pd.DataFrame(pd.read_csv("/home/viki/kaggle/house_price_data/test.csv"))


#check the shape of train
print train.shape
print test.shape

#check for NA more than 80 percent.
columns=train.columns[train.isnull().mean()>0.8]


# check whether the columns has any effect on sale price
# Alley
i=0
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for col in columns:
    fig.add_subplot(2, 2, i)
    sns.boxplot(x=col, y="SalePrice", data=train)
    i=i+1
plt.show()


'''
Based on the obserservation poolqc effects the price of house
Removing Fence MiscFeatures,Alley (doesnt effect the price toomuch
'''
for col in columns:
    del train[col]

#Analyse Sales Price

#check for corr with greter than 0.5 with saleprice
corr_mat=train.corr()
selected_col=[]
col_corr=corr_mat["SalePrice"]>0.5
for col,correlation in col_corr.iteritems():
    if correlation==True:
        selected_col.append(col)

corr_selected=train[selected_col].corr()


sns.heatmap(corr_selected,
                xticklabels=corr_selected.columns.values,
                yticklabels=corr_selected.columns.values)
plt.show()

'''
get the pair of features which are highly correlated
from the selected features

It can indicate Multicollinearity
'''

indices = np.where(corr_selected>0.5)
highest_correlated_col= [(corr_selected.index[x], corr_selected.columns[y]) for x, y in zip(*indices)
           if x != y and x < y]


'''
Lets check for VIF for checking multicollinearity
'''
print highest_correlated_col
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(corr_selected.values, i) for i in range(corr_selected.shape[1])]
vif["features"] = corr_selected.columns


'''
Keeping 1stFlrSF in comparison to TotalBsmtSF becuase they explains nearly the same variance
'''
del corr_selected["TotalBsmtSF"]

final_selected_columns=corr_selected.columns




#Preprocessing of features

print train.shape



selected_test_features=["Id"]
selected_train_features=["Id"]
for col in final_selected_columns:
    if col !="SalePrice":
        selected_train_features.append(col)
        selected_test_features.append(col)

# preprocessing

'''
check for saleprice whether it is skewed or normal
'''
sns.distplot(train["SalePrice"])
plt.show()

skewness=train["SalePrice"].skew()
kurtosis=train["SalePrice"].kurt()
print "Skewness %s and Kurt %s"%(skewness,kurtosis)



'''
Checking the distribution of other features
'''



train_X=pd.get_dummies(train[selected_train_features])

target=train["SalePrice"]

test=test.fillna(0)

test_X=pd.get_dummies(test[selected_test_features])






from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(train_X, target)

preds = rf.predict(test_X)


my_submission = pd.DataFrame({'Id': test_X["Id"], 'SalePrice': preds})
my_submission.to_csv('submission.csv', index=False)
