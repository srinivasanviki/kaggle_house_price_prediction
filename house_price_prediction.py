import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
quantitative_features=[]
qualitative_features=[]
house_attributes=pd.DataFrame(pd.read_csv("/Users/vigneshsrinivasan/kaggle/train.csv"))
test=pd.DataFrame(pd.read_csv("/Users/vigneshsrinivasan/kaggle/test.csv"))

def plot_corr(df,size=10):
    import seaborn as sns
    corr = df.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show()

def scatter_plot(df,var1,var2):
    data = pd.concat([df[var1], df[var2]], axis=1)
    data.plot.scatter(x=var2, y=var1)
    plt.show()


def get_all_quantitative_features(df):
    for column in df:
        if df[column].dtype != object:
            quantitative_features.append(column)
        else:
            qualitative_features.append(column)


# Only the Count Of Na
def get_missing_attributes(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0.90*len(df)]
    return missing

def get_missing_attributes_more_than_ninety(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0.90*len(df)]
    return missing


def check_skewness(df):
    skweness={}
    for key,value in df.iteritems():
        if df[key].dtypes !=object:
            mean=house_attributes[key].mean()
            median=house_attributes[key].median()

            if mean>median:
                skweness[key]="positive"
            elif mean<median:
                skweness[key]="negative"
    return skweness


def correlation_between_quantitative(df,columns):
    y=df['SalePrice']
    for column in columns:
        sns.regplot(x=df[column], y=y)
        plt.show()

def boxplot(df,x):
   sns.boxplot(x=x, y="SalePrice", data=df)
   plt.show()


def check_categorical(attribute):
    if house_attributes[attribute].dtypes !=object:
        return False
    elif house_attributes[attribute].dtypes == object:
        return True

def check_normality_using_shipro_wilk(df):
    test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01
    normal = pd.DataFrame(df)
    normal = normal.apply(test_normality)
    return normal.any()

#for imputing lot frontage by check if the data is normaly distributed
def impute_missing_values():
    missing_attributes=get_missing_attributes(house_attributes)
    for key,attributes in missing_attributes.iteritems():
        if not check_categorical(key):
            if check_normality_using_shipro_wilk(house_attributes[key]):
                house_attributes[key].fillna(house_attributes[key].mode)


#Based on assumption since there are 90% houses which doesn't have either pool,alley or Miscfeatures
#(So it doesn't effect the sales price)
def remove_missing_data():
    missing_attributes=get_missing_attributes_more_than_ninety(house_attributes)
    for key,attributes in missing_attributes.iteritems():
        del house_attributes[key]

def count_outliers(df):
    outliers={}
    for key,value in df.iteritems():
        q75, q25 = np.percentile(df[key], [75 ,25])
        iqr = q75 - q25
        count=0
        for value in df[key]:
            if value > (1.5*iqr):
                count+=1
        if count > 0.80 *len(df[key]):
            outliers[key]=count

    return outliers


def transform_features(features):
    attributes=check_skewness(house_attributes[features])
    for attribute,value in attributes.iteritems():
        if attribute not in ("YrSold",'MoSold','YearBuilt'):
            if value =="positive":
                # Log tranformation
                house_attributes[attribute]=np.log(house_attributes[attribute]+1)
            elif value=="negative":
                #square transformation for left squwenes
                house_attributes[attribute]=np.square(house_attributes[attribute])
       #     house_attributes[attribute]=(house_attributes[attribute] - house_attributes[attribute].mean())/house_attributes[attribute].std(ddof=0)

def check_vif(df):
    from vif import ReduceVIF
    transformer = ReduceVIF()
    # Only use 10 columns for speed in this example
    X = transformer.fit_transform(df[[col for col in df.columns if col !="SalePrice"]],df['SalePrice'])
    return X.head()

def boxplot_categorical(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

def check_categorical_features():
    for c in qualitative_features:
        house_attributes[c] = house_attributes[c].astype('category')
        if house_attributes[c].isnull().any():
            house_attributes[c] = house_attributes[c].cat.add_categories(['MISSING'])
            house_attributes[c] = house_attributes[c].fillna('MISSING')
    f = pd.melt(house_attributes, id_vars=['SalePrice'], value_vars=qualitative_features)
    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
    g = g.map(boxplot_categorical, "value", "SalePrice")
    plt.show()

def check_influence(frame):
    import math
    anv = pd.DataFrame()
    anv['feature'] = qualitative_features
    pvals = []
    for c in qualitative_features:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    a=anv.sort_values('pval')
    a['disparity'] = np.log(1./a['pval'].values)

    exclude_features= a[pd.isna(a['disparity'])]['feature'].values
    extracted_features=[]
    for key,value in a['feature'].iteritems():
        if value not in exclude_features:
            extracted_features.append(value)
    return extracted_features


def one_hot_encoding(qualitative_features):
    for feature in qualitative_features:
        house_attributes[feature]=house_attributes[feature].factorize()[0]


def one_hot_encoding_test(qualitative_features):
    for feature in qualitative_features:
        test[feature]=test[feature].factorize()[0]


def fit(X,y,test_X):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    regr = RandomForestRegressor()
    regr.fit(X, y)
    prediction=regr.predict(X)
    print mean_squared_error(y,prediction)
    from sklearn.metrics import r2_score
    print r2_score(y,prediction)


    # df = pd.DataFrame()

    # df['Id']=test_X['Id']
    # df['SalePrice']=prediction
    # df.to_csv("/Users/vigneshsrinivasan/Desktop/df.csv")






house_attributes=house_attributes.rename(index=str, columns={"Condition1": "prxcond", "Condition2": "prxcondothers",
                                            "Exterior1st":"extcov","Exterior2nd":"extcovothers",
                                            "BsmtFinType1":"bsmtrtngfirst","BsmtFinType2":"bsmtrtngsec",
                                            "BsmtFinSF1":"bsmtfirstsqrfeet","BsmtFinSF2":"bsmtsecondsqrfeet",
                                            "1stFlrSF":"firstflrsf","2ndFlrSF":"secondflrsf",
                                            "3SsnPorch":"threeseasonporcharea"})

test=test.rename(index=int, columns={"Condition1": "prxcond", "Condition2": "prxcondothers",
                                                             "Exterior1st":"extcov","Exterior2nd":"extcovothers",
                                                             "BsmtFinType1":"bsmtrtngfirst","BsmtFinType2":"bsmtrtngsec",
                                                             "BsmtFinSF1":"bsmtfirstsqrfeet","BsmtFinSF2":"bsmtsecondsqrfeet",
                                                             "1stFlrSF":"firstflrsf","2ndFlrSF":"secondflrsf",
                                                             "3SsnPorch":"threeseasonporcharea"})

impute_missing_values()
get_all_quantitative_features(house_attributes)
transform_features(quantitative_features)
quantitive_features_selected=list(check_vif(house_attributes[quantitative_features]).keys())
qualitative_extracted_features=check_influence(house_attributes)

selected_features= list(set().union(quantitive_features_selected,qualitative_extracted_features))
#
# house_attributes=house_attributes.rename(index=str, columns={"prxcond":"Condition1","prxcondothers":"Condition2","extcov":"Exterior1st","extcovothers":"Exterior2nd","bsmtrtngfirst":"BsmtFinType1","bsmtrtngsec":"BsmtFinType2","bsmtfirstsqrfeet":"BsmtFinSF1","bsmtsecondsqrfeet":"BsmtFinSF2","firstflrsf":"1stFlrSF","secondflrsf":"2ndFlrSF","threeseasonporcharea":"3SsnPorch"})
#

for feature in quantitive_features_selected:
    if np.any(np.isnan(house_attributes[feature])):
        house_attributes[feature].fillna(0, inplace=True)

    if np.any(np.isnan(test[feature])):
        test[feature].fillna(0,inplace=True)




one_hot_encoding(qualitative_extracted_features)


train_X=house_attributes[selected_features]
target=house_attributes['SalePrice']

one_hot_encoding_test(qualitative_extracted_features)
test_X=test[selected_features]
# test_Y=test['SalePrice']
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(train_X, target)

preds = rf.predict(test_X)

my_submission = pd.DataFrame({'Id': test_X["Id"], 'SalePrice': preds})
my_submission.to_csv('submission.csv', index=False)
