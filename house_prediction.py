
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV



warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x:'%.3f'%x)
pd.set_option('display.max_rows', None)

def load():
    dataframe1= pd.read_csv("datasets/train.csv")
    dataframe2=pd.read_csv("datasets/test.csv")
    return dataframe1, dataframe2

df_train, df_test = load()
df_train.head()
df_test.head()
#Bağımlı değişken SalePrice

df= df_train.append(df_test, ignore_index=False).reset_index()

df= df.drop("index", axis=1)
#YrSold, GarageYearBuilt, YearBuilt
def check_df(dataframe):
    print("########### Shape #############")
    print(dataframe.shape)
    print("########### Columns ###############")
    print(dataframe.columns)
    print("######### data type ##############")
    print(dataframe.dtypes)
    print("############## Na number #########:")
    print(dataframe.isnull().sum())
    print("######## QUANTILE ##############")
    print(dataframe.quantile([0.00, 0.05, 0.50, 0.95,0.99, 1.00]).T)

check_df(df)

df = df.loc[df["SalePrice"]<=400000,]

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_col= [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != 'O' and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == 'O' and dataframe[col].nunique() > car_th]
    cat_col= cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    #Num cols
    num_col= [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_col = [col for col in num_col if col not in num_but_cat]

    print(f'Observations : {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_col)}')
    print(f'num_cols: {len(num_col)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_col, num_col, cat_but_car

cat_col, num_col, cat_but_car= grab_col_names(df)

df.head()
df.tail()

def cat_summary(dataframe, cat_col, plot=False):
    print(pd.DataFrame({"Value_Number": dataframe[cat_col].value_counts(),
                        "Ratio": dataframe[cat_col].value_counts()/len(dataframe)*100}))
    if plot:
        sns.countplot(x= dataframe[cat_col], data=dataframe)
        plt.show()

for col in cat_col:
    cat_summary(df, col)


def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_col].describe(quantiles).T)
    if plot:
        dataframe[num_col].hist()
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show()

for col in num_col:
    num_summary(df, col, True)


##### Analise to target ##########

def target_analyse(dataframe,target , col_name):
    print(dataframe.groupby(col_name).agg({target : ['mean', 'count']}))

for col in cat_col:
    target_analyse(df,'SalePrice', col)

############## KORELASYON ################


df.corr() #Korelasyonu 1  ve -1 e yakın olanların arasında bir doğrusallık vardır negatif veya pozitif bir doğrusallık

# Korelasyon matrisi

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block= True)


# MİSSİNG VALUES ############



def missing_value_tables(dataframe, na_name= False):
    na_col= [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_col].isnull().sum().sort_values(ascending=False)
    ratio= dataframe[na_col].isnull().sum()/len(dataframe[na_col])*100
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n\n')
    if na_name:
        return na_col


na_columns= missing_value_tables(df, True)
no_columns= ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]


for col in no_columns:
    df[col].fillna("No", inplace=True)

na_columns= missing_value_tables(df)

def missing_values_fill(data, num_method="median", cat_th=20, target="SalePrice"):
    na_var= [col for col in data.columns if data[col].isnull().sum() > 0]
    temp= data[target]
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= cat_th) else x, axis = 0)
    if num_method == "mean":
        data =data.apply(lambda x: x.fillna(x.mean()) if (x.dtype !='O') else x, axis=0)
    elif num_method =="median":
        data= data.apply(lambda x: x.fillna(x.median()) if x.dtype != 'O' else x, axis=0)
    data[target]= temp
    print(data[na_var].isnull().sum(), "\n\n")
    return data

df= missing_values_fill(df, num_method="median", cat_th=17)



missing_value_tables(df)


##################################
# AYKIRI DEĞER ANALİZİ
##########################


def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quantile1= dataframe[col_name].quantile(q1)
    quantile3= dataframe[col_name].quantile(q3)
    IQR= quantile3-quantile1
    lower_lim= quantile1-1.5*IQR
    upper_lim= quantile3+1.5*IQR
    return lower_lim,upper_lim

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.10, q3=0.90):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.10, q3=0.90)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_col:
    if col != "SalePrice":
        print(col, check_outlier(df, col))
        if check_outlier(df, col):
            replace_with_thresholds(df, col)


##################DEĞİŞKEN ÜRETİMİ################################
df["TotalQuality"]= df[["OverallQual", "OverallCond", "ExterQual","ExterCond", "BsmtCond","BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageCond", "Fence"]].sum(axis=1)

df["NEW_TotalFlrSF"]=df["1stFlrSF"] +df["2ndFlrSF"]
df["NEEW_TotalBsmtFin"]=df.BsmtFinSF1+df.BsmtFinSF2
df["NEW_PorchArea"]= df["OpenPorchSF"]+df["EnclosedPorch"]+df["ScreenPorch"]+df["3SsnPorch"]+ df["WoodDeckSF"]
df["NEW_TotalHouseArea"]= df.NEW_TotalFlrSF + df.TotalBsmtSF
df["NEW_RatioArea"]= df.NEW_TotalHouseArea / df.LotArea
df["NEW_Restoration"]=df.YearRemodAdd -df.YearBuilt
df["NEW_HouseAge"]=df.YrSold-df.YearBuilt

drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]
df.drop(drop_list, axis=1, inplace=True)


######################################
# RARE
######################################

# Kategorik kolonların dağılımının incelenmesi
cat_col, num_col, cat_but_car= grab_col_names(df)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_col)


# Sınıfların oranlarına göre diğer sınıflara dahil edilmesi

df["ExterCond"] = np.where(df.ExterCond.isin(["Fa", "Po"]), "FaPo", df["ExterCond"])
df["ExterCond"] = np.where(df.ExterCond.isin(["Ex", "Gd"]), "Ex", df["ExterCond"])

df["LotShape"] = np.where(df.LotShape.isin(["IR1", "IR2", "IR3"]), "IR", df["LotShape"])


df["GarageQual"] = np.where(df.GarageQual.isin(["Fa", "Po"]), "FaPo", df["GarageQual"])
df["GarageQual"] = np.where(df.GarageQual.isin(["Ex", "Gd", "TA"]), "ExGd", df["GarageQual"])


df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["GLQ", "ALQ"]), "RareExcellent", df["BsmtFinType2"])
df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["BLQ", "LwQ", "Rec"]), "RareGood", df["BsmtFinType2"])

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


rare_encoder(df,0.01)


##################################
# ENCODING
##################################

cat_col, num_col, cat_but_car= grab_col_names(df)
binary_col= [col for col in df.columns if df[col].nunique() == 2 and df[col].dtypes == 'O']

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_col:
    df = label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_col, drop_first=True)

df.head()



#DATA SCALİNG





##################################
# BASE MODEL KURULUMU
##################################


train_df=df[df['SalePrice'].notnull()]
test_df=df[df['SalePrice'].isnull()]
train_df.head(20)



###############TOPLU DENEME ########
#Veri sayısal olduğu için regression modeli kullanıyoruz kategorik olsaydı classifier kullanacaktık.
y= np.log1p(train_df['SalePrice'])
X= train_df.drop(['SalePrice', 'Id'], axis=1)

# Verinin eğitim ve tet verisi olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          #('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

#MODEL OLUŞTUR VE PARAMETRE AYARLA


lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500]
               #"colsample_bytree": [0.5, 0.7, 1]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
