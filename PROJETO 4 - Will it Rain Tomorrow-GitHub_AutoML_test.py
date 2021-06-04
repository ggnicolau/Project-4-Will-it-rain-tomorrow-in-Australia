#%% Libraries
import pandas as pd
#pyforest auto-imports
import warnings
import pandas as pds
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import resample
from pandas_profiling import ProfileReport
#import pyforest
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from lightgbm import plot_importance
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, plot_roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import RFECV as RFECV_SKYLEARN
pd.options.display.max_columns = 100
#pd.set_option('display.max_columns', None)
from IPython.display import Audio, display
def allDone():
    display(Audio(url='https://sound.peal.io/ps/audios/000/000/537/original/woo_vu_luvub_dub_dub.wav', autoplay=True))


# %% Tables
#Import key_table
rain_aus = pd.read_csv("C:/Users/user/Documents/1. GitHub/Projeto 4 - Itau/case_guide/data/rain_data_aus.csv")
rain_aus = rain_aus.rename(columns={"amountOfRain": "amntraintmrw"})
rain_aus['raintoday'].replace({'No': 0, 'Yes': 1},inplace = True)
rain_aus['raintomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
rain_aus.head()


# Import side_tables and concatenate in one
wind1 = pd.read_csv("C:/Users/user/Documents/1. GitHub/Projeto 4 - Itau/case_guide/data/wind_table_01.csv")
wind2 = pd.read_csv("C:/Users/user/Documents/1. GitHub/Projeto 4 - Itau/case_guide/data/wind_table_02.csv")
wind3= pd.read_csv("C:/Users/user/Documents/1. GitHub/Projeto 4 - Itau/case_guide/data/wind_table_03.csv")
wind4 = pd.read_csv("C:/Users/user/Documents/1. GitHub/Projeto 4 - Itau/case_guide/data/wind_table_04.csv")
wind5 = pd.read_csv("C:/Users/user/Documents/1. GitHub/Projeto 4 - Itau/case_guide/data/wind_table_05.csv")
wind6 = pd.read_csv("C:/Users/user/Documents/1. GitHub/Projeto 4 - Itau/case_guide/data/wind_table_06.csv")
wind7 = pd.read_csv("C:/Users/user/Documents/1. GitHub/Projeto 4 - Itau/case_guide/data/wind_table_07.csv")
wind8 = pd.read_csv("C:/Users/user/Documents/1. GitHub/Projeto 4 - Itau/case_guide/data/wind_table_08.csv")
wind = pd.concat([wind1, wind2, wind3, wind4, wind5, wind6, wind7, wind8])

#Correct merged side_tables
cont = 2
for col in wind.columns[8:14]:
    wind.loc[~wind[col].isnull(), wind.columns[cont]] = wind.loc[~wind[col].isnull(), col]
    cont +=1
wind = wind.drop(['windgustdir', 'windgustspeed', 'winddir9am', 'winddir3pm', 'windspeed9am', 'windspeed3pm'], axis=1)

allDone()
# %% Correct
#Merge all tables and apply conditions to correct it
rain_merge = pd.merge(left=rain_aus, right=wind, how='left', on=['date', 'location'])
rain_merge['date'] = pd.to_datetime(rain_merge['date'].str.strip(), format='%Y/%m/%d')
rain_merge.loc[(rain_merge.amntraintmrw < 0.4),'amntraintmrw']=0
#Duplicates
    #rain_merge.groupby(rain_merge.columns.tolist(),as_index=False).size())

allDone()
# %% Correct More
#Correct type from columns
rain_merge['wind_gustdir'] = rain_merge['wind_gustdir'].astype(str)
rain_merge['wind_dir9am'] = rain_merge['wind_dir9am'].astype(str)
rain_merge['wind_dir3pm'] = rain_merge['wind_dir3pm'].astype(str)
#turn it into a scale
encoder = LabelEncoder()
encoder.fit(rain_merge['wind_gustdir'])
#transform
rain_merge['wind_gustdir'] = encoder.transform(rain_merge['wind_gustdir'])
rain_merge['wind_dir9am'] = encoder.transform(rain_merge['wind_dir9am'])
rain_merge['wind_dir3pm'] = encoder.transform(rain_merge['wind_dir3pm'])

# see min and max from table
print(rain_merge['date'].min())
print(rain_merge['date'].max())

allDone()
# %% codecell
#Create a table by your current season (apply one month ago + actual month + next month)
seasoned_rain = rain_merge[(rain_merge['date'].dt.month == 5) | (rain_merge['date'].dt.month == 6) | (rain_merge['date'].dt.month == 7)]
seasoned_rain = seasoned_rain[~(seasoned_rain['date'].dt.year <= 2007)]
#seasoned_rain = seasoned_rain[~(seasoned_rain['date'].dt.year >= 2017)]

#Your pipeline to clean your data for your problem and manage it:
#seasoned_rain.drop(['raintomorrow', 'amntraintmrw', 'modelo_vigente', 'temp', 'temp9am', 'temp3pm', 'humidity'], axis=1)

allDone()
# %% See result
seasoned_rain['raintomorrow'].value_counts()
seasoned_rain.info()
seasoned_rain

allDone()
# %% MODEL IT: PyCaret
from pycaret.classification import *
clf1 = setup(data = seasoned_rain, target = 'raintomorrow'
             , silent = True
             , log_experiment = True, experiment_name = 'rain_tomorrow_exp'
             , log_plots = True, log_profile = True, log_data = True
             , profile = True #, profile_kwargs = True
             , train_size = 0.3
             #, sampling = True
             , numeric_imputation = 'median', categorical_imputation = 'constant'
             , normalize = True, normalize_method = 'zscore'
             , handle_unknown_categorical = True, unknown_categorical_method = 'most_frequent'
             , fix_imbalance = True
             , transformation = True, transformation_method = 'yeo-johnson'
             , combine_rare_levels = True, rare_level_threshold = 0.1
             , feature_selection = True, feature_selection_threshold = 0.8
             , remove_multicollinearity = True, multicollinearity_threshold = 0.95
             , pca = False
             , ignore_low_variance = True
             , fold_strategy = 'stratifiedkfold'
             , fold = 10
             , use_gpu = False
              )

logs = get_logs(save=True)

allDone()
#%%  Set Unseen Data
data = seasoned_rain
data.shape
data = seasoned_rain.sample(frac=0.95, random_state=786)
data_unseen = seasoned_rain.drop(data.index)

data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions ' + str(data_unseen.shape))
logs = get_logs(save=True)

allDone()
#%% Choose Models
# return best model
best = compare_models(sort = 'AUC') #default is 'Accuracy'
allDone()
print(best)
allDone()
# return top 3 models based on 'Accuracy'
top3 = compare_models(n_select = 3, sort = 'Prec.', round = 2)
allDone()
print(top3)
allDone()
# compare specific models
#best_specific = compare_models(include = ['dt','rf','xgboost'])
# blacklist certain models
#best_specific = compare_models(exclude = ['catboost', 'svm'])
logs = get_logs(save=True)

allDone()
#%% Your Model
model = create_model(best,fold = 10)
plot_model(best)
best_results = pull()
top3_results = pull()
logs = get_logs(save=True)

allDone()
#%% Tune Model
tuned_model = tune_model(best, optimize = 'Prec.', n_iter = 50)
allDone()
plot_model(tuned_model)
allDone()
plot_model(tuned_model, plot = 'parameter')
allDone()
logs = get_logs(save=True)

allDone()
#%% Choose Your Ensemble Model
# With Bagging
bagged_tuned_model = ensemble_model(tuned_model, method = 'Bagging', n_estimators = 100)
allDone()
plot_model(bagged_tuned_model)
allDone()
print(bagged_tuned_model.estimators_)
logs = get_logs(save=True)
allDone()
# With Boosting
boosted_tuned_model = ensemble_model(tuned_model, method = 'Boosting', n_estimators = 100)
allDone()
plot_model(boosted_tuned_model)
allDone()
print(boosted_tuned_model.estimators_)
allDone()
logs = get_logs(save=True)

# Blend Models
blender = blend_models()
plot_model(blender)
allDone()
print(blender.estimators_)
allDone()
blender_tuned_model = blend_models(tuned_model)
allDone()
plot_model(blender_tuned_model)
allDone()
print(blender_tuned_model.estimators_)
allDone()
blender_specific = blend_models(estimator_list = compare_models(n_select = 3), method = 'hard')
allDone()
plot_model(blender_specific)
print(blender_specific.estimators_)
allDone()
logs = get_logs(save=True)

# Stack Model
stacker = stack_models(estimator_list = top3[1:], meta_model = top3[0])
allDone()
plot_model(stacker)
allDone()
print(blender_top3.estimators_)
logs = get_logs(save=True)

#%% Choose Calibrate Model

plot_model(tuned_model, plot='calibration')
allDone()
calibrated_tuned_model = calibrate_model(tuned_model)
allDone()
plot_model(calibrated_tuned_model, plot='calibration')
allDone()
print(calibrated_tuned_model.estimators_)
logs = get_logs(save=True)

calibrated_tuned_model_isotonic = calibrate_model(calibrated_tuned_model, method = 'isotonic')
allDone()
plot_model(calibrated_rf_isotonic, plot='calibration')
allDone()
print(calibrated_tuned_model_isotonic.estimators_)
logs = get_logs(save=True)

#%% Optimize threshold
optimize_threshold(calibrated_tuned_model, true_negative = 1500, false_negative = -5000)
allDone()
print(calibrated_tuned_model_isotonic.estimators_)
logs = get_logs(save=True)

#%% Finalize Model
final_rf = finalize_model(rf, probability_threshold =)
allDone()
save_model(dt, 'dt_saved_07032020')
allDone()
logs = get_logs(save=True)
get_system_logs()

# See MLflow
!mlflow ui

#%% Interpret Model
interpret_model(tuned_model)
allDone()
interpret_model(tuned_model, plot = 'correlation')
allDone()
interpret_model(tuned_model, plot = 'reason', observation = 10)
allDone()
logs = get_logs(save=True)

get_system_logs()
allDone()


#%% Predict Model
pred_holdout = predict_model(xgboost, probability_threshold = )
allDone()
logs = get_logs(save=True)
