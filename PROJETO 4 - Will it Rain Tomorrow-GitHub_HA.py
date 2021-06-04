{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T01:35:48.031962Z",
     "start_time": "2021-05-10T01:35:48.023984Z"
    }
   },
   "source": [
    "# Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:27:31.237849Z",
     "start_time": "2021-05-10T02:27:26.126826Z"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\AppData\\Roaming\\Python\\Python38\\site-packages\\sklearn\\utils\\deprecation.py:143: FutureWarning: The sklearn.utils.testing module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.utils. Anything that cannot be imported from sklearn.utils is now part of the private API.\n",
      "  warnings.warn(message, FutureWarning)\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#import modin.pandas as pd\n",
    "import warnings\n",
    "import pandas as pds\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from matplotlib import pyplot as plt\n",
    "from sklearn.utils import resample\n",
    "from pandas_profiling import ProfileReport\n",
    "import pyforest\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from lightgbm import LGBMClassifier\n",
    "from lightgbm import plot_importance\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from pandas.plotting import scatter_matrix\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import ExtraTreesRegressor\n",
    "from sklearn import metrics\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.metrics import roc_auc_score, plot_roc_curve\n",
    "from sklearn.metrics import precision_score, recall_score, f1_score\n",
    "from lazypredict.Supervised import LazyRegressor\n",
    "pd.options.display.max_columns = 100\n",
    "#pd.set_option('display.max_columns', None)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Import, Clean, Merge your Key Table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:27:31.853527Z",
     "start_time": "2021-05-10T02:27:31.238818Z"
    }
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Import key_table\n",
    "rain_aus = pd.read_csv(\"C:/Users/user/2. GIT_PROJECTS/Desafios/Projeto 4 - Itau/case/data/rain_data_aus.csv\")\n",
    "rain_aus = rain_aus.rename(columns={\"amountOfRain\": \"amntraintmrw\"})\n",
    "rain_aus['raintoday'].replace({'No': 0, 'Yes': 1},inplace = True)\n",
    "rain_aus['raintomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)\n",
    "rain_aus.head()\n",
    "#Resample and resize variable\n",
    "no = rain_aus[rain_aus.raintomorrow == 0]\n",
    "yes = rain_aus[rain_aus.raintomorrow == 1]\n",
    "yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)\n",
    "rain_oversample = pd.concat([no, yes_oversampled])\n",
    "#Plot and see resample\n",
    "fig = plt.figure(figsize = (8,5))\n",
    "rain_oversample.raintomorrow.value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)\n",
    "plt.title('RainTomorrow Indicator No(0) and Yes(1) after Oversampling (Balanced Dataset)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:27:32.123781Z",
     "start_time": "2021-05-10T02:27:31.854502Z"
    }
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Import side_tables and concatenate in one\n",
    "wind1 = pd.read_csv(\"C:/Users/user/2. GIT_PROJECTS/Desafios/Projeto 4 - Itau/case/data/wind_table_01.csv\")\n",
    "wind2 = pd.read_csv(\"C:/Users/user/2. GIT_PROJECTS/Desafios/Projeto 4 - Itau/case/data/wind_table_02.csv\")\n",
    "wind3= pd.read_csv(\"C:/Users/user/2. GIT_PROJECTS/Desafios/Projeto 4 - Itau/case/data/wind_table_03.csv\")\n",
    "wind4 = pd.read_csv(\"C:/Users/user/2. GIT_PROJECTS/Desafios/Projeto 4 - Itau/case/data/wind_table_04.csv\")\n",
    "wind5 = pd.read_csv(\"C:/Users/user/2. GIT_PROJECTS/Desafios/Projeto 4 - Itau/case/data/wind_table_05.csv\")\n",
    "wind6 = pd.read_csv(\"C:/Users/user/2. GIT_PROJECTS/Desafios/Projeto 4 - Itau/case/data/wind_table_06.csv\")\n",
    "wind7 = pd.read_csv(\"C:/Users/user/2. GIT_PROJECTS/Desafios/Projeto 4 - Itau/case/data/wind_table_07.csv\")\n",
    "wind8 = pd.read_csv(\"C:/Users/user/2. GIT_PROJECTS/Desafios/Projeto 4 - Itau/case/data/wind_table_08.csv\")\n",
    "wind = pd.concat([wind1, wind2, wind3, wind4, wind5, wind6, wind7, wind8])\n",
    "#Correct merged side_tables\n",
    "cont = 2\n",
    "for col in wind.columns[8:14]:\n",
    "    wind.loc[~wind[col].isnull(), wind.columns[cont]] = wind.loc[~wind[col].isnull(), col]\n",
    "    cont +=1\n",
    "wind = wind.drop(['windgustdir', 'windgustspeed', 'winddir9am', 'winddir3pm', 'windspeed9am', 'windspeed3pm'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:27:32.398075Z",
     "start_time": "2021-05-10T02:27:32.124780Z"
    },
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "        if (window._pyforest_update_imports_cell) { window._pyforest_update_imports_cell('import pandas as pd'); }\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Merge all tables and apply conditions to correct it\n",
    "rain_merge = pd.merge(left=rain_aus, right=wind, how='left', on=['date', 'location'])\n",
    "rain_merge['date'] = pd.to_datetime(rain_merge['date'].str.strip(), format='%Y/%m/%d')\n",
    "rain_merge.loc[(rain_merge.amntraintmrw < 0.4),'amntraintmrw']=0\n",
    "#Duplicates\n",
    "    #rain_merge.groupby(rain_merge.columns.tolist(),as_index=False).size())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:27:32.514760Z",
     "start_time": "2021-05-10T02:27:32.399045Z"
    }
   },
   "outputs": [],
   "source": [
    "#Correct type from columns\n",
    "rain_merge['wind_gustdir'] = rain_merge['wind_gustdir'].astype(str)\n",
    "rain_merge['wind_dir9am'] = rain_merge['wind_dir9am'].astype(str)\n",
    "rain_merge['wind_dir3pm'] = rain_merge['wind_dir3pm'].astype(str)\n",
    "#turn it into a scale\n",
    "encoder = LabelEncoder()\n",
    "encoder.fit(rain_merge['wind_gustdir'])\n",
    "#transform\n",
    "rain_merge['wind_gustdir'] = encoder.transform(rain_merge['wind_gustdir'])\n",
    "rain_merge['wind_dir9am'] = encoder.transform(rain_merge['wind_dir9am'])\n",
    "rain_merge['wind_dir3pm'] = encoder.transform(rain_merge['wind_dir3pm'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:27:32.520731Z",
     "start_time": "2021-05-10T02:27:32.515734Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2007-11-01 00:00:00\n",
      "2017-06-25 00:00:00\n"
     ]
    }
   ],
   "source": [
    "#see min and max from table\n",
    "print(rain_merge['date'].min())\n",
    "print(rain_merge['date'].max())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:27:34.851580Z",
     "start_time": "2021-05-10T02:27:32.521718Z"
    }
   },
   "outputs": [],
   "source": [
    "#Create a table by your current season (apply one month ago + actual month + next month)\n",
    "seasoned_rain = rain_merge[(rain_merge['date'].dt.month == 4) | (rain_merge['date'].dt.month == 5) | (rain_merge['date'].dt.month == 6)]\n",
    "seasoned_rain = seasoned_rain[~(seasoned_rain['date'].dt.year <= 2007)]\n",
    "#seasoned_rain = seasoned_rain[~(seasoned_rain['date'].dt.year >= 2017)]\n",
    "#Export your table cleaned\n",
    "rain_merge.to_csv('C:/Users/user/2. GIT_PROJECTS/Desafios/Projeto 4 - Itau/case/data/rain_cleaned.csv', index = False, header=True)\n",
    "seasoned_rain.to_csv('C:/Users/user/2. GIT_PROJECTS/Desafios/Projeto 4 - Itau/case/data/seasoned_rain.csv', index = False, header=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:27:34.880436Z",
     "start_time": "2021-05-10T02:27:34.853482Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>date</th>\n",
       "      <th>location</th>\n",
       "      <th>mintemp</th>\n",
       "      <th>maxtemp</th>\n",
       "      <th>rainfall</th>\n",
       "      <th>evaporation</th>\n",
       "      <th>sunshine</th>\n",
       "      <th>humidity9am</th>\n",
       "      <th>humidity3pm</th>\n",
       "      <th>pressure9am</th>\n",
       "      <th>pressure3pm</th>\n",
       "      <th>cloud9am</th>\n",
       "      <th>cloud3pm</th>\n",
       "      <th>temp9am</th>\n",
       "      <th>temp3pm</th>\n",
       "      <th>raintoday</th>\n",
       "      <th>amntraintmrw</th>\n",
       "      <th>raintomorrow</th>\n",
       "      <th>temp</th>\n",
       "      <th>humidity</th>\n",
       "      <th>precipitation3pm</th>\n",
       "      <th>precipitation9am</th>\n",
       "      <th>modelo_vigente</th>\n",
       "      <th>wind_gustdir</th>\n",
       "      <th>wind_gustspeed</th>\n",
       "      <th>wind_dir9am</th>\n",
       "      <th>wind_dir3pm</th>\n",
       "      <th>wind_speed9am</th>\n",
       "      <th>wind_speed3pm</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>120</th>\n",
       "      <td>2009-04-01</td>\n",
       "      <td>Albury</td>\n",
       "      <td>12.20</td>\n",
       "      <td>30.60</td>\n",
       "      <td>0.00</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>59.00</td>\n",
       "      <td>22.00</td>\n",
       "      <td>1022.60</td>\n",
       "      <td>1019.40</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>16.60</td>\n",
       "      <td>29.50</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>38.72</td>\n",
       "      <td>28.40</td>\n",
       "      <td>9</td>\n",
       "      <td>14.68</td>\n",
       "      <td>0.00</td>\n",
       "      <td>9</td>\n",
       "      <td>22.00</td>\n",
       "      <td>16</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>6.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>121</th>\n",
       "      <td>2009-04-02</td>\n",
       "      <td>Albury</td>\n",
       "      <td>14.30</td>\n",
       "      <td>32.10</td>\n",
       "      <td>0.00</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>59.00</td>\n",
       "      <td>30.00</td>\n",
       "      <td>1022.60</td>\n",
       "      <td>1018.40</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>18.40</td>\n",
       "      <td>30.90</td>\n",
       "      <td>0.00</td>\n",
       "      <td>8.60</td>\n",
       "      <td>1</td>\n",
       "      <td>40.52</td>\n",
       "      <td>38.00</td>\n",
       "      <td>17</td>\n",
       "      <td>11.36</td>\n",
       "      <td>0.37</td>\n",
       "      <td>4</td>\n",
       "      <td>28.00</td>\n",
       "      <td>16</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>11.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>122</th>\n",
       "      <td>2009-04-03</td>\n",
       "      <td>Albury</td>\n",
       "      <td>18.40</td>\n",
       "      <td>28.10</td>\n",
       "      <td>8.60</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>92.00</td>\n",
       "      <td>49.00</td>\n",
       "      <td>1018.80</td>\n",
       "      <td>1012.90</td>\n",
       "      <td>8.00</td>\n",
       "      <td>6.00</td>\n",
       "      <td>19.00</td>\n",
       "      <td>27.40</td>\n",
       "      <td>1.00</td>\n",
       "      <td>12.60</td>\n",
       "      <td>1</td>\n",
       "      <td>35.72</td>\n",
       "      <td>60.80</td>\n",
       "      <td>14</td>\n",
       "      <td>6.21</td>\n",
       "      <td>0.61</td>\n",
       "      <td>13</td>\n",
       "      <td>98.00</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>7.00</td>\n",
       "      <td>17.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>123</th>\n",
       "      <td>2009-04-04</td>\n",
       "      <td>Albury</td>\n",
       "      <td>10.70</td>\n",
       "      <td>21.40</td>\n",
       "      <td>12.60</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>60.00</td>\n",
       "      <td>33.00</td>\n",
       "      <td>1019.80</td>\n",
       "      <td>1019.30</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>13.90</td>\n",
       "      <td>20.30</td>\n",
       "      <td>1.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>27.68</td>\n",
       "      <td>41.60</td>\n",
       "      <td>5</td>\n",
       "      <td>12.26</td>\n",
       "      <td>0.02</td>\n",
       "      <td>13</td>\n",
       "      <td>43.00</td>\n",
       "      <td>16</td>\n",
       "      <td>15</td>\n",
       "      <td>0.00</td>\n",
       "      <td>13.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>124</th>\n",
       "      <td>2009-04-05</td>\n",
       "      <td>Albury</td>\n",
       "      <td>7.80</td>\n",
       "      <td>21.70</td>\n",
       "      <td>0.00</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>72.00</td>\n",
       "      <td>37.00</td>\n",
       "      <td>1020.40</td>\n",
       "      <td>1016.50</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>12.90</td>\n",
       "      <td>21.20</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>28.04</td>\n",
       "      <td>46.40</td>\n",
       "      <td>11</td>\n",
       "      <td>17.19</td>\n",
       "      <td>0.02</td>\n",
       "      <td>15</td>\n",
       "      <td>31.00</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>6.00</td>\n",
       "      <td>19.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>164381</th>\n",
       "      <td>2017-06-22</td>\n",
       "      <td>Uluru</td>\n",
       "      <td>3.60</td>\n",
       "      <td>25.30</td>\n",
       "      <td>0.00</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>56.00</td>\n",
       "      <td>21.00</td>\n",
       "      <td>1023.50</td>\n",
       "      <td>1019.10</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>10.90</td>\n",
       "      <td>24.50</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>32.36</td>\n",
       "      <td>27.20</td>\n",
       "      <td>9</td>\n",
       "      <td>19.72</td>\n",
       "      <td>0.02</td>\n",
       "      <td>6</td>\n",
       "      <td>22.00</td>\n",
       "      <td>9</td>\n",
       "      <td>3</td>\n",
       "      <td>13.00</td>\n",
       "      <td>9.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>164382</th>\n",
       "      <td>2017-06-23</td>\n",
       "      <td>Uluru</td>\n",
       "      <td>5.40</td>\n",
       "      <td>26.90</td>\n",
       "      <td>0.00</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>53.00</td>\n",
       "      <td>24.00</td>\n",
       "      <td>1021.00</td>\n",
       "      <td>1016.80</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>12.50</td>\n",
       "      <td>26.10</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>34.28</td>\n",
       "      <td>30.80</td>\n",
       "      <td>12</td>\n",
       "      <td>0.99</td>\n",
       "      <td>0.01</td>\n",
       "      <td>3</td>\n",
       "      <td>37.00</td>\n",
       "      <td>9</td>\n",
       "      <td>14</td>\n",
       "      <td>9.00</td>\n",
       "      <td>9.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>164383</th>\n",
       "      <td>2017-06-23</td>\n",
       "      <td>Uluru</td>\n",
       "      <td>5.40</td>\n",
       "      <td>26.90</td>\n",
       "      <td>0.00</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>53.00</td>\n",
       "      <td>24.00</td>\n",
       "      <td>1021.00</td>\n",
       "      <td>1016.80</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>12.50</td>\n",
       "      <td>26.10</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>34.28</td>\n",
       "      <td>30.80</td>\n",
       "      <td>12</td>\n",
       "      <td>0.99</td>\n",
       "      <td>0.01</td>\n",
       "      <td>3</td>\n",
       "      <td>37.00</td>\n",
       "      <td>9</td>\n",
       "      <td>14</td>\n",
       "      <td>9.00</td>\n",
       "      <td>9.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>164384</th>\n",
       "      <td>2017-06-24</td>\n",
       "      <td>Uluru</td>\n",
       "      <td>7.80</td>\n",
       "      <td>27.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>51.00</td>\n",
       "      <td>24.00</td>\n",
       "      <td>1019.40</td>\n",
       "      <td>1016.50</td>\n",
       "      <td>3.00</td>\n",
       "      <td>2.00</td>\n",
       "      <td>15.10</td>\n",
       "      <td>26.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>34.40</td>\n",
       "      <td>30.80</td>\n",
       "      <td>15</td>\n",
       "      <td>4.38</td>\n",
       "      <td>0.02</td>\n",
       "      <td>9</td>\n",
       "      <td>28.00</td>\n",
       "      <td>10</td>\n",
       "      <td>3</td>\n",
       "      <td>13.00</td>\n",
       "      <td>7.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>164385</th>\n",
       "      <td>2017-06-24</td>\n",
       "      <td>Uluru</td>\n",
       "      <td>7.80</td>\n",
       "      <td>27.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>51.00</td>\n",
       "      <td>24.00</td>\n",
       "      <td>1019.40</td>\n",
       "      <td>1016.50</td>\n",
       "      <td>3.00</td>\n",
       "      <td>2.00</td>\n",
       "      <td>15.10</td>\n",
       "      <td>26.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0</td>\n",
       "      <td>34.40</td>\n",
       "      <td>30.80</td>\n",
       "      <td>15</td>\n",
       "      <td>4.38</td>\n",
       "      <td>0.02</td>\n",
       "      <td>9</td>\n",
       "      <td>28.00</td>\n",
       "      <td>10</td>\n",
       "      <td>3</td>\n",
       "      <td>13.00</td>\n",
       "      <td>7.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>45302 rows × 29 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "             date location  mintemp  maxtemp  rainfall  evaporation  sunshine  \\\n",
       "120    2009-04-01   Albury    12.20    30.60      0.00          nan       nan   \n",
       "121    2009-04-02   Albury    14.30    32.10      0.00          nan       nan   \n",
       "122    2009-04-03   Albury    18.40    28.10      8.60          nan       nan   \n",
       "123    2009-04-04   Albury    10.70    21.40     12.60          nan       nan   \n",
       "124    2009-04-05   Albury     7.80    21.70      0.00          nan       nan   \n",
       "...           ...      ...      ...      ...       ...          ...       ...   \n",
       "164381 2017-06-22    Uluru     3.60    25.30      0.00          nan       nan   \n",
       "164382 2017-06-23    Uluru     5.40    26.90      0.00          nan       nan   \n",
       "164383 2017-06-23    Uluru     5.40    26.90      0.00          nan       nan   \n",
       "164384 2017-06-24    Uluru     7.80    27.00      0.00          nan       nan   \n",
       "164385 2017-06-24    Uluru     7.80    27.00      0.00          nan       nan   \n",
       "\n",
       "        humidity9am  humidity3pm  pressure9am  pressure3pm  cloud9am  \\\n",
       "120           59.00        22.00      1022.60      1019.40       nan   \n",
       "121           59.00        30.00      1022.60      1018.40       nan   \n",
       "122           92.00        49.00      1018.80      1012.90      8.00   \n",
       "123           60.00        33.00      1019.80      1019.30       nan   \n",
       "124           72.00        37.00      1020.40      1016.50       nan   \n",
       "...             ...          ...          ...          ...       ...   \n",
       "164381        56.00        21.00      1023.50      1019.10       nan   \n",
       "164382        53.00        24.00      1021.00      1016.80       nan   \n",
       "164383        53.00        24.00      1021.00      1016.80       nan   \n",
       "164384        51.00        24.00      1019.40      1016.50      3.00   \n",
       "164385        51.00        24.00      1019.40      1016.50      3.00   \n",
       "\n",
       "        cloud3pm  temp9am  temp3pm  raintoday  amntraintmrw  raintomorrow  \\\n",
       "120          nan    16.60    29.50       0.00          0.00             0   \n",
       "121          nan    18.40    30.90       0.00          8.60             1   \n",
       "122         6.00    19.00    27.40       1.00         12.60             1   \n",
       "123          nan    13.90    20.30       1.00          0.00             0   \n",
       "124          nan    12.90    21.20       0.00          0.00             0   \n",
       "...          ...      ...      ...        ...           ...           ...   \n",
       "164381       nan    10.90    24.50       0.00          0.00             0   \n",
       "164382       nan    12.50    26.10       0.00          0.00             0   \n",
       "164383       nan    12.50    26.10       0.00          0.00             0   \n",
       "164384      2.00    15.10    26.00       0.00          0.00             0   \n",
       "164385      2.00    15.10    26.00       0.00          0.00             0   \n",
       "\n",
       "        temp  humidity  precipitation3pm  precipitation9am  modelo_vigente  \\\n",
       "120    38.72     28.40                 9             14.68            0.00   \n",
       "121    40.52     38.00                17             11.36            0.37   \n",
       "122    35.72     60.80                14              6.21            0.61   \n",
       "123    27.68     41.60                 5             12.26            0.02   \n",
       "124    28.04     46.40                11             17.19            0.02   \n",
       "...      ...       ...               ...               ...             ...   \n",
       "164381 32.36     27.20                 9             19.72            0.02   \n",
       "164382 34.28     30.80                12              0.99            0.01   \n",
       "164383 34.28     30.80                12              0.99            0.01   \n",
       "164384 34.40     30.80                15              4.38            0.02   \n",
       "164385 34.40     30.80                15              4.38            0.02   \n",
       "\n",
       "        wind_gustdir  wind_gustspeed  wind_dir9am  wind_dir3pm  wind_speed9am  \\\n",
       "120                9           22.00           16            1           0.00   \n",
       "121                4           28.00           16            1           0.00   \n",
       "122               13           98.00            0            5           7.00   \n",
       "123               13           43.00           16           15           0.00   \n",
       "124               15           31.00            0            7           6.00   \n",
       "...              ...             ...          ...          ...            ...   \n",
       "164381             6           22.00            9            3          13.00   \n",
       "164382             3           37.00            9           14           9.00   \n",
       "164383             3           37.00            9           14           9.00   \n",
       "164384             9           28.00           10            3          13.00   \n",
       "164385             9           28.00           10            3          13.00   \n",
       "\n",
       "        wind_speed3pm  \n",
       "120              6.00  \n",
       "121             11.00  \n",
       "122             17.00  \n",
       "123             13.00  \n",
       "124             19.00  \n",
       "...               ...  \n",
       "164381           9.00  \n",
       "164382           9.00  \n",
       "164383           9.00  \n",
       "164384           7.00  \n",
       "164385           7.00  \n",
       "\n",
       "[45302 rows x 29 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check your final table!\n",
    "seasoned_rain"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# If you want to explore your data with graphs (TAKES TIME!!)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:27:34.884400Z",
     "start_time": "2021-05-10T02:27:34.881407Z"
    }
   },
   "outputs": [],
   "source": [
    "#Check pairplot if you want to analyze it(takes time to process it!!)\n",
    "    #pairplot = sns.pairplot(rain_merge)\n",
    "    #fig = pairplot.get_figure()\n",
    "    #fig.savefig(\"rain_merge.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:27:34.888412Z",
     "start_time": "2021-05-10T02:27:34.885397Z"
    }
   },
   "outputs": [],
   "source": [
    "#Check pairplot if you want to analyze it(takes time to process it!!)\n",
    "    #pairplot = sns.pairplot(seasoned_rain)\n",
    "    #fig = pairplot.get_figure()\n",
    "    #fig.savefig(\"seasoned_rain.png\") "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:30:56.004985Z",
     "start_time": "2021-05-10T02:27:34.889387Z"
    }
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9447d4a4d2ef4bd291db9f0f761af33f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Summarize dataset:   0%|          | 0/42 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ebdc81358b6b4936935f5e21c486ac28",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Generate report structure:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "938312f3d83942029cbdbade8c015680",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Render HTML:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ceb590d3406f476ea67a7f724f05e3bf",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Export report to file:   0%|          | 0/1 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Export an automated exploratory analysys (takes time to process it!!)\n",
    "#prof = ProfileReport(rain_merge, title=\"Rain in Austraia Seasoned Report\", explorative=True)\n",
    "#prof.to_file(output_file='Australia Rain Seasoned Report.html')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Predict the future!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:46:20.194329Z",
     "start_time": "2021-05-10T02:46:19.956964Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9336688298760467\n",
      "ROC_AUC_SCORE TEST IS:   0.9038391543531873\n"
     ]
    }
   ],
   "source": [
    "#Create you best model variables\n",
    "X = seasoned_rain[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = seasoned_rain['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=9, n_estimators=50, learning_rate=0.2, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:30:56.314158Z",
     "start_time": "2021-05-10T02:30:56.311166Z"
    }
   },
   "outputs": [],
   "source": [
    "#If you want to compare with other model (REALLY HEAVY JOB, TAKES TIME!)\n",
    "    #reg = LazyRegressor(ignore_warnings=False, custom_metric=None)\n",
    "    #models, predictions = reg.fit(X_train, X_test, y_train, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:30:56.319146Z",
     "start_time": "2021-05-10T02:30:56.315156Z"
    }
   },
   "outputs": [],
   "source": [
    "#Try to undersample or oversample more your table by historical series\n",
    "    #no = rain_merge[rain_merge.date > 3000]\n",
    "    #yes = rain_merge[rain_merge.date <= 3000]\n",
    "    #yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)\n",
    "    #rain_oversample = pd.concat([no, yes_oversampled])\n",
    "\n",
    "    #fig = plt.figure(figsize = (8,5))\n",
    "    #rain_oversample.date.value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)\n",
    "    #plt.title('RainTomorrow Indicator No(0) and Yes(1) after Oversampling (Balanced Dataset)')\n",
    "    #plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# By City"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:30:56.344078Z",
     "start_time": "2021-05-10T02:30:56.321140Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Adelaide', 'Albany', 'Albury', 'AliceSprings', 'BadgerysCreek', 'Ballarat', 'Bendigo', 'Brisbane', 'Cairns', 'Canberra', 'Cobar', 'CoffsHarbour', 'Dartmoor', 'Darwin', 'GoldCoast', 'Hobart', 'Katherine', 'Launceston', 'Melbourne', 'MelbourneAirport', 'Mildura', 'Moree', 'MountGambier', 'MountGinini', 'Newcastle', 'Nhil', 'NorahHead', 'NorfolkIsland', 'Nuriootpa', 'PearceRAAF', 'Penrith', 'Perth', 'PerthAirport', 'Portland', 'Richmond', 'Sale', 'SalmonGums', 'Sydney', 'SydneyAirport', 'Townsville', 'Tuggeranong', 'Uluru', 'WaggaWagga', 'Walpole', 'Watsonia', 'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera']\n"
     ]
    }
   ],
   "source": [
    "dfs = dict(tuple(seasoned_rain.groupby('location')))\n",
    "cities_list = list(dfs)\n",
    "print(cities_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:47:02.680153Z",
     "start_time": "2021-05-10T02:47:02.645247Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9265736754083194\n",
      "ROC_AUC_SCORE TEST IS:   0.9089503953989935\n"
     ]
    }
   ],
   "source": [
    "Adelaide = dfs['Adelaide']\n",
    "#Create you best model variables\n",
    "X = Adelaide[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Adelaide['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:48:05.945070Z",
     "start_time": "2021-05-10T02:48:05.917146Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9121512091416422\n",
      "ROC_AUC_SCORE TEST IS:   0.826026820614589\n"
     ]
    }
   ],
   "source": [
    "Albany = dfs['Albany']\n",
    "#Create you best model variables\n",
    "X = Albany[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Albany['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:48:51.294222Z",
     "start_time": "2021-05-10T02:48:51.268291Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9705774927965587\n",
      "ROC_AUC_SCORE TEST IS:   0.9086619718309858\n"
     ]
    }
   ],
   "source": [
    "Albury =  dfs['Albury']\n",
    "X = Albury[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Albury['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:49:16.925842Z",
     "start_time": "2021-05-10T02:49:16.898913Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9845467032967034\n",
      "ROC_AUC_SCORE TEST IS:   0.927830596369922\n"
     ]
    }
   ],
   "source": [
    "AliceSprings = dfs['AliceSprings']\n",
    "X = AliceSprings[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = AliceSprings['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:49:30.374099Z",
     "start_time": "2021-05-10T02:49:30.349166Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9404373762256678\n",
      "ROC_AUC_SCORE TEST IS:   0.9197945845004669\n"
     ]
    }
   ],
   "source": [
    "BadgerysCreek = dfs['BadgerysCreek']\n",
    "X = BadgerysCreek[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = BadgerysCreek['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:49:57.582673Z",
     "start_time": "2021-05-10T02:49:57.557739Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.915791119811878\n",
      "ROC_AUC_SCORE TEST IS:   0.878921568627451\n"
     ]
    }
   ],
   "source": [
    "Ballarat = dfs['Ballarat']\n",
    "X = Ballarat[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Ballarat['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:50:13.926863Z",
     "start_time": "2021-05-10T02:50:13.899935Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.929648465880512\n",
      "ROC_AUC_SCORE TEST IS:   0.8893211920529801\n"
     ]
    }
   ],
   "source": [
    "Bendigo = dfs['Bendigo']\n",
    "X = Bendigo[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Bendigo['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:50:41.745153Z",
     "start_time": "2021-05-10T02:50:41.718225Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9571124062381955\n",
      "ROC_AUC_SCORE TEST IS:   0.9132765957446808\n"
     ]
    }
   ],
   "source": [
    "Brisbane = dfs['Brisbane']\n",
    "X = Brisbane[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Brisbane['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:50:53.407313Z",
     "start_time": "2021-05-10T02:50:53.380410Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9191104064090796\n",
      "ROC_AUC_SCORE TEST IS:   0.8857622305408545\n"
     ]
    }
   ],
   "source": [
    "Cairns = dfs['Cairns']\n",
    "X = Cairns[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Cairns['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:51:36.507930Z",
     "start_time": "2021-05-10T02:51:36.481003Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9495533153020245\n",
      "ROC_AUC_SCORE TEST IS:   0.8746592075912615\n"
     ]
    }
   ],
   "source": [
    "Canberra = dfs['Canberra']\n",
    "X = Canberra[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Canberra['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:52:44.051365Z",
     "start_time": "2021-05-10T02:52:44.025435Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9533266813858694\n",
      "ROC_AUC_SCORE TEST IS:   0.9166666666666667\n"
     ]
    }
   ],
   "source": [
    "Cobar = dfs['Cobar']\n",
    "X = Cobar[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Cobar['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,  random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:53:14.671164Z",
     "start_time": "2021-05-10T02:53:14.640247Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9378254114880753\n",
      "ROC_AUC_SCORE TEST IS:   0.9046963859842231\n"
     ]
    }
   ],
   "source": [
    "CoffsHarbour = dfs['CoffsHarbour']\n",
    "X = CoffsHarbour[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = CoffsHarbour['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:53:38.511576Z",
     "start_time": "2021-05-10T02:53:38.486642Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9290161247786611\n",
      "ROC_AUC_SCORE TEST IS:   0.9108278672232161\n"
     ]
    }
   ],
   "source": [
    "Dartmoor = dfs['Dartmoor']\n",
    "X = Dartmoor[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Dartmoor['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:53:43.108241Z",
     "start_time": "2021-05-10T02:53:43.082311Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9693387547836326\n",
      "ROC_AUC_SCORE TEST IS:   0.8960638024183175\n"
     ]
    }
   ],
   "source": [
    "Darwin = dfs['Darwin']\n",
    "X = Darwin[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Darwin['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:54:06.091681Z",
     "start_time": "2021-05-10T02:54:06.059768Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9326968085744806\n",
      "ROC_AUC_SCORE TEST IS:   0.9064177019775049\n"
     ]
    }
   ],
   "source": [
    "GoldCoast = dfs['GoldCoast']\n",
    "X = GoldCoast[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = GoldCoast['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:55:52.014642Z",
     "start_time": "2021-05-10T02:55:51.989737Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9003269347531643\n",
      "ROC_AUC_SCORE TEST IS:   0.8124281609195403\n"
     ]
    }
   ],
   "source": [
    "Hobart = dfs['Hobart']\n",
    "X = Hobart[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Hobart['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=2, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 224,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:57:41.364250Z",
     "start_time": "2021-05-10T02:57:41.340314Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.992552471225457\n",
      "ROC_AUC_SCORE TEST IS:   0.9356884057971014\n"
     ]
    }
   ],
   "source": [
    "Katherine = dfs['Katherine']\n",
    "X = Katherine[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Katherine['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=4, n_estimators=5, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 225,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:57:46.005312Z",
     "start_time": "2021-05-10T02:57:45.979351Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9163699609350102\n",
      "ROC_AUC_SCORE TEST IS:   0.9010494752623689\n"
     ]
    }
   ],
   "source": [
    "Launceston = dfs['Launceston']\n",
    "X = Launceston[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Launceston['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 233,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:58:58.095308Z",
     "start_time": "2021-05-10T02:58:58.069378Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9368702504925415\n",
      "ROC_AUC_SCORE TEST IS:   0.8904986522911051\n"
     ]
    }
   ],
   "source": [
    "Melbourne = dfs['Melbourne']\n",
    "X = Melbourne[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Melbourne['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.2, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 239,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:59:35.686370Z",
     "start_time": "2021-05-10T02:59:35.660440Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9075727513227513\n",
      "ROC_AUC_SCORE TEST IS:   0.8403903903903904\n"
     ]
    }
   ],
   "source": [
    "MelbourneAirport = dfs['MelbourneAirport']\n",
    "X = MelbourneAirport[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = MelbourneAirport['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 250,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:00:36.866143Z",
     "start_time": "2021-05-10T03:00:36.835227Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9306969334926325\n",
      "ROC_AUC_SCORE TEST IS:   0.8527379053694844\n"
     ]
    }
   ],
   "source": [
    "Mildura = dfs['Mildura']\n",
    "X = Mildura[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Mildura['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=5, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 252,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:01:07.804231Z",
     "start_time": "2021-05-10T03:01:07.776307Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.975095785440613\n",
      "ROC_AUC_SCORE TEST IS:   0.9682986967241987\n"
     ]
    }
   ],
   "source": [
    "More = dfs['Moree']\n",
    "X = More[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = More['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 254,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:01:22.246553Z",
     "start_time": "2021-05-10T03:01:22.220623Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9242276710631141\n",
      "ROC_AUC_SCORE TEST IS:   0.9296802054154996\n"
     ]
    }
   ],
   "source": [
    "MountGambier = dfs['MountGambier']\n",
    "X = MountGambier[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = MountGambier['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 256,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:01:33.505081Z",
     "start_time": "2021-05-10T03:01:33.479150Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9186484214459182\n",
      "ROC_AUC_SCORE TEST IS:   0.8888979725310661\n"
     ]
    }
   ],
   "source": [
    "MountGinini = dfs['MountGinini']\n",
    "X = MountGinini[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = MountGinini['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 266,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:02:27.278282Z",
     "start_time": "2021-05-10T03:02:27.252369Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.8866447390961729\n",
      "ROC_AUC_SCORE TEST IS:   0.8204199590283874\n"
     ]
    }
   ],
   "source": [
    "Newcastle = dfs['Newcastle']\n",
    "X = Newcastle[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Newcastle['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.2, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 270,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:02:47.121645Z",
     "start_time": "2021-05-10T03:02:47.096710Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.968034217877095\n",
      "ROC_AUC_SCORE TEST IS:   0.9197190611664295\n"
     ]
    }
   ],
   "source": [
    "Nhil = dfs['Nhil']\n",
    "X = Nhil[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Nhil['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 272,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:03:02.111363Z",
     "start_time": "2021-05-10T03:03:02.084437Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9096599037586162\n",
      "ROC_AUC_SCORE TEST IS:   0.867873651771957\n"
     ]
    }
   ],
   "source": [
    "NorahHead = dfs['NorahHead']\n",
    "X = NorahHead[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = NorahHead['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 274,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:03:18.567043Z",
     "start_time": "2021-05-10T03:03:18.541145Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.8655091927607009\n",
      "ROC_AUC_SCORE TEST IS:   0.80419921875\n"
     ]
    }
   ],
   "source": [
    "NorfolkIsland = dfs['NorfolkIsland']\n",
    "X = NorfolkIsland[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = NorfolkIsland['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 275,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:03:26.822551Z",
     "start_time": "2021-05-10T03:03:26.796621Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9384497936398121\n",
      "ROC_AUC_SCORE TEST IS:   0.9257494540401032\n"
     ]
    }
   ],
   "source": [
    "Nuriootpa = dfs['Nuriootpa']\n",
    "X = Nuriootpa[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Nuriootpa['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 278,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:03:43.583286Z",
     "start_time": "2021-05-10T03:03:43.556359Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9612354312354312\n",
      "ROC_AUC_SCORE TEST IS:   0.9111236802413273\n"
     ]
    }
   ],
   "source": [
    "PearceRAAF = dfs['PearceRAAF']\n",
    "X = PearceRAAF[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = PearceRAAF['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 280,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:03:52.789691Z",
     "start_time": "2021-05-10T03:03:52.763737Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.927654024920886\n",
      "ROC_AUC_SCORE TEST IS:   0.8761867088607596\n"
     ]
    }
   ],
   "source": [
    "Penrith = dfs['Penrith']\n",
    "X = Penrith[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Penrith['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 282,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:04:10.539892Z",
     "start_time": "2021-05-10T03:04:10.512963Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9617730382975519\n",
      "ROC_AUC_SCORE TEST IS:   0.9336548488008342\n"
     ]
    }
   ],
   "source": [
    "Perth = dfs['Perth']\n",
    "X = Perth[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Perth['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 284,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:04:20.764550Z",
     "start_time": "2021-05-10T03:04:20.733633Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9648160749309531\n",
      "ROC_AUC_SCORE TEST IS:   0.9540537017726798\n"
     ]
    }
   ],
   "source": [
    "PerthAirport = dfs['PerthAirport']\n",
    "X = PerthAirport[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = PerthAirport['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 286,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:04:46.021514Z",
     "start_time": "2021-05-10T03:04:45.993588Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9054750115035678\n",
      "ROC_AUC_SCORE TEST IS:   0.8706097823744883\n"
     ]
    }
   ],
   "source": [
    "Portland = dfs['Portland']\n",
    "X = Portland[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Portland['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 290,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:05:11.478056Z",
     "start_time": "2021-05-10T03:05:11.396274Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9463112793074949\n",
      "ROC_AUC_SCORE TEST IS:   0.8975361288794125\n"
     ]
    }
   ],
   "source": [
    "Richmond = dfs['Richmond']\n",
    "X = Richmond[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Richmond['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 292,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:05:25.107429Z",
     "start_time": "2021-05-10T03:05:25.077510Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9050472861842105\n",
      "ROC_AUC_SCORE TEST IS:   0.8894736842105264\n"
     ]
    }
   ],
   "source": [
    "Sale = dfs['Sale']\n",
    "X = Sale[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Sale['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 299,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:06:01.193424Z",
     "start_time": "2021-05-10T03:06:01.165523Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9480778246601032\n",
      "ROC_AUC_SCORE TEST IS:   0.8614433811802232\n"
     ]
    }
   ],
   "source": [
    "SalmonGums = dfs['SalmonGums']\n",
    "X = SalmonGums[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = SalmonGums['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 301,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:06:14.230818Z",
     "start_time": "2021-05-10T03:06:14.205884Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9400998336106489\n",
      "ROC_AUC_SCORE TEST IS:   0.8507313045529605\n"
     ]
    }
   ],
   "source": [
    "Sydney = dfs['Sydney']\n",
    "X = Sydney[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Sydney['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.2, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 304,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:06:29.031951Z",
     "start_time": "2021-05-10T03:06:29.004026Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9294218580383093\n",
      "ROC_AUC_SCORE TEST IS:   0.9120006828269033\n"
     ]
    }
   ],
   "source": [
    "SydneyAirport = dfs['SydneyAirport']\n",
    "X = SydneyAirport[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = SydneyAirport['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:45:20.614257Z",
     "start_time": "2021-05-10T02:45:20.589324Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9331312677742853\n",
      "ROC_AUC_SCORE TEST IS:   0.9118297175603767\n"
     ]
    }
   ],
   "source": [
    "Townsville = dfs['Townsville']\n",
    "X = Townsville[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Townsville['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=5, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 306,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:06:46.162790Z",
     "start_time": "2021-05-10T03:06:46.136860Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9274938107808675\n",
      "ROC_AUC_SCORE TEST IS:   0.8613871635610766\n"
     ]
    }
   ],
   "source": [
    "Tuggeranong = dfs['Tuggeranong']\n",
    "X = Tuggeranong[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Tuggeranong['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:30:57.543870Z",
     "start_time": "2021-05-10T02:30:57.519934Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9835677904438516\n",
      "ROC_AUC_SCORE TEST IS:   0.9761363636363636\n"
     ]
    }
   ],
   "source": [
    "Uluru = dfs['Uluru']\n",
    "X = Uluru[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Uluru['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 308,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:07:07.084154Z",
     "start_time": "2021-05-10T03:07:07.060218Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9443685521944395\n",
      "ROC_AUC_SCORE TEST IS:   0.926961926961927\n"
     ]
    }
   ],
   "source": [
    "WaggaWagga = dfs['WaggaWagga']\n",
    "X = WaggaWagga[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = WaggaWagga['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 311,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:07:23.677544Z",
     "start_time": "2021-05-10T03:07:23.650616Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9242108308723761\n",
      "ROC_AUC_SCORE TEST IS:   0.8299299568965517\n"
     ]
    }
   ],
   "source": [
    "Walpole = dfs['Walpole']\n",
    "X = Walpole[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Walpole['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 317,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:07:59.334103Z",
     "start_time": "2021-05-10T03:07:59.308173Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9253230231426169\n",
      "ROC_AUC_SCORE TEST IS:   0.8724701445631678\n"
     ]
    }
   ],
   "source": [
    "Watsonia = dfs['Watsonia']\n",
    "X = Watsonia[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Watsonia['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 327,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:09:08.889356Z",
     "start_time": "2021-05-10T03:09:08.863426Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9282944277108434\n",
      "ROC_AUC_SCORE TEST IS:   0.8274147727272725\n"
     ]
    }
   ],
   "source": [
    "Williamtow = dfs['Williamtown']\n",
    "X = Williamtow[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Williamtow['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 329,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:09:30.703113Z",
     "start_time": "2021-05-10T03:09:30.676208Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9291497975708503\n",
      "ROC_AUC_SCORE TEST IS:   0.8920682730923696\n"
     ]
    }
   ],
   "source": [
    "Witchcliffe = dfs['Witchcliffe']\n",
    "X = Witchcliffe[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Witchcliffe['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:30:57.710425Z",
     "start_time": "2021-05-10T02:30:57.682499Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9309890681936492\n",
      "ROC_AUC_SCORE TEST IS:   0.9041446358519529\n"
     ]
    }
   ],
   "source": [
    "Wollongong = dfs['Wollongong']\n",
    "X = Wollongong[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Wollongong['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 331,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T03:09:46.350177Z",
     "start_time": "2021-05-10T03:09:46.323249Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC_AUC_SCORE TRAIN IS:   0.9569068996975081\n",
      "ROC_AUC_SCORE TEST IS:   0.9467604218985435\n"
     ]
    }
   ],
   "source": [
    "Woomera = dfs['Woomera']\n",
    "X = Woomera[['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine', 'humidity9am', 'humidity3pm', 'pressure9am',\n",
    "       'pressure3pm', 'cloud9am', 'cloud3pm', 'raintoday', 'precipitation3pm', 'precipitation9am',\n",
    "       'wind_gustdir', 'wind_gustspeed', 'wind_dir9am', 'wind_dir3pm', 'wind_speed9am', 'wind_speed3pm']]\n",
    "y = Woomera['raintomorrow']\n",
    "#Split test and train\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "#Normalize your values for the model\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_std = scaler.transform(X_train)\n",
    "X_test_std = scaler.transform(X_test)\n",
    "#Apply your best model!\n",
    "lgbm = LGBMClassifier(max_depth=3, n_estimators=10, learning_rate=0.1, colsample_bytree=.7, n_jobs=-1)\n",
    "lgbm.fit(X_train, y_train)\n",
    "#SEE YOUR SCORE!\n",
    "print('ROC_AUC_SCORE TRAIN IS:  ', roc_auc_score(y_train, lgbm.predict_proba(X_train)[:,1]))\n",
    "print('ROC_AUC_SCORE TEST IS:  ', roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# If you want to see tables by each city"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:30:57.746329Z",
     "start_time": "2021-05-10T02:30:57.739348Z"
    }
   },
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (<ipython-input-65-bed1189f191f>, line 32)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-65-bed1189f191f>\"\u001b[1;36m, line \u001b[1;32m32\u001b[0m\n\u001b[1;33m    PerthAirport = dfs['PerthAirport']\u001b[0m\n\u001b[1;37m    ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "Albany = dfs['Albany']\n",
    "Albury =  dfs['Albury']\n",
    "AliceSprings = dfs['AliceSprings']\n",
    "BadgerysCreek = dfs['BadgerysCreek']\n",
    "Ballarat = dfs['Ballarat']\n",
    "Bendigo = dfs['Bendigo']\n",
    "Brisbane = dfs['Brisbane']\n",
    "Cairns = dfs['Cairns']\n",
    "Canberra = dfs['Canberra']\n",
    "Cobar = dfs['Cobar']\n",
    "CoffsHarbour = dfs['CoffsHarbour']\n",
    "Dartmoor = dfs['Dartmoor']\n",
    "Darwin = dfs['Darwin']\n",
    "GoldCoast = dfs['GoldCoast']\n",
    "Hobart = dfs['Hobart']\n",
    "Katherine = dfs['Katherine']\n",
    "Launceston = dfs['Launceston']\n",
    "Melbourne = dfs['Melbourne']\n",
    "MelbourneAirport = dfs['MelbourneAirport']\n",
    "Mildura = dfs['Mildura']\n",
    "More = dfs['Moree']\n",
    "MountGambier = dfs['MountGambier']\n",
    "MountGinini = dfs['MountGinini']\n",
    "Newcastle = dfs['Newcastle']\n",
    "Nhil = dfs['Nhil']\n",
    "NorahHead = dfs['NorahHead']\n",
    "NorfolkIsland = dfs['NorfolkIsland']\n",
    "Nuriootpa = dfs['Nuriootpa']\n",
    "PearceRAAF = dfs['PearceRAAF']\n",
    "Penrith = dfs['Penrith']\n",
    "Perth = dfs['Perth'\n",
    "PerthAirport = dfs['PerthAirport']\n",
    "Portland = dfs['Portland']\n",
    "Richmond = dfs['Richmond']\n",
    "Sale = dfs['Sale']\n",
    "SalmonGums = dfs['SalmonGums']\n",
    "Sydney = dfs['Sydney']\n",
    "SydneyAirport = dfs['SydneyAirport']\n",
    "Townsville = dfs['Townsville']\n",
    "Tuggeranong = dfs['Tuggeranong']\n",
    "Uluru = dfs['Uluru']\n",
    "WaggaWagga = dfs['WaggaWagga']\n",
    "Walpole = dfs['Walpole']\n",
    "Watsonia = dfs['Watsonia']\n",
    "Williamtow = dfs['Williamtown']\n",
    "Witchcliffe = dfs['Witchcliffe']\n",
    "Wollongong = dfs['Wollongong']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-05-10T02:30:57.754308Z",
     "start_time": "2021-05-10T02:27:26.337Z"
    }
   },
   "outputs": [],
   "source": [
    "#Else\n",
    "\n",
    "#for i, g in seasoned_rain.groupby('location'):\n",
    "#    globals()['seasoned_rain_city_' + str(i)] =  g\n",
    "\n",
    "# sort the dataframe\n",
    "#seasoned_rain.sort_values(by='location', axis=1, inplace=True)\n",
    "\n",
    "# set the index to be this and don't drop\n",
    "#seasoned_rain.set_index(keys=['location'], drop=False,inplace=True)\n",
    "\n",
    "# get a list of names\n",
    "#locations = seasoned_rain['location'].unique().tolist()\n",
    "#locations\n",
    "# now we can perform a lookup on a 'view' of the dataframe\n",
    "#joe = df.loc[df.name=='joe']\n",
    "\n",
    "# now you can query all 'joes'"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
