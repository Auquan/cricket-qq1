from backtester.trading_system_parameters import TradingSystemParameters
from backtester.features.feature import Feature
from datetime import datetime, timedelta
from problem1_data_source import Problem1DataSource
from problem1_execution_system import Problem1ExecutionSystem
from backtester.orderPlacer.backtesting_order_placer import BacktestingOrderPlacer
from backtester.trading_system import TradingSystem
from backtester.version import updateCheck
from backtester.constants import *
from backtester.features.feature import Feature
from backtester.logger import *
import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics as sm
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import TimeSeriesSplit
from problem1_trading_params import MyTradingParams

##################################################################################
##################################################################################
## Template file for problem 1.                                                 ##
##################################################################################
## Make your changes to the functions below.
## SPECIFY features you want to use in getInstrumentFeatureConfigDicts()
## Create your fairprice using these features in getPrediction()
## SPECIFY any custom features in getCustomFeatures() below
## Don't change any other function
## The toolbox does the rest for you, from downloading and loading data to running backtest
##################################################################################


class MyTradingFunctions():

    def __init__(self):  #Put any global variables here
        self.lookback = 120  ## max number of historical datapoints you want at any given time
        self.targetVariable = 'Out'
        self.__dataParser = None
        self.dataSetId = 'p1'
        self.instrumentIds = ['trainingData']
        self.startDate = '2010/01/01'
        self.endDate = '2015/12/31'
        self.params = {}

        # for example you can import and store an ML model from scikit learn in this dict
        self.model = {}

        # and set a frequency at which you want to update the model

        self.updateFrequency = 300
        self.__featureKeys = []
        self.__featureDict = self.convertCategoricalVariablesAndTrain()
        self.predictionLogFile = open('predictions.csv', 'a')
        self.headerNotSet = True

    ###########################################
    ## ONLY FILL THE FOUR FUNCTIONS BELOW    ##
    ###########################################

    def convertCategoricalVariablesAndTrain(self):
        ds = self.getDataParser()
        dataDict = ds.getAllInstrumentUpdatesDict()

        ids = self.instrumentIds 

        print('Converting Text Variables to Categorical Variables')
        for i in range(len(ids)):
            s = ids[i]

            data = dataDict[s]

            feature_dict = {}

            for feature in data.columns:
                if data[feature].dtype==object:    
                    fs = data[feature].unique()
                    f_dict = {}
                    count = 0
                    for f in fs:
                        data[feature][data[feature]==f] = count
                        f_dict[f] = count
                        count = count+1
                    feature_dict[feature] = f_dict

            ########################################################
            ####    If you are training a model at the start    ####
            ########################################################

            print('Training a classifier')

            self.model[s]= DecisionTreeClassifier(max_depth = 10)

            training_data = data.copy()
            training_data['run_last_6_balls'] = training_data['innings_runs_before_ball'].rolling(6).sum()
            y = training_data[self.targetVariable]
            del training_data[self.targetVariable]

            import pdb;pdb.set_trace()

            training_data.fillna(0, inplace=True)

            self.model[s].fit(training_data, y) 

            #############################################################
            ####    If you want a train/test split to see metrics    ####
            #############################################################

            # from sklearn.model_selection import train_test_split
            # # dividing X, y into train and test data 
            # X_train, X_test, y_train, y_test = train_test_split(training_data, y, random_state = 10) 

            #self.model[s].fit(X_train, y_train) 

            lg = log_loss(y, self.model[s].predict_proba(training_data))
            print(lg)

            print('Done, moving now')

        return feature_dict

    '''
    Specify all Features you want to use by  by creating config dictionaries.
    Create one dictionary per feature and return them in an array.
    Feature config Dictionary have the following keys:
        featureId: a str for the type of feature you want to use
        featureKey: {optional} a str for the key you will use to call this feature
                    If not present, will just use featureId
        params: {optional} A dictionary with which contains other optional params if needed by the feature
    msDict = {'featureKey': 'ms_5',
              'featureId': 'moving_sum',
              'params': {'period': 5,
                         'featureName': 'basis'}}
    return [msDict]
    You can now use this feature by in getPRediction() calling it's featureKey, 'ms_5'
    '''

    def getInstrumentFeatureConfigDicts(self):

    ##############################################################################
    ### TODO 1a: FILL THIS FUNCTION TO CREATE DESIRED FEATURES for each symbol. ###
    ### USE TEMPLATE BELOW AS EXAMPLE                                          ###
    ##############################################################################

        sumDict = {'featureKey': 'run_last_6_balls',
                     'featureId': 'sum',
                     'params': {'period': 6,
                                'featureName': 'innings_runs_before_ball'}}
        return [sumDict]


    def getMarketFeatureConfigDicts(self):
    ###############################################################################
    ### TODO 2b: FILL THIS FUNCTION TO CREATE features that use multiple symbols ###
    ### USE TEMPLATE BELOW AS EXAMPLE                                           ###
    ###############################################################################

        # customFeatureDict = {'featureKey': 'custom_mrkt_feature',
        #                      'featureId': 'my_custom_mrkt_feature',
        #                      'params': {'param1': 'value1'}}
        return []

    '''
    Combine all the features to create the desired 0/1 predictions for each symbol.
    'predictions' is Pandas Series with symbol as index and predictions as values
    We first call the holder for all the instrument features for all symbols as
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()
    Then call the dataframe for a feature using its feature_key as
        ms5Data = lookbackInstrumentFeatures.getFeatureDf('ms_5')
    This returns a dataFrame for that feature for ALL symbols for all times upto lookback time
    Now you can call just the last data point for ALL symbols as
        ms5 = ms5Data.iloc[-1]
    You can call last datapoint for one symbol 'ABC' as
        value_for_abs = ms5['ABC']
    Output of the prediction function is used by the toolbox to make further trading decisions and evaluate your score.
    '''


    def getPrediction(self, time, updateNum, instrumentManager,predictions):

        # holder for all the instrument features for all instruments
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()
        # holder for all the market features
        lookbackMarketFeatures = instrumentManager.getDataDf()

        #############################################################################################
        ###  TODO 3 : FILL THIS FUNCTION TO RETURN the probability that targetVariable is 0       ###
        ###  You can use all the features created above and combine then using any logic you like ###
        ###  USE TEMPLATE BELOW AS EXAMPLE                                                        ###
        #############################################################################################

        # if you don't enough data yet, don't make a prediction
        
        if updateNum<=2*self.updateFrequency:
            return predictions

        import pdb;pdb.set_trace()
        # Once you have enough data, start making predictions

        # Loading the target Variable
        Y = lookbackInstrumentFeatures.getFeatureDf(self.getTargetVariableKey())

        #Creating an array to load and hold all features
        X = []         # 3D array timestamp x featureNames x instrumentIds
        x_star = []                             # Data point at time t (whose Value will be predicted) featureKeys x instrumentIds
        for f in self.__featureKeys:
            data = lookbackInstrumentFeatures.getFeatureDf(f).fillna(0)
            if data.dtype==object:
                fs = data.unique()
                for fss in fs:
                    data[data==fss] = self.feature_dict[f][fss]        #DF with rows=timestamp and columns=instrumentIds
            X.append(data.values)
            x_star.append(np.array(data.iloc[-1]))

        X = np.nan_to_num(np.array(X))                                             # shape = featureKeys x timestamp x instrumentIds
        x_star = np.nan_to_num(np.array(x_star))                                       # shape = featureKeys x instrumentIds
        
        # import pdb;pdb.set_trace()
        # Now looping over all stocks:
        ids = self.instrumentIds 
        for i in range(len(ids)):
            s = ids[i]
            
            
            #####################################################################
            ##### If you are training on the go, use the code below to train ####
            #####################################################################

            # # if this is the first time we are training a model, start by creating a new model
            # if s not in self.model:
            #         self.model[s]= DecisionTreeClassifier(max_depth = 10)

            # # we will update this model during further runs

            # # if you are at the update frequency, update the model
            # if (updateNum-1)%self.updateFrequency==0:
            #     try:
            #         # drop nans and infs from X
            #         X_train = X[:,:,i]
            #         # create a target variable vector with same index as X
            #         y_s = Y.values #Y.loc[Y.index.isin(X.index)]

            #         print('Training...')


            #         # make numpy arrays with the right shape
            #         x_train = np.array(X_train).T[:-1]                         # shape = timestamps x numFeatures
            #         y_train = np.array(y_s)[1:].astype(float).reshape(-1)        # shape = timestamps x 1

            #         self.model[s].fit(x_train, y_train)
            #     except ValueError:
            #         print('not fitting')


            #####################################
            ####    Making Predictions      #####
            #####################################

            # make your prediction using your model
            # first verify none of the features are nan or inf
            # import pdb;pdb.set_trace()
            if np.isnan(x_star).any():
                y_predict = 0.5
            else:
                try:
                    y_predict = self.model[s].predict_proba(x_star.reshape(1,-1))

                except Exception as e: 
                    print(e)
                    y_predict = 0.5 

            predictions[s] = y_predict
            print('prediction for %s %s :%.3f'%(s, self.targetVariable, y_predict))
        self.logPredictions(time, predictions)
        return predictions

    ###########################################
    ##         DONOT CHANGE THESE            ##
    ###########################################

    def getDataParser(self):
        if self.__dataParser is None:
            self.__dataParser = self.initDataParser()
        return self.__dataParser

    def initDataParser(self):
        ds = Problem1DataSource(cachedFolderName='historicalData/',
                         dataSetId=self.dataSetId,
                         instrumentIds=self.instrumentIds,
                         downloadUrl = 'https://s3.us-east-2.amazonaws.com/qq11-data',
                         targetVariable = self.targetVariable,
                         timeKey = 'date',
                         timeStringFormat = '%Y-%m-%d',
                         startDateStr=self.startDate,
                         endDateStr=self.endDate,
                         liveUpdates=True,
                         pad=True)
        return ds

    def setDataParser(self):
        self.__dataParser = self.initDataParser()
        return self.__dataParser

    def getLookbackSize(self):
        return self.lookback

    def getDataSetId(self):
        return self.dataSetId

    def setDataSetId(self, dataSetId):
        self.dataSetId = dataSetId

    def getInstrumentIds(self):
        return self.instrumentIds

    def setInstrumentIds(self, instrumentIds):
        self.instrumentIds = instrumentIds

    def getTargetVariableKey(self):
        return self.targetVariable

    def setTargetVariableKey(self, targetVariable):
        self.targetVariable = targetVariable

    def getFeatureKeys(self):
        return self.__featureKeys

    def setFeatureKeys(self, featureList):
        self.__featureKeys = featureList

    def getFeatureList(self):
        return self.featureList

    def getTargetVariableType(self):
        return self.targetVariableType

    # c for continuous, b for binary
    def setTargetVariableType(self, targetVariableType):
        self.targetVariableType = targetVariableType

    # upper case is continuous, lower is binary
    def setTargetVariableList(self, targetVariableList):
        self.targetVariableList = targetVariableList

    # upper case is continuous, lower is binary
    def getTargetVariableList(self):
        return self.targetVariableList

    # upper case is continuous, lower is binary
    def setPredictionLogFile(self, logFileName):
        self.predictionLogFile = open(logFileName, 'a')

    def logPredictions(self, time, predictions):
        if (self.predictionLogFile != None):
            if(self.headerNotSet):
                header = 'datetime'
                for index in predictions.index:
                    header = header + ',' + index
                self.predictionLogFile.write(header + '\n')
                self.headerNotSet = False

            lineData = str(time)

            for prediction in predictions.get_values():
                lineData = lineData + ',' + str(prediction)

            self.predictionLogFile.write(lineData + '\n')

    ###############################################
    ##  CHANGE ONLY IF YOU HAVE CUSTOM FEATURES  ##
    ###############################################

    def getCustomFeatures(self):
        return {'my_custom_feature_identifier': MyCustomFeatureClassName}

####################################################
##   YOU CAN DEFINE ANY CUSTOM FEATURES HERE      ##
##  If YOU DO, MENTION THEM IN THE FUNCTION ABOVE ##
####################################################
class MyCustomFeatureClassName(Feature):
    ''''
    Custom Feature to implement for instrument. This function would return the value of the feature you want to implement.
    1. create a new class MyCustomFeatureClassName for the feature and implement your logic in the function computeForInstrument() -
    2. modify function getCustomFeatures() to return a dictionary with Id for this class
        (follow formats like {'my_custom_feature_identifier': MyCustomFeatureClassName}.
        Make sure 'my_custom_feature_identifier' doesnt conflict with any of the pre defined feature Ids
        def getCustomFeatures(self):
            return {'my_custom_feature_identifier': MyCustomFeatureClassName}
    3. create a dict for this feature in getInstrumentFeatureConfigDicts() above. Dict format is:
            customFeatureDict = {'featureKey': 'my_custom_feature_key',
                                'featureId': 'my_custom_feature_identifier',
                                'params': {'param1': 'value1'}}
    You can now use this feature by calling it's featureKey, 'my_custom_feature_key' in getPrediction()
    '''
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        # Custom parameter which can be used as input to computation of this feature
        param1Value = featureParams['param1']

        # A holder for the all the instrument features
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()

        # dataframe for a historical instrument feature (basis in this case). The index is the timestamps
        # atmost upto lookback data points. The columns of this dataframe are the symbols/instrumentIds.
        lookbackInstrumentValue = lookbackInstrumentFeatures.getFeatureDf('symbolVWAP')

        # The last row of the previous dataframe gives the last calculated value for that feature (basis in this case)
        # This returns a series with symbols/instrumentIds as the index.
        currentValue = lookbackInstrumentValue.iloc[-1]

        if param1Value == 'value1':
            return currentValue * 0.1
        else:
            return currentValue * 0.5


if __name__ == "__main__":
    if updateCheck():
        print('Your version of the auquan toolbox package is old. Please update by running the following command:')
        print('pip install -U auquan_toolbox')
    else:
        print('Loading your config dicts and prediction function')
        tf = MyTradingFunctions()
        print('Loaded config dicts and prediction function, Loading Problem Params')
        tf.setInstrumentIds(['training_data'])
        tsParams = MyTradingParams(tf)
        print('Loaded Problem Params, Loading Backtester and Data')
        tradingSystem = TradingSystem(tsParams)
        print('Loaded Backtester and Data Loaded, Backtesting')
    # Set onlyAnalyze to True to quickly generate csv files with all the features
    # Set onlyAnalyze to False to run a full backtest
    # Set makeInstrumentCsvs to False to not make instrument specific csvs in runLogs. This improves the performance BY A LOT
        tradingSystem.startTrading(onlyAnalyze=False, shouldPlot=True, makeInstrumentCsvs=True)
