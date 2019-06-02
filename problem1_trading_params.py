from backtester.trading_system_parameters import TradingSystemParameters
from backtester.features.feature import Feature
from datetime import timedelta
from problem1_data_source import Problem1DataSource
from problem1_execution_system import Problem1ExecutionSystem
from backtester.timeRule.custom_time_rule import CustomTimeRule
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
from sklearn import metrics as sm

## Make your changes to the functions below.
## SPECIFY the symbols you are modeling for in getSymbolsToTrade() below
## You need to specify features you want to use in getInstrumentFeatureConfigDicts() and getMarketFeatureConfigDicts()
## and create your predictions using these features in getPrediction()

## Don't change any other function
## The toolbox does the rest for you, from downloading and loading data to running backtest


class MyTradingParams(TradingSystemParameters):
    '''
    initialize class
    place any global variables here
    '''
    def __init__(self, tradingFunctions):
        self.__tradingFunctions = tradingFunctions
        self.__dataSetId = self.__tradingFunctions.dataSetId
        self.__instrumentIds = self.__tradingFunctions.instrumentIds
        self.__targetVariable = self.__tradingFunctions.getTargetVariableKey()
        self.__priceKey = self.__tradingFunctions.getTargetVariableKey()
        self.__additionalInstrumentFeatureConfigDicts = []
        self.__additionalMarketFeatureConfigDicts = []
        self.__fees = {'brokerage': 0.00,'spread': 0.00}
        self.__startDate = self.__tradingFunctions.startDate
        self.__endDate = self.__tradingFunctions.endDate
        super(MyTradingParams, self).__init__()


    '''
    Returns an instance of class DataParser. Source of data for instruments
    '''

    def getDataParser(self):
        # instrumentIds = ['trainingData']
        ds = self.__tradingFunctions.setDataParser()
        ds.loadLiveUpdates()
        return ds
        # return Problem2DataSource(cachedFolderName='historicalData/',
        #                      dataSetId=self.__dataSetId,
        #                      instrumentIds=instrumentIds,
        #                      downloadUrl = 'https://s3.us-east-2.amazonaws.com/qq10-data',
        #                      targetVariableList=self.__targetVariableList,
        #                      targetVariable = self.__tradingFunctions.getTargetVariableKey(),
        #                      timeKey = 'time',
        #                      timeStringFormat = '%Y-%m-%d',
        #                      startDateStr=self.__startDate,
        #                      endDateStr=self.__endDate,
        #                      liveUpdates=True,
        #                      pad=True)
    '''
    Returns an instance of class TimeRule, which describes the times at which
    we should update all the features and try to execute any trades based on
    execution logic.
    For eg, for intra day data, you might have a system, where you get data
    from exchange at a very fast rate (ie multiple times every second). However,
    you might want to run your logic of computing features or running your execution
    system, only at some fixed intervals (like once every 5 seconds). This depends on your
    strategy whether its a high, medium, low frequency trading strategy. Also, performance
    is another concern. if your execution system and features computation are taking
    a lot of time, you realistically wont be able to keep upto pace.
    '''
    def getTimeRuleForUpdates(self):
        return CustomTimeRule(startDate=self.__startDate, endDate=self.__endDate, frequency='S', sample='10', startTime='00:00', endTime='17:00')

    '''
    Returns a timedetla object to indicate frequency of updates to features
    Any updates within this frequncy to instruments do not trigger feature updates.
    Consequently any trading decisions that need to take place happen with the same
    frequency
    '''

    def getFrequencyOfFeatureUpdates(self):
        return timedelta(60, 0)  # minutes, seconds

    def getStartingCapital(self):
        return 100*len(self.__instrumentIds)

    '''
    This is a way to use any custom features you might have made.
    Returns a dictionary where
    key: featureId to access this feature (Make sure this doesnt conflict with any of the pre defined feature Ids)
    value: Your custom Class which computes this feature. The class should be an instance of Feature
    Eg. if your custom class is MyCustomFeature, and you want to access this via featureId='my_custom_feature',
    you will import that class, and return this function as {'my_custom_feature': MyCustomFeature}
    '''

    def getCustomFeatures(self):
        customFeatures = {'prediction': TrainingPredictionFeature,
                'fees_and_spread': FeesCalculator,
                'ScoreCalculator' : ScoreCalculator,
                'Pnl' : PnLCalculator,
                'AccuracyCalculator' : AccuracyCalculator,
                'TPCalculator' : TPCalculator,
                'TNCalculator' : TNCalculator,
                'FPCalculator' : FPCalculator,
                'FNCalculator' : FNCalculator,
                'PrecisionCalculator' : PrecisionCalculator,
                'RecallCalculator' : RecallCalculator,
                'F1Calculator' : F1Calculator}
        customFeatures.update(self.__tradingFunctions.getCustomFeatures())

        return customFeatures


    def getInstrumentFeatureConfigDicts(self):
        # ADD RELEVANT FEATURES HERE

        predictionDict = {'featureKey': 'prediction',
                                'featureId': 'prediction',
                                 'params': {'function': self.__tradingFunctions}}
        feesConfigDict = {'featureKey': 'fees',
                          'featureId': 'fees_and_spread',
                          'params': {'feeDict': self.__fees,
                                    'price': self.__priceKey,
                                    'position' : 'position'}}
        # profitlossConfigDict = {'featureKey': 'pnl',
        #                         'featureId': 'PnL',
        #                         'params': {'price': self.__priceKey,
        #                             'position' : 'position',
        #                                    'fees': 'fees'}}
        capitalConfigDict = {'featureKey': 'capital',
                             'featureId': 'capital',
                             'params': {'price': self.__priceKey,
                                        'fees': 'fees',
                                        'capitalReqPercent': 0.95}}
        scoreDict = {'featureKey': 'score',
                     'featureId': 'ScoreCalculator',
                     'params': {'predictionKey': 'prediction',
                                'targetVariable' : self.__tradingFunctions.getTargetVariableKey()}}
        accuracyDict = {'featureKey': 'accuracy',
                     'featureId': 'AccuracyCalculator',
                     'params': {'predictionKey': 'prediction',
                                'targetVariable' : self.__tradingFunctions.getTargetVariableKey()}}
        tpDict = {'featureKey': 'truePositive',
                     'featureId': 'TPCalculator',
                     'params': {'predictionKey': 'prediction',
                                'targetVariable' : self.__tradingFunctions.getTargetVariableKey(),
                                'threshold':self.__tradingFunctions.threshold}}
        tnDict = {'featureKey': 'trueNegative',
                     'featureId': 'TNCalculator',
                     'params': {'predictionKey': 'prediction',
                                'targetVariable' : self.__tradingFunctions.getTargetVariableKey(),
                                'threshold':self.__tradingFunctions.threshold}}
        fpDict = {'featureKey': 'falsePositive',
                     'featureId': 'FPCalculator',
                     'params': {'predictionKey': 'prediction',
                                'targetVariable' : self.__tradingFunctions.getTargetVariableKey(),
                                'threshold':self.__tradingFunctions.threshold}}
        fnDict = {'featureKey': 'falseNegative',
                     'featureId': 'FNCalculator',
                     'params': {'predictionKey': 'prediction',
                                'targetVariable' : self.__tradingFunctions.getTargetVariableKey(),
                                'threshold':self.__tradingFunctions.threshold}}
        precisionDict = {'featureKey': 'precision',
                     'featureId': 'PrecisionCalculator',
                     'params': {'predictionKey': 'prediction',
                                'truePositive':'truePositive',
                                'falsePositive':'falsePositive',
                                'targetVariable' : self.__tradingFunctions.getTargetVariableKey()}}
        recallDict = {'featureKey': 'recall',
                     'featureId': 'RecallCalculator',
                     'params': {'predictionKey': 'prediction',
                                'truePositive':'truePositive',
                                'falseNegative':'falseNegative',
                                'targetVariable' : self.__tradingFunctions.getTargetVariableKey()}}
        f1Dict = {'featureKey': 'f1',
                     'featureId': 'F1Calculator',
                     'params': {'predictionKey': 'prediction',
                                'precision':'precision',
                                'recall':'recall',
                                'targetVariable' : self.__tradingFunctions.getTargetVariableKey()}}
        stockFeatureConfigs = self.__tradingFunctions.getInstrumentFeatureConfigDicts()


        return {INSTRUMENT_TYPE_STOCK: stockFeatureConfigs + [predictionDict,
                feesConfigDict,capitalConfigDict,scoreDict,
                accuracyDict, tpDict, tnDict, fpDict, fnDict, precisionDict, recallDict, f1Dict]
                + self.__additionalInstrumentFeatureConfigDicts}

    '''
    Returns an array of market feature config dictionaries
        market feature config Dictionary has the following keys:
        featureId: a string representing the type of feature you want to use
        featureKey: a string representing the key you will use to access the value of this feature.this
        params: A dictionary with which contains other optional params if needed by the feature
    '''

    def getMarketFeatureConfigDicts(self):
    # ADD RELEVANT FEATURES HERE
        scoreDict = {'featureKey': 'score',
                     'featureId': 'score_ll',
                     'params': {'featureName': self.getPriceFeatureKey(),
                                'instrument_score_feature': 'score'}}

        marketFeatureConfigs = self.__tradingFunctions.getMarketFeatureConfigDicts()
        return marketFeatureConfigs + [scoreDict] +self.__additionalMarketFeatureConfigDicts

    '''
    Returns the type of execution system we want to use. Its an implementation of the class ExecutionSystem
    It converts prediction to intended positions for different instruments.
    '''

    def getExecutionSystem(self):
        return Problem1ExecutionSystem(enter_threshold=0.7,
                                    exit_threshold=0.55,
                                    longLimit=1,
                                    shortLimit=1,
                                    capitalUsageLimit=0.10 * self.getStartingCapital(),
                                    enterlotSize=1, exitlotSize = 1,
                                    limitType='L', price=self.__priceKey)

    '''
    Returns the type of order placer we want to use. its an implementation of the class OrderPlacer.
    It helps place an order, and also read confirmations of orders being placed.
    For Backtesting, you can just use the BacktestingOrderPlacer, which places the order which you want, and automatically confirms it too.
    '''

    def getOrderPlacer(self):
        return BacktestingOrderPlacer()

    '''
    Returns the amount of lookback data you want for your calculations. The historical market features and instrument features are only
    stored upto this amount.
    This number is the number of times we have updated our features.
    '''

    def getLookbackSize(self):
        return max(150, self.__tradingFunctions.getLookbackSize())

    def getPriceFeatureKey(self):
        return self.__priceKey

    def setPriceFeatureKey(self, priceKey='Adj_Close'):
        self.__priceKey = priceKey

    def getDataSetId(self):
        return self.__dataSetId

    def setDataSetId(self, dataSetId):
        self.__dataSetId = dataSetId

    def getInstrumentsIds(self):
        return self.__instrumentIds

    def setInstrumentsIds(self, instrumentIds):
        self.__instrumentIds = instrumentIds

    def getDates(self):
        return {'startDate':self.__startDate,
                'endDate':self.__endDate}

    def setDates(self, dateDict):
        self.__startDate = dateDict['startDate']
        self.__endDate = dateDict['endDate']

    def getTargetVariableKey(self):
        return self.__targetVariable

    def setFees(self, feeDict={'brokerage': 0.00,'spread': 0.00}):
        self.__fees = feeDict

    def setAdditionalInstrumentFeatureConfigDicts(self, dicts = []):
        self.__additionalInstrumentFeatureConfigDicts = dicts

    def setAdditionalMarketFeatureConfigDicts(self, dicts = []):
        self.__additionalMarketFeatureConfigDicts = dicts

class TrainingPredictionFeature(Feature):

    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        tf = featureParams['function']
        val = 0.5 
        predictions = pd.Series(val, index = instrumentManager.getAllInstrumentsByInstrumentId())
        predictions = tf.getPrediction(time, updateNum, instrumentManager, predictions)
        return predictions

class FeesCalculator(Feature):

    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()


        return 0

# class BuyHoldPnL(Feature):
#     @classmethod
#     def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
#         instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()

#         priceData = instrumentLookbackData.getFeatureDf(featureParams['price'])
#         pnlData = instrumentLookbackData.getFeatureDf(featureKey)
#         if len(priceData)>1:
#             currentPrice = priceData.iloc[-1]
#             previousPnl = pnlData.iloc[-1]
#         else:
#             currentPrice = 0
#             previousPnl = 0
#         bhpnl = pd.Series(0,index = instrumentManager.getAllInstrumentsByInstrumentId())
#         if len(priceData)>1:
#             # bhpnl += (100*(1+previousPnl)*(1+currentPrice) - 100)/100
#             bhpnl += (100+previousPnl)*(1+currentPrice) - 100
#         print('Buy Hold Pnl: %.3f'%bhpnl.iloc[0])
#         # printdf = pd.DataFrame(index=instrumentManager.getAllInstrumentsByInstrumentId())
#         # printdf['previousPnl'] = previousPnl
#         # printdf['currentPrice'] = currentPrice
#         # printdf['bhpnl'] = bhpnl
#         # print(printdf)

#         return bhpnl

class PnLCalculator(Feature):

    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()

        priceData = instrumentLookbackData.getFeatureDf(featureParams['price'])
        positionData = instrumentLookbackData.getFeatureDf(featureParams['position'])
        pnlData = instrumentLookbackData.getFeatureDf(featureKey)
        currentPosition = positionData.iloc[-1]
        previousPosition = 0 if updateNum < 2 else positionData.iloc[-2]
        previousPnl = 0 if updateNum < 2 else pnlData.iloc[-1]
        changeInPosition = currentPosition - previousPosition
        feesData = instrumentLookbackData.getFeatureDf(featureParams['fees'])
        if len(priceData)>2:
            currentPrice = priceData.iloc[-1]
            previousPrice = priceData.iloc[-2]
        else:
            currentPrice = 0
            previousPrice = 0
        zeroSeries = currentPrice * 0

        cumulativePnl = zeroSeries
        fees = feesData.iloc[-1]
        tradePrice = pd.Series([instrumentManager.getInstrument(x).getLastTradePrice() for x in priceData.columns], index=priceData.columns)
        tradeLoss = pd.Series([instrumentManager.getInstrument(x).getLastTradeLoss() for x in priceData.columns], index=priceData.columns)

        # cumulativePnl += (100*(1+previousPnl)*(1+currentPosition*currentPrice) - 100)/100
        cumulativePnl += (100+previousPnl)*(1+currentPosition*currentPrice) - (100)
        return cumulativePnl

class ScoreCalculator(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        ids = list(instrumentManager.getAllInstrumentsByInstrumentId())
        if updateNum <2 :
            return pd.Series(0.5, index=ids)
            
        predictionData = instrumentLookbackData.getFeatureDf(featureParams['predictionKey']).iloc[-2]
        trueValue = instrumentLookbackData.getFeatureDf(featureParams['targetVariable']).iloc[-1]

        previousValue = instrumentLookbackData.getFeatureDf(featureKey).iloc[-1]
        currentScore = pd.Series(0.5, index=previousValue.index)
        yp = 1-predictionData
        currentScore[predictionData!=0.5] = -(trueValue*np.log(yp) + (1 - trueValue)*np.log(1 - yp))
        score = (previousValue*(updateNum-1)+currentScore)/updateNum#sm.accuracy_score(predictionData, trueValue)
        print(score)
        return score

class AccuracyCalculator(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        ids = list(instrumentManager.getAllInstrumentsByInstrumentId())
        if updateNum <2 :
            return pd.Series(0.5, index=ids)
            
        predictionData = instrumentLookbackData.getFeatureDf(featureParams['predictionKey']).iloc[-2]
        trueValue = instrumentLookbackData.getFeatureDf(featureParams['targetVariable']).iloc[-1]

        previousValue = instrumentLookbackData.getFeatureDf(featureKey).iloc[-1]
        currentScore = pd.Series(0.5, index=previousValue.index)
        currentScore[predictionData!=0.5] = currentScore +(0.5 -  np.abs(predictionData - trueValue))
        score = (previousValue*(updateNum-1)+currentScore)/updateNum#sm.accuracy_score(predictionData, trueValue)
        print(score)
        return score

class TPCalculator(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        ids = list(instrumentManager.getAllInstrumentsByInstrumentId())
        if updateNum <2 :
            return pd.Series(0, index=ids)
            
        predictionData = instrumentLookbackData.getFeatureDf(featureParams['predictionKey']).iloc[-2]
        trueValue = instrumentLookbackData.getFeatureDf(featureParams['targetVariable']).iloc[-1]

        previousValue = instrumentLookbackData.getFeatureDf(featureKey).iloc[-1]

        currentScore = 1 if ((trueValue==1) & (predictionData<=featureParams['threshold'])).bool() else 0
        score = (previousValue+currentScore)
        print('True Positive')
        print(score)
        return score

class TNCalculator(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        ids = list(instrumentManager.getAllInstrumentsByInstrumentId())
        if updateNum <2 :
            return pd.Series(0, index=ids)
            
        predictionData = instrumentLookbackData.getFeatureDf(featureParams['predictionKey']).iloc[-2]
        trueValue = instrumentLookbackData.getFeatureDf(featureParams['targetVariable']).iloc[-1]

        previousValue = instrumentLookbackData.getFeatureDf(featureKey).iloc[-1]
        currentScore = 1 if ((trueValue==0) & (predictionData>featureParams['threshold'])).bool() else 0
        score = (previousValue+currentScore)
        print('True Negative')
        print(score)
        return score

class FPCalculator(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        ids = list(instrumentManager.getAllInstrumentsByInstrumentId())
        if updateNum <2 :
            return pd.Series(updateNum, index=ids)
            
        predictionData = instrumentLookbackData.getFeatureDf(featureParams['predictionKey']).iloc[-2]
        trueValue = instrumentLookbackData.getFeatureDf(featureParams['targetVariable']).iloc[-1]

        previousValue = instrumentLookbackData.getFeatureDf(featureKey).iloc[-1]
        currentScore = 1 if ((trueValue==0) & (predictionData<=featureParams['threshold'])).bool() else 0
        score = (previousValue+currentScore)
        print('False Positive')
        print(score)
        return score

class FNCalculator(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        ids = list(instrumentManager.getAllInstrumentsByInstrumentId())
        if updateNum <2 :
            return pd.Series(0, index=ids)
            
        predictionData = instrumentLookbackData.getFeatureDf(featureParams['predictionKey']).iloc[-2]
        trueValue = instrumentLookbackData.getFeatureDf(featureParams['targetVariable']).iloc[-1]

        previousValue = instrumentLookbackData.getFeatureDf(featureKey).iloc[-1]
        currentScore = 1 if ((trueValue==1) & (predictionData>featureParams['threshold'])).bool() else 0
        score = (previousValue+currentScore)
        print('False Negative')
        print(score)
        return score

class PrecisionCalculator(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        ids = list(instrumentManager.getAllInstrumentsByInstrumentId())
        if updateNum <2 :
            return pd.Series(0.5, index=ids)
        # import pdb;pdb.set_trace()            
        tp = instrumentLookbackData.getFeatureDf(featureParams['truePositive']).iloc[-1]

        fp = instrumentLookbackData.getFeatureDf(featureParams['falsePositive']).iloc[-1]
        score = tp / (tp + fp)#sm.accuracy_score(predictionData, trueValue)
        print('precision:')
        print( score)
        return score

class RecallCalculator(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        ids = list(instrumentManager.getAllInstrumentsByInstrumentId())
        if updateNum <2 :
            return pd.Series(0.5, index=ids)
            
        tp = instrumentLookbackData.getFeatureDf(featureParams['truePositive']).iloc[-1]

        fn = instrumentLookbackData.getFeatureDf(featureParams['falseNegative']).iloc[-1]
        score = tp / (tp + fn)#sm.accuracy_score(predictionData, trueValue)
        print('recall:')
        print(score)
        return score

class F1Calculator(Feature):
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        instrumentLookbackData = instrumentManager.getLookbackInstrumentFeatures()
        ids = list(instrumentManager.getAllInstrumentsByInstrumentId())
        if updateNum <2 :
            return pd.Series(0.5, index=ids)
            
        precision = instrumentLookbackData.getFeatureDf(featureParams['precision']).iloc[-2]
        recall = instrumentLookbackData.getFeatureDf(featureParams['recall']).iloc[-1]

        score = 2 * (precision * recall) / (precision + recall)
        print('F1:')
        print(score)
        return score
