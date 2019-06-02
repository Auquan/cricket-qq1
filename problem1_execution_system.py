from backtester.executionSystem.simple_execution_system import SimpleExecutionSystem
from backtester.logger import *
import numpy as np
import pandas as pd
from scipy.stats import norm


class Problem1ExecutionSystem(SimpleExecutionSystem):
    def __init__(self, enter_threshold=0.7, exit_threshold=0.55, longLimit=10,
                 shortLimit=10, capitalUsageLimit=0, enterlotSize=1, exitlotSize = 1, limitType='L', price='close'):
        self.priceFeature = price
        super(Problem1ExecutionSystem, self).__init__(enter_threshold=enter_threshold,
                                                                 exit_threshold=exit_threshold,
                                                                 longLimit=longLimit, shortLimit=shortLimit,
                                                                 capitalUsageLimit=capitalUsageLimit,
                                                                 enterlotSize=enterlotSize, exitlotSize=exitlotSize, limitType=limitType, price=price)

    def getPriceDf(self, instrumentsManager):
        instrumentLookbackData = instrumentsManager.getLookbackInstrumentFeatures()
        try:
            price = instrumentLookbackData.getFeatureDf(self.priceFeature)
            return price
        except KeyError:
                logError('You have specified Dollar Limit but Price Feature Key %s does not exist'%self.priceFeature)


    def exitPosition(self, time, instrumentsManager, currentPredictions, closeAllPositions=False):

        instrumentLookbackData = instrumentsManager.getLookbackInstrumentFeatures()
        positionData = instrumentLookbackData.getFeatureDf('position')
        executions = pd.Series([0] * len(positionData.columns), index=positionData.columns)

        # print('exit?',self.exitCondition(currentPredictions, instrumentsManager))
        return executions

    def enterPosition(self, time, instrumentsManager, currentPredictions, capital):
        instrumentLookbackData = instrumentsManager.getLookbackInstrumentFeatures()
        positionData = instrumentLookbackData.getFeatureDf('position')
        # import pdb;pdb.set_trace()
        executions = pd.Series([0] * len(positionData.columns), index=positionData.columns)
        return executions

    def getBuySell(self, currentPredictions, instrumentsManager):
        return np.sign(currentPredictions - 0.5)
        

    def enterCondition(self, currentPredictions, instrumentsManager):
        return pd.Series(False, index=currentPredictions.index)

    def exitCondition(self, currentPredictions, instrumentsManager):
        return pd.Series(False, index=currentPredictions.index)


    def hackCondition(self, currentPredictions, instrumentsManager):
        return pd.Series(False, index=currentPredictions.index)