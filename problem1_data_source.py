from backtester.dataSource.data_source import DataSource
from backtester.dataSource.csv_data_source import CsvDataSource
from backtester.instrumentUpdates import *
import os
import pandas as pd
from datetime import datetime
import csv
from backtester.logger import *
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class Problem1DataSource(DataSource):
    def __init__(self, cachedFolderName, dataSetId, instrumentIds, downloadUrl = None, targetVariableList=[], targetVariable = '', timeKey = None, timeStringFormat = None, startDateStr=None, endDateStr=None, liveUpdates=True, pad=True):
        self._cachedFolderName = cachedFolderName
        self._dataSetId = dataSetId
        self._downloadUrl = downloadUrl
        self._targetVariable = targetVariable
        self._targetVariableList = list(targetVariableList)
        self._targetVariableList.remove(self._targetVariable) if self._targetVariable in self._targetVariableList else None
        self._targetVariableList.remove(self._targetVariable.upper()) if self._targetVariable.upper() in self._targetVariableList else None
        self._timeKey = timeKey
        self._timeStringFormat = timeStringFormat
        self.ensureDirectoryExists(self._cachedFolderName, self._dataSetId)
        self.ensureAllInstrumentsFile(dataSetId)
        self.featureList=[]
        super(Problem1DataSource, self).__init__(cachedFolderName, dataSetId, instrumentIds, startDateStr, endDateStr)
        print(self._instrumentIds)
        # if liveUpdates:
        #     self._allTimes, self._groupedInstrumentUpdates = self.getGroupedInstrumentUpdates()
        # else:
        #     self._allTimes, self._bookDataByInstrument = self.getAllInstrumentUpdates()
        #     for ids in self._instrumentIds:
        #         if ids=='allData':
        #             self._bookDataByInstrument[ids].drop([col for col in self._bookDataByInstrument[ids] if col in self._targetVariableList], axis=1, inplace=True)
        #     self._bookDataFeatureKeys = list(self._bookDataByInstrument[self._instrumentIds[0]].columns)
        #     if pad:
        #         self.padInstrumentUpdates()
        #     if (startDateStr is not None) and (endDateStr is not None):
        #         self.filterUpdatesByDates([(startDateStr, endDateStr)])

    def getAllInstrumentUpdates(self, chunks=None):
        allInstrumentUpdates = {instrumentId : None for instrumentId in self._instrumentIds}
        timeUpdates = []
        for instrumentId in self._instrumentIds:
            print('Processing data for stock: %s' % (instrumentId))
            fileName = self.getFileName(instrumentId)
            if not self.downloadAndAdjustData(instrumentId, fileName):
                continue
            ### TODO: Assumes file is a csv, this is should not be in base class but ds type specific
            allInstrumentUpdates[instrumentId] = pd.read_csv(fileName, index_col=0, parse_dates=True)
            timeUpdates = allInstrumentUpdates[instrumentId].index.union(timeUpdates)
            allInstrumentUpdates[instrumentId].dropna(inplace=True)
            # NOTE: Assuming data is sorted by timeUpdates and all instruments have same columns
        timeUpdates = list(timeUpdates)
        return timeUpdates, allInstrumentUpdates


    def getAllInstrumentUpdatesDict(self):
        times, bookDataByInstrument = self.getAllInstrumentUpdates()
        for ids in self._instrumentIds:
            if ids=='allData':
                bookDataByInstrument[ids].drop([col for col in bookDataByInstrument[ids] if col in self._targetVariableList], axis=1, inplace=True)
        return bookDataByInstrument

    def loadLiveUpdates(self, featureList):
        self.featureList = featureList
        self._allTimes, self._groupedInstrumentUpdates = self.getGroupedInstrumentUpdates()

    def getInstrumentUpdateFromRow(self, instrumentId, row):
        bookData = row
        timeKey = self._timeKey
        for key in list(bookData):
            if is_number(bookData[key]):
                bookData[key] = float(bookData[key])

        timeOfUpdate = datetime.strptime(row[timeKey], self._timeStringFormat)
        print('Processing for: '+row[timeKey])
        bookData.pop(timeKey, None)
        # print(bookData.keys())

        inst = StockInstrumentUpdate(stockInstrumentId=instrumentId,
                                     tradeSymbol=instrumentId,
                                     timeOfUpdate=timeOfUpdate,
                                     bookData=bookData)

        if self._bookDataFeatureKeys is None:
            self._bookDataFeatureKeys = bookData.keys()  # just setting to the first one we encounter
            print(self._bookDataFeatureKeys)
        return inst

    def getFileName(self, instrumentId):
        return self._cachedFolderName + self._dataSetId + '/' + instrumentId + '.csv'

    def ensureAllInstrumentsFile(self, dataSetId):
        return True

    def downloadFile(self, instrumentId, downloadLocation):
        url = ''
        if self._dataSetId != '':
            url = '%s/%s/%s.csv' % (self._downloadUrl, self._dataSetId, instrumentId)
        else:
            url = '%s/%s.csv' % (self._downloadUrl, instrumentId)

        response = urlopen(url)
        status = response.getcode()
        if status == 200:
            print('Downloading %s data to file: %s' % (instrumentId, downloadLocation))
            with open(downloadLocation, 'w') as f:
                f.write(response.read().decode('utf8'))
            return True
        else:
            logError('File not found. Please check settings!')
            return False

    def downloadAndAdjustData(self, instrumentId, fileName):
        if not os.path.isfile(fileName):
            if not self.downloadFile(instrumentId, fileName):
                logError('Skipping %s:' % (instrumentId))
                return False
        return True




if __name__ == "__main__":
    ds = Problem1DataSource(cachedFolderName='historicalData/',
                             dataSetId='p1',
                             instrumentIds=['trainingData'],
                             downloadUrl = 'https://qq11-data.s3.amazonaws.com',
                             targetVariableList=[],
                             targetVariable = 'Out',
                             timeKey = 'date',
                             timeStringFormat = '%Y-%m-%d',
                             startDateStr='1993/01/31',
                             endDateStr='2012/12/31',
                             liveUpdates=True,
                             pad=True)
    t = ds.emitAllInstrumentUpdates()
    fl = []
    ds.loadLiveUpdates(fl)
    groupedInstrumentUpdates = ds.emitInstrumentUpdates()
    timeOfUpdate, instrumentUpdates = next(groupedInstrumentUpdates)
    print(timeOfUpdate, instrumentUpdates[0].getBookData())
    while True:
        try:
            timeOfUpdate, instrumentUpdates = next(groupedInstrumentUpdates)
            print(timeOfUpdate, instrumentUpdates[0].getBookData())
        except StopIteration:
            break
