"""
Implement Value at Risk and conditional Value at Risk using:
    1. Historical Method
    2. Parametric Method (Variance-Covariance)
    3. Monte Carlo Method
"""
import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
from scipy.stats import norm, t

# Import data
def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks,start=start,end=end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

# Portfolio performance
def portfolioPerformance(weights, meanReturns, CovMatrix, Time):
    returns = np.sum(meanReturns*weights)*Time
    std = np.sqrt(np.dot(weights.T, np.dot(CovMatrix, weights))) * np.sqrt(Time)
    return returns, std

stocks = ['TSLA', 'GOOG', 'NFLX', 'NVDA', 'AAPL']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=800)

returns, meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
returns = returns.dropna()

weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)

returns['portfolio'] = returns.dot(weights)

def HistoricalVaR(returns,alpha=5):
    """
    Read in a pandas dataframe of returns or a pandas series of returns
    Output: Percentile of the distribution at that given alpha confidence level
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)

    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(HistoricalVaR, alpha=5)
    
    else:
        raise TypeError('Expected returns to be dataframe or series')


def HistoricalCVaR(returns,alpha=5):
    """
    Read in a pandas dataframe of returns or a pandas series of returns
    Output: CVar for dataframe or series
    """
    if isinstance(returns, pd.Series):
        belowVar = returns <= HistoricalVaR(returns, alpha=alpha)
        return  returns[belowVar].mean()

    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(HistoricalCVaR, alpha=5)
    
    else:
        raise TypeError('Expected returns to be dataframe or series')

# 1 Day
Time = 1

Var = -HistoricalVaR(returns['portfolio'], alpha=5)*np.sqrt(Time)
CVar = -HistoricalCVaR(returns['portfolio'], alpha=5)*np.sqrt(Time)

pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)

initialInvestment = 100000
print('Expected Portfolio Return:       ', round(initialInvestment*pRet,2))
print('Value at Risk 95th CI    :       ', round(initialInvestment*Var,2))
print('Conditional VaR 95th CI  :       ', round(initialInvestment*CVar,2))

def var_parametric(portfolioReturn,portfolioStd, distribution='normal', alpha=5, dof=6):
    """
        calculate portfolio VaR given a distribution, with known parameters
    """
    if distribution=='normal':
        VaR = norm.ppf(1-alpha/100)*portfolioStd - portfolioReturn
    
    elif distribution=='t-distribution':
        nu = dof
        VaR = np.sqrt((nu-2)/nu) * t.ppf(1-alpha/100,nu)*portfolioStd - portfolioReturn
    else:
        raise TypeError('Expected distribution to be "normal' or 't')
    return VaR

def cvar_parametric(portfolioReturn,portfolioStd, distribution='normal', alpha=5, dof=6):
    """
        calculate portfolio VaR given a distribution, with known parameters
    """
    if distribution=='normal':
        CVar = (alpha/100)**-1*norm.pdf(norm.ppf(alpha/100))*portfolioStd - portfolioReturn
    
    elif distribution=='t-distribution':
        nu = dof
        x_anu = t.ppf(alpha/100,nu)
        CVar = -1/(alpha/100) * (1-nu)**-1 * (nu-2+ x_anu**2)* t.pdf(x_anu, nu)*portfolioStd - portfolioReturn
    else:
        raise TypeError('Expected distribution to be "normal' or 't')
    return CVar

normVar = var_parametric(pRet,pStd)
normCVar = cvar_parametric(pRet,pStd)

tVar = var_parametric(pRet,pStd,distribution='t-distribution')
tCVar = cvar_parametric(pRet,pStd,distribution='t-distribution')

print('normal VaR 95th CI       :       ', round(initialInvestment*normVar,2))
print('normal CVaR VaR 95th CI  :       ', round(initialInvestment*normCVar,2))
print('t-dist VaR 95th CI       :       ', round(initialInvestment*tVar,2))
print('t-dist CVaR VaR 95th CI  :       ', round(initialInvestment*tCVar,2))