
"""
Implement Monte Carlo method to simulate a stock portfolio
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr

# import data
def get_data(stocks,start,end):
    stockData=pdr.get_data_yahoo(stocks,start,end)
    stockData=stockData['Close']
    returns = stockData.pct_change()
    meanReturns= returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stocks = ['TSLA', 'GOOG', 'NFLX', 'NVDA', 'AAPL']
endDate = dt.datetime.now()
startDate = endDate-dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks,startDate,endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

print(weights)

# Monte Carlo method
# number of simulations
mc_sims = 100
T = 1000 #timeframe in days

meanM = np.full(shape=(T,len(weights)), fill_value=meanReturns)
meanM=meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initial_portfolio = 10000

for m in range(0,mc_sims):
    # MC loops
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L,Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initial_portfolio


def mcVar(returns, alpha=5):
    """
    Input: pandas series of returns
    Output: percentile on return distributions to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")

def mcCVar(returns, alpha=5):
    """
    Input: pandas series of returns
    Output: CVar or expected shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVar = returns <= mcVar(returns, alpha=alpha)
        return returns[belowVar].mean()
    else:
        raise TypeError("Expected a pandas data series.")

portfolioResults = pd.Series(portfolio_sims[-1,:])

Var = initial_portfolio - mcVar(portfolioResults, alpha=5)
CVar = initial_portfolio - mcCVar(portfolioResults, alpha=5)

print('VaR ${}'.format(round(Var,2)))
print('CVaR ${}'.format(round(CVar,2)))

plt.plot(portfolio_sims)
plt.axhline(y=initial_portfolio-Var)
plt.axhline(y=initial_portfolio-CVar)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulations of a stock portfolio')
plt.show()