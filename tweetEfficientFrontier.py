#!/usr/bin/python3

# Imports  
import argparse
import os

import pymysql
from datetime import datetime

import numpy as np
import math
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from adjustText import adjust_text

import cvxopt as opt
from cvxopt import blas, solvers

# Turn off progress printing
solvers.options['show_progress'] = False
from twitter import Twitter, OAuth

import random
from copy import deepcopy
import operator

import config


def getDbConnection():
    """
    Open connection to the database
    """
    return pymysql.connect(host=config.mysql['host'],
                           port=config.mysql['port'],
                           user=config.mysql['user'],
                           passwd=config.mysql['passwd'],
                           db=config.mysql['db'],
                           charset='utf8',
                           autocommit=True,
                           cursorclass=pymysql.cursors.DictCursor)


def getCoinNames():
    """
    Generate a map of coin symbol to coin name
    """

    # Obtain observations from database
    conn = getDbConnection()
    cur = conn.cursor()

    # Get names
    sql = "SELECT DISTINCT symbol, name FROM coinmarketcap"
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()

    return {r['symbol']: r['name'] for r in rows}


def getPrices(hr, minMarketCap=1E9):
    """
    Generate a dataframe of the top coins

    Input:
        hr (int): number of hours for calculation
        minMarketCap (float): minimum market cap

    Output:
        Dataframe of prices of top coins over duration, hr
    """

    # Obtain observations from database
    conn = getDbConnection()
    cur = conn.cursor()

    # Get list of coins that were over the minimum market cap at any point in the past hr hours
    earliest_timestamp = (pd.Timestamp.utcnow() - pd.Timedelta(f'+{hr}:00:00'))
    earliest_date = earliest_timestamp.strftime('%Y-%m-%d')
    earliest_datetime = earliest_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    sql = f"SELECT DISTINCT symbol FROM coinmarketcap WHERE symbol IN " + \
          f"(SELECT DISTINCT symbol FROM coinmarketcap WHERE DATE(timestamp) = '{earliest_date}') " + \
          f"AND symbol NOT IN " + \
          f"(SELECT DISTINCT symbol FROM coinmarketcap WHERE timestamp > '{earliest_datetime}' " + \
          f"AND market_cap_USD < {minMarketCap})"
    cur.execute(sql)
    symbols = cur.fetchall()
    symbolStr = "(" + ','.join(["'" + s['symbol'] + "'" for s in symbols]) + ")"
    # E.g. "('BTC','ETH','XRP','BCH','ADA','LTC','XEM','NEO')"

    # Get price history for those coins
    sql = f"SELECT timestamp, symbol, name, price_usd FROM coinmarketcap " + \
          f"WHERE symbol IN {symbolStr} AND timestamp >= '{earliest_date}' " + \
          f"ORDER BY timestamp"
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()

    dfPrice = pd.DataFrame(rows)
    dfPrice['timestamp'] = dfPrice['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
    dfPrice = dfPrice.pivot(values='price_usd', columns='symbol', index='timestamp')

    # Back-fill missing observations
    dfPrice.fillna(method='backfill', axis='rows', inplace=True)
    dfPrice.fillna(method='ffill', axis='rows', inplace=True)

    return dfPrice


def getReturns(prices):
    """
    Calculate daily returns from 5-minute prices
    """

    return prices.pct_change()


def getRiskReturn(returns):
    """
    Calculates the returns, volatility, Sharpe Ratio for coins

    Input:
       prices (dataframe): coin prices arranged by coin (columns) and time (rows)

    Output:
       Dataframe of returns, volatility, Sharpe ratios
    """

    oph = 12  # Observations per hour
    hpd = 24  # Hours per day

    mean = returns.mean() * oph * hpd  # mean of returns of coins
    covariance = returns.cov() * oph * hpd  # covariance of returns of coins
    covariance_array = covariance.as_matrix()
    covariance_id = covariance_array.diagonal()
    volatility = np.sqrt(covariance_id)

    sharpe = mean / volatility

    portfolio = {'Return': mean,
                 'Volatility': volatility,
                 'Sharpe Ratio': sharpe}

    df = pd.DataFrame(portfolio)
    df = df[['Return', 'Volatility', 'Sharpe Ratio']]
    df = df.sort_values(by=['Sharpe Ratio'], ascending=False)
    return df


def getOptimalPortfolio(returnsDf):
    returns = returnsDf.as_matrix()[1:].T

    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks, portfolios


def getSimulatedPortfolios(returns, num_simulations, starting_pfs):
    """
    Calculates the returns, volatility, Sharpe Ratio for various portfolios with randomly assigned coin weights

    Input:
       num_simulations (int): number of simulations
       starting_pfs (list): optimal portfolios around which to simulate

    Output:
       Dataframe of returns, volatility, Sharpe Ratio, and weights of top n coins over duration, hr, for each simulation
    """

    topCoins = list(returns)

    oph = 12  # Observations per hour
    hpd = 24  # Hours per day
    returns_mean = returns.mean() * oph * hpd
    covariance = returns.cov() * oph * hpd

    # empty lists to store returns, volatility and weights of simulated portfolios
    coin_weights = []
    port_returns = []
    port_volatility = []
    sharpe_ratio = []

    # populate the empty lists with each portfolio's returns, risk and weights
    for portfolio in range(num_simulations):
        starting_pf = starting_pfs[random.randint(0, len(starting_pfs) - 1)]
        starting_weights = np.array(list(starting_pf))

        perturbed_weights = deepcopy(starting_weights)
        for i in range(random.randint(0, len(perturbed_weights))):
            random_source = random.randint(0, len(perturbed_weights) - 1)
            random_destination = random.randint(0, len(perturbed_weights) - 1)
            random_amount = np.random.random() * perturbed_weights[random_source]
            perturbed_weights[random_source] -= random_amount
            perturbed_weights[random_destination] += random_amount

        weights = perturbed_weights

        coin_weights.append(weights)

        returns = np.dot(weights, returns_mean)
        port_returns.append(returns)

        volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        port_volatility.append(volatility)

        sharpe = returns / volatility
        sharpe_ratio.append(sharpe)

    # Create a dictionary for returns and risk values of each portfolio
    portfolio = {'Return': port_returns,
                 'Volatility': port_volatility,
                 'Sharpe Ratio': sharpe_ratio}

    for i, coin in enumerate(topCoins):
        portfolio[coin] = [weight[i] for weight in coin_weights]

    dfTopCoins = pd.DataFrame(portfolio)
    dfTopCoins = dfTopCoins[['Return', 'Volatility', 'Sharpe Ratio'] + [coin for coin in topCoins]]

    return dfTopCoins


def plotSimulatedPortfolios(df_coins, df_simulated, color_map='coolwarm_r', label_period='7d', hours=168,
                            filename='EFTopCoins.png'):
    """
    Plot the coins and the simulated portfolios in the specified color scheme
    """

    # Plot simulated portfolios
    plt.style.use('seaborn-dark')
    ef_plot = df_simulated.plot.scatter(x='Volatility',
                                        y='Return',
                                        c='Sharpe Ratio',
                                        s=7,
                                        alpha=0.5,
                                        cmap=color_map,
                                        edgecolors=None,
                                        figsize=(10, 8),
                                        grid=True)

    plt.xlabel(f'Daily volatility ({label_period} average)')
    plt.ylabel(f'Daily return ({label_period} average)')
    plt.title('Cryptocurrency Risk-Return')

    # Overlay coins
    plt.scatter(x=df_coins['Volatility'],
                y=df_coins['Return'],
                c=df_coins['Sharpe Ratio'],
                s=10,
                cmap=color_map,
                edgecolors='black')

    # y axis
    y_min = df_coins['Return'].min()
    y_max = max(df_coins['Return'].max(), df_simulated['Return'].max())
    y_range = y_max - y_min
    y_min -= y_range * 0.01
    y_max += y_range * 0.01
    plt.ylim(y_min, y_max)
    y_ticks = ef_plot.get_yticks()
    ef_plot.set_yticklabels(['{:3.0f}%'.format(y * 100) for y in y_ticks])

    # x axis
    plt.xlim(0, df_coins['Volatility'].max() * 1.01)
    x_ticks = ef_plot.get_xticks()
    ef_plot.set_xticklabels(['{:3.0f}%'.format(x * 100) for x in x_ticks])

    # Find optimal portfolio
    sorted_pfs = df_simulated[['Volatility', 'Return']].sort_values(by=['Volatility', 'Return'], ascending=True)
    frontier = dict()
    frontier[0] = (sorted_pfs.iloc[0]['Volatility'], sorted_pfs.iloc[0]['Return'])
    for i in range(1, len(sorted_pfs)):
        if True or sorted_pfs.iloc[i]['Return'] >= sorted_pfs.iloc[i - 1]['Return']:
            frontier[i] = (sorted_pfs.iloc[i]['Volatility'], sorted_pfs.iloc[i]['Return'])
    rf_return = 0.015 * hours / (24 * 365)
    max_slope = 0
    max_i = -1
    for i in frontier:
        r, sigma = frontier[i][1], frontier[i][0]
        slope = (r - rf_return) / sigma
        if slope > max_slope:
            max_slope = slope
            max_i = i
    optimal_port = df_simulated.iloc[max_i].drop(['Return', 'Volatility', 'Sharpe Ratio']).to_dict()
    optimal_port = sorted(optimal_port.items(), key=operator.itemgetter(1), reverse=True)
    optimal_portfolio_vol, optimal_portfolio_r = frontier[max_i][0], frontier[max_i][1]

    # Plot tangent line and optimal portfolio
    plt.plot([0.0, 2.0 * optimal_portfolio_vol], [rf_return, 2.0 * max_slope * optimal_portfolio_vol],
             '-', color='gray', lw=2, zorder=10)
    plt.scatter(x=optimal_portfolio_vol, y=optimal_portfolio_r, c='red', marker='o', s=20, zorder=20)

    # Add labels and arrows
    labels = [plt.text(optimal_portfolio_vol, optimal_portfolio_r, "Optimal\nportfolio")]
    for symbol, row in df_coins.iterrows():
        labels.append(plt.text(row['Volatility'], row['Return'], symbol))
    adjust_text(labels, only_move='y', arrowprops=dict(arrowstyle="->", color='black', lw=0.5))

    plt.savefig(filename, bbox_inches='tight')
    return optimal_port


def plotOptimalDoughnut(optimal_portfolio, label_period, color_map, filename="doughnut.png"):
    # Group minions into "Others"
    min_pct = 0.024
    opt = [(coin, pct) for (coin, pct) in optimal_portfolio if pct > min_pct]
    opt.append(("Others", 1 - sum([pct for (coin, pct) in opt])))

    # Create pie chart
    labels = [k for (k, v) in opt]
    sizes = [v for (k, v) in opt]
    n = len(opt)
    plt.figure()
    plt.pie(sizes, labels=labels, colors = matplotlib.cm.get_cmap(color_map)(np.arange(n)/float(n)),
            autopct='%1.1f%%', shadow=False, startangle=90)
    plt.axis('equal') # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.title(f'Optimal portfolio for the past {label_period}')

    # Draw a circle in the center of pie to make it a doughnut
    centre_circle = plt.Circle((0,0),0.75, fc='white',linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Save file
    plt.savefig(filename, bbox_inches = 'tight')

    return opt


def getCoinDetails():
    """
    Return coin details:
    symbol, name, hashtag, handle
    """
    dir = os.path.dirname(os.path.realpath(__file__))
    return pd.read_csv(f'{dir}/coin_names.csv')


def tweetToBestCoin(top_coin_symbol, period):
    """
    Tweets to the Twitter account of the best coin
    """
    df = getCoinDetails()
    coin_handles = {r['symbol']: r['handle'] for i, r in df.iterrows()}
    coin_hashtags = {r['symbol']: r['hashtag'] for i, r in df.iterrows()}

    if top_coin_symbol in coin_handles:
        handle = coin_handles[top_coin_symbol]
        hashtag = coin_hashtags[top_coin_symbol] or f"${top_coin_symbol}"
        if handle:
            status = f"{handle}, {hashtag} is the coin with the best risk-adjusted returns" + \
                     f" of the past {period}"
            tweet(status)


def tweet(status, image=None):
    """
    Tweets plot
    """

    print(f"Tweeting: {status}")

    # Access parameters
    TOKEN = config.twitter['token']
    TOKEN_SECRET = config.twitter['token_secret']

    CONSUMER_KEY = config.twitter['consumer_key']
    CONSUMER_SECRET = config.twitter['consumer_secret']

    t = Twitter(auth=OAuth(TOKEN, TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET))

    if image:
        # Send image along with tweet:
        with open(image, "rb") as imagefile:
            imagedata = imagefile.read()

        params = {"media[]": imagedata, "status": status}
        t.statuses.update_with_media(**params)
    else:
        # Just a status
        t.statuses.update(status=status)


def main():
    print("Starting")
    parser = argparse.ArgumentParser(description='Crunch coinmarketcap data')
    parser.add_argument('--hours', type=int, action="store", dest="hours",
                        help='Number of hours over which to calculate')
    args = parser.parse_args()

    # Settings
    hours = 168 if not args.hours else args.hours
    min_market_cap = 1E9
    num_simulations = 30000

    if hours == 168:
        period = "week"
        color_map = 'coolwarm_r'
        label_period = '7d'
    elif hours == 720:
        period = "30 days"
        color_map = 'plasma'
        label_period = '30d'
    else:
        period = f"{hours} hours"
        color_map = 'viridis'
        label_period = f'{hours}h'

    # Generate efficicent frontier plot
    names = getCoinNames()
    print("Getting prices")
    prices = getPrices(hours, min_market_cap)
    returns = getReturns(prices)
    riskReturnDf = getRiskReturn(returns)
    print("Calculating efficient frontier")
    optimal_weights, optimal_returns, optimal_risks, optimal_portfolios = getOptimalPortfolio(returns)
    print("Simulating portfolios")
    simulatedPortfolios = getSimulatedPortfolios(returns, num_simulations, optimal_portfolios)
    print("Generating efficient frontier")
    plot_name = f"EFTopCoins_{hours}h.png"
    optimal_portfolio = plotSimulatedPortfolios(riskReturnDf, simulatedPortfolios, color_map, label_period, hours, plot_name)

    # Generate optimal portfolio doughnut chart
    print("Baking doughnut")
    doughnut_name = f"Doughnut_{hours}h.png"
    doughnut = plotOptimalDoughnut(optimal_portfolio, label_period, color_map, doughnut_name)
    print(doughnut)

    # Tweet
    print("Tweeting efficient frontier")
    top3 = list(riskReturnDf.iloc[:3].index)
    coin_details = getCoinDetails()
    hashtags = {r['symbol']: r['hashtag'] for i, r in coin_details.iterrows()}
    hashtags.update({k: k for k in set(names.keys()).difference(hashtags.keys())})
    status = f"Best #cryptocurrency risk-adjusted returns in the past {period}:\n" + \
             f"1. {hashtags[top3[0]]} ${top3[0]}\n" + \
             f"2. {hashtags[top3[1]]} ${top3[1]}\n" + \
             f"3. {hashtags[top3[2]]} ${top3[2]}"
    tweet(status, plot_name)

    print("Tweeting to winner")
    tweetToBestCoin(top3[0], period)

    print("Tweeting doughnut")
    status = f"Best #cryptocurrency portfolio in the past {period}"
    if len(doughnut) > 2:
        status += ":\n" + \
             f"{'%3.1f' % (100.0*doughnut[0][1])}% ${doughnut[0][0]}\n" + \
             f"{'%3.1f' % (100.0*doughnut[1][1])}% ${doughnut[1][0]}\n" + \
             f"{'%3.1f' % (100.0*doughnut[2][1])}% ${doughnut[2][0]}\n" + \
             f"{'%3.1f' % (100.0*(1 - doughnut[0][1] - doughnut[1][1] - doughnut[2][1]))}% other coins"
    print(status)
    tweet(status, doughnut_name)



    print("Done.")

if __name__ == "__main__":
    main()
