# Tweet cryptocurrency efficient frontier
---
**Function:**
Run a Twitter bot [@CryptoWatchBot](https://twitter.com/CryptoWatchBot), which tweets simulated efficient frontier of cryptocurrencies and identifies top 3 currencies with the best risk-adjusted returns.  


## tweetEfficientFrontier.py
---
**Procedures:**
* Access cryptocurrency data stored on Amazon RDS MySQL database
* Calculate the daily returns, volatility, Sharpe Ratio for coins and determine optimal portfolio
* Perform Monte Carlo simulation for various portfolios with randomly assigned coin weights
* Plot return-volatility of the coins and the simulated portfolios
* Tweet plot to Crypto Watch Bot and tweet to the Twitter account of the cryptocurrency with the best risk-adjusted returns.


This code is run by AWS EC2:
* Once daily to show 7d-average return-volatility

![](https://github.com/nalin-t/tweet-efficient-frontier/blob/master/Weekly.png)

* Once every 4 hours to show 1d-average return-volatility

![](https://github.com/nalin-t/tweet-efficient-frontier/blob/master/Daily.png)



### References:
---
* Starke, T.; Edwards, D.; Wiecki, T. (2015, March) *The Efficient Frontier: Markowitz portfolio optimization in Python*. Retrieved from https://blog.quantopian.com/markowitz-portfolio-optimization-2/ (2018, January)
* Boyd, S.; Vandenberghe, L. *Convex Optimization*.
Cambridge University Press, 2009
