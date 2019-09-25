"""
Jack Harrison - 16/05/2019 17:50


A program to find the value of a call option given known variables. The program also 
includes a function to return the implied volatility given a known value, which relies on 
a Newton-Raphson process.

"""
#need imports for the functions and cdfs
import numpy as np
import scipy.stats as si



#function to return the value of a call option using the Black-Scholes equation
def optionValue(sigma,T,t,S,K,r):
     d1 = (1 /(sigma * np.sqrt(T - t)))*((np.log(S / K)) + (r + sigma**2 / 2 )*(T-t))
     d2 = d1 - sigma * np.sqrt(T-t)
     PV = K* np.exp(-r*(T-t))
     value = si.norm.cdf(d1, 0.0, 1.0)*S -si.norm.cdf(d2, 0.0, 1.0)*PV
     return value


#implied volatility is the inverse of the options value:
#use iteration to find the value
def impliedVolatility(V,T,t,S,K,r):
    #starting test sigma
    sigma = 0.2
    #get a starting value
    testValue = optionValue(sigma,T,t,S,K,r)
    error = np.abs(testValue - V)
    #recalculate using newton raphson process (error can be changed dependent on accuracy required)
    while error > 0.001:
     d1 = (1 /(sigma * np.sqrt(T - t)))*((np.log(S / K)) + (r + sigma**2 / 2 )*(T-t))
     d2 = d1 - sigma * np.sqrt(T-t)
     fx =  S * si.norm.cdf(d1, 0.0, 1.0) - K*np.exp(-r*T) *si.norm.cdf(d2, 0.0, 1.0) - V
     dfx = (1 / np.sqrt(2 * np.pi))*S*np.sqrt(T)*np.exp(-((d1)**2)/2)
     #adjust the sigma
     sigma = sigma - fx/dfx
     #retest for how close to the desired value it is
     testValue = optionValue(sigma,T,t,S,K,r)
     error = np.abs(testValue - V)
    return sigma
      
#set variables to test the functions with
sigma = 0.9
T = 1
t = 0
S = 25
K = 20
r = 0.05


value = optionValue(sigma,T,t,S,K,r)
print('Value found : ', value)

checkSigma = impliedVolatility(value,T,t,S,K,r)
print('Sigma value found: (should be',sigma,') : ',checkSigma)

"""
This solver uses the Newton Raphson process to find the solution for sigma. It is good as it 
is quick to implement, but has the disadvantage of not finding a solution if you do not start
within a reasonable range of the solution (since the function have a linear derivative for example )
Different iterating functions have different advantages, such as the convergence time,
that depend on the accuracy of the starting position and the form of the function involved. Choosing
the right function for the situation is important.

"""