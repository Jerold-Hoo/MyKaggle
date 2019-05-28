# -*- coding: utf-8 -*-

# @File       : titanic.py
# @Date       : 2019-05-09
# @Author     : Jerold
# @Description: for kaggle titanic competition

from numpy import *

def Mod(sep, NorP, X):
    def M(x):
        if x < sep:
            return 1.0 * NorP
        else:
            return -1.0 * NorP
    Yp = array(list(map(M,X)))
    return Yp

def fMod(Mods, X, Y):
    Yp = array([0]*10,dtype='float64')
    for mod in Mods:
        Yp += mod[2] * Mod(mod[0],mod[1],X)
    return sum(where((Yp*Y)<0,1,0))

def get_e(Y, Yp, W):
    error = 0
    for i in range(len(Y)):
        if Y[i] != Yp[i]:
            error += W[i]
    return error

def get_bestSep(X, Y, W):
    minres = 1
    minidx = -1
    minNP = 1
    minYp = []
    for i in range(10):
        for j in [1,-1]:
            Yp = Mod(i + 0.5, j, X)
            e = get_e(Y, Yp, W)
            if e < minres:
                minres = e
                minidx = i + 0.5
                minNP = j
                minYp = Yp.copy()
    return minidx, minNP, minres, minYp

def test_adb():
    import math
    X = arange(10)
    Y = array([1,1,1,-1,-1,-1,1,1,1,-1],dtype='float64')
    W = array([0.1]*10)
    mods = []
    for KK in range(3):
        idx,NP,e,Yp = get_bestSep(X,Y,W)
        a = 0.5 * math.log((1-e)/e,math.e)
        # update W
        for i in range(len(X)):
            W[i] = W[i] * math.exp(-1 * a * Y[i] * Yp[i])
        Z = sum(W)
        for i in range(len(W)):
            W[i] /= Z
        print(KK+1,':',idx,NP,e,a,W)
        mods.append([idx,NP,a])
        print('eNum:',fMod(mods,X,Y),mods)
        print('-------------------------------')

if __name__ == '__main__':
    test_adb()
