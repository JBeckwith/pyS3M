#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def LRtest(datain):
    """
    Likelihood ratio test for finding one change point

    Returns:
    tuple: (Cpt,  LM) where:
        Cpt is the change point index (0-based)
        LM is the maximum likelihood ratio value
    """
    nd = len(datain)  # nd=length(datain);
    S = np.zeros(nd)  # S=zeros(1,nd);
    SS = 0  # SS=0;
    L = np.zeros(nd)  # L=zeros(1,nd);

    for k in range(nd):  # for k=1:nd
        SS += datain[k]  # SS=SS+datain(k);
        S[k] = SS  # S(k)=SS;
    LM = 0  # LM=0;
    Cpt = 0  # Cpt=0;

    for k in range(nd - 1):  # for k=1:(nd-1)
        if (
            S[k] > 0 and SS - S[k] > 0
        ):  # if (S(k) > 0 & SS-S(k) > 0) %this keeps from taking log(0)
            L[k] = (
                S[k] * np.log(S[k] / (k + 1))  # L(k)=S(k)*log(S(k)/k)+
                + (SS - S[k])
                * np.log(
                    (SS - S[k]) / (nd - (k + 1))
                )  # (SS-S(k))*log((SS-S(k))/(nd-k))-
                - SS * np.log(SS / nd)
            )  # SS*log(SS/nd);

            if LM < L[k]:  # if(LM<L(k))
                LM = L[k]  # LM=L(k);
                Cpt = k  # Cpt=k; %0-based index

    return Cpt, LM


"""
Find change points
Yan Jiang 09/09/07
Algorithm based on Watkins and Yang, J. Phys. Chem. B, Vol. 109, No. 1,
617-628(2005) and Boudjellaba et al, Commun. Statist. - Theory Meth., 30(3),407-434(2001)
Edited on 10/22/07 by Yan to include comments and make the change points storing vector and the name of several variables more reasonable.
Edited on 2/2/08 by Yan to include the Poisson fitting for the stepsize distribution and the visualization of the found events.
"""
# fprintf('\n finding changepoints   ')
# global L;
# Read in the raw data and choose part of it to be analyzed.
# Specify the data you want to analyze.  It should be in a 1D vector/numpy array called "binnedC".
binnedC = np.array([...])  # insert data
# The main output containing the found changepoints and levels will be in the vector "filtered".
nData = len(binnedC)  # nData=length(binnedC);
data = binnedC[int(nData * 0) : int(nData * 1)]  # data=binnedC((nData*0+1):(nData*1));
# Initialize vector ChangePnt, in which each 1 indicate a change point.
ChangePnt = np.zeros(nData, dtype=int)  # ChangePnt=zeros(1,nData);
ChangePnt[0] = 1  # ChangePnt(1)=1;
ChangePnt[-1] = 1  # ChangePnt(nData)=1;
# Find the change points
# WinSize is the smallest segment that you want to look for change point in it.
WinSize = 10  # WinSize=10;
crntChange = 0  # crntChange=1;
nxtChange = nData - 1  # nxtChange=nData;
nCPnt = 2  # nCPnt=2;
while (
    crntChange < nData - 1
):  # while(crntChange<nData) % when not every change points are found
    if (
        nxtChange - crntChange
    ) > WinSize:  # if((nxtChange-crntChange)>WinSize) % if the current data segment is long enough, try to find a change point inside this segment
        datain = data[
            crntChange + 1 : nxtChange + 1
        ]  # datain=data((crntChange+1):nxtChange);
        Cpt, LM = LRtest(datain)  # [Cpt LM]=LRtest2(datain);
        # If the change point found just now is real, update the ChangePnt vector and cut the data segment at this change point.
        # Otherwise go on the the next segment of data.
        # LM is a threshold. For now you have to try.
        if LM > 400:  # if(LM>400)
            global_Cpt = crntChange + Cpt + 1  # datalength(nCPnt)=length(datain);
            ChangePnt[global_Cpt] = 1  # ChangePnt(crntChange+Cpt)=1;
            nCPnt += 1  # nCPnt=nCPnt+1;
            nxtChange = global_Cpt  # nxtChange=crntChange+Cpt;
        else:  # else
            crntChange = nxtChange  # crntChange=nxtChange;
            if crntChange < nData - 1:  # if(crntChange<nData)
                nxtChange += 1  # nxtChange=nxtChange+1;
                while (
                    nxtChange < nData and ChangePnt[nxtChange] == 0
                ):  # while(ChangePnt(nxtChange)==0)
                    nxtChange += 1  # nxtChange=nxtChange+1;
    else:  # else % if the segment is too short, go on to the next segment of data.
        crntChange = nxtChange  # crntChange=nxtChange;
        if crntChange < nData - 1:  # if(crntChange<nData)
            nxtChange += 1  # nxtChange=nxtChange+1;
            while (
                nxtChange < nData and ChangePnt[nxtChange] == 0
            ):  # while(ChangePnt(nxtChange)==0)
                nxtChange += 1  # nxtChange=nxtChange+1;
# Organize the position of the change points to a new vector tCPnt.
tCPnt = []  # tCPnt=zeros(1,nCPnt);
# i=1;
t = 0  # t=1;
while t < nData:  # while(t<nData)
    while t < nData and ChangePnt[t] == 0:  # while(ChangePnt(t)==0)
        t += 1  # t=t+1;
    if t < nData:  # tCPnt(i)=t;
        tCPnt.append(t)  # t=t+1;
        t += 1  # i=i+1;
# Below is one set of Data analysis. Use the value in the 'if' sentence to choose.
# Visualize the result by drawing the raw data and the stepped data in a same figure.
filtered = np.zeros(nData)  # filtered=zeros(1,nData);
Ilevel = np.zeros(len(tCPnt) - 1)
for j in range(len(tCPnt) - 1):  # for j=1:(nCPnt-1)
    start = tCPnt[
        j
    ]  # Ilevel(j)=sum(data((tCPnt(j)+1):tCPnt(j+1)))/(tCPnt(j+1)-tCPnt(j));
    end = tCPnt[j + 1]  # for i=(tCPnt(j)+1):(tCPnt(j+1))
    segment = data[start:end]  # filtered(i)=Ilevel(j);
    Ilevel[j] = np.sum(segment) / len(segment)
    filtered[start:end] = Ilevel[j]
plt.figure()  # figure()
plt.plot(binnedC, label="Original Data")  # plot(binnedC)
plt.plot(filtered, "r", label="Step Filtered")  # hold on
plt.legend()  # plot(filtered, 'r')
plt.show()
