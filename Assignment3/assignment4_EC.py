# vsr266
# Virendra  S Rajpurohit
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:41:54 2019

@author: vvviren
"""
import numpy as np

class K_Means:
    def __init__(self, k=2, n=5000, threshold=0.01):
        self.k = k
        self.n = n
        self.threshold = threshold
        
    def distance(self, featureA, featureB): 
        diffs = (featureA - featureB)**2 
        return np.sqrt(diffs.sum())

    def fit(self,X):
        self.cds = {}

        for i in range(self.k):
            a = np.random.randint(0,len(X))
            self.cds[i] = X[a]

        for i in range(self.n):
            old_cds = dict(self.cds)
            self.classes = {}
            
            for i in range(self.k):
                self.classes[i] = []

            for xi in X:
                distances = [ self.distance(xi, self.cds[cd]) for cd in self.cds]
                cl = distances.index(min(distances))
                self.classes[cl].append(xi)

            for cl in self.classes:
                self.cds[cl] = np.average(self.classes[cl],axis=0)
            limit_reached = True

            for c in self.cds:
                o_cds = old_cds[c]
                new_cds = self.cds[c]
                if np.sum((new_cds-o_cds)/o_cds*100.0) > self.threshold:
                    limit_reached = False

            if limit_reached is True:
                break

    def predict(self,X):
        distances = [self.distance(X, self.cds[cd]) for cd in self.cds]
        cl = distances.index(min(distances))
        return cl
