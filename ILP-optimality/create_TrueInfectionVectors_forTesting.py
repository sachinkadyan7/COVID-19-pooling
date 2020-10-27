#initialize 384*100 matrix, which represents 100 vectors, each with length 384

import numpy as np
import pandas as pd

def createTrueInfVectors(num_samples, num_trials, infectionRate):
    rows = num_samples #384
    cols = num_trials #100
    f = infectionRate
    
    InfVectors = []
    for row in range(rows):
        InfVectors += [[7]*cols] #Weird initial number to make sure every cell gets touched by next loop
    
    for r in range(rows):
        for c in range(cols):
            InfVectors[r][c] = int(np.random.binomial(1, f, 1))  #binomial(n,p,trials)
    
    
    final = np.array(InfVectors) #convert to numpy array (So it can be used as input to Sachin's solver)
    print(final)
    print(final.shape)
    DF = pd.DataFrame(final)  #convert to pandas dataframe
    DF.to_csv("data4.csv")    #convert to csv


createTrueInfVectors(384,100,2/384) #Can change these numbers


