import time
import numpy as np
from dataEntropy import *
from deepSets import *

def main():
    print("Generating data...")
    train = generate_data(1000)
    validation = generate_data(1000)
    test = generate_data(1000)
    #Train the model
    print("Training model...")
    weights = model(train,validation)
    #Predict for test set
    print("Predicting for test set...")
    prediction = predict(test, weights)
    #Evaluate predictions 
    eval_preds(prediction, test.labels)
if __name__ == "__main__":
	main()