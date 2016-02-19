from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]

    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    
    #added molecule id with the corresponding probability values, was not present in original code 
    predicted_probs = [[(index + 1), x[1]] for index, x in enumerate(rf.predict_proba(test))]
    
    #append header name as required for submission on kaggle
    result = [['MoleculeId', 'PredictedProbability']]
    result.extend(predicted_probs)

    #changed format to string ('%s') as opposed to float '%f' in original code
    savetxt('Data/submission.csv', result, delimiter=',', fmt='%s')

if __name__=="__main__":
    main()