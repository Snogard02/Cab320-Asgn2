
'''

Some partially defined functions for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls different functions to perform the required tasks.

'''
import numpy
import random
from sklearn import neighbors
from sklearn import svm
from sklearn import naive_bayes
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    #raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    ''' 
    #numpy.set_printoptions(threshold=numpy.inf) #Shows entire array when printing

    ##	Loads data required for training
    x = numpy.genfromtxt(dataset_path,dtype = numpy.float_,delimiter = ',',usecols = range(2,31)) 

    ##	Load class label from file and replace 'M' with 1 and 'B' with 0
    y = numpy.genfromtxt(dataset_path,dtype = numpy.str,delimiter = ',',usecols = (1,)) 
    y[y=='M'] = 1
    y[y=='B'] = 0
    y = y.astype(numpy.int)

    #Return arrays
    return x, y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def split_dataset(x_input, y_input):
    '''
    Split the data set into training, validation and testing
    @param
    x_input: A array to be split
    y_input: The class label of the array to be split

    @return
    X_training: Data to be used for training
    y_training: Class label for the data to be used for training
    X_test: Data to be used for testing
    y_test: Class label for the data to be used for testing
    '''
    #Note: Need to split the data set sepearatly from classifiers so all the classifier train,
    # and test against the same data and so the perfomance can be compared.

    n = x_input.shape[0]
    split = int(n*0.8)
    p = numpy.random.permutation(n)
    X, y = x_input[p], y_input[p]

    X_training, X_test = X[:split], X[split:]
    y_training, y_test = y[:split], y[split:]
    return X_training, y_training, X_test, y_test
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def test_classifier(X_test, y_test, clf_nn, clf_svm, clf_nb):
    nn_score = clf_nn.score(X_test,y_test)
    svm_score = clf_svm.score(X_test,y_test)
    nb_score = clf_nb.score(X_test,y_test)
    #Saves to file now, no need to print
    #print('Test accuracy for nn ­ > ', nn_score)
    #print('test accuracy for svm ­> ', svm_score)
    #print('test accuracy for nb  ­> ', nb_score)
    return nn_score, svm_score, nb_score
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    clf = naive_bayes.GaussianNB()
    clf.fit(X_training,y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    ''' 
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_training,y_training)
    return clf
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    clf = svm.SVC(kernel='linear')
    clf.fit(X_training,y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    max_tests = 10
    score_array = [0,0,0]
    for i in range (0,max_tests):
        data = prepare_dataset("./medical_records.data")
        [X_training, y_training, X_test, y_test] = split_dataset(data[0],data[1])
        clf_nn = build_NN_classifier(X_training, y_training)
        clf_svm = build_SVM_classifier(X_training, y_training)
        clf_nb = build_NB_classifier(X_training,y_training)
        score_array = numpy.vstack((score_array,[test_classifier(X_test, y_test, clf_nn, clf_svm, clf_nb)]))
#Save a array of the test results to a file called 'data'
#Results saved in the format of [Nearest Neighbour,Support Machine Vector, Naive Bayes]
    numpy.savetxt('data',score_array, fmt='%1.5f')
    print('Done')
