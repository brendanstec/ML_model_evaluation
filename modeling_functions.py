#Helpful model evaluation functions for scikit-learn (will be adding more)



def my_cross_validate(X,y,k,mod_type,scoring_type):
    '''
    B. Stec - version 1.2 - February 2018
    Function that runs k-fold cross validation on X and y. X can either be a num rows-by-num features numpy array or a 
    num rows-by-num features pandas DataFrame. y can be either a num row-by-1 numpy array or a num row-by-1 DataFrame.
    k is a list of cross-validation set sizes (e.g. k can be [10], [2,5,10], etc.). Improved to include many different
    estimators and different scoring schemes beyond MSE. 
    
    regression metrics: explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error,
    median_absolute_error, r2_score
    
    classification metrics (ones supported): accuracy_score, classification_report, confusion_matrix, f1_score, 
    hamming_loss, jaccard_similarity_score, log_loss, matthews_corrcoef, precision_score, recall_score, zero_one_loss
    '''
 
    #return a list of average metrics by different k values
    average_errors = list()
    for n in k:
        kf = KFold(n_splits=n,random_state=101)
        #create a blank list for mse's per fold
        errors = list()
        for train_index, test_index in kf.split(X):
            #if the data is a numpy array use normal indexing
            if type(X).__name__ == 'ndarray':
            #type(X).__name__ or type(X).__module__ are helpful here.
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            #if the data is a pandas DataFrame use iloc indexing
            else:
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            #fit the model
            mod_type.fit(X_train,y_train)
            #make predictions
            pred = mod_type.predict(X_test)
            #calculate mean squared error
            error=scoring_type(y_test,pred)
            #add each fold mse to list
            errors.append(error)
        #find the average mse per fold
        average_error = np.mean(errors)
        #add average mse per fold to list of other averages
        average_errors.append(average_error)
    #return list
    return average_errors
    
####

def mse_polynomial(X,y,degrees):
    '''
    B. Stec - version 1.1 - February 2018
    Function that calculates the mean squared error (via 10-fold cross-validation) of a linear model with of varying
    degrees (aka flexibility). X and y are num rows-by-num features and num rows-by-1 DataFrames or numpy arrays. 
    degrees is a list of integer corresponding to different flexibilities (e.g. [1,2,3] or [2])
    '''
    
    mse_results = list()
    for n in degrees:
        poly = pp.PolynomialFeatures(degree=n, include_bias=False)
        lm.fit(poly.fit_transform(X),y)
        #pred = lm.predict(poly.fit_transform(X))
        #estimate test mse via 10-fold cross validation (can change to k if needed)
        mse = cross_validate(poly.fit_transform(X),y,[10])
        mse_results.append(mse[0])
    
    return mse_results
