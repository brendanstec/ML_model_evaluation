#Helpful model evaluation functions for scikit-learn (will be adding more)



def cross_validate(X,y,k):
    '''
    B. Stec - version 1.1 - February 2018
    Function that runs k-fold cross validation on X and y. X can either be a num rows-by-num features numpy array or a 
    num rows-by-num features pandas DataFrame. y can be either a num row-by-1 numpy array or a num row-by-1 DataFrame.
    k is a list of cross-validation sizes (e.g. k can be [10], [2,5,10], etc.).
    '''
    #return a list of average mse by different k values
    average_mses = list()
    for n in k:
        kf = KFold(n_splits=n,random_state=101)
        #create a blank list for mse's per fold
        mean_squared_errors = list()
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
            lm.fit(X_train,y_train)
            #make predictions
            pred = lm.predict(X_test)
            #calculate mean squared error
            mse=mean_squared_error(y_test,pred)
            #add each fold mse to list
            mean_squared_errors.append(mse)
        #find the average mse per fold
        average_mse = np.mean(mean_squared_errors)
        #add average mse per fold to list of other averages
        average_mses.append(average_mse)
    #return list
    return average_mses
    
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