'''
 Author      : Shiva Agrawal
 Date        : 06.09.2018
 Version     : 1.0
 Description : Regression model development using machine learning algorithm for Boston Housing Price Dataset.
			   The model is used to predict the price of the house for the given new sample of features.
'''

'''
Used Dataset - Boston Housing Price (housing.data.csv)

13 features
1 output (PRICE)
506 samples

Attribute Information:

    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                 by town
    13. LSTAT    % lower status of the population
    14. HOUSE_PRICE     Median value of owner-occupied homes in $1000's  (the name is changed from original)


'''

import pandas as pd
import matplotlib.pyplot as pyplt
import numpy as np
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pickle


'''
Function: regressionModel
Description: develop the regression model
@param: CsvFileName - string varible for dataset csv file
@return: -
'''
def regressionModel(CsvFileName):

    # step 1: Import the Dataset from CSV to python
    #--------------------------------------------------

    header_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                    'HOUSE_PRICE']

    ML_data = pd.read_csv(CsvFileName, sep='\s+',
                          names=header_names)  # sep='\s+' is used to have multi-space separated values
    print(ML_data)


    # step 2: Separate input and output data for ML
    #---------------------------------------------------

    ML_data_array = ML_data.values
    ML_data_input = ML_data_array[:, 0:13]  # all rows and columns from index 0 to 12 (all 13 input features)
    ML_data_output = ML_data_array[:, 13]  # all rows and column index 13 (last column - House_price (output))


    # step 3: Desciptive analysis of the dataset
    #---------------------------------------------------
    print(ML_data.shape)  # dimensions of the dataset (rows, cols)
    print(ML_data.dtypes) # dataypes of all the features and outcome
    print(ML_data.head(20))  # print first 20 samples (just for knowing the data)

    dataStatistics = ML_data.describe() # find mean, std dev, min value, max value, 25th, 50th, 75th percentile of each feature
    print(dataStatistics)
    pd.set_option('precision', 2)
    print(ML_data.corr(method = 'pearson')) # correlation matrix for all the features


    # step 4: Data anaylysis using different plots
    #---------------------------------------------------
    ML_data.hist(sharex = False, sharey = False, xlabelsize = 1, ylabelsize = 1)  # histogram plots
    ML_data.plot(kind = 'density', subplots = True, layout = (4,4), sharex = False, sharey = False) # density plots
    ML_data.plot(kind = 'box', subplots = True, layout = (4,4), sharex = False, sharey = False) # box and whisker plots

    # correlation matrix
    fig = pyplt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(ML_data.corr(), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    ticks = np.arange(0, 14, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(header_names)
    ax.set_yticklabels(header_names)

    # step 5: Data preprocessing - Standardize the input features
    #------------------------------------------------------------
    scalar = StandardScaler().fit(ML_data_input)
    ML_data_input = scalar.transform(ML_data_input)


    # step 6: separate train (70 %) and validation (30 %) dataset
    # -----------------------------------------------------------
    validation_size = 0.3
    seed  = 7
    [X_train, X_validation, Y_train, Y_validation] = train_test_split(ML_data_input,ML_data_output,
                                                                      test_size=validation_size, random_state=seed)

    # step 7: As at this moment, it is difficult to predict the right choice of algorithm, several regresssion algorithms are
    #         are used to develop different models and then all are compared
    # 10 fold cross validation and Rsquare metric is selected for accuracy check

    # six different algorithms tried for regression
    print('\n')
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('EN', ElasticNet()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))
    models.append(('SVM', SVR()))

    # evaluate each model
    all_cv_results = []
    all_names = []
    k_folds = 10

    for name, model in models:
        kfold = KFold(n_splits=k_folds,random_state=seed)
        cv_results = cross_val_score(model, X_train, Y_train,cv = kfold,scoring='r2')
        all_cv_results.append(cv_results)
        all_names.append(name)
        print(name + ' : ' + str(cv_results.mean()) + ' (' + str(cv_results.std()) + ')')

    # Compare Algorithms
    fig = pyplt.figure()
    fig.suptitle(' Algorithm Comparison ')
    ax = fig.add_subplot(111)
    pyplt.boxplot(all_cv_results)
    ax.set_xticklabels(all_names)


    '''
    
    LR     : 0.7223775457359257 (0.08511067020729662)
    LASSO  : 0.6697772207483847 (0.06454538188399106)
    EN     : 0.6517399040563703 (0.06214458636866283)
    KNN    : 0.7272884836121088 (0.10546843781370058)
    CART   : 0.661835475725195 (0.19080425526001163)
    SVM    : 0.6184153729995604 (0.10975744626878726)
    
    Hence from the above results, KNN and LR fits best
    
    '''

    # Step 8: Validating the two selected models with validation dataset to find best fit

    print('-----KNN model validation--------')
    knn_model_tuple = models[3]
    knn_model = knn_model_tuple[1]
    print(knn_model)
    knn_model.fit(X_train, Y_train)
    print(knn_model.score(X_validation,Y_validation))

    print('-----LR model validation--------')
    lr_model_tuple = models[0]
    lr_model = lr_model_tuple[1]
    print(lr_model)
    lr_model.fit(X_train, Y_train)
    print(lr_model.score(X_validation,Y_validation))

    '''
    scores:
    KNN:  0.7686788821385517
    LR: 0.6508417720329543
    '''

    # step 9: save the final model (KNN algorithm) using pickel package

    model_filename = 'KNN_model.sav'
    pickle.dump(knn_model, open(model_filename, 'wb'))


    # show() function is kept last so that complete funcion can run and then plots are available
    pyplt.show()
# end of function




