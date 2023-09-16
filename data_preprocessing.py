#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:59:01 2023

@author: diego
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import preprocessing
from utils import find_pairs_of_features_based_on_correlation,visualize_data,drop_redundant_features,plot_attributes
from seaborn import heatmap

def label_function(val):
    return f'{val / 100 * len(df):.0f}\n{val:.0f}%'

def compute_statistics_for_nominal_features(dataset,feature,statistical_description,correlation_with_the_target):
    ### The most common value within this attribute
    mode=dataset[feature].mode()[0]
    ### Entropy of a variable is the measure of uncertainty if it is 0 it means that the variable is deterministic
    entropy=stats.entropy(dataset[feature].replace(dataset[feature].unique(),list(range(len(dataset[feature].unique())))))
    ### Chi-square test p-value that is less than or equal to your significance level indicates there is sufficient evidence to conclude that the observed distribution is not the same as the expected distribution
    
    p=chi_square_test(dataset,feature)
    ### Correlation with TARGET
    corr=correlation_with_the_target[feature]
    
    statistical_description[feature]={'Mode':mode,
                                      'Entropy':entropy,
                                      'Chi_square_test_p':p,
                                      'Correlation':corr}
    return statistical_description


def compute_statistics(dataset,feature,statistical_description,correlation_with_the_target):

    min_val= dataset[feature].min()
    max_value= dataset[feature].max()
    median=dataset[feature].median()
    
    #A percentile rank indicates the percentage of data points in a dataset that are less than or equal to a particular value.
    #For example, the 25th percentile  represents the value below which 25% of the data points fall
    P25=dataset[feature].quantile(0.25)
    P50=dataset[feature].quantile(0.5)
    P75=dataset[feature].quantile(0.75)
    
    corr=correlation_with_the_target[feature]
    ### The most common value within this attribute
    mode=dataset[feature].mode()[0]
    
    statistical_description[feature]={'Median':median,
                                      'Min_value':min_val,
                                      'Max_value':max_value,
                                      'Correlation':corr,
                                      'Mode':mode,
                                      'P25':P25,
                                      'P50':P50,
                                      'P75':P75}
    return statistical_description

def compute_statistics_for_ratio_features(dataset,feature,statistical_description):
    mean= dataset[feature].mean()
    std= dataset[feature].std()
    statistical_description[feature]['Mean']=mean
    statistical_description[feature]['Standard_deviation']=std
    return statistical_description
    
def chi_square_test(dataset,attribute,attribute2='TARGET'):
    # Perform the chi-square test
    
    observed = pd.crosstab(dataset[attribute], dataset[attribute2])
    print(f'TARGET VS {attribute}')
    print(observed)
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    
    # Interpret the results
   
    print(f"P-Value: {p}")
    
    # significance level 
    alpha = 0.05
    
    if p < alpha:
        print(f"There is a significant association between {attribute} and TARGET.")
    else:
        print(f"There is no significant association between {attribute} and TARGET")
        
    return p

    

def plot_features_3D(f1,f2,f3,dataset,ag1=None, ag2=None):

    dataset=dataset.sort_values(by=['TARGET'])
    
    fig = plt.figure(figsize=(12,8))
    
    ax = fig.add_subplot(111, projection='3d')
    
    color_dict = {0:'red', 1:'green'}
    
    names = dataset['TARGET'].unique()
    
    for s in names:
        if s == 1:
            l='Can Pay'
        else:
            l='Cannot Pay'
        data = dataset.loc[dataset['TARGET'] == s]
        sc = ax.scatter(data[f1], data[f2], data[f3], s=25,
        c=[color_dict[i] for i in data['TARGET']], marker='x',  label=l)
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    if(ag1 and ag2):
        ax.view_init(ag1, ag2)   
    ax.set_xlabel(f1, rotation=150)
    ax.set_ylabel(f2)
    ax.set_zlabel(f3, rotation=60)
    
    
    plt.show()    


def equi_width_binning(n_bins,df,attribute):
    #get the data values of the original variable
    data=df[attribute]
    #Determine the difference between the maximum and minimum value of the attribute
    diff=max(data)-min(data)
    #Initializes an empthy list to  save the bin names
    bin_names=[]
    #Finds the minimum value of the dataset that will be saved as first element in
    #the list bins which will contain the ranges, which always have to start with the minimum value 
    c=min(data)
    bins=[c]
    #Calculate the width of the bins
    step=diff/n_bins
    #Create the bin names and bin ranges based on the specified number of bines 
    for i in range(n_bins):
        if i!=n_bins:
            bin_names.append(f'Bin {i+1}')
            
        #create the bin ranges
        c=c+step
        bins.append(int(np.floor(c)))
    
    #The last element of the ranges has to be the maximum value of the attribute
    bins[-1]=max(data)
    
    #Use the method cut to divide the dataset in bines of the same width
    df[f'{attribute} (Eq Width n={n_bins})'] = pd.cut(df[attribute],
                                                           bins,
                                                           labels=bin_names,
                                                           include_lowest=True)
    #Sort the elements of the attribute
    df[[attribute, f'{attribute} (Eq Width n={n_bins})']].sort_values(attribute)
    return df

def equi_depth_binning(n_bins,df,attribute):
    #Sort the elements of the attribute
    df=df.sort_values(attribute)
    #Create an empthy list to save the names of the bins
    bin_names=[]
    #Create an alias for each bin
    for i in range(n_bins):
        bin_names.append(f'Bin {i+1}')
    
    try:
        #Divide the dataset in n bins with the same amount of elements
        df[f'{attribute} (Eq depth n={n_bins})'] = pd.qcut(df[attribute],
                                                               q=n_bins,
                                                               labels=bin_names,
                                                               duplicates='drop')
    except:
        #In case that it is not possible to divide the dataset in n bins with the same number of elements,
        #divide the dataset with the maximum amount of bins which will be less than the n bins specified.
        #For example, if the data is [-2,-1,0,1,2,3,3,3,3,3,3,3,3,3] and we want to divide it into 7 bins
        #it will not be possible because each bin should have 2 elements but we cannot separate two 
        #equal numbers in two different bins. That is why the resultant bins will be [-2,-1],[0,1] and 
        #[2,3,3,3,3,3,3,3,3,3]. So, for this case the maximum numer of bins if we want to divide 
        #the dataset into 7 bins will be just 3 bins.
        df[f'{attribute} (Eq depth n={n_bins})'] = pd.qcut(df[attribute],
                                                               q=n_bins,
                                                               duplicates='drop') 
    return df
    
    
def smooth_attribute_based_on_bins(df,attribute,attribute_binned,n_bins,method,crop_hist=-1):
    # Calculate the mean of each bin
    bin_mean = df[[attribute, attribute_binned]].groupby(attribute_binned).mean()

    # then merge on bins
    df1 = df.merge(bin_mean, on = attribute_binned, suffixes=('',f'_{method}_smoothed_n={n_bins}'))
    # sort the values
    df1.sort_values(attribute)
    
    #Get the name of the feature that was smoothed
    smooth_feature_name=df1.columns[-1]
    
    #Plot the histogram of the original variable
    df[attribute].plot(kind='hist', bins=400,title=attribute,align='mid')
    plt.show()

    #Plot the histogram of the variable smoothed
    df1[smooth_feature_name].iloc[:crop_hist].plot(kind='hist', bins=400,title=smooth_feature_name,align='mid')
    plt.show()
    
    #return the dataframe with the smoothed variable
    return df1


df = pd.read_csv('32130_AT2_24686103.csv')
binning=0
normalization=0
discretize=0
binarize=0
if(binning):
    attribute='DAYS_EMPLOYED'
    
    #aa=df[['DAYS_EMPLOYED',f'DAYS_EMPLOYED (Eq depth n={n_bins})',f'DAYS_EMPLOYED_smoothed_n={n_bins}']]
    
    
    #Filter outlier
    #df = df[df['DAYS_EMPLOYED'] <0]
    
    
    n_bins1=100
    df=equi_depth_binning(n_bins1,df,attribute)
    df=smooth_attribute_based_on_bins(df,attribute,f'{attribute} (Eq depth n={n_bins1})',n_bins1,'equi_depth',2500)
    
    
    
    n_bins2=4000
    df=equi_width_binning(n_bins2,df,attribute)
    df=smooth_attribute_based_on_bins(df,attribute,f'{attribute} (Eq Width n={n_bins2})',n_bins2,'equi_width',2500)
    
    
    
    result1=df[[f'{attribute}',
                f'{attribute} (Eq depth n={n_bins1})',
                f'{attribute} (Eq Width n={n_bins2})',
                f'{attribute}_equi_depth_smoothed_n={n_bins1}',
                f'{attribute}_equi_width_smoothed_n={n_bins2}']].sort_values(attribute)
    
    
    
    
    attribute='DAYS_ID_PUBLISH'
    n_bins1=300
    df=equi_depth_binning(n_bins1,df,attribute)
    df=smooth_attribute_based_on_bins(df,attribute,f'{attribute} (Eq depth n={n_bins1})',n_bins1,'equi_depth')
    
    
    
    n_bins2=200
    df=equi_width_binning(n_bins2,df,attribute)
    df=smooth_attribute_based_on_bins(df,attribute,f'{attribute} (Eq Width n={n_bins2})',n_bins2,'equi_width')
    
    
    result2=df[[f'{attribute}',
                f'{attribute} (Eq depth n={n_bins1})',
                f'{attribute} (Eq Width n={n_bins2})',
                f'{attribute}_equi_depth_smoothed_n={n_bins1}',
                f'{attribute}_equi_width_smoothed_n={n_bins2}']].sort_values(attribute)
    
    result1.to_excel('result1.xlsx')
    result2.to_excel('result2.xlsx')
    
elif(normalization):
    attribute='AMT_INCOME_TOTAL'
    data=df[[attribute]]
    
    #########Z-score
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    print(f'The mean of {attribute} is {scaler.mean_[0]}')
    print(f'The variance of {attribute} is {scaler.var_[0]}')
    
    df[f'{attribute}_normalized_Z_Score'] = scaler.transform(data)
    ######## min-max normalization
    
    data_max=data.max()[0]
    data_min=data.min()[0]
    
    normalized_data=(data-data_min)/(data_max-data_min)
    
    df[f'{attribute}_normalized_min_max'] = normalized_data[attribute]
    
    
    
    
    
    result=df[[f'{attribute}_normalized_Z_Score',f'{attribute}_normalized_min_max', f'{attribute}']]
    result.to_excel('result.xlsx')
elif(discretize):
    attribute='DAYS_BIRTH'
    # & bitwise operator
    conditions = [
    ((df[attribute] >= -10000) & (df[attribute] < 0)),
    ((df[attribute] >= -20000) & (df[attribute] < -10000)),
    ((df[attribute] >= -30000) & (df[attribute] < -20000)),
    ]
    discrete_values = ['Children and Young adults', 'Adults', 'Elderlies']
    
    df[f'{attribute}_discretized'] = np.select(conditions, discrete_values)
    
    result=df[[f'{attribute}_discretized', f'{attribute}']]
    result.to_excel('result.xlsx')
elif(binarize):
    attribute='CODE_GENDER'
    df[f'{attribute}_binarized'] = (df[attribute] == 'F').astype(int)
    result=df[[f'{attribute}_binarized', f'{attribute}']]
    result.to_excel('result.xlsx')
else:

    labels=df['TARGET']
    
    #df= df.drop(['TARGET'],axis=1)
    
    
    ## low variance
    df=df.drop(['AMT_REQ_CREDIT_BUREAU_HOUR'],axis=1)
    
    nan_count = df.isna().sum()
    
    #print(nan_count)
    
    df_new=df.dropna()
    
    cols=df_new.columns
    num_cols = df_new._get_numeric_data().columns
    categorical=set(cols)-set(num_cols)
    
    
    
    

    
    ###### ZERO or LOW VARIANCE ATTRIBUTES ##############
    
    df_new=df_new.drop(['FLAG_MOBIL', 'FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_10','FLAG_DOCUMENT_12','FLAG_DOCUMENT_17'], axis=1)
    

    ## irrelevant
    df_new=df_new.drop(['SK_ID_CURR'],axis=1)
    
    #FILTER OUTLIER OF DAYS EMPLOYED
    #df_new = df_new[df_new['DAYS_EMPLOYED'] <0]
    
    for attribute in categorical:
        df_new[attribute].replace(df_new[attribute].unique(),list(range(len(df_new[attribute].unique()))), inplace=True)
        
    
    correlation_matrix=df_new.corr()
    
    
    heatmap(np.array(correlation_matrix),cmap="YlGnBu")
    plt.title('Correlation Matrix')
    plt.show()
    
    ###### FIND HIGH CORRELATED VARIABLES #################
    threshold=0.5
    
    high_correlated_attributes=find_pairs_of_features_based_on_correlation(threshold,
                                                                           correlation_matrix,
                                                                           True)
    ###### FIND LOW CORRELATED VARIABLES #################
    threshold=0.1
    
    low_correlated_attributes=find_pairs_of_features_based_on_correlation(threshold,
                                                                           correlation_matrix,
                                                                           False)
    ###################### plot High Correlated variables
    
    
    visualize_data(high_correlated_attributes,
                   df_new,
                   correlation_matrix,
                   ['can pay','cannot pay'],
                   len(high_correlated_attributes))
    
    ###################### plot Low Correlated variables
    
    
    visualize_data(low_correlated_attributes,
                   df_new,
                   correlation_matrix,
                   ['can pay','cannot pay'],
                   10)
    
    

    
    selected_features,correlation_with_the_target=drop_redundant_features(correlation_matrix,
                                              high_correlated_attributes)
    
    
    A='DAYS_BIRTH'
    B='EXT_SOURCE_2'
    correlation_value=correlation_matrix.loc[A, B]
    plot_attributes(df_new[A],
                   df_new[B],
                    A,
                    B,
                    correlation_value,
                    df_new['TARGET'],
                    ['can pay','cannot pay'],
                    )
    
    
    
    f1=correlation_with_the_target.index[0]
    f2=correlation_with_the_target.index[1]
    f3=correlation_with_the_target.index[2]
    

    plot_features_3D(f1,f2,f3,df_new,40,130)
    
    
    # import seaborn as sns
    
    # pp = sns.pairplot(df_new[selected_features], height=1.8, aspect=1.8,
    #               plot_kws=dict(edgecolor="k", linewidth=0.5),
    #               diag_kind="kde", diag_kws=dict(fill=True),hue="TARGET")

    # fig = pp.fig
    # fig.subplots_adjust(top=0.93, wspace=0.3)
    # t = fig.suptitle('Import data Pairwise Plots', fontsize=14) 
        
    
    

    
    selected_features.append('TARGET')
    dataset=df[selected_features]
    
    nan_count = dataset.isna().sum()
    
    #print(nan_count)
    
    dataset=dataset.dropna()
    
    
    statistical_description={}
    
    
    ordinal_features=['REGION_RATING_CLIENT_W_CITY',]
    ratio_variables=['AMT_REQ_CREDIT_BUREAU_YEAR','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE','AMT_GOODS_PRICE']
    nominal=['FLAG_DOCUMENT_3','REG_CITY_NOT_WORK_CITY','REG_CITY_NOT_LIVE_CITY','FLAG_WORK_PHONE','FLAG_EMAIL']
    
    for f in df_new.columns:
        # for categorical_feature in nominal:
        #     sns.boxplot(x=categorical_feature, y=numerical_feature, data=df_new)
        if  len(df_new[f].unique())>30 and f!='ORGANIZATION_TYPE': 
            df_new[f].plot(kind = 'box').set_title(f)
            plt.grid()
            plt.show()
    
    
    
    
    
    categorical=[]
    for feature in selected_features:
        try:
            if len(dataset[feature].unique())>50:
                bins=int(len(dataset[feature].unique())/10)
            else:
                bins=int(len(dataset[feature].unique()))
                
            if(feature in nominal):
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
                dataset.groupby(feature).size().plot(kind='pie', autopct=label_function, textprops={'fontsize': 8},
                                      colors=['tomato', 'gold', 'skyblue','green'], ax=ax1)
                
                dataset[feature].value_counts().plot(kind='bar',title=feature,ax=ax2)
                ax1.set_ylabel('', size=8)
                ax2.set_ylabel('Frequency', size=8)
                plt.tight_layout()
                plt.show()
            else:
                dataset[feature].plot(kind='hist', bins=bins,title=feature,align='mid')
                plt.show()
            
            if (feature in ordinal_features) or (feature in ratio_variables):
                statistical_description=compute_statistics(dataset,
                                                           feature,
                                                           statistical_description,
                                                           correlation_with_the_target)
            if feature in nominal:
                statistical_description=compute_statistics_for_nominal_features(dataset,
                                                                                feature,
                                                                                statistical_description,
                                                                                correlation_with_the_target)
            if feature in ratio_variables:
                statistical_description=compute_statistics_for_ratio_features(dataset,
                                                                              feature,
                                                                              statistical_description)
    
            
                   
        except:
            #dataset.groupby(['NAME_INCOME_TYPE']).sum().plot(kind='pie', y='NAME_INCOME_TYPE', autopct='%1.0f%%')

            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
            dataset.groupby(feature).size().plot(kind='pie', autopct=label_function, textprops={'fontsize': 8},
                                  colors=['tomato', 'gold', 'skyblue','green'], ax=ax1)
            
            dataset[feature].value_counts().plot(kind='bar',title=feature,ax=ax2)
            ax1.set_ylabel('', size=8)
            ax2.set_ylabel('Frequency', size=8)
            plt.tight_layout()
            plt.show()
            
            
            statistical_description=compute_statistics_for_nominal_features(dataset,
                                                                            feature,
                                                                            statistical_description,
                                                                            correlation_with_the_target)

        

            
            
       
        
       