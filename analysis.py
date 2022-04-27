#!/usr/bin/env python
# coding: utf-8

#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
# import os
import seaborn as sns
# from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import qqplot
import plotly.express as px
# from scipy.stats.stats import pearsonr
from pandas.plotting import scatter_matrix
from matplotlib import axis, rcParams
# from sklearn.neighbors import KernelDensity
# from numpy import array, linspace
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-notebook')
# ### Dataset Description

#%%


play_store_app_df=pd.read_csv("data/Google-Playstore.csv")


#%%


print("Shape of the dataset: ",play_store_app_df.shape)


#%%


print("1st 5 rows of the dataset:\n",play_store_app_df.head(5))


# Dataset Description:
# The dataset is from kaggle(https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps).  This dataset contain approximately 2.3 million google app store data.
# It contains 24 columns. In which 19 columns are categorical and 5 columns are numerical.
# 
# Numerical columns list:
# - Rating
# - Rating Count
# - Minimum Installs
# - Maximum Installs
# - Price 
# 
# Categorical Columns List:
# - Object
#     - App Name
#     - App Id
#     - Category
#     - Installs              
#     - Currency              
#     - Size                  
#     - Minimum Android
#     - Developer Id
#     - Developer Website
#     - Developer Email
#     - Released  
#     - Last Updated
#     - Content Rating
#     - Privacy Policy
#     - Scraped Time
# - Bool
#     - Free                    
#     - Ad Supported
#     - In App Purchases
#     - Editors Choice

#%%


print("Dataset Columns and Data types:")
print(play_store_app_df.info())


#%%

print("Dataset Column Statistics:")
print(play_store_app_df.describe())


# # Pre-Processing Dataset
# ## Statistics

#%%

print(f"Number of rows: {play_store_app_df.shape[0]}")
print(f"Number of columns: {play_store_app_df.shape[1]}")

print(f"Total Cells in the dataset is : {play_store_app_df.shape[0]*play_store_app_df.shape[1]}")
total_number_of_missing_data=play_store_app_df.isnull().sum().sum()
print(f"Total Number of missing data: {total_number_of_missing_data}")
percentage=(total_number_of_missing_data/(play_store_app_df.shape[0]*play_store_app_df.shape[1]))*100
print(f"Percentage of Missing values in play store dataset: {percentage:.2f}%")


#%%


column_wise_missing_values=play_store_app_df.isnull().sum()
print("Missing values in columns: \n",column_wise_missing_values)


print("From the upper section we can see that 'Developer Website' and 'Privacy Policy'\
 both has lot of missing value. If we delete this row then almost 1.1 million data row\
 will be missing. So our plan is to drop these columns from the dataset.")
#%%


Column_wise_missing_values_percentage=column_wise_missing_values*100/len(play_store_app_df)
print(f"Missing value percentages: \n{Column_wise_missing_values_percentage}")


#%%

play_store_app_df=play_store_app_df.drop(['Developer Website','Privacy Policy'],axis=1)
print("Dataset after dropping Developer Website and Privacy Policy:\n",play_store_app_df.head(5))


# Updated Dataset statistics

#%%

print(f"Number of rows: {play_store_app_df.shape[0]}")
print(f"Number of columns: {play_store_app_df.shape[1]}")
print(f"Total Cells in the dataset is : {play_store_app_df.shape[0]*play_store_app_df.shape[1]}")
total_number_of_missing_data=play_store_app_df.isnull().sum().sum()
print(f"Total Number of missing data: {total_number_of_missing_data}")
percentage=(total_number_of_missing_data/(play_store_app_df.shape[0]*play_store_app_df.shape[1]))*100
print(f"Percentage of Missing values in play store dataset: {percentage:.2f}%")

#%%


play_store_app_df.isnull().sum()


print("For removing missing values from the dataset, for the categorical data I\
 will remove the row from the dataset and for the numerical data I will replace\
 it with average value.")

#%%

# Imputing missing values
play_store_app_df['Rating'].fillna((play_store_app_df['Rating'].mean()), inplace=True)
play_store_app_df['Rating Count'].fillna((play_store_app_df['Rating Count'].mean()), inplace=True)
play_store_app_df['Minimum Installs'].fillna((play_store_app_df['Minimum Installs'].mean()), inplace=True)


#%%

# Drop all rows with missing values
play_store_app_df=play_store_app_df.dropna()


#%%

print("Final Dataset: \n",play_store_app_df.isnull().sum())

#%%

# Final dataset
print(f"Number of rows: {play_store_app_df.shape[0]}")
print(f"Number of columns: {play_store_app_df.shape[1]}")
print(f"Total Cells in the dataset is : {play_store_app_df.shape[0]*play_store_app_df.shape[1]}")
total_number_of_missing_data=play_store_app_df.isnull().sum().sum()
print(f"Total Number of missing data: {total_number_of_missing_data}")
percentage=(total_number_of_missing_data/(play_store_app_df.shape[0]*play_store_app_df.shape[1]))*100
print(f"Percentage of Missing values in play store dataset: {percentage:.2f}%")


# # Outlier detection and removal
# 
# Normally outliers can be found in the numerical columns. 
# In out dataset there is 5 numerical dataset which are 
# - Rating,
# - Rating Count ,
# - Minimum Installs, 
# - Maximum Installs,
# - Price.
#  
# At First We will visualize the this 5 columns so that we can check that if there is any outliers.
# 

#%%

# Rating
plt.figure()
sns.boxplot(play_store_app_df['Rating'])
plt.title("Boxplot of App Rating")
plt.show()

#%%

# Rating count
plt.figure()
sns.boxplot(play_store_app_df['Rating Count'])
plt.title("Boxplot of App Rating Count")
plt.show()

#%%

plt.figure()
sns.boxplot(play_store_app_df['Minimum Installs'])
plt.title("Boxplot of App Minimum Installs")
plt.show()

#%%
plt.figure()
sns.boxplot(play_store_app_df['Maximum Installs'])
plt.title("Boxplot of App Maximum Installs")
plt.show()
#%%

plt.figure()
sns.boxplot(play_store_app_df['Price'])
plt.title("Boxplot of App Price")
plt.show()

#%%

plt.figure()
sns.scatterplot(data=play_store_app_df, x='Rating', y='Rating Count')
plt.title("Rating Count Vs Rating")
plt.grid()
plt.show()

#%%

# Scatterplot
plt.figure()
sns.scatterplot(data=play_store_app_df, x='Maximum Installs', y='Minimum Installs')
plt.title("Minimum Installs Vs Maximum Installs")
plt.grid()
plt.show()


#%%


# print(play_store_app_df.shape)

# From Upper Graphs you can see that some of the numerical columns has outlier.
# Rating Count, Maximum Installs and Minimum Installs have some data which are
# very far from the most of the data. So For getting the better visualization we
# need to remove those data from the dataset and check the outliers again.

#%%


def outliers_detection(play_df,columns):
    Q1=play_df[columns].quantile(0.25)
    Q3=play_df[columns].quantile(0.75)
    IQR=Q3-Q1
    Lower_limit=Q1-1.5*IQR
    Upper_limit=Q3+1.5*IQR
    ls=play_df.index[(play_df[columns]<Lower_limit) | (play_df[columns]>Upper_limit)]
    return ls


#%%


def remove_outliers(play_df,detected_index):
    detected_index=sorted(set(detected_index))
    play_df=play_df.drop(detected_index)
    return play_df


#%%

outlier_index_list=[]
# outlier_index_list.extend(outliers_detection(play_store_app_df,'Minimum Installs'))
for column in ['Minimum Installs','Rating Count','Maximum Installs']:
    outlier_index_list.extend(outliers_detection(play_store_app_df,column))
    
    
#%%

print("Total Outliers: ",len(outlier_index_list))  


#%%

# Remove outliers
play_store_app_df_removed_outlier=remove_outliers(play_store_app_df,outlier_index_list)
print("After removing outliers shape of the dataset:",play_store_app_df_removed_outlier.shape)


#%%


f,axes=plt.subplots(1,2)
sns.boxplot(play_store_app_df_removed_outlier['Minimum Installs'],ax=axes[1])
axes[0].set_title("Before removing outlier")
sns.boxplot(play_store_app_df['Minimum Installs'],ax=axes[0])
axes[1].set_title("After removing outlier")
plt.suptitle("Minimum Installs")
plt.tight_layout()
plt.show()

#%%


f,axes=plt.subplots(1,2)
sns.boxplot(play_store_app_df['Maximum Installs'],ax=axes[0])
axes[0].set_title("Before removing outlier")
sns.boxplot(play_store_app_df_removed_outlier['Maximum Installs'],ax=axes[1])
axes[1].set_title("After removing outlier")
plt.suptitle("Maximum Installs")
plt.tight_layout()
plt.show()

#%%

f,axes=plt.subplots(1,2)
sns.boxplot(play_store_app_df['Rating Count'],ax=axes[0])
axes[0].set_title("Before removing outlier")
sns.boxplot(play_store_app_df_removed_outlier['Rating Count'],ax=axes[1])
axes[1].set_title("After removing outlier")
plt.suptitle("Rating Count")
plt.tight_layout()
plt.show()



#%%

print("Shape of dataset after removing outliers: ",play_store_app_df_removed_outlier.shape)


# # Principal Component Analysis (PCA)


# We are using only those columns which are numerical for PCA . 
# So total 5 features have used in PCA. 
# Features are :Rating,  Rating Count,Minimum Installs,Maximum Installs,Price
# 

#%%


numerical_columns=play_store_app_df[['Rating','Rating Count','Minimum Installs','Maximum Installs','Price']]
Converted_dataframe=StandardScaler().fit_transform(numerical_columns)
# Converted_dataframe


#%%


Singular_values=np.matmul(numerical_columns[numerical_columns.columns[1:]].values.T,numerical_columns[numerical_columns.columns[1:]].values)
Condition_number=np.linalg.svd(Singular_values)
print(f'Singular Values:\n{Singular_values}')
print(f'Condition Number:\n{np.linalg.cond(numerical_columns[numerical_columns.columns[1:]].values)}')


#%%


originals=numerical_columns[numerical_columns.columns[:]]
plt.figure(figsize=(6,4))
sns.heatmap(originals.corr(),annot=True,linewidths=1)
plt.suptitle("Correlation Coefficient between features-original feature space")
plt.show()


#%%


pca=PCA(n_components='mle',svd_solver='full')
pca.fit(Converted_dataframe)
playstore_pca=pca.transform(Converted_dataframe)

print("Original Dimension:",Converted_dataframe.shape)
print("Transformed dimension:", playstore_pca.shape)
print("Explained variance ration:\n",pca.explained_variance_ratio_)
x=np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1,1)
plt.plot(x,np.cumsum(pca.explained_variance_ratio_))
plt.xticks(x)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance vs Number of Components")
plt.grid()
plt.show()


#%%


print('Eventhough PCA already reduced the features from from 5 to 4,one more feature can be removed as with 3 features we are getting almost 90%explained variance. ')
# Making new reduced feature space with 3 components
pcaf=PCA(n_components=3,svd_solver='full')
pcaf.fit(Converted_dataframe)
PlayStore_pcaf=pcaf.transform(Converted_dataframe)

print("Original Dimension:",Converted_dataframe.shape)
print("Transformed Dimension:",PlayStore_pcaf.shape)
print("Explained variance ratio:\n",pcaf.explained_variance_ratio_)


#%%


x=np.arange(1,len(np.cumsum(pcaf.explained_variance_ratio_))+1,1)
plt.plot(x,np.cumsum(pcaf.explained_variance_ratio_))
plt.xticks(x)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance Vs Number of Components")
plt.suptitle("Reduced Feature Space")
plt.grid()
plt.show()


#%%


PlayStore_pcaf_df=pd.DataFrame(PlayStore_pcaf).corr()
column=[]
for i in range(PlayStore_pcaf.shape[1]):
    column.append(f'Priciple Column{i+1}')
plt.figure(figsize=(8,6))
sns.heatmap(PlayStore_pcaf_df,annot=True, xticklabels=column,yticklabels=column)
plt.title("Correlation Coefficient of Reduced Feature Space")
plt.show()


#%%


PlayStore_pcaf_df_with_target=pd.DataFrame(data=PlayStore_pcaf,columns=column)
PlayStore_pcaf_df_with_target

PlayStore_pcaf_df_with_target.insert(0,'Category',play_store_app_df['Category'])
print("Reduced Feature Dataframe:\n",PlayStore_pcaf_df_with_target.head())


#%%

# # Normality Test
# We will use at first Graphical method. We will use histogram and QQ plot.

# Histogram
plt.figure(figsize=(14, 8))
columns=['Minimum Installs','Maximum Installs','Rating','Rating Count','Price',
         'Installs']
for n,column in enumerate(columns):
    ax = plt.subplot(2, 3, n + 1)
    play_store_app_df_removed_outlier[column].hist(bins=30).plot(ax=ax)
    ax.set_title(f"Histogram of {column.upper()}")
    ax.set_ylabel("Frequency")
    ax.set_xlabel(f"{column.upper()}")

plt.tight_layout()
plt.show()


#%%

# qqplot(play_store_app_df_removed_outlier['Maximum Installs'],line='s')

# Histogram
plt.figure(figsize=(14, 8))
columns=['Minimum Installs','Maximum Installs','Rating','Rating Count','Price',
         'Installs']
for n,column in enumerate(columns):
    ax = plt.subplot(2, 3, n + 1)
    qqplot(play_store_app_df_removed_outlier[column],ax=ax)
    ax.set_title(column.upper())
    ax.set_title(f"QQ-Plot of {column.upper()}")
    ax.grid()

plt.tight_layout()
plt.show()


# From the both graph, we can see that this dataset does not come from
# gaussian distribution.


#%%

# # Data Transformation 

# I am doing only 1 data Transformation in this project. 
# One is total install number which is string and will convert to int.
play_store_app_df_removed_outlier['Install']=play_store_app_df_removed_outlier.Installs.str.replace('[+,","]', '').astype(float)
play_store_app_df['Install']=play_store_app_df.Installs.str.replace('[+,","]', '').astype(float)
#%%
import scipy.stats as st
play_store_app_df2 = play_store_app_df.copy()
play_store_app_df2['Install'] = st.norm.ppf(st.rankdata(play_store_app_df2['Install'])/(len(play_store_app_df2['Install']) + 1))
play_store_app_df2['Minimum Installs'] = st.norm.ppf(st.rankdata(play_store_app_df2['Minimum Installs'])/(len(play_store_app_df2['Minimum Installs']) + 1))
play_store_app_df2['Maximum Installs'] = st.norm.ppf(st.rankdata(play_store_app_df2['Maximum Installs'])/(len(play_store_app_df2['Maximum Installs']) + 1))
play_store_app_df2['Rating'] = st.norm.ppf(st.rankdata(play_store_app_df2['Rating'])/(len(play_store_app_df2['Rating']) + 1))
play_store_app_df2['Rating Count'] = st.norm.ppf(st.rankdata(play_store_app_df2['Rating Count'])/(len(play_store_app_df2['Rating Count']) + 1))
play_store_app_df2['Price'] = st.norm.ppf(st.rankdata(play_store_app_df2['Price'])/(len(play_store_app_df2['Price']) + 1))

#%%
plt.figure(figsize=(14, 8))
columns=['Minimum Installs','Maximum Installs','Rating','Rating Count','Price',
         'Installs']
for n,column in enumerate(columns):
    ax = plt.subplot(2, 3, n + 1)
    play_store_app_df2[column].hist(bins=30).plot(ax=ax)
    ax.set_title(f"Histogram of {column.upper()}")
    ax.set_ylabel("Frequency")
    ax.set_xlabel(f"{column.upper()}")
    ax.tick_params(axis='x', labelrotation=90)

plt.tight_layout()
plt.show()
#%%


# # HeatMap & Pearson Co-relation Matrix

corr_matrix=play_store_app_df.corr(method='pearson')
sns.heatmap(corr_matrix,xticklabels=corr_matrix.columns,yticklabels=corr_matrix.columns,cmap='RdBu_r',
           annot=True,vmin=-1,vmax=1,linewidths=0.5, annot_kws={"fontsize":8})

plt.title("Correlation Between Features")
plt.show()

#%%


axs=scatter_matrix(corr_matrix, alpha = 0.3, figsize = (17, 12), diagonal = 'kde')
for ax in axs.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')
plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.suptitle("Pairplot")
plt.show()

#%%


sns.pairplot(corr_matrix)
plt.show()


#%%
# Question 12: Statistics:

#%%

plt.figure(figsize = (8,4))
sns.kdeplot(data=play_store_app_df, x="Price",
    hue='Ad Supported', legend=True, fill=True)
# plt.xticks(rotation=90)
plt.suptitle("KDE of Price")
plt.grid(axis='y')
# plt.legend(loc='upper right')
plt.show()


#%%

price_morethan_0 = play_store_app_df[play_store_app_df['Price'] > 0]
plt.figure(figsize = (8,4))
sns.kdeplot(data=price_morethan_0, x="Price", y='Rating',
    hue='Ad Supported', legend=True, fill=True)
# plt.xticks(rotation=90)
plt.suptitle("bivariate distribution between App Price more than 0 and Rating")
plt.grid(axis='y')
# plt.legend(loc='upper right')
plt.show()




#%%

# Mean and median mode of each column
play_store_app_df_removed_outlier.describe()


#%%


corr_matrix=play_store_app_df.corr()
sns.heatmap(corr_matrix,xticklabels=corr_matrix.columns,yticklabels=corr_matrix.columns,cmap='RdBu_r',
           annot=True,linewidths=0.5, annot_kws={"fontsize":8})
plt.show()




# # Data Visualization:

#%%

# play_store_app_df


# ### Line Plot
# Plot a relationship Rating vs Installs for apps based on Apps are free or not.

#%%

plt.figure()
sns.lineplot(data=play_store_app_df_removed_outlier,x="Rating",
             y=play_store_app_df_removed_outlier.Installs.str.replace('[+,","]', '').astype(float),hue='Free')
plt.title("Installs Vs Rating")
plt.grid(axis='y')
# plt.legend()
plt.show()


# We can see that pattern is same for both free and paid apps.
# Trends are low in 0 and 5 Ratings app.

#%%

play_store_app_df_removed_outlier['Released'] = pd.to_datetime(play_store_app_df_removed_outlier['Released'])
play_store_app_df_removed_outlier['year']=play_store_app_df_removed_outlier['Released'].dt.year
plt.figure()
sns.lineplot(data=play_store_app_df_removed_outlier,x="year",
             y='Rating', hue='Content Rating')
plt.title("App Rating Vs Released Year")
plt.grid(axis='y')
# plt.legend()
plt.show()


#%%
# # Bar-Plot : Stack, group
# Give the statistics of app category count


unique_count = play_store_app_df_removed_outlier.groupby(['Category','Free'], as_index=False).agg(Total_Count=('Category', pd.Series.count))
print(unique_count)
rcParams['figure.figsize'] = 15,8

sns.barplot(x='Category',y='Total_Count',data=unique_count, hue='Free',palette=['r','b'])

# plt.figure(figsize=(15,8))
plt.xticks(rotation=90,fontsize=18)
plt.title("Count of App Categories",fontsize=22)
plt.tight_layout()
plt.xlabel("App Categories", fontsize=18)
plt.ylabel("Total Count of Apps", fontsize=18)
plt.grid(axis='y')
plt.show()

#%%


unique_count_Supported = play_store_app_df_removed_outlier.groupby(['Ad Supported','Category'], as_index=False).agg(Total_Count=('Category', pd.Series.count))
rcParams['figure.figsize'] = 13,7
sns.barplot(x='Category',y='Total_Count',hue='Ad Supported',data=unique_count_Supported)
plt.xticks(rotation=90,fontsize=18)
plt.title("Count of App Categories by Ad Supported",fontsize=22)
plt.xlabel("App Categories", fontsize=18)
plt.ylabel("Total Count of Apps", fontsize=18)


plt.grid(axis='y')
plt.show()
#%%



#%%
# # Count-plot
plt.figure(figsize=(8,5))
sns.countplot(x='Content Rating',hue='Ad Supported',data=play_store_app_df_removed_outlier)
plt.title("Count of Content Rating by Ad Supported")
# plt.legend()
plt.grid(axis='y')
plt.show()

# # Cat-Plot

#%%


unique_count_Free = play_store_app_df_removed_outlier.groupby(['Ad Supported','year'], as_index=False).agg(Total_Count=('year', pd.Series.count))
plt.figure(figsize=(7,13))
sns.catplot(x='year',y='Total_Count',
        data=unique_count_Free, hue='Ad Supported',
        kind="bar", legend=False,
        )
plt.xticks(rotation=90)
plt.title("Count Apps Over the Years by Ad Supported")
plt.legend(loc='upper left', title='Add supported')
plt.tight_layout()
plt.grid(axis='y')
plt.show()
# unique_count_Free


# # Pie-chart
#%%


# How many apps are free?
unique_count_Free = play_store_app_df_removed_outlier.groupby(['Free'], 
                                                              as_index=False).agg(Total_Count=('Free', pd.Series.count))

fig = plt.figure(figsize =(6, 4))
explode = [0.02, 0.2]
plt.pie(unique_count_Free['Total_Count'],  labels=unique_count_Free['Free'], 
        explode = explode, autopct = '%.1f%%')
plt.axis('square')
plt.axis('equal')
plt.title("Percentage of Free apps")

plt.show()


#%%


# Content Rating allocation?
unique_count_Free = play_store_app_df_removed_outlier.groupby(['Content Rating'], 
                                                              as_index=False).agg(Total_Count=('Content Rating', pd.Series.count))
unique_count_Free = unique_count_Free.drop(unique_count_Free[unique_count_Free['Content Rating'] =='Unrated'].index)
print(unique_count_Free)
explode = [0.02, 0.1, 0.03,0.04,0.05]
fig = plt.figure(figsize =(10, 7))
plt.pie(unique_count_Free['Total_Count'],  labels=unique_count_Free['Content Rating'], autopct='%1.1f%%',
        # wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},textprops={'size': 'x-large'},
        explode=explode,
        shadow=True, startangle=45)
plt.axis('equal') 
plt.axis('square') 
plt.title("Content Rating allocation")

plt.show()


# # Displot

#%%

plt.figure(figsize = (8,4))
sns.displot(data=play_store_app_df, x="Rating",
    hue='Free', legend=True, 
    stat='probability', kde=True)
# plt.xticks(rotation=90)
plt.suptitle("Distribution of App Rating")
plt.grid(axis='y')
# plt.legend(loc='upper right')
plt.show()




# # Pair Plot
#%%
# execution hidden as criterion already satisfied above
# sns.pairplot(play_store_app_df, kind="kde")
# corr_matrix=play_store_app_df.corr()
# sns.pairplot(corr_matrix,kind="kde")
# plt.show()

pairplot_df = play_store_app_df.copy()
pairplot_df.drop("Maximum Installs", axis=1, inplace=True)
pairplot_df.drop("Minimum Installs", axis=1, inplace=True)
pairplot_df.drop("Installs", axis=1, inplace=True)
# pairplot_df['Installs'] = pairplot_df['Installs'].astype(float)
# pairplot_df['Maximum Installs'] = pairplot_df['Maximum Installs'].astype(float)
sns.pairplot(pairplot_df.corr(),kind="kde")
plt.suptitle("Pairplot of Correlation between features", y=1.02)
plt.show()

# # HeatMap
#%%
# execution hidden as criterion already satisfied above
# sns.heatmap(corr_matrix,xticklabels=corr_matrix.columns,yticklabels=corr_matrix.columns,cmap='RdBu_r',
#            annot=True,linewidths=0.5)
# plt.show()



# # Histplot
#%%
plt.figure()
sns.histplot(data=play_store_app_df,x='Installs', hue='Free')
plt.xticks(rotation = 90)
plt.title("Histogram of App Installs")
plt.grid(axis='y')
plt.show()

#%%
plt.figure()
sns.histplot(data=play_store_app_df,x='Rating', hue='Ad Supported')
plt.xticks(rotation = 90)
plt.title("Histogram of App Ratings by Ad Supported")
plt.grid(axis='y')
plt.show()


# # QQ Plot

#%%


plt.figure(figsize=(14, 8))
columns=['Minimum Installs','Rating','Installs']
for n,column in enumerate(columns):
    ax = plt.subplot(1, 3, n + 1)
    qqplot(play_store_app_df_removed_outlier[column],ax=ax)
    ax.set_title(column.upper())
    ax.set_xlabel("")
    # ax.tick_params(labelrotation=90)
plt.tight_layout()
plt.show()


# # KDE 

#%%


plt.figure(figsize=(8, 6))
sns.kdeplot(data=play_store_app_df,x='Rating',hue='In App Purchases', fill=True)
plt.title("KDE plot of App Ratings by In App Purchase")
plt.grid(axis='y')
plt.show()



#%%
# # Scatter plot and regression line using sklearn
### # execution hidden as it takes lot of time to execute, output shown in the report

# ax = sns.regplot(x="Install", y="Rating", data=play_store_app_df,
#         fit_reg=True)
# plt.title("Scatterplot of Rating Vs Install with Regression Line")
# plt.show()




# # Multivariate Box Plot
#%%

plt.figure(figsize=(24, 8))
sns.boxplot(x="Category", y="Rating", data=play_store_app_df_removed_outlier)
plt.ylabel('Rating')
plt.xlabel('Category')
plt.title("Box plot of Rating Vs Categories")
plt.xticks(rotation=90)
plt.grid()
plt.show()

#%%

plt.figure(figsize=(10, 8))
sns.boxplot(x="Editors Choice", y="Rating",hue="Free", data=play_store_app_df)
plt.ylabel('Rating')
plt.xlabel('Editors Choice')
plt.title("Box plot of Rating Vs Editors Choice by Free Apps")
plt.grid()
plt.show()

# # Violin Plot

#%%
plt.figure(figsize=(14, 8))
columns=['Minimum Installs','Rating','Maximum Installs','Price','Rating Count','Install']
for n,column in enumerate(columns):
    ax = plt.subplot(2, 3, n + 1)
    sns.violinplot(x=play_store_app_df_removed_outlier[column],ax=ax)
    ax.set_title(column.upper())
    ax.set_xlabel("")
    ax.tick_params(labelrotation=90)
plt.tight_layout()
plt.show()

# # SubPlots

#%%

top5Apps=play_store_app_df.sort_values(by=['Install'],ascending=False).head(5)
Category_count = play_store_app_df_removed_outlier.groupby(['Category'], 
                                                              as_index=False).agg(Total_Count=('Category', pd.Series.count))
top5Category=Category_count.sort_values(by=['Total_Count'],ascending=False).head(5)
Minimum_android = play_store_app_df_removed_outlier.groupby(['Minimum Android'], 
                                                              as_index=False).agg(Total_Count=('Minimum Android', pd.Series.count))
top5Androidcoverage=Minimum_android.sort_values(by=['Total_Count'],ascending=False).head(5)
top5Androidcoverage


#%%

# In this subplots history we will discuss about the app history statistics.

plt.figure(figsize=(10, 8))
ax = plt.subplot(2, 2, 1)
sns.barplot(x="App Name", y="Install", data=top5Apps)
plt.xticks(rotation=90)
ax.set_title("Top 5 Apps")
# ax.xticks(rotation=90)
ax = plt.subplot(2, 2, 2)
sns.barplot(x="Category", y="Total_Count", data=top5Category)
plt.xticks(rotation=12)
ax.set_title("Top 5 Category")
ax = plt.subplot(2, 2, 3)
sns.histplot(data=play_store_app_df,x='Rating')
ax.set_title("Ratings Distributions")

ax = plt.subplot(2, 2, 4)
sns.barplot(x="Minimum Android", y="Total_Count", data=top5Androidcoverage)
plt.xticks(rotation=12)
ax.set_title("Top 5 Minimum Android")


plt.tight_layout()
plt.show()

#%%