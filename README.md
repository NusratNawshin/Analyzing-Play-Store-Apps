# Analyzing-Play-Store-Apps
Analyzing Google Play Store Apps

## DATS_6401 Visualization of Complex Data Final Project
#### With Prof. Reza Jafari
### George Washington University
#### Spring 2022

The project is to analyze Google Play Store Apps by Visualizing using Python Seaborn, Plotly and Dash packages. The dataset is collected from Kaggle. (Link: [https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps])
#
It contains 24 columns and 2.3+ records. In which 19 columns are categorical, and 5 columns are numerical.
 
Numerical columns list:
- Rating
- Rating Count
- Minimum Installs
- Maximum Installs
- Price 

Categorical Columns List:
- Object
- App Name
- App Id
- Category
- Installs              
- Currency              
- Size                  
- Minimum Android
- Developer Id
- Developer Website
- Developer Email
- Released  
- Last Updated
- Content Rating
- Privacy Policy
- Scraped Time
Boolean Column List:
- Free                    
- Ad Supported
- In App Purchases
- Editors Choice

The dataset is inside of this repository's 'data; folder named as 'Google-Playstore.csv'. However, for faster dash implementation I have the dataset reduced the dataset to 50775 rows and 22 columns by considering only the popular apps. I selected app installs count more than 100k and rating count more than 4500. Farther, I have dropped Scrapped time and Last updated column as it had high missing values. The code for this reduction is inside 'Data-Cleaning-For-Dash.py' and the dataset is stored into data folder named as 'Play_Store_Dash_new.csv'.
#
In order to run this full project these packages need to be pre-installed into the python environment. 
- dash 
- pandas 
- scipy 
- dash_bootstrap_components 
- statsmodels 
- plotly 
- datetime


The command to install all packages together is:
#### - pip install dash pandas scipy dash_bootstrap_components statsmodels plotly datetime
#
- **Data-Cleaning-For-Dash.py:** Reads the original dataset 'Google-Playstore.csv' and generates final dataset for dash application 'Play_Store_Dash_new.csv'
- **analysis.py:** Reads the original dataset 'Google-Playstore.csv' and performs EDA by visualizing using Seaborn and Matplotlib
- **Visualization.py:** Reads the original dataset 'Play_Store_Dash_new.csv' and implements dash application using Dash and Plotly packages
- **NN_Final_Report.pdf:** Final project report pdf

I have deployed the dash app in Google Cloud Platform as well.
#
## Link to the dash app: [https://dashapp-s4afy3ytoq-rj.a.run.app/]
#
#
### Note: Some output in the dash takes sometime to run, try by re-selecting the component or reloading the link. 
#
