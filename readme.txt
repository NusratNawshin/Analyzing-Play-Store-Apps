In order to run this full project these packages need to be pre-installed into the python environment.

- dash
- pandas
- scipy
- dash_bootstrap_components
- statsmodels
- plotly
- datetime

The command to install all packages together is :-

pip install dash pandas scipy dash_bootstrap_components statsmodels plotly datetime

____________
datasets :-
____________
All datasets are inside 'data' folder
- Google-Playstore.csv
- Play_Store_Dash_new.csv

____________
.py files :-
____________
- Data-Cleaning-For-Dash.py: 
Reads the original dataset 'Google-Playstore.csv' and generates final dataset for dash application 'Play_Store_Dash_new.csv'

- analysis.py: 
Reads the original dataset 'Google-Playstore.csv' and performs EDA by visualizing using Seaborn and Matplotlib

- Visualization.py: Reads the original dataset 'Play_Store_Dash_new.csv' and implements dash application using Dash and Plotly packages