
from dash import dash_table
import plotly.express as px
import plotly.graph_objs as go
import dash as dash
from dash import dcc
from dash import html
import pandas as pd
import dash_bootstrap_components as dbc
from datetime import date, datetime 
# import time
import plotly.figure_factory as ff
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as st


from dash.dependencies import Input, Output, State

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
play_store_apps_df=pd.read_csv('data/Play_Store_Dash_new.csv')

Categorylist=play_store_apps_df.Category.unique()
content_rating=play_store_apps_df['Content Rating'].unique()
# print(play_store_apps_df)

temp_df=pd.DataFrame()
csv_df=pd.DataFrame()
filter_df=pd.DataFrame()
filter_df=play_store_apps_df

my_app = dash.Dash('Visualizing Google Play Store App Market', external_stylesheets=external_stylesheets)

my_app.layout = html.Div([ html.H1('Visualizing Google Play Store App Market', style = {'textAlign':'center'}),
                html.Br(),
                dcc.Tabs(id = 'project', 
                    children = [
                        dcc.Tab(label='Statistics', value ='app'),
                        dcc.Tab(label='Comparision', value ='compare'),
                        dcc.Tab(label='App Details', value ='details'),
                        dcc.Tab(label='Graph Visualization', value ='graph'),
                        dcc.Tab(label='Normality Test', value ='normality'),
                        
                        ]),
                html.Div(id = 'layout')

])
########## Apps Statistics Layout ###########
app_statistics_layout=html.Div([
            html.H6("App Statistics Layout",style={'textAlign': 'center'}),
            html.Div([ html.H6('Filters'),
            html.Div([

                html.Div([
                    # html.P("Category"),
                    dcc.Dropdown(placeholder="Category",
                    options=[{'label':category, 'value':category} for category in Categorylist],id='Category-Selector'
                    )
                    ],
                    style={'width': '15%','display': 'inline-block'}),
                html.Div([
                    html.P("Free"),
                    dcc.RadioItems(['Yes','No'],id="free-radio")
                    ],style={'width': '15%','display': 'inline-block'}),
                html.Div([
                    html.P("Ad Supported"),
                    dcc.RadioItems(['Yes','No'],id='ad-radio'),
                ],style={'width': '15%','display': 'inline-block'}),
                html.Div([
                    html.P("In App Purchase"),
                    dcc.RadioItems(['Yes','No'],id='purchase-radio'),
                ],style={'width': '15%','display': 'inline-block'}),
                html.Div([
                    html.P("Editor's Choice"),
                    dcc.RadioItems(['Yes','No'],id='editor-radio'),
                ],style={'width': '15%','display': 'inline-block'}),
                html.Div([
                    html.P("Relase Date"),
                    # dcc.DatePickerRange(
                    # id='my-date-picker-range',
                    # max_date_allowed=dt.today,
                    # start_date_placeholder_text='YYYY-MM-DD',
                    # end_date=dt.today
                    # ),
                    dcc.DatePickerRange(
                        display_format='YYYY-MM-DD',
                        clearable=True,
                        max_date_allowed=date.today(),
                        # end_date=date.today(),
                        # start_date=date(2010,1,28),
                        end_date=date.today(),
                        id='realse-picker1'
                    ),
                ],style={'width': '25%','display': 'inline-block'}),
                           
            ],style={'margin-bottom':'20px'}),
            html.Div([
                # html.Div([
                #     html.P("App Ratings"),
                #     dcc.RangeSlider(min=-1, max=5, step=1, value=[-1], id='app-ratings-slider'),
                # ],style={'width': '20%','display': 'inline-block','margin-right':'20px'}),
                html.Div([
                    html.P("Content Rating"),
                    # dcc.RangeSlider(min=-1, max=5, step=1, value=[-1], id='my-range-slider'),
                    # dcc.Checklist(['New York City', 'Montréal', 'San Francisco','New York City', 'Montréal', 'San Francisco'],inline=True)
                    dcc.Checklist(content_rating,inline=True,id='content-ratings-checklist2')

                ],style={'width': '80%','display': 'inline-block'}),
                html.Div([
                    # html.P("App Ratings"),
                    html.Button('Reset Filter',id='reset-button',n_clicks=0)
                ],style={'width': '20%','display': 'inline-block'}),
                html.Hr(style={'border':'2px solid black','border-radius':'5px'}),

            ]),

            ]),
            html.Div([ 
                # dcc.Graph(id = 'firstsubplot')
            ],id="Statistics",style={'textAlign': 'center'})
])
########## Apps Statistics Layout ###########

@my_app.callback(Output(component_id='Category-Selector', component_property='value'),
                Output(component_id='free-radio', component_property='value'),
                Output(component_id='ad-radio', component_property='value'),
                Output(component_id='purchase-radio', component_property='value'),
                Output(component_id='editor-radio', component_property='value'),
                Output(component_id='realse-picker1', component_property='start_date'),
                Output(component_id='realse-picker1', component_property='end_date'),
                Output(component_id='content-ratings-checklist2', component_property='value'),
                Input('reset-button','n_clicks'),)
def reset_button(n_clicks):
    return None,None,None,None,None,None,date.today(),[]

###################### Filter Selection Functions ################
# Output(component_id='First_graph', component_property='figure'),
#                 Output(component_id='Second_graph', component_property='figure'),
#                 Output(component_id='Third_graph', component_property='figure'),
#                 Output(component_id='Fourth_graph', component_property='figure')

@my_app.callback(Output(component_id='Statistics', component_property='children'),
                [Input(component_id='Category-Selector', component_property='value'),
                Input(component_id='free-radio', component_property='value'),
                Input(component_id='ad-radio', component_property='value'),
                Input(component_id='purchase-radio', component_property='value'),
                Input(component_id='editor-radio', component_property='value'),
                Input(component_id='realse-picker1', component_property='start_date'),
                Input(component_id='realse-picker1', component_property='end_date'),
                Input(component_id='content-ratings-checklist2', component_property='value'),
                # Input(component_id='app-ratings-slider', component_property='value')
                ]
                )
def Filter(Category,free,ad,purchase,editor,start_date,end_date,contentRatings):
    play_store_apps_df_filtered=play_store_apps_df
    # print(play_store_apps_df_filtered)
    if Category is not None:
        play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered.Category==Category]
        print(play_store_apps_df_filtered.shape)
    else:
        print("Category Empty")    
    if free is not None:
        if(free == 'Yes'):
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered.Free==True]
            print(play_store_apps_df_filtered.shape)
        else:
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered.Free==False]
            print(play_store_apps_df_filtered.shape)
    else:
        print("free empty")
    if ad is not None:
        if(ad == 'Yes'):
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['Ad Supported']==True]
            print(play_store_apps_df_filtered.shape)
        else:
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['Ad Supported']==False]
            print(play_store_apps_df_filtered.shape)
    else:
        print("ad empty")
    if purchase is not None:
        if(purchase == 'Yes'):
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['In App Purchases']==True]
            print(play_store_apps_df_filtered.shape)
        else:
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['In App Purchases']==False]
            print(play_store_apps_df_filtered.shape)
    else:
        print("purchase empty")
    if editor is not None:
        if(editor == 'Yes'):
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['Editors Choice']==True]
            print(play_store_apps_df_filtered.shape)
        else:
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['Editors Choice']==False]
            print(play_store_apps_df_filtered.shape)
    else:
        print("editor empty")
    if start_date is not None:
        if end_date is None:
            print(date.today())
            end_date=date.today()
        play_store_apps_df_filtered=play_store_apps_df_filtered.loc[(play_store_apps_df_filtered['Released']>start_date) &(play_store_apps_df_filtered['Released']<end_date) ]
        print(play_store_apps_df_filtered.shape)   
    else:
        print("start_date empty")
    if contentRatings is not None and (len(contentRatings)>0):
        play_store_apps_df_filtered=play_store_apps_df_filtered.loc[(play_store_apps_df_filtered['Content Rating'].isin(contentRatings))]
        print(play_store_apps_df_filtered.shape)
    else:
        print("contentRatings empty")
    # if(appratings[0]!=-1):
    #     start=appratings[0]
    #     end=start+1
    #     play_store_apps_df_filtered=play_store_apps_df_filtered.loc[((play_store_apps_df_filtered['Rating']>=start) & (play_store_apps_df_filtered['Rating']<end) )]
    #     print(play_store_apps_df_filtered.shape)
    # else:
    #     print("appratings empty")
    # print(play_store_apps_df)
    print(play_store_apps_df_filtered.shape)
    # if(play_store_apps_df_filtered.shape[0]>5):
    #     print("Enongh Data")
    # else:
    #     print("Not Enough Data")
    
    if(play_store_apps_df_filtered.shape[0]<=1):
        filter_df=play_store_apps_df_filtered
        filtered_div=html.Div([
            html.H6("There is no data with selected filter")
        ])
    else:
        # temp_year=pd.DataFrame()
        # play_store_apps_df[]
        play_store_apps_df_filtered['Released'] = pd.to_datetime(play_store_apps_df['Released'])
        play_store_apps_df_filtered['year']=play_store_apps_df['Released'].dt.year
        sorted_df=play_store_apps_df_filtered
        sorted_df=sorted_df.sort_values(by='Installs',ascending=False)
        sorted_df=sorted_df.head(5)

        temp_years=pd.DataFrame()
        temp_years = play_store_apps_df_filtered.year.value_counts(sort=True,ascending=False).rename_axis('year').reset_index(name='counts')
        temp_years=temp_years.sort_values(by=['year'])
        # print(sorted_df)
        # print(play_store_apps_df_filtered.describe())
        describe_df=play_store_apps_df_filtered.describe()
        describe_df=describe_df.drop(['year','Unnamed: 0'],axis=1)
        columns_list=describe_df.columns
        # columns_list=columns_list.drop(['Unnamed 0'])
        print(columns_list)
        figure1=px.bar(sorted_df,x='App Name',y='Installs',title="Top Apps based on Installation")
        # figure2=px.line(describe_df,x=describe_df.index,y='Installs',title="Dataset Statistics based on ")
        figure3=px.box(temp_years,x='year',y='counts',points='all',title="Year Wise App Counts")
        # figure3=px.box(temp_years,x='year')
        # figure4=px.histogram(play_store_apps_df_filtered,x='Rating')
        hist_data1=play_store_apps_df_filtered['Rating']
        hist_data2=play_store_apps_df_filtered['Installs']
        hist_data=[hist_data1]
        group_labels=['Rating']
        fig=ff.create_distplot(hist_data=hist_data,group_labels=group_labels)
        fig.layout.update({'title': 'Distribution Plot for Ratings'})
        filtered_div=html.Div([
        html.H6('Statistics'),
        html.Div([
                    html.Div([
                        # html.P("First Subplot"),
                        # dcc.Graph(id="First_graph"),
                        dcc.Graph(figure=figure1),
                    ],style={'width': '50%','display': 'inline-block'},id='first_plot'),
                    html.Div([
                        # html.P("Second Subplot"),
                        dcc.Dropdown(placeholder="Column",
                    options=[{'label':columnss, 'value':columnss} for columnss in columns_list],id='column-Selector',value='Rating'
                    ),
                        # dcc.Graph(id="Second_graph"),
                        dcc.Graph(id='filter_graph2'),
                    ],style={'width': '50%','display': 'inline-block'},id="second_plot"),
                ],id="first_subplot_div"),

                html.Div([
                    html.Div([
                        # html.P("Third Subplot"),
                        
                        # dcc.Graph(id="Third_graph"),
                        dcc.Graph(figure=figure3),
                    ],style={'width': '50%','display': 'inline-block'},id="third_plot"),
                    html.Div([
                        # html.P("Fourth Subplot"),
                        # dcc.Graph(id="Fourth_graph"),
                        # html.P(""),
                        dcc.Graph(figure=fig),
                    ],style={'width': '50%','display': 'inline-block'},id="fourth_plot"),
                ],id="second_subplot_div"),

    ])

   
    
    return filtered_div

    
 
###################### Filter Selection Functions ################

########## Apps Comparision Layout ###########
app_comparision_layout=html.Div([
            
            html.Div([html.H5("App Comparision Layout",style={'textAlign': 'center'}),
            
            html.Div([
                    # html.P("Category",style={'width':'50%','left':'50%','right':'auto'}),
                    dcc.Dropdown(placeholder="Category",
                    options=[{'label':category, 'value':category} for category in Categorylist],id='Category-Selector-comparision',style={'width':'50%','left':'50%','right':'auto'}
                    ),],style={"padding-bottom":'20px'}),
            html.Div([
                html.P("Please select the first apps to compare"),
            ],style={'width': '40%','display': 'inline-block'},id="first_app_dropdown"),
            html.Div([
                html.P("Please select the second apps to compare"),
            ],style={'width': '40%','display': 'inline-block'},id="second_app_dropdown"),
            html.Div([
                # html.P("Please select the second apps to compare"),
                html.Button('Compare',id='compare_button',n_clicks=0)
            ],style={'width': '20%','display': 'inline-block'}),
            ]),
            html.Hr(style={'border':'2px solid black','border-radius':'5px'}),
            html.Div([html.H6("Comparision Panel",style={'textAlign': 'center'}),
            
            
            ],id="comaprision_result"),
    
])
########## Apps Comparision Layout ###########

#####column selector callback ##########
@my_app.callback(Output(component_id='filter_graph2', component_property='figure'),
                    [Input(component_id='column-Selector', component_property='value'),])

def column_updater(column_value):
    # print(play_store_apps_df)
    # print(filter_df)
    describes_df=filter_df.describe()
    describes_df=describes_df.drop(['year'],axis=1)
    figure2=px.line(describes_df,x=describes_df.index,y=column_value,title=f"Dataset Statistics based on {column_value} ")

    return figure2

    
#####column selector callback ##########
############# Comparision CallBack for Category ##############
@my_app.callback([Output(component_id='first_app_dropdown',component_property='children'),
                  Output(component_id='second_app_dropdown',component_property='children')],
                  [Input(component_id='Category-Selector-comparision',component_property='value')])
def categoryComparisionUpdate(category):
    print("Comparision pdf")
    print(category)
    play_store_compare=play_store_apps_df
    if category is None:
        appslist=play_store_compare['App Name'].unique()
        print(appslist.shape)
    else:
        play_store_compare=play_store_compare.loc[play_store_compare.Category==category]
        appslist=play_store_compare['App Name'].unique()
        print(appslist.shape)
    first_app_dropdown=html.Div([
    html.P("Please select the first apps to compare"),
    dcc.Dropdown(placeholder="First App",
                options=[{'label':app, 'value':app} for app in appslist],id='First_App_Comparision',style={'width':'80%'})
    ])
    second_app_dropdown=html.Div([
    html.P("Please select the second apps to compare"),
    dcc.Dropdown(placeholder="Second App",
                options=[{'label':app, 'value':app} for app in appslist],id='Second_App_Comparision',
                style={'width':'80%'})])
    # time.sleep(2)
    return first_app_dropdown,second_app_dropdown
########## call back for comparision layer ############
@my_app.callback(Output(component_id='comaprision_result',component_property='children'),
                [Input(component_id="compare_button",component_property="n_clicks")],
                [State("First_App_Comparision","value"),
                State("Second_App_Comparision","value"),])
def comparision_result(n_clicks,apps1,apps2):
    print(n_clicks)
    print("Comapre Button Clicks")
    if(apps1 is None or apps2 is None):
        if(apps1 is None and apps2 is None):
            divs=html.Div([
                html.Hr(style={'border':'2px solid black','border-radius':'5px'}),
                html.H5("Comparision Panel",style={'textAlign': 'center'}),
                html.P("You have not selected any apps for comparing")
            ])
            return divs
        elif(apps2 is None):
            divs=html.Div([
                html.Hr(style={'border':'2px solid black','border-radius':'5px'}),
                html.H5("Comparision Panel",style={'textAlign': 'center'}),
                html.P("Please select the Second Apps")
            ])
            return divs
        else:
            divs=html.Div([
                html.Hr(style={'border':'2px solid black','border-radius':'5px'}),
                html.H5("Comparision Panel",style={'textAlign': 'center'}),
                html.P("Please select the First Apps")
                
            ])
            return divs
    elif(apps1==apps2):
        divs=html.Div([
            html.Hr(style={'border':'2px solid black','border-radius':'5px'}),
            html.H5("Comparision Panel",style={'textAlign': 'center'}),
            html.P("You have selected the same apps")
            ])
        return divs
    else:
        print(apps1)
        print(apps2)
        df1=play_store_apps_df.loc[play_store_apps_df['App Name']==apps1]
        df2=play_store_apps_df.loc[play_store_apps_df['App Name']==apps2]
        df1=df1.sort_values(by='Installs',ascending=False)
        df2=df2.sort_values(by='Installs',ascending=False)
        df1=df1[:1]
        df2=df2[:1]
        final_df=pd.concat([df1,df2])
        apps1_score=0
        apps2_score=0
        if(df1.Rating.values[0]>df2.Rating.values[0]):
            apps1_score=apps1_score+1
        elif(df2.Rating.values[0]>df1.Rating.values[0]):
            apps2_score=apps2_score+1
        if(df1.Free.values[0]==True):
            apps1_score=apps1_score+1
        else:
            apps1_score=apps1_score-1
        if(df2.Free.values[0]==True):
            apps2_score=apps2_score+1
        else:
            apps2_score=apps2_score-1
        if(df1.Installs.values[0]>df2.Installs.values[0]):
            apps1_score=apps1_score+1
        elif(df2.Installs.values[0]>df1.Installs.values[0]):
            apps2_score=apps2_score+1
        if(df1.Price.values[0]>df2.Price.values[0]):
            apps1_score=apps1_score+1
        elif(df2.Price.values[0]>df1.Price.values[0]):
            apps2_score=apps2_score+1
        if(df1['Rating Count'].values[0]>df2['Rating Count'].values[0]):
            apps1_score=apps1_score+1
        elif(df2['Rating Count'].values[0]>df1['Rating Count'].values[0]):
            apps2_score=apps2_score+1
        
        print(apps1_score)
        print(apps2_score)
        if(apps1_score>apps2_score):
            score=f"{apps1} has the higher score than {apps2} based on some criteria."
        elif(apps2_score>apps1_score):
            score=f"{apps2} has the higher score than {apps1} based on some criteria."
        else:
            score=f"{apps1} has the same score as {apps2}"
        fig_rating_color={}
        fig_rating_color[0]='red'
        fig_rating_color[1]='green'
        fig_ratings=px.bar(final_df,x='App Name',y='Rating',color='App Name',color_discrete_map={f'{apps1}':'navy',f'{apps2}':'aqua'})
        fig_ratings_count=px.bar(final_df,x='App Name',y='Rating Count',color='App Name',color_discrete_map={f'{apps1}':'maroon',f'{apps2}':'purple'})
        fig_Installs=px.bar(final_df,x='App Name',y='Installs',color='App Name',color_discrete_map={f'{apps1}':'red',f'{apps2}':'green'})
        fig_maximum_installs=px.bar(final_df,x='App Name',y='Maximum Installs',color='App Name',color_discrete_map={f'{apps1}':'blueviolet',f'{apps2}':'bisque'})
        html.Hr(style={'border':'2px solid black','border-radius':'5px'}),
        divs=html.Div([html.H5("Comparision Panel",style={'textAlign': 'center'}),

        html.Div([
            html.Div([
                # html.H6(f"Comaprision of Ratings between {apps1} and {apps2}",style={'textAlign': 'center'}),
                html.H6(f"Comaprision of Ratings ",style={'textAlign': 'center'}),
                dcc.Graph(figure=fig_ratings)
            ],style={'width': '50%','display': 'inline-block'}),
            html.Div([
                # html.H6(f"Comaprision of Ratings Count between {apps1} and {apps2}",style={'textAlign': 'center'}),
                html.H6(f"Comaprision of Ratings Count ",style={'textAlign': 'center'}),
                dcc.Graph(figure=fig_ratings_count)
            ],style={'width': '50%','display': 'inline-block'}),
        ]),
        html.Div([
            html.Div([
                # html.H6(f"Comaprision of Installs between {apps1} and {apps2}",style={'textAlign': 'center'}),
                html.H6(f"Comaprision of Installs",style={'textAlign': 'center'}),
                dcc.Graph(figure=fig_Installs)
            ],style={'width': '50%','display': 'inline-block'}),
            html.Div([
                # html.H6(f"Comaprision of Maximum Installs between {apps1} and {apps2}",style={'textAlign': 'center'}),
                html.H6(f"Comaprision of Maximum Installs",style={'textAlign': 'center'}),
                dcc.Graph(figure=fig_maximum_installs)
            ],style={'width': '50%','display': 'inline-block'}),
        ]),
        #['Unnamed: 0', 'App Name', 'App Id', 'Category', 'Rating',
    #    'Rating Count', 'Installs', 'Minimum Installs', 'Maximum Installs',
    #    'Free', 'Price', 'Currency', 'Size', 'Minimum Android', 'Developer Id',
    #    'Developer Website', 'Developer Email', 'Released', 'Last Updated',
    #    'Content Rating', 'Privacy Policy', 'Ad Supported', 'In App Purchases',
    #    'Editors Choice', 'Scraped Time']
    html.Div([
                html.H6("Comparision Result",style={'textAlign': 'center'}),
                html.B(f"{score}",style={'textAlign': 'center'})

    ]),
        html.Div([
                html.H6("Comparision Table",style={'textAlign': 'center'}),
                html.Table([
                    html.Tr([
                        html.Th(''),
                        html.Th(f'{apps1}'),
                        html.Th(f'{apps2}'),

                    ]),
                    html.Tr([
                        html.Td('App Id'),
                        html.Td(f'{df1["App Id"].values[0]}'),
                        html.Td(f'{df2["App Id"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Category'),
                        html.Td(f'{df1["Category"].values[0]}'),
                        html.Td(f'{df2["Category"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Free'),
                        html.Td(f'{df1["Free"].values[0]}'),
                        html.Td(f'{df2["Free"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Price'),
                        html.Td(f'{df1["Price"].values[0]}'),
                        html.Td(f'{df2["Price"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Currency'),
                        html.Td(f'{df1["Currency"].values[0]}'),
                        html.Td(f'{df2["Currency"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Size'),
                        html.Td(f'{df1["Size"].values[0]}'),
                        html.Td(f'{df2["Size"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Minimum Android'),
                        html.Td(f'{df1["Minimum Android"].values[0]}'),
                        html.Td(f'{df2["Minimum Android"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Developer Id'),
                        html.Td(f'{df1["Developer Id"].values[0]}'),
                        html.Td(f'{df2["Developer Id"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Developer Website'),
                        # html.Td(f'{df1["Developer Website"].values[0]}'),
                        # html.Td(f'{df2["Developer Website"].values[0]}'),
                        html.Td(html.A(children=f'{df1["Developer Website"].values[0]}',href=f'{df1["Developer Website"].values[0]}',style={'color': 'blue', 'text-decoration': 'none'}  ),),
                        html.Td(html.A(children=f'{df2["Developer Website"].values[0]}',href=f'{df2["Developer Website"].values[0]}',style={'color': 'blue', 'text-decoration': 'none'}  ))  

                    ]),
                    html.Tr([
                        html.Td('Developer Email'),
                        html.Td(f'{df1["Developer Email"].values[0]}'),
                        html.Td(f'{df2["Developer Email"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Released'),
                        html.Td(f'{df1["Released"].values[0]}'),
                        html.Td(f'{df2["Released"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Content Rating'),
                        html.Td(f'{df1["Content Rating"].values[0]}'),
                        html.Td(f'{df2["Content Rating"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Privacy Policy'),
                        # html.Td(f'{df1["Privacy Policy"].values[0]}'),
                        # html.Td(f'{df2["Privacy Policy"].values[0]}'),
                        html.Td(html.A(children=f'{df1["Privacy Policy"].values[0]}',href=f'{df1["Privacy Policy"].values[0]}',style={'color': 'blue', 'text-decoration': 'none'}  ),),
                        html.Td(html.A(children=f'{df2["Privacy Policy"].values[0]}',href=f'{df2["Privacy Policy"].values[0]}',style={'color': 'blue', 'text-decoration': 'none'}  ))  

                    ]),
                    html.Tr([
                        html.Td('Ad Supported'),
                        html.Td(f'{df1["Ad Supported"].values[0]}'),
                        html.Td(f'{df2["Ad Supported"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('In App Purchases'),
                        html.Td(f'{df1["In App Purchases"].values[0]}'),
                        html.Td(f'{df2["In App Purchases"].values[0]}'),

                    ]),
                    html.Tr([
                        html.Td('Editors Choice'),
                        html.Td(f'{df1["Editors Choice"].values[0]}'),
                        html.Td(f'{df2["Editors Choice"].values[0]}'),

                    ]),
                    # html.Tr([
                    #     html.Td('Last Updated'),
                    #     html.Td(f'{df1["Last Updated"].values[0]}'),
                    #     html.Td(f'{df2["Last Updated"].values[0]}'),

                    # ])
                    

                ],style={'margin':'0px auto','text-align':'center','padding-bottom':'30px'})

        ])

        ])
        return divs


########## call back for comparision layer ############
############# Comparision CallBack for Category ##############

############### code block for app details layout ##########################

app_details_layout=html.Div([
            html.H6("App Details Details",style={'textAlign': 'center'}),
            html.Div([ html.H6('Filters'),
            html.Div([

                html.Div([
                    # html.P("Category"),
                    dcc.Dropdown(placeholder="Category",
                    options=[{'label':category, 'value':category} for category in Categorylist],id='Category-Selector1'
                    )
                    ],
                    style={'width': '15%','display': 'inline-block'}),
                html.Div([
                    html.P("Free"),
                    dcc.RadioItems(['Yes','No'],id="free-radio1")
                    ],style={'width': '15%','display': 'inline-block'}),
                html.Div([
                    html.P("Ad Supported"),
                    dcc.RadioItems(['Yes','No'],id='ad-radio1'),
                ],style={'width': '15%','display': 'inline-block'}),
                html.Div([
                    html.P("In App Purchase"),
                    dcc.RadioItems(['Yes','No'],id='purchase-radio1'),
                ],style={'width': '15%','display': 'inline-block'}),
                html.Div([
                    html.P("Editor's Choice"),
                    dcc.RadioItems(['Yes','No'],id='editor-radio1'),
                ],style={'width': '15%','display': 'inline-block'}),
                html.Div([
                    html.P("Relase Date"),
                    # dcc.DatePickerRange(
                    # id='my-date-picker-range',
                    # max_date_allowed=dt.today,
                    # start_date_placeholder_text='YYYY-MM-DD',
                    # end_date=dt.today
                    # ),
                    dcc.DatePickerRange(
                        display_format='YYYY-MM-DD',
                        clearable=True,
                        max_date_allowed=date.today(),
                        end_date=date.today(),
                        id='realse-picker'
                    ),
                ],style={'width': '25%','display': 'inline-block'}),
                           
            ],style={'margin-bottom':'20px'}),
            html.Div([
                html.Div([
                    html.P("App Ratings"),
                    dcc.RangeSlider(min=0, max=5, step=1, value=[-1], id='app-ratings-slider'),
                ],style={'width': '20%','display': 'inline-block','margin-right':'20px'}),
                html.Div([
                    html.P("Content Rating"),
                    # dcc.RangeSlider(min=-1, max=5, step=1, value=[-1], id='my-range-slider'),
                    # dcc.Checklist(['New York City', 'Montréal', 'San Francisco','New York City', 'Montréal', 'San Francisco'],inline=True)
                    dcc.Checklist(content_rating,inline=True,id='content-ratings-checklist')

                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    # html.P("App Ratings"),
                    html.Button('Reset Filter',id='reset-button1',n_clicks=0)
                ],style={'width': '20%','display': 'inline-block'})
            ]),]),

            html.Div([

            ],id='app_list'),

            html.Div([],id='singleappdetails')
])
############### code block for app details layout ##########################
@my_app.callback(Output(component_id='Category-Selector1', component_property='value'),
                Output(component_id='free-radio1', component_property='value'),
                Output(component_id='ad-radio1', component_property='value'),
                Output(component_id='purchase-radio1', component_property='value'),
                Output(component_id='editor-radio1', component_property='value'),
                Output(component_id='app-ratings-slider', component_property='value'),
                Output(component_id='content-ratings-checklist', component_property='value'),
                Output(component_id='realse-picker', component_property='start_date'),
                Output(component_id='realse-picker', component_property='end_date'),
                Input('reset-button1','n_clicks'),)
def reset_button(n_clicks):
    return None,None,None,None,None,[0],[],None,date.today()
##################### callback for app list ##################
@my_app.callback(Output(component_id='app_list',component_property='children'),
                [Input(component_id='Category-Selector1', component_property='value'),
                Input(component_id='free-radio1', component_property='value'),
                Input(component_id='ad-radio1', component_property='value'),
                Input(component_id='purchase-radio1', component_property='value'),
                Input(component_id='editor-radio1', component_property='value'),
                Input(component_id='realse-picker', component_property='start_date'),
                Input(component_id='realse-picker', component_property='end_date'),
                Input(component_id='content-ratings-checklist', component_property='value'),
                Input(component_id='app-ratings-slider', component_property='value')]
                )
def Filter(Category,free,ad,purchase,editor,start_date,end_date,contentRatings,appratings):
    play_store_apps_df_filtered=play_store_apps_df
    # print(play_store_apps_df_filtered)
    if Category is not None:
        play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered.Category==Category]
        print(play_store_apps_df_filtered.shape)
    else:
        print("Category Empty")    
    if free is not None:
        if(free == 'Yes'):
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered.Free==True]
            print(play_store_apps_df_filtered.shape)
        else:
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered.Free==False]
            print(play_store_apps_df_filtered.shape)
    else:
        print("free empty")
    if ad is not None:
        if(ad == 'Yes'):
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['Ad Supported']==True]
            print(play_store_apps_df_filtered.shape)
        else:
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['Ad Supported']==False]
            print(play_store_apps_df_filtered.shape)
    else:
        print("ad empty")
    if purchase is not None:
        if(purchase == 'Yes'):
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['In App Purchases']==True]
            print(play_store_apps_df_filtered.shape)
        else:
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['In App Purchases']==False]
            print(play_store_apps_df_filtered.shape)
    else:
        print("purchase empty")
    if editor is not None:
        if(editor == 'Yes'):
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['Editors Choice']==True]
            print(play_store_apps_df_filtered.shape)
        else:
            play_store_apps_df_filtered=play_store_apps_df_filtered.loc[play_store_apps_df_filtered['Editors Choice']==False]
            print(play_store_apps_df_filtered.shape)
    else:
        print("editor empty")
    if start_date is not None:
        if end_date is None:
            print(date.today())
            end_date=date.today()
        play_store_apps_df_filtered=play_store_apps_df_filtered.loc[(play_store_apps_df_filtered['Released']>start_date) &(play_store_apps_df_filtered['Released']<end_date) ]
        print(play_store_apps_df_filtered.shape)   
    else:
        print("start_date empty")
    if contentRatings is not None and (len(contentRatings)>0):
        play_store_apps_df_filtered=play_store_apps_df_filtered.loc[(play_store_apps_df_filtered['Content Rating'].isin(contentRatings))]
        print(play_store_apps_df_filtered.shape)
    else:
        print("contentRatings empty")
    if(appratings[0]!=0):
        start=appratings[0]
        end=start+1
        play_store_apps_df_filtered=play_store_apps_df_filtered.loc[((play_store_apps_df_filtered['Rating']>=start) & (play_store_apps_df_filtered['Rating']<end) )]
        print(play_store_apps_df_filtered.shape)
    else:
        print("appratings empty")
    
    score=play_store_apps_df_filtered.shape[0]
    print(type(play_store_apps_df))
    
    global temp_df
    global csv_df
    csv_df=play_store_apps_df_filtered
    # temp_df=play_store_apps_df_filtered[['Index','App Name','Category','Rating','Rating Count','Installs','Maximum Installs'
    # ,'Free','Price','Size','Minimum Android','Released','Content Rating','In App Purchases','Editors Choice','Ad Supported']]

    temp_df=play_store_apps_df_filtered[['App Name','Category','Rating','Rating Count','Installs'
    ,'Free','Price','Size','Minimum Android','Released','Content Rating']]
    # temp_df=play_store_apps_df_filtered
    
    div=html.Div([
        html.Hr(style={'border':'2px solid black','border-radius':'5px'}),
        html.H6(f"Total Apps Count: {score}",style={'text-align':'center',}),
        html.H5("Details of Application",style={'text-align':'center'}),
        dash_table.DataTable(
            id="app_details_table",
            columns=[{"name":i,"id":i} for i in (temp_df.columns) ],
            page_current=0,
            page_size=10,
            page_action='custom',
            sort_mode='multi',
            # filter_action='native',
            sort_by=[],
            selected_rows=['App Name'],
            style_data_conditional=[
               {'if':{
                    'filter_query': '{Free} contains "true"',
                    'column_id': 'Free'
                },
                'backgroundColor': 'green',
                'color':'white'
               },
               {'if':{
                    'filter_query': '{Free} contains "false"',
                    'column_id': 'Free'
                },
                'backgroundColor': 'red',
                'color':'white'
               }
            ]

        ),
        html.P("**** Click on the App to see details ****",style={'color':'red'}),
        html.Div([
            
            html.P("Download the Filterd CSV   :  ",style={'display': 'inline-block'}),
            html.Button('Download CSV',id='csv_download_Button',style={'display': 'inline-block'}),
            dcc.Download(id='save_csv_promot'),
        ],style={'display':'flex','justify-content':'flex-end','padding-top':'30px'}),

        # dbc.Pagination(max_value=page,fully_expanded=False)

    ])
    return div
################# dash table #################
@my_app.callback(Output('app_details_table', "data"),
                Output(component_id='singleappdetails',component_property='children'),
                Input('app_details_table', "page_current"),
                Input('app_details_table', "page_size"),
                Input('app_details_table', "sort_by"),
                Input('app_details_table', 'active_cell'))
def update_dash_table(page_current,page_size,sort_by,active_cell):
    # print(temp_df)
    # active_row_id=active_cell['row_id'] if active_cell else None
    print("Row table")
    div2=html.Div([
        html.Hr(style={'border':'2px solid black','border-radius':'5px'}),
        ])
    print(active_cell)
    if active_cell is not None:
        print(active_cell['row'])
        single_df=csv_df.iloc[[active_cell['row']]]
        print(single_df)
        div2=html.Div([
            html.Hr(style={'border':'2px solid black','border-radius':'5px'}),
            html.H6(f"Let's See the all the details of {single_df['App Name'].values[0]}",style={'text-align':'center',}),
            html.Table([
            html.Tr([html.Td('App Id'),
            html.Td(f'{single_df["App Id"].values[0]}'),
            html.Td('Category'),
            html.Td(f'{single_df["Category"].values[0]}'),]),

            html.Tr([
                html.Td('Free'),
                html.Td(f'{single_df["Free"].values[0]}'),
                html.Td('Rating'),
                html.Td(f'{single_df["Rating"].values[0]}'),
            ]),
            html.Tr([
                html.Td('Rating Count'),
                html.Td(f'{single_df["Rating Count"].values[0]}'),
                html.Td('Installs'),
                html.Td(f'{single_df["Installs"].values[0]}'),
            ]),
            html.Tr([
                html.Td('Rating Count'),
                html.Td(f'{single_df["Rating Count"].values[0]}'),
                html.Td('Installs'),
                html.Td(f'{single_df["Installs"].values[0]}'),
            ]),
            html.Tr([
                html.Td('Minimum Installs'),
                html.Td(f'{single_df["Minimum Installs"].values[0]}'),
                html.Td('Maximum Installs'),
                html.Td(f'{single_df["Maximum Installs"].values[0]}'),
            ]),
            html.Tr([
                html.Td('Price'),
                html.Td(f'{single_df["Price"].values[0]}'),
                html.Td('Currency'),
                html.Td(f'{single_df["Currency"].values[0]}'),
            ]),
            html.Tr([
                html.Td('Size'),
                html.Td(f'{single_df["Size"].values[0]}'),
                html.Td('Minimum Android'),
                html.Td(f'{single_df["Minimum Android"].values[0]}'),
            ]),
            html.Tr([
                html.Td('Developer Id'),
                html.Td(f'{single_df["Developer Id"].values[0]}'),
                html.Td('Developer Website'),
                html.Td(html.A(children=f'{single_df["Developer Website"].values[0]}',href=f'{single_df["Developer Website"].values[0]}',style={'color': 'blue', 'text-decoration': 'none'}  )  ),
            ]),
            html.Tr([
                html.Td('Rating'),
                html.Td(f'{single_df["Rating"].values[0]}'),
                html.Td('Installs'),
                html.Td(f'{single_df["Installs"].values[0]}'),
            ]),
            html.Tr([
                html.Td('Developer Email'),
                html.Td(f'{single_df["Developer Email"].values[0]}'),
                html.Td('Released'),
                html.Td(f'{single_df["Released"].values[0]}'),
            ]),
            # html.Tr([
            #     html.Td('Last Updated'),
            #     html.Td(f'{single_df["Last Updated"].values[0]}'),
            #     html.Td('Content Rating'),
            #     html.Td(f'{single_df["Content Rating"].values[0]}'),
            # ]),
            html.Tr([
                html.Td('Privacy Policy'),
                html.Td(html.A(children=f'{single_df["Privacy Policy"].values[0]}',href=f'{single_df["Privacy Policy"].values[0]}',style={'color': 'blue', 'text-decoration': 'none'}   ) ),
                html.Td('Ad Supported'),
                html.Td(f'{single_df["Ad Supported"].values[0]}'),
            ]),
            html.Tr([
                html.Td('In App Purchases'),
                html.Td(f'{single_df["In App Purchases"].values[0]}'),
                html.Td('Editors Choice'),
                html.Td(f'{single_df["Editors Choice"].values[0]}'),
            ]),
            
            ],style={'margin':'0px auto','text-align':'center','padding-bottom':'30px'})

        ],)
    else:
        div2=html.Div([
        html.Hr(style={'border':'2px solid black','border-radius':'5px'}),
        ])
    
    if len(sort_by):
        dff=temp_df.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )
    else:
        # No sort is applied
        dff = temp_df
    return dff.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records'),div2
################# dash table #################

############ CSV download Code Blocks##########
@my_app.callback(Output('save_csv_promot','data'),
                Input('csv_download_Button','n_clicks'),
                prevent_initial_call=True,)
def save_csv(n_clicks):
    return dcc.send_data_frame(csv_df.to_csv,'filtered_csv_from_dash.csv')
############ CSV download Code Blocks##########
##################### callback for app list ##################




############################ Graph Visualization Layer #########################
app_graph_visualization_layer=html.Div([
    dcc.Dropdown(
        options=[
       {'label': 'Line-plot', 'value': 'lineplot'},
       {'label': 'Bar-plot ', 'value': 'barplot'},
       {'label': 'Count-plot', 'value': 'countplot'},
        {'label': 'Cat-plot', 'value': 'catplot'},
       {'label': 'Pie-chart', 'value': 'piechart'},
       {'label': 'Displot', 'value': 'displot'},
       {'label': 'Heatmap', 'value': 'heatmap'},
        {'label': 'Histogram and Distplot', 'value': 'histplot'},
        {'label': 'Scatter plot', 'value': 'scatter'},
       {'label': 'Multivariate Box plot', 'value': 'multiboxplot'},
       {'label': 'Violin plot', 'value': 'violinplot'},
   ],placeholder="Select Your Favorite Graph", searchable=False, id='graph_selector',style={'margin-top':'20px'}
    ),
    html.Hr(style={'border':'2px solid black','border-radius':'5px'}),

    html.Div(id="graph_div")

])

############################ Graph Visualization Callback #########################

@my_app.callback(Output(component_id='graph_div',component_property='children'),
                [Input(component_id='graph_selector',component_property='value')])
def graph_selector(graph_selector):
    # print(play_store_apps_df)
    temp_year=pd.DataFrame()
    # play_store_apps_df[]
    play_store_apps_df['Released'] = pd.to_datetime(play_store_apps_df['Released'])
    play_store_apps_df['year']=play_store_apps_df['Released'].dt.year
    temp_year = play_store_apps_df.year.value_counts(sort=True,ascending=False).rename_axis('year').reset_index(name='counts')
    temp_year=temp_year.sort_values(by=['year'])
    released_year=play_store_apps_df.groupby('year')['Installs'].sum().rename_axis('year').reset_index(name='Installs Count')
    released_maximum=play_store_apps_df.groupby('year')['Maximum Installs'].sum().rename_axis('year').reset_index(name='Maximum Installs Count')
    if graph_selector=="lineplot":
        line_fig1= px.line(temp_year, x="year", y="counts",title='Release year vs count of released apps')
        line_fig2= px.line(released_year, x="year", y="Installs Count",title='Release year vs count of total Installations of Apps')
        line_fig3= px.line(released_maximum, x="year", y="Maximum Installs Count",title='Release year vs count of Maximum Installations of Apps')
        line_div=html.Div([
            html.Div([
                html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=line_fig1)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=line_fig2)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
            html.Div([
                    dcc.Graph(figure=line_fig3)
            ]),
        ])
        return line_div
    elif graph_selector=="barplot":
        bar_fig1=px.bar(play_store_apps_df,x='Category', y='Installs',color='Content Rating',title='App Installs Vs Categories by Content rating')
        bar_fig2=px.bar(play_store_apps_df,x='Category', y='Installs',color='Free',title='App Installs Vs Categories by Free')
        bar_fig3=px.bar(play_store_apps_df,x='Category', y='Installs',color='In App Purchases',title='App Installs Vs Categories by In App Purchases')
        bar_fig4=px.bar(play_store_apps_df,x='Category', y='Installs',color='Editors Choice',title='App Installs Vs Categories by Editors Choice')
        # bar_fig1=px.bar(play_store_apps_df,x='Category', y='Installs')
        # bar_fig2=px.bar(play_store_apps_df,x='Category', y='Installs')
        # bar_fig3=px.bar(play_store_apps_df,x='Category', y='Installs')
        # bar_fig4=px.bar(play_store_apps_df,x='Category', y='Installs')
        bar_div=html.Div([
            html.Div([
                html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=bar_fig1)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=bar_fig2)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
            html.Div([
                    html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=bar_fig3)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=bar_fig4)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
        ])
        return bar_div
    elif graph_selector=="countplot":
        minimum_androind=play_store_apps_df.groupby('Minimum Android').count().reset_index()
        minimum_androind=minimum_androind.rename(columns={"Developer Id":"Count"})
        category=play_store_apps_df.groupby('Category').count().reset_index()
        category=category.rename(columns={"Developer Id":"Count"})
        released_year=play_store_apps_df.groupby('year').count().reset_index()
        released_year=released_year.rename(columns={"Developer Id":"Count"})
        content_rating=play_store_apps_df.groupby('Content Rating').count().reset_index()
        content_rating=content_rating.rename(columns={"Developer Id":"Count"})
        count_plot1=px.bar(minimum_androind,x='Minimum Android',y='Count',title="Count of Minimum Android Version",color_discrete_sequence=['chocolate']*len(minimum_androind))
        count_plot2=px.bar(category,x='Category',y='Count',title="Count of Categories",color_discrete_sequence=['darkmagenta']*len(category))
        count_plot3=px.bar(released_year,x='year',y='Count',title="App releases over the Years",color_discrete_sequence=['crimson']*len(released_year))
        count_plot4=px.bar(content_rating,x='Content Rating',y='Count',title="Count of Content Rating",color_discrete_sequence=['coral']*len(content_rating))
        count_div=html.Div([
            html.Div([
                html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=count_plot1)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=count_plot2)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
            html.Div([
                    html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=count_plot3)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=count_plot4)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
        ])
        return count_div
    elif graph_selector=="catplot":
        free=play_store_apps_df.groupby('Free').count().reset_index()
        free=free.rename(columns={"Developer Id":"Count"})
        editor_choice=play_store_apps_df.groupby('Editors Choice').count().reset_index()
        editor_choice=editor_choice.rename(columns={"Developer Id":"Count"})
        in_app_purchase=play_store_apps_df.groupby('In App Purchases').count().reset_index()
        in_app_purchase=in_app_purchase.rename(columns={"Developer Id":"Count"})
        ad_supported=play_store_apps_df.groupby('Ad Supported').count().reset_index()
        ad_supported=ad_supported.rename(columns={"Developer Id":"Count"})
        cat_plot1=px.bar(free,x='Free',y='Count',title="Free Apps",color_discrete_sequence=['darkorange']*len(free))
        cat_plot2=px.bar(editor_choice,x='Editors Choice',y='Count',title="Editor's Choice",color_discrete_sequence=['darksalmon']*len(editor_choice))
        cat_plot3=px.bar(in_app_purchase,x='In App Purchases',y='Count',title="In App Purchase",color_discrete_sequence=['darkslateblue']*len(in_app_purchase))
        cat_plot4=px.bar(ad_supported,x='Ad Supported',y='Count',title="Ad Supported",color_discrete_sequence=['darkcyan']*len(ad_supported))
        cat_div=html.Div([
            html.Div([
                html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=cat_plot1)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=cat_plot2)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
            html.Div([
                    html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=cat_plot3)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=cat_plot4)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
        ])
        return cat_div
    elif graph_selector=="piechart":
        category=play_store_apps_df.groupby('Category').count().reset_index()
        category=category.rename(columns={"Developer Id":"Categories"})
        category['Price']=category['Price'].astype(float)
        category=category.loc[category.Price > 2000]
        # print(category.dtypes)
        content_rating=play_store_apps_df.groupby('Content Rating').count().reset_index()
        content_rating=content_rating.rename(columns={"Developer Id":"Content Ratings"})
        released_year=play_store_apps_df.groupby('year').count().reset_index()
        released_year=released_year.rename(columns={"Developer Id":"Years"})
        currency=play_store_apps_df.groupby('Free').count().reset_index()
        currency=currency.rename(columns={"Developer Id":"Frees"})
        # print(category)
        pie_chart1=px.pie(category,values='Categories',names='Category',title="App Categories",color_discrete_sequence=px.colors.sequential.Reds)
        pie_chart2=px.pie(content_rating,values='Content Ratings',names='Content Rating',title="App Content Ratings",color_discrete_sequence=px.colors.sequential.Darkmint)
        pie_chart3=px.pie(released_year,values='Years',names="year",title="App Released Year",color_discrete_sequence=px.colors.sequential.Electric)
        pie_chart4=px.pie(currency,values='Frees',names='Free',title="Free or Paid",color_discrete_sequence=px.colors.sequential.Burgyl)
        pie_div=html.Div([
            html.Div([
                html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=pie_chart1)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=pie_chart2)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
            html.Div([
                    html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=pie_chart3)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=pie_chart4)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
        ])
        return pie_div
    elif graph_selector=="displot":
        # Price (Distribution of App Prices)
        # Installs (Distribution of App Installs)
        # Rating (Distribution of App Ratings)
        dis_fig1=px.histogram(play_store_apps_df,x='year',y='Price',color='Free', hover_data=play_store_apps_df.columns,
        title="Distributions of App Prices Over the Year Based on App Free or not",opacity=0.5)
        dis_fig2=px.histogram(play_store_apps_df,x='year',y='Installs',color='Ad Supported', 
        hover_data=play_store_apps_df.columns,title="Distributions of App Installs Over the Year Based on App Ad Supported or not",opacity=0.5)
        dis_fig3=px.histogram(play_store_apps_df,x='year',y='Rating',color='In App Purchases',
        hover_data=play_store_apps_df.columns,title="Distributions of App Ratings Over the Year Based on In App Purchases present or not",opacity=0.5)
        dis_fig4=px.histogram(play_store_apps_df,x='year',y='Rating Count',color='Editors Choice',
        hover_data=play_store_apps_df.columns,title="Distributions of App Rating Count Over the Year Based on Editor's choice or not",opacity=0.5)

        #  dis_fig1=px.histogram(play_store_apps_df,x='year',y='Price',color='Free',marginal="box", hover_data=play_store_apps_df.columns,
        # title="Distributions of App Prices Over the Year Based on App Free or not")
        # dis_fig2=px.histogram(play_store_apps_df,x='year',y='Installs',color='Ad Supported',marginal="violin", 
        # hover_data=play_store_apps_df.columns,title="Distributions of App Installs Over the Year Based on App Ad Supported or not")
        # dis_fig3=px.histogram(play_store_apps_df,x='year',y='Rating',color='In App Purchases',marginal="rug", 
        # hover_data=play_store_apps_df.columns,title="Distributions of App Ratings Over the Year Based on In App Purchases present or not")
        # dis_fig4=px.histogram(play_store_apps_df,x='year',y='Rating Count',color='Editors Choice',marginal="violin",
        #  hover_data=play_store_apps_df.columns,title="Distributions of App Rating Count Over the Year Based on Editor's choice or not")
        # dis_fig2=ff.create_distplot(hisdata,datalabels)
        dis_div=html.Div([
            html.Div([
                html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=dis_fig1)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=dis_fig2)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
            html.Div([
                    html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=dis_fig3)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=dis_fig4)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
        ])

        return dis_div
    elif graph_selector=="heatmap":
        temp_df_heatmap=pd.DataFrame()
        temp_df_heatmap=play_store_apps_df.drop(['Unnamed: 0'],axis=1)
        # temp_df_heatmap=play_store_apps_df
        # print(temp_df_heatmap)
        mat_corr=temp_df_heatmap.corr()
        
        heat_fig1=px.imshow(mat_corr.round(2),text_auto=True,aspect="auto",title='Correlation Between the Numeric Features')
        heat_div=html.Div([
            html.Div([
                html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=heat_fig1)
                ]),
                
            ]),
            # html.Div([
            #         html.Div([
            #         # html.P("Release year vs count of released apps"),
            #         dcc.Graph(figure=heat_fig1)
            #     ],style={'width': '50%','display': 'inline-block'}),
            #     html.Div([
            #         dcc.Graph(figure=heat_fig1)
            #     ],style={'width': '50%','display': 'inline-block'}),
            # ]),
        ])
        return heat_div
    elif graph_selector=="histplot":
        # histgram_fig1=px.histogram(play_store_apps_df,x='Rating',title='Histogram of Ratings')
        # temp_price=play_store_apps_df[play_store_apps_df['Price']>0.0]
        # print(temp_price)
        histgram_fig2=px.histogram(play_store_apps_df,x='Rating',title='Histogram of Rating',color_discrete_sequence=['hotpink']*len(play_store_apps_df))
        histgram_fig4=px.histogram(play_store_apps_df,x='year',title='Histogram of Year',color_discrete_sequence=['indianred']*len(play_store_apps_df))
        # histgram_fig4=px.histogram(play_store_apps_df[play_store_apps_df['Price']>0.0],x='Price',title='Histogram of Price',color_discrete_sequence=['mediumorchid']*len(play_store_apps_df))
        hist_data11=play_store_apps_df['Rating']
        hist_data1=[hist_data11]
        group_labels1=['Rating']
        histgram_fig1=ff.create_distplot(hist_data=hist_data1,group_labels=group_labels1)
        histgram_fig1.layout.update({'title': 'Distribution Plot for Ratings'})

        hist_data12=play_store_apps_df['year']
        hist_data2=[hist_data12]
        group_labels2=['year']
        histgram_fig3=ff.create_distplot(hist_data=hist_data2,group_labels=group_labels2)
        histgram_fig3.layout.update({'title': 'Distribution Plot for year'})

        # hist_data13=play_store_apps_df['Installs']
        # hist_data3=[hist_data13]
        # group_labels3=['Installs']
        # histgram_fig3=ff.create_distplot(hist_data=hist_data3,group_labels=group_labels3)
        # histgram_fig3.layout.update({'title': 'Distribution Plot for Installs'})

        # hist_data14=play_store_apps_df['Price']
        # hist_data4=[hist_data14]
        # group_labels4=['Price']
        # histgram_fig4=ff.create_distplot(hist_data=hist_data4,group_labels=group_labels4)
        # histgram_fig4.layout.update({'title': 'Distribution Plot for Ratings Count'})


        histgram_div=html.Div([
        html.Div([
                html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=histgram_fig1)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=histgram_fig2)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
            html.Div([
                    html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=histgram_fig3)
                ],style={'width': '50%','display': 'inline-block'}),
                html.Div([
                    dcc.Graph(figure=histgram_fig4)
                ],style={'width': '50%','display': 'inline-block'}),
            ]),
        ])
        return histgram_div
    elif graph_selector=="qqplot":
        # qqplot_fig1=qqplot(play_store_apps_df['Installs'])
        
        # qqplot_div=html.Div([
        #     html.Div([
        #         html.Div([
        #             # html.P("Release year vs count of released apps"),
        #             dcc.Graph(figure=qqplot_fig1)
        #         ],style={'width': '50%','display': 'inline-block'}),
        #         html.Div([
        #             dcc.Graph(figure=qqplot_fig1)
        #         ],style={'width': '50%','display': 'inline-block'}),
        #     ]),
        #     html.Div([
        #             html.Div([
        #             # html.P("Release year vs count of released apps"),
        #             dcc.Graph(figure=qqplot_fig1)
        #         ],style={'width': '50%','display': 'inline-block'}),
        #         html.Div([
        #             dcc.Graph(figure=qqplot_fig1)
        #         ],style={'width': '50%','display': 'inline-block'}),
        #     ]),
        # ])
        
        return html.P([f'{graph_selector} qqplot'])
    elif graph_selector=="kde":
        return html.P([f'{graph_selector} kde'])
    elif graph_selector=="scatter":
        # X = price Y=rating color = rating_count (Scatterplot between App Price and App Ratings by Rating Counts)
        # X = price Y=installs color = rating_count  (Scatterplot between App Price and App Installs by Rating Counts)
        sca_fig1=px.scatter(play_store_apps_df,x='Price',y='Rating',color='Rating Count', trendline='ols',title='Scatterplot between App Price and App Ratings by Rating Counts')
        sca_fig2=px.scatter(play_store_apps_df,x='Price',y='Installs',color='Rating Count',trendline='ols',title='Scatterplot between App Price and App Installs by Rating Counts')
        scatter_div=html.Div([
            html.Div([
                html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=sca_fig1)
                ]),
                html.Div([
                    dcc.Graph(figure=sca_fig2)
                ]),
            ]),
        ])
        return scatter_div
    elif graph_selector=="multiboxplot":
        # Category, rating (Box Plot of App Categories by Ratings)
        # category , price (Box Plot of App Categories by Price)
        box_fig1=px.box(play_store_apps_df,x='Category',y='Rating',title='Box Plot of App Categories by Ratings',color_discrete_sequence=['mediumpurple']*len(play_store_apps_df))
        box_fig2=px.box(play_store_apps_df,x='Content Rating',y='Rating',title='Box Plot of App Content Rating by Ratings',color_discrete_sequence=['olivedrab']*len(play_store_apps_df))

        multibox_div=html.Div([
            html.Div([
                html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=box_fig1)
                ]),
                html.Div([
                    dcc.Graph(figure=box_fig2)
                ]),
            ]),
        ])
        return multibox_div
    elif graph_selector=="areaplot":
        return html.P([f'{graph_selector} areaplot'])
    elif graph_selector=="violinplot":
        # Category, rating (Violin Plot of App Categories by Ratings)
        # category , price (Violin Plot of App Categories by Price)
        vio_fig1=px.violin(play_store_apps_df,x='Category',y='Rating',color='Ad Supported',title='Violin Plot of App Categories and Ratings divide into Ad Supported or not')
        vio_fig2=px.violin(play_store_apps_df,x='Content Rating',y='Rating',color='In App Purchases',title='Violin Plot of App Content Ratings and Ratings divide into In App Purchases or not')
        viloin_div=html.Div([
            html.Div([
                html.Div([
                    # html.P("Release year vs count of released apps"),
                    dcc.Graph(figure=vio_fig1)
                ]),
                html.Div([
                    dcc.Graph(figure=vio_fig2)
                ]),
            ]),
        ])

        return viloin_div
        

    # return html.P([f'{graph_selector}'])
############################ Graph Visualization Callback #########################


############################ Graph Visualization Layer #########################

################### Normality Test #######################
# trace=go.Histogram(
#     x=play_store_apps_df['Rating']
# )
# fig=iplot([trace],filename='JJJ')

# print(play_store_apps_df.dtypes)
app_graph_normality_layer=html.Div([
    html.Div([
        html.Div([
            html.P("Select the Column"),
            dcc.Dropdown(
                options=[
                    {'label': 'Installs', 'value': 'Installs'},
                    {'label': 'Rating Count', 'value': 'Rating Count'},
                    {'label': 'Rating', 'value': 'Rating'},
                    {'label': 'Minimum Installs', 'value': 'Minimum Installs'},
                    {'label': 'Maximum Installs ', 'value': 'Maximum Installs'},
                    {'label': 'Price', 'value': 'Price'},
                ],
                value='Rating',
                id='normal_dropdown_1'
            )
        ],style={'width': '45%','display': 'inline-block'}),
        html.Div([],style={'width': '10%','display': 'inline-block'}),
        html.Div([
            html.P("Select the Normality Test"),
            dcc.Dropdown(
                options=[
                    {'label': 'Quantile-Quantile Plot', 'value': 'qqplot'},
                    {'label': 'Shapiro-Wilk Test', 'value': 'shapiro'},
                    {'label': 'Kolmogorov-Smirnov (K-S) Test', 'value': 'anderson'},
                    {'label': "D'Agostino's Test", 'value': 'k2'},
                ],
                value='qqplot',
                id='normal_dropdown_2'
            )
        ],style={'width': '45%','display': 'inline-block'})
        
    ]),

    html.Div([

    ],id='Normality_div',style={'padding-top':'50px'})

    

])

########### normality callback #############
@my_app.callback(
        Output(component_id='Normality_div',component_property='children'),
        [Input(component_id='normal_dropdown_1',component_property='value'),
        Input(component_id='normal_dropdown_2',component_property='value')]
)

def normality_test(drop1,drop2):
    print(drop1)
    print(drop2)
    if (drop1 is None ) and (drop2 is None):
        return html.H3("Please Select Both of the options. If Any one of the dropdown is not selected then it will not work",style={'text-align':'center'})
    else:
        if(drop2=='qqplot'):
            qqplot_data = qqplot(play_store_apps_df[drop1], line='s').gca().lines
            fig = go.Figure()

            fig.add_trace({
                'type': 'scatter',
                'x': qqplot_data[0].get_xdata(),
                'y': qqplot_data[0].get_ydata(),
                'mode': 'markers',
                'marker': {
                    'color': '#19d3f3'
                }
            })

            fig.add_trace({
                'type': 'scatter',
                'x': qqplot_data[1].get_xdata(),
                'y': qqplot_data[1].get_ydata(),
                'mode': 'lines',
                'line': {
                    'color': '#636efa'
                }

            })


            fig['layout'].update({
                'title': 'Quantile-Quantile Plot',
                'xaxis': {
                    'title': 'Theoritical Quantities',
                    'zeroline': False
                },
                'yaxis': {
                    'title': 'Sample Quantities'
                },
                'showlegend': False,
                'width': 800,
                'height': 700,
            })
            fig.update_layout(
                title=f"Quantile-Quantile Plot of {drop1}"
            )
            return dcc.Graph(figure=fig)
        elif drop2=='shapiro':
            stat,p=st.shapiro(play_store_apps_df[drop1])
            alpha=0.01
            if p > alpha:
                result='Sample looks Gaussian (fail to reject H0)'
            else:
                result='Sample doesnot look Gaussian (reject H0)'
            result_mat = [
            ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
            [len(play_store_apps_df), stat, p, result]
            ]

            dash_table_shapiro=dbc.Container([
                dbc.Label('Shapiro-Wilk Test',style={'text-align':'center'}),
                dash_table.DataTable(pd.DataFrame(result_mat).to_dict('result'),[{"name": i, "id": i} for i in pd.DataFrame(result_mat).columns]),

            ])
            return dash_table_shapiro
        elif drop2=='anderson':
            stat,p=st.kstest(play_store_apps_df[drop1],'norm')
            alpha=0.01
            if p > alpha:
                result='Sample looks Gaussian (fail to reject H0)'
            else:
                result='Sample doesnot look Gaussian (reject H0)'
            result_mat = [
            ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
            [len(play_store_apps_df), stat, p, result]
            ]

            dash_table_ks=dbc.Container([
                dbc.Label('Kolmogorov-Smirnov (K-S) Test',style={'text-align':'center'}),
                dash_table.DataTable(pd.DataFrame(result_mat).to_dict('result'),[{"name": i, "id": i} for i in pd.DataFrame(result_mat).columns]),

            ])
            return dash_table_ks
        elif drop2=='k2':
            stat,p=st.normaltest(play_store_apps_df[drop1])
            alpha=0.01
            if p > alpha:
                result='Sample looks Gaussian (fail to reject H0)'
            else:
                result='Sample doesnot look Gaussian (reject H0)'
            result_mat = [
            ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
            [len(play_store_apps_df), stat, p, result]
            ]

            dash_table_k2=dbc.Container([
                dbc.Label("D'Agostino's Test",style={'text-align':'center'}),
                dash_table.DataTable(pd.DataFrame(result_mat).to_dict('result'),[{"name": i, "id": i} for i in pd.DataFrame(result_mat).columns]),

            ])
            return dash_table_k2

################### Normality Test #######################
############################ Code Blocks for Tabs ################################
@my_app.callback(Output(component_id='layout',component_property='children'),
                [Input(component_id='project',component_property='value')])

def update_layout(tabs):
    if tabs == 'app':
        return app_statistics_layout
    elif tabs == 'compare':
        return app_comparision_layout
    elif tabs=='details':
        return app_details_layout
    elif tabs=='graph':
        return app_graph_visualization_layer
    elif tabs=='normality':
        return app_graph_normality_layer

############################ Code Blocks for Tabs ################################
my_app.server.run(debug = 'True')