import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from dash.dependencies import Input, Output, State, ALL
import pandas as pd
import numpy as np

#reading data
ratings = pd.read_csv('ratings.dat', sep=':', header=None)
ratings = ratings.dropna(axis=1)
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

movies = pd.read_csv('movies.dat', sep='::', header=None, encoding='latin-1')
movies.columns = ['MovieID', 'Title', 'Genres']

users = pd.read_csv('users.dat', sep=':', header=None, encoding='latin-1')
users = users.dropna(axis=1)
users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']


small_image_url = "https://liangfgithub.github.io/MovieImages/"

movies['image_url'] = movies['MovieID'].apply(
    lambda x: small_image_url + str(x)+'.jpg?raw=true')

genres = movies['Genres'].str.split('|')
genres_set = set(np.concatenate(genres).flat)
genres_list = list(genres_set)
key_list = ['label', 'value']

genres_dic = []
for idx in range(0, len(genres_list)):
    genres_dic.append({key_list[0]: genres_list[idx], key_list[1]: genres_list[idx]})
    
# algrithm of recommander by genre
def recc_genre_by_rating(movies, ratings, genre):
    movies = movies.copy()
    ratings = ratings.copy()
    mov_rating = movies.set_index('MovieID').join(
        ratings.set_index('MovieID'), how='left', on='MovieID')
    mov_rating = mov_rating[mov_rating['Genres'].str.contains(genre)]
    mov_rating['count'] = mov_rating['UserID'].groupby(
        mov_rating['Title']).transform('count')

    # only movies that received more number of ratings than 90% of all the movies in the genre are considered
    cutoff = mov_rating['count'].quantile(0.8)
    mov_rating = mov_rating[mov_rating['count'] >=
                            cutoff].drop(['UserID', 'Timestamp'], axis=1)
    # mov_rating_mean = mov_rating.groupby('Title').mean().sort_values(by=['Rating','count'], ascending=False)
    mov_rating_mean = mov_rating.groupby('MovieID').mean().sort_values(
        by=['Rating', 'count'], ascending=False).reset_index()
    movies_list = list(mov_rating_mean['MovieID'])
    return movies_list[:6]


#reuseable component to display movies
def display_movies(movieID_list):
    #the size of movieID_list should be 6 or greater than 6 -- TBD
    if (len(movieID_list) < 6):
        return html.P('the size of Movie List is smaller than 6')
    
    cards = []
    for ID in movieID_list:
        movieCard = dbc.Card(
            [
                dbc.CardImg(src=movies.loc[ID, 'image_url'], top=True),
                dbc.CardBody(
                    html.P(movies.loc[ID, 'Title'], className="card-text")
                ),
            ],
            color = 'info',
            inverse = True,
            style={'width': '22rem'},
        )
        cards.append(movieCard)
    cardsDIV = html.Div(
        [
            html.H2('Here are some movies you might like'),
            dbc.Row(
                [dbc.Col(cards[0]), dbc.Col(cards[1]), dbc.Col(cards[2])],
                align="start",
            ),
            dbc.Row(
                [dbc.Col(cards[3]), dbc.Col(cards[4]), dbc.Col(cards[5])],
                align="start",
            ),
        ]
    )
    return cardsDIV


#reuseable component to displat movies to rate
#Do we want randomly to give movieID? How many movies do we want user to rate?--- TBD
movieID_toRate_list = list(range(1,21))
def display_movies_to_rate(movieID_toRate_list):
    cards = []
    for ID in movieID_toRate_list:
        movieCard = dbc.Card(
            [
                dbc.CardImg(src=movies.loc[ID, 'image_url'], top=True),
                dbc.CardBody(
                    html.P(movies.loc[ID, 'Title'], className="card-text")
                ),
                dcc.Slider(
                    id={
                        'type': 'rating_value',
                        'index': ID
                        },
                    min=1,
                    max=5,
                    step=1,
                    marks={
                        1: {'label': '1', 'style': {'color': '#F5ECCE'}},
                        2: {'label': '2', 'style': {'color': '#F5ECCE'}},
                        3: {'label': '3', 'style': {'color': '#F5ECCE'}},
                        4: {'label': '4', 'style': {'color': '#F5ECCE'}},
                        5: {'label': '5', 'style': {'color': '#F5ECCE'}},
                    },
                ),
            ],
            color = 'info',
            inverse = True,
            style={'width': '22rem'},
        )
        cards.append(movieCard)
    rows = []
    cols = []
    for index in range(len(movieID_toRate_list)):
        if((index)%4 == 3):
            cols.append(dbc.Col(cards[index]))
            rows.append(dbc.Row(cols,align="start"))
            cols=[]
        else:
            cols.append(dbc.Col(cards[index]))
    
    cardsDIV = html.Div(rows)
    return cardsDIV

    

#System1 layout
tab1_content = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H2('Movie Recommandation'),
                                 html.P('Please select your favorite genre:'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(
                                                      id='favorite_genre',
                                                      options=genres_dic,
                                                      style={'backgroundColor': '#1E1E1E'},
                                                      ),
                                     ],
                                     style={'color': '#1E1E1E'})
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                            children=[
                                 #html.H2('Here are some movies you might like'),
                                 html.Div(id='movie_recommandation1')
                             ])
            ])

        ]
)

#System2 layout
tab2_content = html.Div(
    children=[
        html.Div(className='div-user-controls',
                 children=[
                     html.H2('Movie Recommandation'),
                     html.P('Please rate these movies as many as possible:'),
                     display_movies_to_rate(movieID_toRate_list),
                     dbc.Button(
                          "Click Here to get your recommendations",
                           id="recommend_button",
                           block=True,
                           color="primary",
                       ),
                     dbc.Row(),
                     html.Div(id='movie_recommandation2'),
                ])
                
    ]
)

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app.server
app.layout = dbc.Tabs([dbc.Tab(tab1_content, label="Recommender by Genre"), dbc.Tab(tab2_content, label="Recommender by Rating"),])

#update movieID_list for system1
@app.callback(
    Output(component_id='movie_recommandation1', component_property='children'),
    Input(component_id='favorite_genre', component_property='value')
)
def update_movieID_list1(selected_genre):
    if selected_genre:
        movieID_list = recc_genre_by_rating(movies, ratings, selected_genre)
        return display_movies(movieID_list)
    
#update movieID_list for system2
@app.callback(
    Output(component_id='movie_recommandation2', component_property='children'),
    Input(component_id='recommend_button', component_property='n_clicks'),
    State({'type': 'rating_value', 'index': ALL}, 'value')
)
def update_movieID_list2(n, values):
    user_rating_list=[]
    for (i, value) in enumerate(values):
        if value:
            user_rating_list.append([i, value])
    if len(user_rating_list) < 1:
        return html.P('Please rate above movies first')
    user_rating_df = pd.DataFrame(user_rating_list, columns = ['MovieID', 'Rating'])
    if n:
        # TODO: add system2 recommandation algorithm based on user_rating_df here
        if len(user_rating_df) < 3:
            movieID_list = [1,1,1,1,1,1]
            return display_movies(movieID_list)
        else:
            movieID_list = [1,1,1,2,2,2]
            return display_movies(movieID_list)

if __name__ == '__main__':
    app.run_server(debug=True)