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

from sklearn.metrics.pairwise import cosine_similarity
import operator
# reading data
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
    genres_dic.append({key_list[0]: genres_list[idx],
                       key_list[1]: genres_list[idx]})


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


# algrithm of recommander by rating

def similar_users(new_user_ratings, matrix, k=3):
    # https://towardsdatascience.com/build-a-user-based-collaborative-filtering-recommendation-engine-for-anime-92d35921f304
    # create a df of just the current user
    new_user_ratings = new_user_ratings.copy()
    other_users = matrix
    # calc cosine similarity between user and each other user
    similarities = cosine_similarity(new_user_ratings, other_users)[0].tolist()

    # create list of indices of these users
    indices = other_users.index.tolist()
    # create key/values pairs of user index and their similarity
    index_similarity = dict(zip(indices, similarities))
    # sort by similarity
    index_similarity_sorted = sorted(
        index_similarity.items(), key=operator.itemgetter(1), reverse=True)

    # grab k users off the top
    top_users_similarities = index_similarity_sorted[:k]
    users = [u[0] for u in top_users_similarities]
    return users


def UBCF_recc_movies(ratings, similar_user_list, k=6):

    similar_ratings = ratings[ratings['UserID'].isin(
        similar_user_list)].drop('Timestamp', axis=1)
    similar_ratings['count'] = similar_ratings['UserID'].groupby(
        similar_ratings['MovieID']).transform('count')
    # only movies that received more number of ratings than 80% of all the movies in the genre are considered
    cutoff = similar_ratings['count'].quantile(0.8)
    similar_ratings = similar_ratings[similar_ratings['count'] >= cutoff]
    movie_avg_ratings = similar_ratings.groupby('MovieID').mean(
    ).sort_values(by=['Rating'], ascending=False).reset_index()
    movies_list = list(movie_avg_ratings['MovieID'])
    return movies_list[:6]

# reuseable component to display movies

def display_movies(movieID_list):
    if not movieID_list:
        return html.P('No movies are recommended')

    cards = []
    for ID in movieID_list:
        movieCard = dbc.Card(
            [
                dbc.CardImg(src=movies.loc[ID, 'image_url'], top=True),
                dbc.CardBody(
                    html.P(movies.loc[ID, 'Title'], className="card-text")
                ),
            ],
            color='info',
            inverse=True,
            style={'width': '22rem'},
        )
        cards.append(movieCard)
    rows = []
    cols = []
    for index in range(len(movieID_list)):
        if((index) % 3 == 2):
            cols.append(dbc.Col(cards[index]))
            rows.append(dbc.Row(cols, align="start"))
            cols = []
        else:
            cols.append(dbc.Col(cards[index], width=4))
    if cols:
        rows.append(dbc.Row(cols, align="start"))

    cardsDIV = html.Div(rows)
    return cardsDIV


# reuseable component to displat movies to rate

movieID_toRate_list = list(range(0, 40))


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
            color='info',
            inverse=True,
            style={'width': '22rem'},
        )
        cards.append(movieCard)
    rows = []
    cols = []
    for index in range(len(movieID_toRate_list)):
        if((index) % 4 == 3):
            cols.append(dbc.Col(cards[index]))
            rows.append(dbc.Row(cols, align="start"))
            cols = []
        else:
            cols.append(dbc.Col(cards[index]))

    cardsDIV = cardsDIV = html.Div(style={'height':'650px', 'overflow-y':'auto','overflow-x':'hidden'},children=rows)
    return cardsDIV


# System1 layout
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
                                              style={
                                                 'backgroundColor': '#1E1E1E'},
                                          ),
                                      ],
                                      style={'color': '#1E1E1E'})
                              ]
                              ),
                     html.Div(className='eight columns div-for-charts bg-grey',
                              children=[
                                  dcc.Loading(
                                    id="loading-1",
                                    type="default",
                                    children=html.Div(id="movie_recommandation1")
                                   )
                              ])
                 ])

    ]
)

# System2 layout
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
                     dcc.Loading(
                        id="loading-2",
                        type="default",
                        children=html.Div(id="movie_recommandation2")
                     ),
                 ])

    ]
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.layout = dbc.Tabs([dbc.Tab(tab1_content, label="Recommender by Genre"), dbc.Tab(
    tab2_content, label="Recommender by Rating"), ])

# update movieID_list for system1


@app.callback(
    Output(component_id='movie_recommandation1',
           component_property='children'),
    Input(component_id='favorite_genre', component_property='value')
)
def update_movieID_list1(selected_genre):
    if selected_genre:
        movieID_list = recc_genre_by_rating(movies, ratings, selected_genre)
        return display_movies(movieID_list)

# update movieID_list for system2


@app.callback(
    Output(component_id='movie_recommandation2',
           component_property='children'),
    Input(component_id='recommend_button', component_property='n_clicks'),
    State({'type': 'rating_value', 'index': ALL}, 'value')
)
def update_movieID_list2(n, values):
    user_rating_list = []
    for (i, value) in enumerate(values):
        if value:
            user_rating_list.append([i+1, value])
            print(user_rating_list)
    if len(user_rating_list) < 1:
        return html.P('Please rate above movies first')
    if n:
        new_user_ratings = [i[1] for i in user_rating_list]
        new_user_movies = [i[0] for i in user_rating_list]
        new_user_ratings_df = pd.DataFrame(
            np.array(new_user_ratings).reshape(1, -1), columns=new_user_movies)
        # find users in the database that also rated the movies rated by new user
        filtered_ratings = ratings[ratings['MovieID'].isin(
            new_user_movies)].drop('Timestamp', axis=1)
        # print(filtered_ratings)
        filtered_rating_matrix = filtered_ratings.pivot_table(
            index='UserID', columns='MovieID', values='Rating').fillna(0)

        similar_user_list = similar_users(
            new_user_ratings_df, filtered_rating_matrix, 5)
        print(similar_user_list)

        movieID_list = UBCF_recc_movies(ratings, similar_user_list, 6)
        print(movieID_list)
        return display_movies(movieID_list)


if __name__ == '__main__':
    app.run_server(debug=True)
