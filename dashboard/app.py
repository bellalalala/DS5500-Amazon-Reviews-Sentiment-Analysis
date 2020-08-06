# Import required libraries
from flask import send_from_directory
import io
import shutil
from urllib.parse import quote as urlquote
import pathlib
import base64
import dash
import pandas as pd
import fastText
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from wordcloud import WordCloud
from datetime import datetime, timedelta
import pickle
import os
import json
import plotly.graph_objs as go
import plotly.express as px


external_stylesheets=[
    {'href':"https://fonts.googleapis.com/css2?family=Fondamento&display=swap", 'rel': "stylesheet"}
]
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=external_stylesheets,
)

server = app.server
app.config["suppress_callback_exceptions"] = True
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

# Load model
model = fastText.load_model("dashboard_data/model.bin")
weights_dict = load_obj('dashboard_data/weights_dict.pkl')

# Global Variables that will be used in following functions
department_list = ['All_Beauty',
 'Arts_Crafts_and_Sewing',
 'Automotive',
 'CDs_and_Vinyl',
 'Cell_Phones_and_Accessories',
 'Clothing_Shoes_and_Jewelry',
 'Digital_Music',
 'Electronics',
 'Grocery_and_Gourmet_Food',
 'Home_and_Kitchen',
 'Industrial_and_Scientific',
 'Kindle_Store',
 'Luxury_Beauty',
 'Movies_and_TV',
 'Musical_Instruments',
 'Office_Products',
 'Patio_Lawn_and_Garden',
 'Pet_Supplies',
 'Prime_Pantry',
 'Software',
 'Sports_and_Outdoors',
 'Tools_and_Home_Improvement',
 'Toys_and_Games',
 'Video_Games']


UPLOAD_DIRECTORY = "file/app_uploaded_files"

if os.path.exists(UPLOAD_DIRECTORY):
    shutil.rmtree(UPLOAD_DIRECTORY, ignore_errors=True)
os.makedirs(UPLOAD_DIRECTORY)


DOWN_DIRECTORY = "file/app_processed_files"
if os.path.exists(DOWN_DIRECTORY):
    shutil.rmtree(DOWN_DIRECTORY, ignore_errors=True)
os.makedirs(DOWN_DIRECTORY)



@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(DOWN_DIRECTORY, path, as_attachment=True)



def processing_txt(input_dict):
    input_df=pd.DataFrame.from_dict(input_dict)
    comments = list(input_df['summary'] + ': ' + input_df['reviewText'])
    comments = [w.replace('\n', '') for w in comments]
    pred = list(map(lambda x: 'positive' if x[0] == '__label__2' else 'negative', model.predict(comments)[0]))
    input_df['pred_label'] = pred
    return input_df


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def save_processed_file(name, input_dict):
    df = processing_txt(input_dict)
    df.to_csv(os.path.join(DOWN_DIRECTORY, name[:-4]+'_processed.csv'), index = False)



def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

def processed_files():
    """List the files in the processed directory."""
    files = []
    for filename in os.listdir(DOWN_DIRECTORY):
        path = os.path.join(DOWN_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


def init_value_store():
    """Initialize stored data."""
    state_dict = {
        "department": "All_Beauty",
        "productID": "None",
        "reviewText": "Test Sample review: I really like this.",
    }
    return state_dict

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def readJson(filePath):
    """Read json file from given file path."""
    with open(filePath) as data: 
        data = json.load(data)
        data = pd.DataFrame(data)
        
        # Convert the string time back to datetime
        data["reviewTime"] = pd.to_datetime(data.reviewTime)
        return data

def count_rating(data):
    """
    Calculate rating distribution from given dataframe, 
    will be used in plotting graphs.
    """
    count_dictionary = {}
    for score in [1.0, 2.0, 3.0, 4.0, 5.0]:
        x_star = str(int(score)) + "-star"
        count_dictionary[x_star] = data[data["overall"] == score].shape[0]
    return count_dictionary

def create_led_display_content(positive_num, negative_num):
    """Given the number that need to be displayed in LED screen, return the string"""
    content_str = ""
    if positive_num < 10:
        content_str += "0" + str(positive_num)
    else:
        content_str += str(positive_num)

    content_str += ":"
    if negative_num < 10:
        content_str += "0" + str(negative_num)
    else:
        content_str += str(negative_num)
    return content_str

def plot_summary_table(df):
    """Functions to draw the summary table in setting menu."""
    product_no = df.productID.unique().shape[0]
    reviewer_no = df.reviewerID.unique().shape[0]
    pos_no = df[df.pred_labels == 1].shape[0]
    neg_no = df[df.pred_labels == 0].shape[0]
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Product', 'Reviewer', "Positive", "Negative"],
                    fill_color="#1e2130",
                    align='left'),
        cells=dict(values=[[product_no],
                           [reviewer_no],
                           [pos_no],
                           [neg_no]],
                   fill_color="#4b506b",
                   align='left'))
    ])
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
        height=100,
        paper_bgcolor="#1e2130",
        font={"color": "white"}
    )
    return fig


def plot_product_rating_chart(df):
    """Plot the rating distribution of given product ID."""
    df["sentiment"] = df[["pred_labels"]].applymap(lambda x: "pos" if x == 1 else "neg")
    graph_df = df.groupby(["overall", "sentiment"], as_index=False).count()
    fig = px.bar(graph_df, x="overall", y="reviewText", color="sentiment", 
                hover_data=["overall", "sentiment"],
                labels={
                     "overall": "Rating",
                     "reviewText": "Count"
                })
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#1e2130",
        plot_bgcolor="#1e2130",
        font={"color": "white"}
    )
    return fig


def plot_product_verified_image_chart(df):
    """
    Plot the verified purchase rate and 
    customer image rate of given product ID.
    """
    test_dict = {
        "verified": {
            "True": df[df["verified"] == True].shape[0],
            "False": df[df["verified"] == False].shape[0],
        },
        "image": {
            "True": df[df["image"] == True].shape[0],
            "False": df[df["image"] == False].shape[0],
        }
    }
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['Customer Image', 'Verified Purchase'],
        x=[test_dict["verified"]["True"], test_dict["image"]["True"]],
        name='TRUE',
        orientation='h',

    ))
    fig.add_trace(go.Bar(
        y=['Customer Image', 'Verified Purchase'],
        x=[test_dict["verified"]["False"], test_dict["image"]["False"]],
        name='FALSE',
        orientation='h',

    ))
    fig.update_layout(
        barmode='stack',
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#1e2130",
        plot_bgcolor="#1e2130",
        font={"color": "white"}
    )
    return fig


def plot_product_summary_table(df):
    """Plot the summary table of given product ID."""
    reviewer_no = df.reviewerID.unique().shape[0]
    pos_no = df[df.pred_labels == "1"].shape[0]
    neg_no = df.shape[0] - pos_no
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Num of Reviews', df.shape[0]],
                    align='left'),
        cells=dict(values=[['Num of Reviewers', "Pos/Neg"],  # 1st column
                           [reviewer_no, "{}/{}".format(pos_no, neg_no)]],  # 2nd column
                   align='left'))
    ])
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        paper_bgcolor="#1e2130",
        font={"color": "white"}
    )
    return fig


def build_product_div(department, productID, time_period, label):
    """
    Return a html div that contains all part of product ID graphs.
    Graphs will be related to the options user choose.

    Args:
        department (str): amazon department user choose in setting menu.
        productID (str): ASIN number user input.
        time_period (int): number of months that user want to use.
        label (str): label chosen by customer to decide 
                        using only positive, negative or all reviews.
    """
    file_path = os.path.join("../subsets_data", department + ".json")
    df = readJson(file_path)
    if productID not in set(df.productID.unique()):
        return html.Div(
            children=[
                html.P(
                    children="There is no record of this product({}) in our dataset.".format(productID),
                    style={"display": "flex", "float": "left"}
                )])
    else:
        filter_data = df[df.reviewTime > max(df.reviewTime) - timedelta(days=30 * time_period)]
        # Filter the data based on select sentiment
        if label == "pos":
            final_data = filter_data[filter_data.pred_labels == 1]
        elif label == "neg":
            final_data = filter_data[filter_data.pred_labels == 0]
        else:
            final_data = filter_data
        return html.Div(children=[
            html.Div(
                children=[
                    html.Label(children="Here is the previous record of product {}:".format(productID)),
                    html.Br(),
                    dcc.Graph(
                        id="product-bar-chart",
                        figure=plot_product_rating_chart(final_data)
                    ),
                ],
                className="five columns",
            ),
            html.Div(
                children=[
                    dcc.Graph(
                        figure=plot_product_verified_image_chart(final_data)
                    )
                ],
                className="five columns",
                style={"margin-top": "2rem", "margin-left": "15rem"}
            ),
        ])


def generate_modal():
    """Generate the pop-up window that user could report wrong predictions."""
    return html.Div(
        id="report-dialog",
        className="modal",
        children=[
            html.Div(
                id="markdown-container",
                className="markdown-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close", id="dialog-close",
                            n_clicks=0, className="closeButton")),

                    html.Div(
                        children=[
                            html.Label("Please type the comment you want to report:"),
                            dcc.Textarea(
                                id="report-text",
                                value="",
                                style={'width': '100%', 'height': 150},
                            ),
                            dcc.RadioItems(
                                id="report-comment-label",
                                options=[
                                    {"label": "Postivie", "value": "pos"},
                                    {"label": "Negative", "value": "neg"},
                                ],
                                value="",
                                labelStyle={'display': 'inline-block'},
                            ),
                            html.Button(id="upload-report-btn", children="Report", n_clicks=0,
                                        style={"margin-top": "2rem"}),
                            html.P(id="report-success-msg"),
                        ]
                    ),
                ],
            ),
        ]
    )


def build_tabs():
    """Return a html div that contains two tabs: setting and results summary"""
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Specs-tab",
                        label="Setting",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children = user_setting
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="Results Summary",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children= build_metrics
                    )
                ]
            )
        ]
    )

# Build user setting html page, which contains the options(department, productID)
# and upload box.
user_setting = [
        html.Div(
            id="set-specs-intro-container",
            children=[html.Div(
                className="one-third column",
                children=[
                    html.P(
                    "Upload information of your comment."
                    ),

                    # Define the department drop down menu
                    html.Div(
                        id="input-menu",
                        children=[
                            html.Label(children="Select Department"),
                            html.Br(),
                            dcc.Dropdown(
                                id="department-dropdown",
                                options=list(
                                    {"label": depart.replace("_", " "), "value": depart} for depart in department_list
                                ),
                                value=department_list[0],
                            ),

                        ],
                        style={"margin-bottom": "2rem", "margin-top": "5rem"}
                    ),
                    
                    # Define input box for product ID
                    html.Div(
                        id="product-menu",
                        children=[
                            html.Label(children="Type product ASIN (Optional)"),
                            html.Br(),
                            dcc.Input(
                                id="product-input-box",
                                value="",
                                placeholder="e.g. B07ZD56WL3",
                                style={"width": '100%', 'height': 20},
                            )
                        ]
                    ),
                    html.Div(id="result-table", style={"margin-top": "5rem"}),
                ],
                style={'float': 'left'}
            ),
                # Define the upload box and text area for document uploading
                html.Div(
                    className="two-thirds column",
                    children=[
                        html.Div(
                            id="comment-menu",
                            children=[
                                html.Label(children="Upload your comment"),
                                html.Br(),
                                dcc.Textarea(
                                    id="comment-input-box",
                                    value="",
                                    placeholder="Type the comment you want to classify",
                                    style={'width': '100%', 'height': 200},
                                ),
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        'OR Drag and Drop your files here ',
                                        html.A('Select Files')
                                    ]),
                                    style={
                                        'width': '100%',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                    },
                                    multiple=False,

                                ),
                                html.Label("Uploaded File:"),
                                html.Ul(id="file-list"),
                                html.Label("Processed File:"),
                                html.Ul(id="file-list_processed"),
                            ],
                            style={"margin-top": "4rem"},
                        )
                    ],
                    style={'float': 'right'}
                )],
        ),

        # Define the Show Results and Report button
        html.Div(
            children=[
                html.Div(
                    id="button-div",
                    children=[
                        html.Button(
                        id="submit-btn", n_clicks=0,
                        children="Show Results",
                            style={'width': '20%', 'border-radius': '8px',
                                   'text-align': 'center',
                                   'margin': '0 auto'}),
                        html.Button(
                            id="report-btn", n_clicks=0,
                            children="Report",
                            style={"display": "none"}
                        ),
                    ]),
            ],
            style={'width': '100%', 'margin-top': '0.5rem', 'display': 'inline-block'}
        ),
    html.Div(id='output-state')
]


# Build the results summary page that contains all the results and graphs
build_metrics = html.Div(
        children = [
            html.Div(
            
            # Left: display the department user choose and the prediction result in precentage.
            className="three columns",
            children=[
                html.Div(
                    id="card-1",
                    children=[
                        html.Label(children="Department you choose:"),
                        html.Br(),
                        html.Div(
                            id="show-department",
                            children=[html.H5(id = "department_name")],
                        ),
                    ],
                ),
                html.Div(
                    id="card-2",
                    children=[
                        html.Label(children="Sentiment Analysis Results(%):"),
                        html.Br(),
                        daq.LEDDisplay(
                            id="result-display",
                            value="00:00",
                            color="#92e0d3",
                            backgroundColor="#1e2130",
                            size=45,
                            label="pos/neg"
                        )
                    ],
                    style={"margin-bottom": "3rem"}
                ),
                html.Button(id="learn-more-btn", children="LEARN MORE", n_clicks=0),
            ],
            style={"margin": "2rem 0 auto"}
            ),

            # Right: show the wordcloud of uploaded comment and 
            # rating distribution of given department
            html.Div(
                className="six columns",
                children = [
                    html.Div(
                        id="department-word-cloud",
                        children = [
                            html.Label(children="Select the sentiment you want to show:"),
                            html.Br(),
                            dcc.RadioItems(
                                id="wordcloud-label",
                                options=[
                                    {"label": "Postivie", "value": "pos"},
                                    {"label": "Negative", "value": "neg"},
                                    {'label': "Show All", "value": "all"},
                                ],
                                value="all",
                                labelStyle={'display': 'inline-block'},
                            ),
                            html.Hr(),
                            html.Img(id='wordcloud-graph')
                        ],
                        style={"margin": "2rem 0 auto"}
                    )
                ],
                style={'float': "left"}
            ),
            html.Div(
                className="three columns",
                children=[
                    html.Div(
                        id="department-time-chart",
                        children=[
                            html.Label(children="Select the most recent months:"),
                            html.Br(),
                            dcc.Slider(
                                id="time-selector",
                                min=3,
                                max=24,
                                step=None,
                                marks={
                                    3: '3',
                                    6: '6',
                                    12: '12',
                                    24: '24',
                                },
                                value=3,
                            ),
                            html.Hr(),
                            html.H4("Department Summary", style={"text-align": "center", "color": "darkgrey"}),
                            dcc.Graph(id="table-chart"),
                            dcc.Graph(id="rating-pie-chart"),
                        ],
                        style={"margin": "2rem 0 auto"}
                    )
                ],
                style={"float": "right"}
            ),
            html.Div(
            id="learn-more-content",
            children = [],
            style={"margin-top": "3rem"}
            ),
        ],
        style={"background-color": "#1e2130"}
    )

# Create global chart template
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"
state_dict = init_value_store()

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)

# Create app layout
app.layout = html.Div(
    [
        html.Div(
            [  
            html.H3(
                "Amazon Review Sentiment Analysis",
                style={"font-family": 'Fondamento', "text-shadow": "1px 1px 3px #cef5ee", 
                "text-align":"center", "font-size": "5rem"},
            ),
            ],
            id="header",
            className="row flex-display",
        ),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                html.Div(id="app-content"),
            ],
        ),
        dcc.Store(id="value-store", data=init_value_store()),
        dcc.Store(id="result-store", data={"pos": 0, "neg": 0}),
        generate_modal(),
    ],
)

# Update the selected department
@app.callback(Output("department_name", "children"),
[Input("department-dropdown", "value")])
def render_tab_content(department):
    return [department.replace("_", " ")]

# Update prediction result and draw the summary table
@app.callback([Output("result-store", "data"), Output("result-table", "children"), Output("report-btn", "style")],
              [Input("value-store", "data"), Input("submit-btn", "n_clicks")], [State("result-store", "data")])
def update_prediction(data, n, result_data):
    if n != 0:

        # Get prediction result and update result-store
        if type(data["reviewText"]) == str:
            pred = model.predict(data["reviewText"])
            pos = 1 if pred[0][0] == "__label__2" else 0
            neg = 1 if pred[0][0] == "__label__1" else 0
            num_reviews = 1
        else:
            df = processing_txt(data["reviewText"])
            pred = df['pred_label']
            num_reviews = len(pred)

            pos = sum([1 if res == 'positive' else 0 for res in pred])
            neg = sum([1 if res == 'negative' else 0 for res in pred])
        result_data["neg"] = neg
        result_data["pos"] = pos

        # Draw the summary table
        values = [["product ID", "department", "positive/negative"],
                  [data["productID"], data["department"].replace("_", " "), "{}/{}".format(pos, neg)]]
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Num of Reviews', num_reviews],
                        align='left',
                        fill_color="#1e2130"),
            cells=dict(values=values,
                       align='left',
                       fill_color="#4b506b", ))
        ])
        fig.update_layout(
            autosize=True,
            margin=dict(l=20, r=20, t=20, b=20),
            height=130,
            paper_bgcolor="#171a27",
            font={"color": "white"}
        )

        # Create the product link
        if str(data["productID"]) != "None":
            link = "https://www.amazon.com/dp/" + str(data["productID"])
            product_link = html.A("More info about this product", href=link, target="_blank",
                                  style={"font-size": "1.5rem", "margin-left": "2rem"})
        else:
            product_link = html.Div()
        return result_data, [html.Div(children=
                                      [html.Label(children="Here is the result summary table: "),
                                       dcc.Graph(figure=fig),
                                       product_link, ])], {"display": "inline", "margin": "0 auto"}
    return result_data, [], {"display": "none"}


# Callbacks for modal popup (report window)
@app.callback(Output("report-dialog", "style"),
              [Input("report-btn", "n_clicks"), Input("dialog-close", "n_clicks")])
def update_click_output(button_click, close_click):
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "report-btn":
            return {"display": "block"}
    return {"display": "none"}

# Callbakcs for showing the success message when user report the error.
@app.callback(Output("report-success-msg", "children"), [Input("upload-report-btn", "n_clicks")])
def show_message(n):
    if n != 0:
        return "Thanks for your reporting!"


# Show prediction result in LED display box
@app.callback(Output("result-display", "value"),
[Input("value-store", "data"), Input("result-store", "data")])
def prediction_display(data, result_data):
    total = result_data["pos"]+result_data["neg"]
    if total == 0 :
        return create_led_display_content(0, 0)
    else:
        pos = round(100 *result_data["pos"]/total)
        neg = round(100 *result_data["neg"]/total)
        return create_led_display_content(pos, neg)


# Draw wordcloud
@app.callback(dash.dependencies.Output("wordcloud-graph","src"), [Input("value-store", "data"), Input("wordcloud-label", "value")])
def draw_wordcloud(data, label):
    department = data["department"]
    weights = weights_dict[department][label]
    wc = WordCloud(
            background_color="#1e2130",
            max_words=2000,
            width = 700,
            height = 450)
    wc.generate_from_frequencies(weights)
    return wc.to_image()

def make_image(data, label):
    img = BytesIO()
    draw_wordcloud(data, label).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

# Callbacks for showing the plots of given productID
# If the user click the learn more button, show the graphs.
@app.callback(Output("learn-more-content", "children"),
              [Input("value-store", "data"),
               Input("learn-more-btn", "n_clicks"),
               Input("time-selector", "value"),
               Input("wordcloud-label", "value")])
def click_learn_more_btn(data, n, time_period, label):
    if n != 0:
        return build_product_div(data["department"], data["productID"], time_period, label)


# Draw rating distribution
@app.callback([Output("rating-pie-chart", "figure"), Output("table-chart", "figure")],
              [Input("wordcloud-label", "value"), Input("value-store", "data"), Input("time-selector", "value")])
def draw_chart(label, data, time_period):
    department = data["department"]
    folder_name = "../subsets_data"
    # Read data
    data = readJson(folder_name + "/" + department + ".json")

    # First select data based on given time_period
    filter_data = data[data.reviewTime > max(data.reviewTime) - timedelta(days=30 * time_period)]

    # Filter the data based on select sentiment
    if label == "pos":
        final_data = filter_data[filter_data.pred_labels == 1]
    elif label == "neg":
        final_data = filter_data[filter_data.pred_labels == 0]
    else:
        final_data = filter_data

    count_rating_dict = count_rating(final_data)
    fig = go.Figure(
        data=[go.Pie(labels=list(count_rating_dict.keys()), values=list(count_rating_dict.values()), sort=False)]
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=False,
        autosize=True,
        width=280,
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#1e2130",

    )
    return fig, plot_summary_table(final_data)


# Update file list
@app.callback(
    Output("file-list", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        save_file(uploaded_filenames, uploaded_file_contents)
    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]



# Update processed file list
@app.callback(
    Output("file-list_processed", "children"),
    [Input("submit-btn", "n_clicks"),Input("value-store", "data")],
    [State("upload-data", "filename"),
     ],
)
def update_output(n_clicks, data, uploaded_filenames):
    """Save uploaded files and regenerate the file list."""
    if n_clicks > 0:
        if uploaded_filenames is not None:

            save_processed_file(uploaded_filenames, data['reviewText'])
        files = processed_files()
        if len(files) == 0:
            return [html.Li("No files yet!")]
        else:
            return [html.Li(file_download_link(filename)) for filename in files]


# Update the state_dict, which save all the user input.
@app.callback(Output("value-store", "data"),
                [Input("submit-btn", "n_clicks")],
                [State("department-dropdown", "value"),
                State("product-input-box", "value"),
                State("comment-input-box", "value"),
                State('upload-data', 'contents'),
                State('upload-data', 'filename'),
                State("value-store", "data")])
def update_state_dict(n_clicks, department, product_id, review_text, uploaded_file_contents, uploaded_filenames, data):
    if n_clicks > 0:
        if uploaded_filenames is not None and uploaded_file_contents is not None:
            content_type, content_string = uploaded_file_contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=[0])
            data["reviewText"] = df.to_dict()
        else:
            data["reviewText"] = review_text

        data["department"] = department
        data["productID"] = product_id if product_id != "" else "None"

    return data



# Main
if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)