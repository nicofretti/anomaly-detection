import json

import dash
import numpy as np
from dash import dcc, html, Output, Input, callback
from flask import Flask, request
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import image as mpimg
import plotly.express as px
from plotly.subplots import make_subplots

SERVER = Flask(__name__)

APP = dash.Dash(
    __name__,
    server=SERVER,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_scripts=[
        # Tailwind CSS
        "https://tailwindcss.com/",
        {
            "src": "https://cdn.tailwindcss.com"
        }
    ],
    external_stylesheets=[
        # Font Awesome
        {
            'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
            'rel': 'stylesheet',
            'integrity': 'sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf',
            'crossorigin': 'anonymous'
        }
    ]
)

MAP_POSITION = [
    [0.0, 0.0, False],
    [0.0, 0.0, False],
    [0.0, 0.0, False],
    [0.00011207591086537239, -4.910984671656989, False],
    [0.001172868609843336, -4.913968294485954, False],
    [0.004431977181544644, -4.920990183339943, False],
    [0.005505823191344517, -4.9229903739006105, False],
    [0.006969767420146211, -4.9280400226494265, False],
    [0.006744008979296612, -4.927540643361569, False],
    [0.006814117989740853, -4.927550390710583, False],
    [0.007028911839472962, -4.927463288394096, False],
    [0.007139086606917311, -4.927401280466934, False],
    [0.008498141063949416, -4.9271223119578105, False],
    [0.019704654064453764, -4.92660281433761, True],
    [0.019704654064453764, -4.92660281433761, True],
    [0.03275116376748721, -4.926643440545843, True],
    [0.03204315444214201, -4.926598092662674, True],
    [0.026290955344286826, -4.92707934146832, True],
    [0.025803095687340627, -4.927067905112203, True],
    [0.026845960134329383, -4.927006103470224, True],
    [0.027532947511068162, -4.927212651214907, True],
    [0.027813562263784508, -4.927220213914992, True],
    [0.027609855786610682, -4.927126813997255, True],
    [0.0275909595161703, -4.9271101247606985, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262179, -4.9271133127634785, True],
    [0.02759457017262179, -4.9271133127634785, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
]
MAP_POSITION = []
VARIABLE_DECOMPOSITION = []


def app_init():
    APP.title = "Anomaly Detection on Kairos"
    APP.layout = html.Div(
        id="app",
        children=[
            # Global interval for refreshing data
            dcc.Interval(
                id="interval-component",
                interval=1000,
                n_intervals=0
            ),
            # Banner
            html.Div(
                id="banner",
                className="banner text-5xl",
                children=[html.Div(className="fa fa-chart-bar text-red-700"),
                          html.H3("Anomaly Detection", className="ml-2 text-gray-700")]
            ),
            # Left column
            html.Div(
                id="left-column",
                className="four columns",
                children=[
                    html.Div(
                        ["initial child"], id="output-clientside", style={"display": "none"}
                    )
                ],
            ),
            # Right column
            html.Div(
                id="right-column",
                className="eight columns",
                children=[
                    # ICE lab chart
                    html.Div(
                        id="first-chart",
                        children=[
                            html.B("First chart"),
                            html.Hr(),
                            dcc.Graph(id="ice_lab", figure={}, responsive=True,
                                      style={"height": "750px", "width": "400px"}),
                        ],
                    ),
                ],
            ),
        ],
    )


@SERVER.route("/map_position_insert", methods=['GET', 'POST'])
def map_position_insert():
    global MAP_POSITION
    if request.method != 'POST':
        return 'Method not allowed', 405
    # bytes to dict
    data = json.loads(request.data.decode('utf-8'))
    MAP_POSITION.append([data["X"], data["Y"], data["anomaly"]])
    print(len(MAP_POSITION))
    # update_map_position()
    # Return 200 OK
    return "OK", 200


@SERVER.route("/variable_decomposition_insert", methods=['GET', 'POST'])
def variable_decomposition_insert():
    global VARIABLE_DECOMPOSITION
    if request.method != 'POST':
        return 'Method not allowed', 405
    data = json.loads(request.data.decode('utf-8'))
    VARIABLE_DECOMPOSITION.append(data)
    # Return 200
    return "OK", 200


@SERVER.route("/commit", methods=['GET', 'POST'])
def commit():
    global MAP_POSITION, VARIABLE_DECOMPOSITION
    if request.method == 'GET':
        # Here we reset the data
        MAP_POSITION, VARIABLE_DECOMPOSITION = [], []
        return "OK", 200
    # TODO: implement save to database
    return "Not implemented", 501


# --------
# Callbacks
# --------
import plotly.graph_objects as go

@callback(
    Output(component_id='ice_lab', component_property='figure'),
    Input(component_id='interval-component', component_property='n_intervals')
)
def update_map_position(n_intervals):
    global MAP_POSITION
    print(str(len(MAP_POSITION))+" chart")
    map = make_subplots(rows=1)
    # Set limits


    # plot map
    # map.imshow(img_backgroung, extent=(-1.5, 2.5, -13, 1), cmap='gray')
    map.add_layout_image(
        dict(
            source="assets/ICE_lab.png",
            y=1,
            x=0,
            sizex=1.3,
            sizey=1.3,
            opacity=1,
            layer="below",
        )
    )

    if len(MAP_POSITION) == 0:
        return map

    # points_copy: {contains X, Y, anomaly}
    points_copy = np.array(MAP_POSITION)
    x0, y0 = points_copy[0][0], points_copy[0][1]
    x_plot, y_plot = points_copy[1:, 0], points_copy[1:, 1]
    anomaly = points_copy[1:, 2]
    green_patch = mpatches.Patch(color='green', label='start')
    red_patch = mpatches.Patch(color='red', label='anomaly')
    blue_patch = mpatches.Patch(color='blue', label='normal')
    color = np.where(anomaly, 'r', 'b')
    #map.add_scatter(x_plot, y_plot, c=color, s=7, zorder=1)
    map.add_scatter(x=[x0], y=[y0])
    map.add_scatter(x=x_plot, y=y_plot)

    # map.legend(handles=[green_patch, blue_patch, red_patch], loc=1)
    map.update_layout(
        xaxis_range=[-1.5, 2.5],
        yaxis_range=[-5, 1]
    )
    return map


if __name__ == "__main__":
    host = '0.0.0.0'
    port = 8050
    debug = True
    # Initialize the app
    app_init()
    # Start the app
    APP.run(debug=debug, host=host, port=port)
