import json

import dash
import numpy as np
import pandas as pd
from dash import dcc, html, Output, Input, callback
from dash.exceptions import PreventUpdate
from flask import Flask, request
import plotly.graph_objects as go
import paho.mqtt.client as mqtt

import debug as debug_constraints

# --------
# Global variables
# --------
SERVER = Flask(__name__)
APP = dash.Dash(
    __name__,
    server=SERVER,
    update_title=None,
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
NEW_DATA = True
VARIABLES = ["X", "Y", "O", "LS", "LC", "LD"]
VARIABLE_DECOMPOSITION_THR = [1.26020238, 6.67861522, 0.4251171, 0.70920265, 0.94272347, 0.89692743]
MAP_CHART, VARIABLE_DECOMPOSITION_CHART = {}, {}
MAP_POSITION, VARIABLE_DECOMPOSITION = [], []


# -------
# Layout methods
# -------

def app_init():
    global MAP_POSITION, VARIABLES, VARIABLE_DECOMPOSITION
    # TODO: remove this
    MAP_POSITION = debug_constraints.DEBUG_MAP_POSITION
    VARIABLE_DECOMPOSITION = debug_constraints.VARIABLE_DECOMPOSITION_DEBUG
    map_chart_init()
    variable_decomposition_chart_init()
    APP.title = "Anomaly Detection"
    APP.layout = html.Div(
        id="app",
        children=[
            # global interval for refreshing data
            dcc.Interval(
                id="interval-component",
                interval=200,
                n_intervals=0,
            ),

            # hidden input for the last position, if value is 0, the chart is not updated
            dcc.Input(
                id="hidden_map_position_trigger",
                type="hidden",
                value=0
            ),
            # hidden input for the last variable decomposition, if value is 0, the chart is not updated
            dcc.Input(
                id="hidden_variable_decomposition_trigger",
                type="hidden",
                value=0
            ),

            # banner
            html.Div(
                id="banner",
                className="banner text-5xl font-bold flex justify-between items-center",
                children=[
                    html.Div(
                        className="flex items-center",
                        children=[
                            html.Div(className="fa fa-chart-bar text-blue-500"),
                            html.H3("Anomaly Detection", className="ml-2 font-bold")
                        ],
                    ),
                    # button to reset the charts
                    html.Button(
                        id="reset_button",
                        className="bg-blue-500 text-xl hover:bg-blue-700 text-white font-bold py-2 px-4 rounded",
                        style={"border": "none"},
                        children=[
                            html.Div(className="fa fa-redo-alt mr-2 text-white"),
                            html.Span(
                                className="text-white",
                                children="Reset"
                            )
                        ]
                    )
                ]
            ),
            # left column
            # html.Div(
            #     id="left-column",
            #     className="three columns",
            #     children=[
            #     ],
            # ),
            # Right column
            html.Div(
                id="right-column",
                className="twelve columns shadow-lg",
                children=[
                    html.Div(
                        id="charts",
                        className="flex bg-white mt-2 rounded-lg shadow-lg",
                        children=[
                            html.Div(
                                className="p-8",
                                children=[
                                    html.P(
                                        className="text-left text-3xl font-bold pl-2 text-blue-500",
                                        children="ICE LAB MAP"
                                    ),
                                    dcc.Graph(
                                        id="map_position_chart",
                                        className="p-2",
                                        config=dict(
                                            displayModeBar=False
                                        ),
                                        figure={},
                                        style={"height": "600px", "width": "400px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                className="p-8 w-full relative",
                                children=[
                                    html.P(
                                        className="text-left text-3xl font-bold pl-2 text-blue-500",
                                        children="HELLINGER DISTANCE DECOMPOSITION"
                                    ),
                                    html.Div(
                                        id="variable_decomposition_semaphore",
                                        className="absolute top-36 left-20 text-center z-10 text-black " + \
                                                  "pl-2 pr-6",
                                        style={
                                            "background-color": "rgba(255, 255, 255, 0.9)",
                                        },
                                        children=semaphore_generator()
                                    ),
                                    dcc.Graph(
                                        id="variable_decomposition_chart",
                                        className="p-2",
                                        style={"height": "600px"},
                                        config=dict(
                                            displayModeBar=True,
                                            modeBarButtonsToRemove=["lasso2d", "select2d"]
                                        )
                                    ),
                                ]
                            )
                        ]
                    ),

                ],
            ),
        ],
    )


def semaphore_generator():
    global VARIABLE_DECOMPOSITION, VARIABLE_DECOMPOSITION_THR
    r = []
    if not VARIABLE_DECOMPOSITION or len(VARIABLE_DECOMPOSITION) == 0:
        # semaphore are all green, no anomaly detected
        last_anomaly = np.zeros(len(VARIABLES))
    else:
        # there are datas, we calculate the last anomaly
        last_anomaly = VARIABLE_DECOMPOSITION[-1]
    for i in range(len(VARIABLES)):
        active = last_anomaly[i] > VARIABLE_DECOMPOSITION_THR[i]
        r.append(
            html.Div(
                className="flex justify-start items-center text-2xl my-1",
                children=[
                    html.Div(
                        className="fa fa-circle mr-2 {}".format("text-red-500" if active else "text-green-500"),
                    ),
                    VARIABLES[i]
                ])
        )
    return r


# Init only at the startup the chart of the ICE-lab
def map_chart_init():
    global MAP_CHART
    MAP_CHART = go.Figure()
    MAP_CHART.add_layout_image(
        source="assets/ICE_lab.png",
        y=1,
        x=-1.5,
        sizex=12,
        sizey=15,
        xref="x",
        yref="y",
        opacity=1,
        layer="below",
        sizing="contain"
    )
    # set limits
    MAP_CHART.update_layout(
        clickmode="event+select",
        xaxis_range=[-1.5, 2.5],
        yaxis_range=[-5, 1],
        xaxis=dict(showgrid=False, fixedrange=True),
        yaxis=dict(showgrid=False, fixedrange=True),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=.6,
            bgcolor="rgba(255, 255, 255, 0.9)"
        ),
        # string to maintain user selections
        uirevision="ICE-LAB",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    MAP_CHART.add_traces([
        # correct behaviour
        go.Scatter(
            x=[], y=[],
            mode="markers",
            marker={"color": "blue"},
            name="correct behaviour"
        ),
        # anomalies
        go.Scatter(
            x=[], y=[],
            mode="markers",
            marker={"color": "red"},
            name="anomaly"
        ),
        # last position
        go.Scatter(
            x=[], y=[],
            mode="markers",
            marker={"color": "orange", "size": 12},
            name="last position",
            showlegend=True
        )
    ])
    # Adding nominal position
    nominal_0 = pd.read_csv('./data/nominal_0.csv').to_numpy()
    nominal_1 = pd.read_csv('./data/nominal_1.csv').to_numpy()
    MAP_CHART.add_traces([
        go.Scatter(
            x=nominal_0[:, 0], y=nominal_0[:, 1],
            mode="lines",
            line={"color": "green"},
            name="nominal position 0",
        ),
        go.Scatter(
            x=nominal_1[:, 0], y=nominal_1[:, 1],
            mode="lines",
            line={"color": "lightgreen"},
            name="nominal position 1",
        )
    ])


def variable_decomposition_chart_init():
    global VARIABLES, VARIABLE_DECOMPOSITION_CHART
    VARIABLE_DECOMPOSITION_CHART = go.Figure()
    VARIABLE_DECOMPOSITION_CHART.update_layout(
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=.87,
            bgcolor="rgba(255, 255, 255, 0.9)"
        ),
        # string to maintain user selections
        uirevision="VARIABLE DECOMPOSITION",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    colors = ["blue", "purple", "green", "yellow", "brown", "orange"]
    for i in range(len(VARIABLES)):
        VARIABLE_DECOMPOSITION_CHART.add_traces(
            [
                # correct behaviour for the i-th variable
                go.Scatter(
                    x=[], y=[],
                    mode="lines+markers",
                    name=VARIABLES[i],
                    marker={
                        "color": colors[i % len(VARIABLES)],
                        "size": 3
                    },
                ),
                # anomaly for the i-th variable
                go.Scatter(
                    x=[], y=[],
                    mode="lines+markers",
                    name=VARIABLES[i],
                    marker={
                        "color": colors[i % len(VARIABLES)],
                        "size": 10,
                        "symbol": "x"
                    },
                    showlegend=False
                ),
            ]
        )


# --------
# Callbacks
# --------
@callback(
    [
        Output(component_id='hidden_map_position_trigger', component_property='value'),
        Output(component_id='hidden_variable_decomposition_trigger', component_property='value')
    ],
    Input(component_id='interval-component', component_property='n_intervals'),
)
def callback_hidden_trigger(_):
    global NEW_DATA, MAP_CHART
    if NEW_DATA:
        NEW_DATA = False
        # update the charts
        return 1, 1
    # we have to update the charts, also them update their data
    return 0, 0


@callback(
    Output(component_id='map_position_chart', component_property='figure'),
    Input(component_id='hidden_map_position_trigger', component_property='value')
)
def callback_map_position(update_chart):
    global MAP_CHART, MAP_POSITION
    if not update_chart:
        # do not update the chart
        raise PreventUpdate
    # n_intervals: not used its given by the default reloader
    if len(MAP_POSITION) == 0:
        # no point to display :( reset the map
        for i in range(3):
            MAP_CHART["data"][i]["x"], MAP_CHART["data"][i]["y"] = [], []
        return MAP_CHART
    # store our points
    points_copy = np.array(MAP_POSITION)
    x_plot, y_plot = points_copy[1:, 0], points_copy[1:, 1]
    anomaly = points_copy[1:, 2]
    # correct behaviour
    MAP_CHART["data"][0]["x"], MAP_CHART["data"][0]["y"] = x_plot[anomaly == 0], y_plot[anomaly == 0]
    # anomaly
    MAP_CHART["data"][1]["x"], MAP_CHART["data"][1]["y"] = x_plot[anomaly == 1], y_plot[anomaly == 1]
    # last position
    MAP_CHART["data"][2]["x"], MAP_CHART["data"][2]["y"] = [points_copy[-1][0]], [points_copy[-1][1]]
    return MAP_CHART


@callback(
    [
        Output(component_id="variable_decomposition_chart", component_property="figure"),
        Output(component_id="variable_decomposition_semaphore", component_property="children")
    ],
    Input(component_id='hidden_variable_decomposition_trigger', component_property='value')
)
def callback_variable_decomposition(update_chart):
    global VARIABLE_DECOMPOSITION, VARIABLE_DECOMPOSITION_THR, VARIABLE_DECOMPOSITION_CHART
    if not update_chart:
        # do not update the chart
        raise PreventUpdate
    # n_intervals: not used its given by the default reloader
    if len(VARIABLE_DECOMPOSITION) == 0:
        # no point, reset the chart
        for i in range(len(VARIABLES) * 2):
            VARIABLE_DECOMPOSITION_CHART["data"][i]["x"], VARIABLE_DECOMPOSITION_CHART["data"][i]["y"] = [], []
        return VARIABLE_DECOMPOSITION_CHART, semaphore_generator()
    points_copy = np.array(VARIABLE_DECOMPOSITION)
    x_axis = [k for k in range(len(VARIABLE_DECOMPOSITION))]
    count = 0
    for i in range(len(VARIABLES)):
        # calculate the period of anomaly
        anomaly = np.where(points_copy[:, i] > VARIABLE_DECOMPOSITION_THR[i], points_copy[:, i], np.nan)
        VARIABLE_DECOMPOSITION_CHART["data"][count]["x"], VARIABLE_DECOMPOSITION_CHART["data"][count]["y"] = \
            x_axis, points_copy[:, i]
        count += 1
        VARIABLE_DECOMPOSITION_CHART["data"][count]["x"], VARIABLE_DECOMPOSITION_CHART["data"][count]["y"] = \
            x_axis, anomaly
        count += 1
    return VARIABLE_DECOMPOSITION_CHART, semaphore_generator()


@callback(
    Output(component_id="banner", component_property="children"),
    Input(component_id='variable_decomposition_chart', component_property='restyleData')
)
def callback_update_variable_decomposition_options(layout):
    global NEW_DATA
    if not layout or len(layout) < 2:
        raise PreventUpdate
    chart = layout[1][0]
    active = layout[0]["visible"][0] == True
    # hide or show the chart (only anomaly)
    VARIABLE_DECOMPOSITION_CHART["data"][chart + 1]["visible"] = active
    # to trigger the update of the chart
    NEW_DATA = True
    # nothing to update
    raise PreventUpdate


@callback(
    Output(component_id="reset_button", component_property="children"),
    Input(component_id='reset_button', component_property='n_clicks')
)
def callback_reset_button(n_clicks):
    global NEW_DATA, MAP_POSITION, VARIABLE_DECOMPOSITION
    if n_clicks is None:
        raise PreventUpdate
    # reset the charts
    core_reset()
    # nothing to update
    raise PreventUpdate


# function to add new data into the map_position chart
def core_map_position_insert(data):
    global NEW_DATA, MAP_POSITION
    MAP_POSITION.append([data["X"], data["Y"], data["anomaly"]])
    # set new data to true, to trigger the update of the charts
    NEW_DATA = True


# function to add new data into the variable_decomposition chart
def core_variable_decomposition_insert(data):
    global NEW_DATA, VARIABLE_DECOMPOSITION
    VARIABLE_DECOMPOSITION.append([data[var] for var in VARIABLES])
    # set new data to true, to trigger the update of the charts
    NEW_DATA = True


def core_reset():
    global NEW_DATA, MAP_POSITION, VARIABLE_DECOMPOSITION
    # reset the charts
    MAP_POSITION, VARIABLE_DECOMPOSITION = [], []
    # to trigger the update of the charts
    NEW_DATA = True


# --------
# Custom apis
# --------

@SERVER.route("/map_position_insert", methods=['GET', 'POST'])
def api_map_position_insert():
    global NEW_DATA, MAP_POSITION
    if request.method != 'POST':
        return 'Method not allowed', 405
    # bytes to dict
    data = json.loads(request.data.decode('utf-8'))
    core_map_position_insert(data)
    # Return 200 OK
    return "OK", 200


@SERVER.route("/variable_decomposition_insert", methods=['GET', 'POST'])
def api_variable_decomposition_insert():
    global NEW_DATA, VARIABLE_DECOMPOSITION
    if request.method != 'POST':
        return 'Method not allowed', 405
    data = json.loads(request.data.decode('utf-8'))
    core_map_position_insert(data)
    # Return 200
    return "OK", 200


@SERVER.route("/commit", methods=['GET', 'POST'])
def api_commit():
    global MAP_POSITION, VARIABLE_DECOMPOSITION
    if request.method == 'GET':
        # Here we reset the data
        MAP_POSITION, VARIABLE_DECOMPOSITION = [], []
        return "OK", 200
    # TODO: implement save to database
    return "Not implemented", 501


# --------
# mqtt client
# --------
def mqtt_init(host_srv, port_srv):
    # initialize the mqtt client
    client = mqtt.Client("web_interface")
    # set the callback
    client.on_message = mqtt_on_message
    client.on_connect = mqtt_on_connect
    # connect to the broker, the callback will be called
    client.connect(host_srv, port_srv)
    client.loop_start()


def mqtt_on_connect(client, userdata, flags, rc):
    # subscribe to the topic
    client.subscribe("map_position_insert")
    client.subscribe("variable_decomposition_insert")


def mqtt_on_message(client, userdata, msg):
    global MAP_POSITION, VARIABLE_DECOMPOSITION, NEW_DATA
    # print(str(msg.topic))
    # check the topic
    if msg.topic == "map_position_insert":
        # update the map position
        core_map_position_insert(json.loads(msg.payload))
    elif msg.topic == "variable_decomposition_insert":
        # update the variable decomposition
        core_variable_decomposition_insert(json.loads(msg.payload))


if __name__ == "__main__":
    # initialize the app variables
    app_init()
    # initialize the mqtt client
    mqtt_init('0.0.0.0', 1883)
    # Start the app
    APP.run(debug=False, host='0.0.0.0', port=8080)
