import configparser
import copy
import json

import dash
import numpy as np
import pandas as pd
from dash import dcc, html, Output, Input, callback
from dash.exceptions import PreventUpdate
from flask import Flask, request
import plotly.graph_objects as go

from controllers.charts_controller import ChartsController
import paho.mqtt.client as mqtt

# --------
# Global variables
# --------
# in the config file set up the server options
CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')
# SERVER = Flask(__name__)

MAP_CHART, VARIABLE_DECOMPOSITION_CHART = {}, {}
MAP_CHART_2, VARIABLE_DECOMPOSITION_CHART_2 = {}, {}

CHARTS_CONTROLLER = ChartsController(
    CONFIG["charts"]["decomposition_variables"].split(","),
    list(map(float, CONFIG["charts"]["decomposition_thr"].strip().split(","))),
    int(CONFIG["app"]["window_size"])
)

CHARTS_CONTROLLER_2 = ChartsController(
    CONFIG["charts"]["decomposition_variables"].split(","),
    list(map(float, CONFIG["charts"]["decomposition_thr"].strip().split(","))),
    int(CONFIG["app"]["window_size"])
)


def semaphore_generator(charts_controller):
    r = []
    if not charts_controller.decomposition or len(charts_controller.decomposition) == 0:
        # semaphore are all green, no anomaly detected
        last_anomaly = np.zeros(len(charts_controller.variables))
    else:
        # there are datas, we calculate the last anomaly
        last_anomaly = charts_controller.decomposition[-1]
    for i in range(len(charts_controller.variables)):
        active = last_anomaly[i] > charts_controller.decomposition_thr[i]
        r.append(
            html.Div(
                className="flex justify-start items-center text-2xl my-1",
                children=[
                    html.Div(
                        className="fa fa-circle mr-2 {}".format("text-red-500" if active else "text-green-500"),
                    ),
                    charts_controller.variables[i]
                ])
        )
    return r


APP = dash.Dash(
    __name__,
    # server=SERVER,
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
    ],
    #suppress_callback_exceptions=True,
)
APP.layout = html.Div(
    id="app",
    children=[
        # global interval for refreshing data
        dcc.Interval(
            id="interval-component",
            interval=500,
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
        # hidden input for the last position, if value is 0, the chart is not updated
        dcc.Input(
            id="hidden_map_position_trigger_2",
            type="hidden",
            value=0
        ),
        # hidden input for the last variable decomposition, if value is 0, the chart is not updated
        dcc.Input(
            id="hidden_variable_decomposition_trigger_2",
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
                        html.H3("Anomaly Detection", className="ml-3 font-bold")
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
                                    className="text-left text-4xl font-bold pl-2 d-flex justify-center",
                                    children=[
                                        # html.Span(className="fas fa-car-side mr-2 text-orange-500"),
                                        "ICE map \"Robot 1\"",
                                    ]
                                ),
                                dcc.Graph(
                                    id="map_position_chart",
                                    className="p-2",
                                    config=dict(
                                        displayModeBar=False
                                    ),
                                    figure={},
                                    style={"height": "400px", "width": "400px"},
                                ),
                            ]
                        ),
                        html.Div(
                            className="p-8 w-full relative",
                            children=[
                                html.P(
                                    className="text-left text-4xl font-bold pl-2 d-flex justify-center",
                                    children=[
                                        # html.Span(className="fas fas fa-chart-line mr-2 text-orange-500"),
                                        "Variables view"
                                    ]
                                ),
                                html.Div(
                                    id="variable_decomposition_semaphore",
                                    className="absolute top-36 left-20 text-center z-10 text-black " + \
                                              "pl-2 pr-6",
                                    style={
                                        "background-color": "rgba(255, 255, 255, 0.9)",
                                    },
                                    children=semaphore_generator(CHARTS_CONTROLLER)
                                ),
                                dcc.Graph(
                                    id="variable_decomposition_chart",
                                    className="p-2",
                                    style={"height": "400px"},
                                    config=dict(
                                        displayModeBar=True,
                                        modeBarButtonsToRemove=["lasso2d", "select2d"]
                                    )
                                ),
                            ]
                        )
                    ]
                ),
                html.Div(
                    id="right-column-2",
                    className="flex bg-white mt-2 rounded-lg shadow-lg",
                    children=[
                        html.Div(
                            className="p-8",
                            children=[
                                html.P(
                                    className="text-left text-4xl font-bold pl-2 d-flex justify-center",
                                    children=[
                                        # html.Span(className="fas fa-car-side mr-2 text-orange-500"),
                                        "ICE map \"Robot 2\"",
                                    ]
                                ),
                                dcc.Graph(
                                    id="map_position_chart_2",
                                    className="p-2",
                                    config=dict(
                                        displayModeBar=False
                                    ),
                                    figure={},
                                    style={"height": "400px", "width": "400px"},
                                ),
                            ]
                        ),
                        html.Div(
                            className="p-8 w-full relative",
                            children=[
                                html.P(
                                    className="text-left text-4xl font-bold pl-2 d-flex justify-center",
                                    children=[
                                        # html.Span(className="fas fas fa-chart-line mr-2 text-orange-500"),
                                        "Variables view"
                                    ]
                                ),
                                html.Div(
                                    id="variable_decomposition_semaphore_2",
                                    className="absolute top-36 left-20 text-center z-10 text-black " + \
                                              "pl-2 pr-6",
                                    style={
                                        "background-color": "rgba(255, 255, 255, 0.9)",
                                    },
                                    children=semaphore_generator(CHARTS_CONTROLLER_2)
                                ),
                                dcc.Graph(
                                    id="variable_decomposition_chart_2",
                                    className="p-2",
                                    style={"height": "400px"},
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


# -------
# Layout methods
# -------

def app_init():
    global CHARTS_CONTROLLER
    map_chart_init()
    variable_decomposition_chart_init()
    APP.title = "Anomaly Detection"


# Init only at the startup the chart of the ICE-lab
def map_chart_init():
    global MAP_CHART
    MAP_CHART = go.Figure()
    MAP_CHART.add_layout_image(
        source="assets/ICE_lab.png",
        y=2.35 + 2.6,
        x=-5.70 + 9.6,
        sizex=21,
        sizey=5.5,
        xref="x",
        yref="y",
        opacity=1,
        layer="below",
        sizing="stretch")

    # set limits
    MAP_CHART.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        clickmode="event+select",
        xaxis_range=[-4 + 9.6, 14 + 9.6],
        yaxis_range=[-5 + 2.6, 6 + 2.6],
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
    nominal_0 = pd.read_csv(CONFIG["app"]["nominal_0"]).to_numpy()
    # nominal_1 = pd.read_csv(CONFIG["app"]["nominal_1"]).to_numpy()
    MAP_CHART.add_traces([
        go.Scatter(
            x=nominal_0[:, 0], y=nominal_0[:, 1],
            mode="lines",
            line={"color": "lightgreen"},
            name="nominal position",
        )
    ])
    # Copying MAP_CHART to MAP_CHART_2
    global MAP_CHART_2
    MAP_CHART_2 = copy.deepcopy(MAP_CHART)


def variable_decomposition_chart_init():
    global CHARTS_CONTROLLER, VARIABLE_DECOMPOSITION_CHART
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
    for i in range(len(CHARTS_CONTROLLER.variables)):
        VARIABLE_DECOMPOSITION_CHART.add_traces(
            [
                # correct behaviour for the i-th variable
                go.Scatter(
                    x=[], y=[],
                    mode="lines+markers",
                    name=CHARTS_CONTROLLER.variables[i],
                    marker={
                        "color": colors[i % len(CHARTS_CONTROLLER.variables)],
                        "size": 3
                    },
                ),
                # anomaly for the i-th variable
                go.Scatter(
                    x=[], y=[],
                    mode="lines+markers",
                    name=CHARTS_CONTROLLER.variables[i],
                    marker={
                        "color": colors[i % len(CHARTS_CONTROLLER.variables)],
                        "size": 10,
                        "symbol": "x"
                    },
                    showlegend=False
                ),
            ]
        )
    # Copying VARIABLE_DECOMPOSITION_CHART to VARIABLE_DECOMPOSITION_CHART_2
    global VARIABLE_DECOMPOSITION_CHART_2
    VARIABLE_DECOMPOSITION_CHART_2 = copy.deepcopy(VARIABLE_DECOMPOSITION_CHART)


# --------
# Callbacks
# --------
@callback(
    [
        Output(component_id='hidden_map_position_trigger', component_property='value'),
        Output(component_id='hidden_variable_decomposition_trigger', component_property='value'),
        Output(component_id='hidden_map_position_trigger_2', component_property='value'),
        Output(component_id='hidden_variable_decomposition_trigger_2', component_property='value'),
    ],
    Input(component_id='interval-component', component_property='n_intervals'),
)
def callback_hidden_trigger(_):
    global CHARTS_CONTROLLER, CHARTS_CONTROLLER_2
    r = [0, 0, 0, 0]
    if CHARTS_CONTROLLER.can_update():
        CHARTS_CONTROLLER.update_charts(False)
        # update the charts of first robot
        r[0] = 1
        r[1] = 1
    if CHARTS_CONTROLLER_2.can_update():
        CHARTS_CONTROLLER_2.update_charts(True)
        # update the charts of second robot
        r[2] = 1
        r[3] = 1
    # update values
    return r[0], r[1], r[2], r[3]


@callback(
    Output(component_id='map_position_chart', component_property='figure'),
    Input(component_id='hidden_map_position_trigger', component_property='value')
)
def callback_map_position(update_chart):
    global MAP_CHART, CHARTS_CONTROLLER
    if not update_chart:
        # do not update the chart
        raise PreventUpdate
    # if len(CHARTS_CONTROLLER.position) == 0:
    #     # no point to display :( reset the map
    #     for i in range(3):
    #         MAP_CHART["data"][i]["x"], MAP_CHART["data"][i]["y"] = [], []
    #     return MAP_CHART
    # # store our points
    # points_copy = np.array(CHARTS_CONTROLLER.position)
    # x_plot, y_plot = points_copy[1:, 0], points_copy[1:, 1]
    # anomaly = points_copy[1:, 2]
    # # correct behaviour
    # MAP_CHART["data"][0]["x"], MAP_CHART["data"][0]["y"] = x_plot[anomaly == 0], y_plot[anomaly == 0]
    # # anomaly
    # MAP_CHART["data"][1]["x"], MAP_CHART["data"][1]["y"] = x_plot[anomaly == 1], y_plot[anomaly == 1]
    # # last position
    # MAP_CHART["data"][2]["x"], MAP_CHART["data"][2]["y"] = [points_copy[-1][0]], [points_copy[-1][1]]
    MAP_CHART = map_chart_update_from_controller(MAP_CHART, CHARTS_CONTROLLER)
    return MAP_CHART


@callback(
    Output(component_id='map_position_chart_2', component_property='figure'),
    Input(component_id='hidden_map_position_trigger_2', component_property='value')
)
def callback_map_position_2(update_chart):
    global MAP_CHART_2, CHARTS_CONTROLLER_2
    if not update_chart:
        # do not update the chart
        raise PreventUpdate
    MAP_CHART_2 = map_chart_update_from_controller(MAP_CHART_2, CHARTS_CONTROLLER_2)
    return MAP_CHART_2


def map_chart_update_from_controller(map_chart, controller):
    if len(controller.position) == 0:
        # no point to display :( reset the map
        for i in range(3):
            map_chart["data"][i]["x"], map_chart["data"][i]["y"] = [], []
        return map_chart
    # store our points
    points_copy = np.array(controller.position)
    x_plot, y_plot = points_copy[1:, 0], points_copy[1:, 1]
    anomaly = points_copy[1:, 2]
    # correct behaviour
    map_chart["data"][0]["x"], map_chart["data"][0]["y"] = x_plot[anomaly == 0], y_plot[anomaly == 0]
    # anomaly
    map_chart["data"][1]["x"], map_chart["data"][1]["y"] = x_plot[anomaly == 1], y_plot[anomaly == 1]
    # last position
    map_chart["data"][2]["x"], map_chart["data"][2]["y"] = [points_copy[-1][0]], [points_copy[-1][1]]
    return map_chart


@callback(
    [
        Output(component_id="variable_decomposition_chart", component_property="figure"),
        Output(component_id="variable_decomposition_semaphore", component_property="children")
    ],
    Input(component_id='hidden_variable_decomposition_trigger', component_property='value')
)
def callback_variable_decomposition(update_chart):
    global CHARTS_CONTROLLER, VARIABLE_DECOMPOSITION_CHART
    if not update_chart:
        # do not update the chart
        raise PreventUpdate
    # if len(CHARTS_CONTROLLER.decomposition) == 0:
    #     # no point, reset the chart
    #     for i in range(len(CHARTS_CONTROLLER.variables) * 2):
    #         VARIABLE_DECOMPOSITION_CHART["data"][i]["x"], VARIABLE_DECOMPOSITION_CHART["data"][i]["y"] = [], []
    #     return VARIABLE_DECOMPOSITION_CHART, semaphore_generator(CHARTS_CONTROLLER)
    # points_copy = np.array(CHARTS_CONTROLLER.decomposition)
    # x_axis = CHARTS_CONTROLLER.decomposition_x
    # count = 0
    # for i in range(len(CHARTS_CONTROLLER.variables)):
    #     # calculate the period of anomaly
    #     anomaly = np.where(points_copy[:, i] > CHARTS_CONTROLLER.decomposition_thr[i], points_copy[:, i], np.nan)
    #     VARIABLE_DECOMPOSITION_CHART["data"][count]["x"], VARIABLE_DECOMPOSITION_CHART["data"][count]["y"] = \
    #         x_axis, points_copy[:, i]
    #     count += 1
    #     VARIABLE_DECOMPOSITION_CHART["data"][count]["x"], VARIABLE_DECOMPOSITION_CHART["data"][count]["y"] = \
    #         x_axis, anomaly
    #     count += 1
    VARIABLE_DECOMPOSITION_CHART, sem = variable_decomposition_update_from_controller(VARIABLE_DECOMPOSITION_CHART,
                                                                                      CHARTS_CONTROLLER)
    return VARIABLE_DECOMPOSITION_CHART, sem


@callback(
    [
        Output(component_id="variable_decomposition_chart_2", component_property="figure"),
        Output(component_id="variable_decomposition_semaphore_2", component_property="children")
    ],
    Input(component_id='hidden_variable_decomposition_trigger_2', component_property='value')
)
def callback_variable_decomposition_2(update_chart):
    global CHARTS_CONTROLLER_2, VARIABLE_DECOMPOSITION_CHART_2
    if not update_chart:
        # do not update the chart
        raise PreventUpdate
    VARIABLE_DECOMPOSITION_CHART_2, sem = variable_decomposition_update_from_controller(VARIABLE_DECOMPOSITION_CHART_2,
                                                                                        CHARTS_CONTROLLER_2)
    return VARIABLE_DECOMPOSITION_CHART_2, sem


def variable_decomposition_update_from_controller(variable_chart, controller):
    if len(controller.decomposition) == 0:
        # no point, reset the chart
        for i in range(len(controller.variables) * 2):
            variable_chart["data"][i]["x"], variable_chart["data"][i]["y"] = [], []
        return variable_chart, semaphore_generator(controller)
    points_copy = np.array(controller.decomposition)
    x_axis = controller.decomposition_x
    count = 0
    for i in range(len(controller.variables)):
        # calculate the period of anomaly
        anomaly = np.where(points_copy[:, i] > controller.decomposition_thr[i], points_copy[:, i], np.nan)
        variable_chart["data"][count]["x"], variable_chart["data"][count]["y"] = \
            x_axis, points_copy[:, i]
        count += 1
        variable_chart["data"][count]["x"], variable_chart["data"][count]["y"] = \
            x_axis, anomaly
        count += 1
    return variable_chart, semaphore_generator(controller)


@callback(
    Output(component_id="right-column", component_property="children"),
    Input(component_id='variable_decomposition_chart', component_property='restyleData')
)
def callback_update_variable_decomposition_options(layout):
    global VARIABLE_DECOMPOSITION_CHART, CHARTS_CONTROLLER
    if not layout or len(layout) < 2:
        raise PreventUpdate
    chart = layout[1][0]
    active = layout[0]["visible"][0] == True
    # hide or show the chart (only anomaly)
    VARIABLE_DECOMPOSITION_CHART["data"][chart + 1]["visible"] = active
    # to trigger the update of the chart
    CHARTS_CONTROLLER.update_charts(True)
    # nothing to update
    raise PreventUpdate


@callback(
    Output(component_id="right-column-2", component_property="children"),
    Input(component_id='variable_decomposition_chart_2', component_property='restyleData')
)
def callback_update_variable_decomposition_options_2(layout):
    global VARIABLE_DECOMPOSITION_CHART_2, CHARTS_CONTROLLER_2
    if not layout or len(layout) < 2:
        raise PreventUpdate
    chart = layout[1][0]
    active = layout[0]["visible"][0] == True
    # hide or show the chart (only anomaly)
    VARIABLE_DECOMPOSITION_CHART_2["data"][chart + 1]["visible"] = active
    # to trigger the update of the chart
    CHARTS_CONTROLLER_2.update_charts(True)
    # nothing to update
    raise PreventUpdate


@callback(
    Output(component_id="reset_button", component_property="children"),
    Input(component_id='reset_button', component_property='n_clicks')
)
def callback_reset_button(n_clicks):
    global CHARTS_CONTROLLER, CHARTS_CONTROLLER_2
    if n_clicks is None:
        raise PreventUpdate
    # reset the charts
    CHARTS_CONTROLLER.reset()
    CHARTS_CONTROLLER_2.reset()
    # nothing to update
    raise PreventUpdate


# --------
# Custom apis
# --------

# @SERVER.route("/map_position_insert", methods=['GET', 'POST'])
# def api_map_position_insert():
#     global CHARTS_CONTROLLER
#     if request.method != 'POST':
#         return 'Method not allowed', 405
#     # bytes to dict
#     CHARTS_CONTROLLER.map_position_insert(json.loads(request.data.decode('utf-8')))
#     # Return 200 OK
#     return "OK", 200


# @SERVER.route("/variable_decomposition_insert", methods=['GET', 'POST'])
# def api_variable_decomposition_insert():
#     global CHARTS_CONTROLLER
#     if request.method != 'POST':
#         return 'Method not allowed', 405
#     CHARTS_CONTROLLER.variable_decomposition_insert(json.loads(request.data.decode('utf-8')))
#     # Return 200
#     return "OK", 200
#
#
# @SERVER.route("/commit", methods=['GET', 'POST'])
# def api_commit():
#     global CHARTS_CONTROLLER
#     if request.method == 'GET':
#         # Here we reset the data
#         CHARTS_CONTROLLER.reset()
#         return "OK", 200
#     # TODO: implement save to database
#     return "Not implemented", 501


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
    global CONFIG
    # subscribe to the topic
    client.subscribe(CONFIG["mqtt"]["map_topic"]+"/robot_1")
    client.subscribe(CONFIG["mqtt"]["map_topic"]+"/robot_2")
    client.subscribe(CONFIG["mqtt"]["decomposition_topic"]+"/robot_1")
    client.subscribe(CONFIG["mqtt"]["decomposition_topic"]+"/robot_2")


def mqtt_on_message(client, userdata, msg):
    global CHARTS_CONTROLLER, CHARTS_CONTROLLER_2
    # map
    if msg.topic == CONFIG["mqtt"]["map_topic"]+"/robot_1":
        CHARTS_CONTROLLER.map_position_insert(json.loads(msg.payload))
    if msg.topic == CONFIG["mqtt"]["map_topic"]+"/robot_2":
        CHARTS_CONTROLLER_2.map_position_insert(json.loads(msg.payload))
    # decomposition
    if msg.topic == CONFIG["mqtt"]["decomposition_topic"]+"/robot_1":
        CHARTS_CONTROLLER.variable_decomposition_insert(json.loads(msg.payload))
    if msg.topic == CONFIG["mqtt"]["decomposition_topic"]+"/robot_2":
        CHARTS_CONTROLLER_2.variable_decomposition_insert(json.loads(msg.payload))


if __name__ == "__main__":
    mqtt_init(CONFIG["mqtt"]["host"], int(CONFIG["mqtt"]["port"]))
    # initialize the app variables
    app_init()
    # initialize the mqtt client
    # Start the app
    APP.run(debug=False, host=CONFIG["app"]["host"], port=int(CONFIG["app"]["port"]))
