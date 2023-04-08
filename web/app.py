import json

import dash
import numpy as np
from dash import dcc, html, Output, Input, callback, ClientsideFunction, State
from dash.exceptions import PreventUpdate
from flask import Flask, request
import plotly.graph_objects as go
import debug as debug_costrants

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
MAP_CHART, VARIABLE_DECOMPOSITION_CHART = {}, {}
MAP_POSITION, VARIABLE_DECOMPOSITION = [], []


# -------
# Layout methods
# -------

def app_init():
    global MAP_POSITION, VARIABLES, VARIABLE_DECOMPOSITION
    map_chart_init()
    variable_decomposition_chart_init()
    APP.title = "Anomaly Detection on Kairos"
    APP.layout = html.Div(
        id="app",
        children=[
            # Global interval for refreshing data
            dcc.Interval(
                id="interval-component",
                interval=200,
                n_intervals=0
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
                className="banner text-5xl font-bold",
                children=[
                    html.Div(className="fa fa-chart-bar text-blue-500"),
                    html.H3("Anomaly Detection", className="ml-2 font-bold uppercase")
                ]
            ),
            # left column
            html.Div(
                id="left-column",
                className="three columns",
                children=[
                ],
            ),
            # Right column
            html.Div(
                id="right-column",
                className="nine columns shadow-lg",
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
    if debug:
        MAP_POSITION = debug_costrants.DEBUG_MAP_POSITION
        VARIABLE_DECOMPOSITION = debug_costrants.VARIABLE_DECOMPOSITION_DEBUG


def semaphore_generator():
    global VARIABLE_DECOMPOSITION
    r = []
    h2_thr = [1.26020238, 6.67861522, 0.4251171, 0.70920265, 0.94272347, 0.89692743]
    if not VARIABLE_DECOMPOSITION or len(VARIABLE_DECOMPOSITION) == 0:
        # semaphore are all green, no anomaly detected
        last_anomaly = np.zeros(len(VARIABLES))
    else:
        # there are datas, we calculate the last anomaly
        last_anomaly = VARIABLE_DECOMPOSITION[-1]
    for i in range(len(VARIABLES)):
        active = last_anomaly[i] > h2_thr[i]
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
            x=.4,
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
    VARIABLE_DECOMPOSITION_CHART.add_traces([
        go.Scatter(
            x=[], y=[],
            mode="lines+markers",
            name=VARIABLES[i],
            marker={"color": colors[i % len(VARIABLES)], "size": 3}
        ) for i in range(len(VARIABLES))
    ])


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
    MAP_POSITION.append([data["X"], data["Y"], data["anomaly"]])
    # set new data to true, to trigger the update of the charts
    NEW_DATA = True
    # Return 200 OK
    return "OK", 200


@SERVER.route("/variable_decomposition_insert", methods=['GET', 'POST'])
def api_variable_decomposition_insert():
    global NEW_DATA, VARIABLE_DECOMPOSITION
    if request.method != 'POST':
        return 'Method not allowed', 405
    data = json.loads(request.data.decode('utf-8'))
    VARIABLE_DECOMPOSITION.append([data[var] for var in VARIABLES])
    # set new data to true, to trigger the update of the charts
    NEW_DATA = True
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
# Callbacks
# --------
@callback(
    [
        Output(component_id='hidden_map_position_trigger', component_property='value'),
        Output(component_id='hidden_variable_decomposition_trigger', component_property='value')
    ],
    Input(component_id='interval-component', component_property='n_intervals')
)
def callback_hidden_trigger(_):
    global NEW_DATA
    if not NEW_DATA:
        return 0, 0
    NEW_DATA = False
    return 1, 1


@callback(
    Output(component_id='map_position_chart', component_property='figure'),
    Input(component_id='hidden_map_position_trigger', component_property='value')
)
def callback_map_position(update_chart):
    global MAP_CHART, MAP_POSITION
    if not update_chart:
        # print("not called map")
        # do not update the chart
        raise PreventUpdate
    print("called map")
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
    # Update only the charts without reloading!
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
    global VARIABLE_DECOMPOSITION, VARIABLE_DECOMPOSITION_CHART
    if not update_chart:
        # print("not called variable")
        # do not update the chart
        raise PreventUpdate
    # n_intervals: not used its given by the default reloader
    if len(VARIABLE_DECOMPOSITION) == 0:
        # no point, reset the chart
        for i in range(6):
            print("resetting variable")
            VARIABLE_DECOMPOSITION_CHART["data"][i]["x"], VARIABLE_DECOMPOSITION_CHART["data"][i]["y"] = [], []
        return VARIABLE_DECOMPOSITION_CHART, semaphore_generator()
    points_copy = np.array(VARIABLE_DECOMPOSITION)
    x_axis = [k for k in range(len(VARIABLE_DECOMPOSITION))]
    for i in range(6):
        VARIABLE_DECOMPOSITION_CHART["data"][i]["x"], VARIABLE_DECOMPOSITION_CHART["data"][i]["y"] = \
            x_axis, points_copy[:, i]
    print("updated variable")
    return VARIABLE_DECOMPOSITION_CHART, semaphore_generator()


if __name__ == "__main__":
    debug = True
    host, port = '0.0.0.0', 8050
    # Initialize the app
    app_init()
    # Start the app
    APP.run(debug=debug, host=host, port=port)
