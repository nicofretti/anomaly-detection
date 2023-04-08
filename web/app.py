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
                value=1
            ),
            # hidden input for the last variable decomposition, if value is 0, the chart is not updated
            dcc.Input(
                id="hidden_variable_decomposition_trigger",
                type="hidden",
                value=1
            ),

            # banner
            html.Div(
                id="banner",
                className="banner text-5xl",
                children=[html.Div(className="fa fa-chart-bar text-red-700"),
                          html.H3("Anomaly Detection", className="ml-2 text-gray-700")]
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
                className="nine columns",
                children=[
                    html.Div(
                        id="charts",
                        className="flex bg-white mt-2",
                        children=[
                            html.Div(
                                className="p-8",
                                children=[
                                    html.P(
                                        className="text-center text-3xl border-b-[4px] border-orange-500",
                                        children="ICE LAB MAP"
                                    ),
                                    dcc.Graph(
                                        id="map_position_chart",
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
                                        className="text-center text-3xl border-b-[4px] border-orange-500",
                                        children="HELLINGER DISTANCE DECOMPOSITION"
                                    ),
                                    html.Div(
                                        id="variable_decomposition_semaphore",
                                        className="absolute top-60 left-44 text-center z-10 text-black " + \
                                                  "pl-2 pr-6",
                                        style={
                                            "background-color": "rgba(255, 255, 255, 0.9)",
                                        },
                                        children=semaphore_generator()
                                    ),
                                    dcc.Graph(
                                        id="variable_decomposition_chart", figure={},
                                        style={"height": "600px"}
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
        # set true and false respectively to red and green with the threshold
        last_anomaly = np.where(last_anomaly > h2_thr, last_anomaly, True)
    for i in range(len(VARIABLES)):
        active = False
        if np.isnan(last_anomaly[i]):
            active = True
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
        uirevision="ICE-LAB"
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
            y=0.99,
            xanchor="left",
            x=.9,
            bgcolor="rgba(255, 255, 255, 0.9)"
        ),
        # string to maintain user selections
        uirevision="VARIABLE DECOMPOSITION"
    )

    # TODO
    # X_dec = h2_variables[:, 0]
    # Y_dec = h2_variables[:, 1]
    # O_dec = h2_variables[:, 2]
    # LS_dec = h2_variables[:, 3]
    # LC_dec = h2_variables[:, 4]
    # LD_dec = h2_variables[:, 5]
    # decomposition.plot(X_dec, 'k', linestyle='-', label='X')
    # decomposition.plot(Y_dec, 'g', linestyle='-', label='Y')
    # decomposition.plot(O_dec, 'y', linestyle='-', label='O')
    # decomposition.plot(LS_dec, 'c', linestyle='-', label='LS')
    # decomposition.plot(LC_dec, 'm', linestyle='-', label='LC')
    # decomposition.plot(LD_dec, 'b', linestyle='-', label='LD')
    colors = ["blue", "purple", "green", "yellow", "brown", "orange"]

    VARIABLE_DECOMPOSITION_CHART.add_traces([
        go.Scatter(
            x=[], y=[],
            mode="lines+markers",
            name=VARIABLES[i],
            marker={"color": colors[i], "size": 3}
        ) for i in range(len(VARIABLES))
    ])

    # # PLOT A RED LINE IF THERE IS AN ANOMALY ON THE CURRENT SENSORS
    # anomalies = np.where(h2_variables > h2_thr, h2_variables, np.nan)
    # X_an = anomalies[:, 0]
    # Y_an = anomalies[:, 1]
    # O_an = anomalies[:, 2]
    # LS_an = anomalies[:, 3]
    # LC_an = anomalies[:, 4]
    # LD_an = anomalies[:, 5]
    # decomposition.plot(X_an, 'kx')
    # decomposition.plot(Y_an, 'gx')
    # decomposition.plot(O_an, 'yx')
    # decomposition.plot(LS_an, 'cx')
    # decomposition.plot(LC_an, 'mx')
    # decomposition.plot(LD_an, 'bx')
    # labels = ['X', 'Y', 'O', 'LS', 'LC', 'LD']
    # colors = np.where(h2_variables[-1] > h2_thr, 'red', 'green')
    # lights = []
    # for i in range(0, len(labels)):
    #     light = mlines.Line2D([], [], color=colors[i], marker='.', linestyle='None', markersize=20, label=labels[i])
    #     lights.append(light)
    # semaphore_legend = decomposition.legend(handles=lights, loc='upper left')
    # decomposition.legend(loc='upper right')
    # decomposition.add_artist(semaphore_legend)


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
    [
        Input(component_id='interval-component', component_property='n_intervals'),
        Input(component_id='hidden_map_position_trigger', component_property='value'),
        Input(component_id='hidden_variable_decomposition_trigger', component_property='value')
    ]
)
def callback_hidden_trigger(n_intervals, map_trigger, variable_trigger):
    global NEW_DATA, MAP_POSITION, VARIABLE_DECOMPOSITION
    if not NEW_DATA:
        raise PreventUpdate
    reload_map = 0
    reload_variable = 0
    if len(MAP_POSITION) > 0 and not map_trigger:
        reload_map = 1
    if len(VARIABLE_DECOMPOSITION) > 0 and not variable_trigger:
        reload_variable = 1
    NEW_DATA = False
    return reload_map, reload_variable


@callback(
    Output(component_id='map_position_chart', component_property='figure'),
    Input(component_id='hidden_map_position_trigger', component_property='value')
)
def callback_map_position(update_chart):
    if not update_chart:
        # do not update the chart
        raise PreventUpdate
    print("called variable")
    # n_intervals: not used its given by the default reloader
    global MAP_CHART, MAP_POSITION
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
    # Input(component_id='interval-component', component_property='n_intervals')
    Input(component_id='hidden_variable_decomposition_trigger', component_property='value')
)
def callback_variable_decomposition(update_chart):
    print("called map")
    if not update_chart:
        # do not update the chart
        raise PreventUpdate
    # n_intervals: not used its given by the default reloader
    global VARIABLE_DECOMPOSITION, VARIABLE_DECOMPOSITION_CHART
    if not VARIABLE_DECOMPOSITION or len(VARIABLE_DECOMPOSITION) == 0:
        # no point, reset the chart
        for i in range(6):
            VARIABLE_DECOMPOSITION_CHART["data"][i]["x"], VARIABLE_DECOMPOSITION_CHART["data"][i]["y"] = [], []
        return VARIABLE_DECOMPOSITION_CHART
    points_copy = np.array(VARIABLE_DECOMPOSITION)
    x_axis = [k for k in range(len(VARIABLE_DECOMPOSITION))]
    for i in range(6):
        VARIABLE_DECOMPOSITION_CHART["data"][i]["x"], VARIABLE_DECOMPOSITION_CHART["data"][i]["y"] = \
            x_axis, points_copy[:, i]
    return [VARIABLE_DECOMPOSITION_CHART, semaphore_generator()]


if __name__ == "__main__":
    debug = True
    host, port = '0.0.0.0', 8050
    # Initialize the app
    app_init()
    # Start the app
    APP.run(debug=debug, host=host, port=port)
