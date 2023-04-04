import json

import dash
import numpy as np
from dash import dcc, html, Output, Input, callback
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
MAP_CHART, VARIABLE_DECOMPOSITION_CHART = {}, {}
MAP_POSITION, VARIABLE_DECOMPOSITION = [], []


# -------
# Layout methods
# -------

def app_init():
    global MAP_POSITION
    map_chart_init()
    variable_decomposition_chart_init()
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
                                        children="ICE LAB"
                                    ),
                                    dcc.Graph(
                                        id="map_position_chart",
                                        figure={},
                                        style={"height": "600px", "width": "400px"}
                                    ),
                                ]
                            ),
                            html.Div(
                                className="p-8 w-full",
                                children=[
                                    html.P(
                                        className="text-center text-3xl border-b-[4px] border-orange-500",
                                        children="ICE Lab"
                                    ),
                                    dcc.Graph(
                                        id="decomposition_chart", figure={},
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
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
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
            x=[],
            y=[],
            mode="markers",
            marker={"color": "blue"},
            name="correct behaviour"
        ),
        # anomalies
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"color": "red"},
            name="anomaly"
        ),
        # start and stop
        go.Scatter(
            x=[],
            mode="markers",
            marker={"color": "orange", "size": 12},
            name="start and end",
            showlegend=False
        )
    ])


def variable_decomposition_chart_init():
    global VARIABLE_DECOMPOSITION_CHART
    VARIABLE_DECOMPOSITION_CHART = go.Figure()
    # TODO
    # X_dec = h2_variables[:, 0]
    #         Y_dec = h2_variables[:, 1]
    #         O_dec = h2_variables[:, 2]
    #         LS_dec = h2_variables[:, 3]
    #         LC_dec = h2_variables[:, 4]
    #         LD_dec = h2_variables[:, 5]
    #         decomposition.plot(X_dec, 'k', linestyle='-', label='X')
    #         decomposition.plot(Y_dec, 'g', linestyle='-', label='Y')
    #         decomposition.plot(O_dec, 'y', linestyle='-', label='O')
    #         decomposition.plot(LS_dec, 'c', linestyle='-', label='LS')
    #         decomposition.plot(LC_dec, 'm', linestyle='-', label='LC')
    #         decomposition.plot(LD_dec, 'b', linestyle='-', label='LD')
    #
    #         # PLOT A RED LINE IF THERE IS AN ANOMALY ON THE CURRENT SENSORS
    #         anomalies = np.where(h2_variables > h2_thr, h2_variables, np.nan)
    #         X_an = anomalies[:, 0]
    #         Y_an = anomalies[:, 1]
    #         O_an = anomalies[:, 2]
    #         LS_an = anomalies[:, 3]
    #         LC_an = anomalies[:, 4]
    #         LD_an = anomalies[:, 5]
    #         decomposition.plot(X_an, 'kx')
    #         decomposition.plot(Y_an, 'gx')
    #         decomposition.plot(O_an, 'yx')
    #         decomposition.plot(LS_an, 'cx')
    #         decomposition.plot(LC_an, 'mx')
    #         decomposition.plot(LD_an, 'bx')
    #         labels = ['X', 'Y', 'O', 'LS', 'LC', 'LD']
    #         colors = np.where(h2_variables[-1] > h2_thr, 'red', 'green')
    #         lights = []
    #         for i in range(0, len(labels)):
    #             light = mlines.Line2D([], [], color=colors[i], marker='.', linestyle='None', markersize=20, label=labels[i])
    #             lights.append(light)
    #         semaphore_legend = decomposition.legend(handles=lights, loc='upper left')
    #         decomposition.legend(loc='upper right')
    #         decomposition.add_artist(semaphore_legend)


# --------
# Custom apis
# --------

@SERVER.route("/map_position_insert", methods=['GET', 'POST'])
def map_position_insert():
    global MAP_POSITION
    if request.method != 'POST':
        return 'Method not allowed', 405
    # bytes to dict
    data = json.loads(request.data.decode('utf-8'))
    MAP_POSITION.append([data["X"], data["Y"], data["anomaly"]])
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

@callback(
    Output(component_id='map_position_chart', component_property='figure'),
    Input(component_id='interval-component', component_property='n_intervals')
)
def update_map_position(n_intervals):
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
    # start and end
    MAP_CHART["data"][2]["x"], MAP_CHART["data"][2]["y"] = \
        [points_copy[0][0], points_copy[-1][0]], [points_copy[0][1], points_copy[-1][1]]
    return MAP_CHART


if __name__ == "__main__":
    debug = True
    host, port = '0.0.0.0', 8050
    # Initialize the app
    app_init()
    # Start the app
    APP.run(debug=debug, host=host, port=port)
