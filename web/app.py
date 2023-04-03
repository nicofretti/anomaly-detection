import json

import dash
from dash import dcc
from dash import html
from flask import Flask, request

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

MAP_POSITION = []
VARIABLE_DECOMPOSITION = []


def app_init():
    APP.title = "Anomaly Detection on Kairos"
    APP.layout = html.Div(
        id="app",
        children=[
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
                    # Patient Volume Heatmap
                    html.Div(
                        id="patient_volume_card",
                        children=[
                            html.B("First chart"),
                            html.Hr(),
                            dcc.Graph(id="patient_volume_hm"),
                        ],
                    ),
                ],
            ),
        ],
    )


@SERVER.route("/map_position_insert", methods=['GET', 'POST'])
def map_position_insert():
    if request.method != 'POST':
        return 'Method not allowed', 405
    # bytes to dict
    data = json.loads(request.data.decode('utf-8'))
    MAP_POSITION.append(data)
    # Return 200 OK
    return "OK", 200


@SERVER.route("/variable_decomposition_insert", methods=['GET', 'POST'])
def variable_decomposition_insert():
    if request.method != 'POST':
        return 'Method not allowed', 405
    data = json.loads(request.data.decode('utf-8'))
    VARIABLE_DECOMPOSITION.append(data)
    print(len(VARIABLE_DECOMPOSITION))
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


if __name__ == "__main__":
    host = '0.0.0.0'
    port = 8050
    debug = True
    # Initialize the app
    app_init()
    # Start the app
    APP.run(debug=debug, host=host, port=port)
