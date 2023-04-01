import dash
from dash import dcc
from dash import html

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_scripts=["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}],
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


def app_start(deb: bool = False):
    app.title = "Anomatext-red-700ly Detection on Kairos"
    app.layout = html.Div(
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
    app.run(debug=deb)


if __name__ == "__main__":
    debug = True
    app_start(debug)
