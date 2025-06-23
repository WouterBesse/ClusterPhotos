from dash import html, callback, Output, Input
from widgets.image_cluster import ImageGraph
import pandas as pd


class StatCards:
    def __init__(self, data: pd.DataFrame) -> None:
        self.cards = html.Div(id="stats-cards", children=self.create_stats_cards(data))

    def create_stats_cards(self, data: pd.DataFrame):
        if data.empty:
            return html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "0", style={"font-size": "2rem", "color": "#6b7280"}
                            ),
                            html.P("No Data Available", style={"color": "#666"}),
                        ],
                        style={
                            "background": "white",
                            "padding": "20px",
                            "border-radius": "8px",
                            "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
                            "text-align": "center",
                        },
                    )
                ]
            )
        total = len(data)
        avg = data["prob"].mean()
        high = len(data[data["prob"] >= 0.7])
        return html.Div(
            [
                html.Div(
                    [
                        html.H3(str(total), style={"color": "#2563eb"}),
                        html.P("Total Images", style={"color": "#666"}),
                    ],
                    style={
                        "background": "white",
                        "padding": "20px",
                        "border-radius": "8px",
                        "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "text-align": "center",
                    },
                ),
                html.Div(
                    [
                        html.H3(f"{avg:.2f}", style={"color": "#059669"}),
                        html.P("Average Probability", style={"color": "#666"}),
                    ],
                    style={
                        "background": "white",
                        "padding": "20px",
                        "border-radius": "8px",
                        "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "text-align": "center",
                    },
                ),
                html.Div(
                    [
                        html.H3(str(high), style={"color": "#dc2626"}),
                        html.P("High Confidence (≥0.7)", style={"color": "#666"}),
                    ],
                    style={
                        "background": "white",
                        "padding": "20px",
                        "border-radius": "8px",
                        "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "text-align": "center",
                    },
                ),
            ],
            style={
                "display": "grid",
                "grid-template-columns": "repeat(auto-fit,minmax(200px,1fr))",
                "gap": "20px",
                "margin-bottom": "30px",
            },
        )

    def register_callbacks(self, dataStore: tuple[str, str] = ("data-store", "data")):
        @callback(
            Output("stats-cards", "children"),
            Input(dataStore[0], dataStore[1]),
        )
        def update_visuals(data):
            df = pd.DataFrame(data)
            stats = self.create_stats_cards(df)

    def get_widget(self):
        return self.cards
