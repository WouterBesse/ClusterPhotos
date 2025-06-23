from dash import html, callback, Output, Input, State, dcc
import pandas as pd

from utils.util import CLUSTER_COLORS


class ImageGrid:
    def __init__(self) -> None:
        self.grid = html.Div(id="cluster-sections")

    def create_image_grid(self, data, cluster_name, max_images=12):
        subset = data[data["cluster"] == cluster_name].head(max_images)
        if subset.empty:
            return html.Div(
                "No images in this cluster", className="text-center text-gray-500"
            )
        tiles = []
        for _, row in subset.iterrows():
            tiles.append(
                html.Div(
                    [
                        html.Img(
                            src=row["uri"],
                            id={"type": "cluster-image", "index": row["plot_index"]},
                            style={
                                "width": "120px",
                                "height": "120px",
                                "object-fit": "cover",
                                "border-radius": "8px",
                                "box-shadow": "0 2px 8px rgba(0,0,0,0.1)",
                                "cursor": "pointer",
                                "transition": "transform 0.2s, box-shadow 0.2s",
                            },
                            className="cluster-image-hover",
                        ),
                        html.P(
                            row["filename"][:15] + "..."
                            if len(row["filename"]) > 15
                            else row["filename"],
                            style={
                                "font-size": "10px",
                                "margin": "5px 0",
                                "text-align": "center",
                                "color": "#666",
                            },
                        ),
                        html.P(
                            f"p={row['prob']:.2f}",
                            style={
                                "font-size": "12px",
                                "font-weight": "bold",
                                "text-align": "center",
                                "color": CLUSTER_COLORS[cluster_name],
                            },
                        ),
                    ],
                    style={"margin": "10px", "text-align": "center"},
                )
            )

        # Simply return the grid without the html.Style() wrapper
        return html.Div(
            tiles,
            style={
                "display": "grid",
                "grid-template-columns": "repeat(auto-fill,minmax(140px,1fr))",
                "gap": "15px",
                "padding": "20px",
            },
        )

    def register_callbacks(self, dataStore: tuple[str, str] = ("data-store", "data")):
        @callback(
            Output("cluster-sections", "children"),
            Input(dataStore[0], dataStore[1]),
            prevent_initial_call=True,
        )
        def update_visuals(data):
            df = pd.DataFrame(data)
            if not df.empty:
                clusters = html.Div(
                    [
                        html.H2(
                            "🖼️ Image Clusters",
                            style={"text-align": "center", "color": "#1f2937"},
                        ),
                        dcc.Tabs(
                            id="cluster-tabs",
                            value="Absolute probability",
                            children=[
                                dcc.Tab(
                                    label=f"✅ Absolute Probability ({len(df[df['cluster'] == 'Absolute probability'])})",
                                    value="Absolute probability",
                                ),
                                dcc.Tab(
                                    label=f"⚖️ In Between ({len(df[df['cluster'] == 'In between'])})",
                                    value="In between",
                                ),
                                dcc.Tab(
                                    label=f"❌ Zero Probability ({len(df[df['cluster'] == 'Zero probability'])})",
                                    value="Zero probability",
                                ),
                            ],
                        ),
                        html.Div(id="cluster-content", style={"margin-top": "20px"}),
                    ]
                )
            else:
                clusters = html.Div()
            return clusters

        @callback(
            Output("cluster-content", "children"),
            Input("cluster-tabs", "value"),
            State(dataStore[0], dataStore[1]),
            prevent_initial_call=True,
        )
        def update_cluster_content(tab, data):
            if not data:
                return html.Div()
            df = pd.DataFrame(data)
            return self.create_image_grid(df, tab, max_images=24)

    def get_widget(self):
        return self.grid
