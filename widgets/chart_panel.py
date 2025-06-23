from dash import html, callback, Output, Input, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from widgets.image_cluster import ImageGraph
from pathlib import Path
from utils.util import CLUSTER_COLORS


class ChartPanel:
    def __init__(self, img_folder: str) -> None:
        self.img_graph = ImageGraph(Path(img_folder), "cyto")
        self.charts_grid = html.Div(
            id="charts-grid",
            children=[self.img_graph.graph, html.Div(id="pie_container")],
            style={
                "display": "grid",
                "grid-template-columns": "repeat(2,1fr)",
                "gap": "20px",
                "margin-bottom": "20px",
            },
        )
        pass

    def create_scatter_plot(self, data):
        if data.empty:
            return px.scatter(title="No data available - Run a job first!")
        fig = px.scatter(
            data,
            x="plot_index",
            y="prob",
            color="cluster",
            size=[15] * len(data),
            color_discrete_map=CLUSTER_COLORS,
            title="📊 Probability Distribution Across Images",
            labels={"plot_index": "Image Index", "prob": "Entailment Probability"},
            # Include plot_index in custom_data for accurate identification
            custom_data=["filename", "cluster", "plot_index"],
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cluster: %{customdata[1]}<br>"
                "Probability: %{y:.2f}<br>"
                "<i>Click to view image</i><extra></extra>"
            )
        )
        fig.update_layout(
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"
            ),
        )
        print(data)
        return fig

    def create_pie_chart(self, data):
        if data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available", xref="paper", yref="paper", x=0.5, y=0.5
            )
            return fig
        counts = data["cluster"].value_counts()
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=counts.index,
                    values=counts.values,
                    hole=0.4,
                    marker_colors=[CLUSTER_COLORS[l] for l in counts.index],
                    textinfo="label+percent+value",
                    textfont_size=12,
                )
            ]
        )
        fig.update_layout(
            title="🥧 Cluster Distribution",
            height=400,
            annotations=[
                dict(
                    text="Image<br>Clusters",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font_size=16,
                )
            ],
        )
        return fig

    def register_callbacks(self, dataStore: tuple[str, str] = ("data-store", "data")):
        @callback(
            Output("pie_container", "children"),
            Output("cyto_container", "children"),
            Input(dataStore[0], dataStore[1]),
        )
        def update_visuals(data: pd.DataFrame):
            df = pd.DataFrame(data)
            print(type(df))
            print(df)
            if not df.empty:
                pie_chart = dcc.Graph(figure=self.create_pie_chart(df))
                cyto = self.img_graph.get_graph(df)
            else:
                pie_chart = html.Div(
                    "No data to display. Run a job to generate results!",
                    style={"text-align": "center", "padding": "40px", "color": "#666"},
                )
                cyto = html.Div(
                    "No data to display. Run a job to generate results!",
                    style={"text-align": "center", "padding": "40px", "color": "#666"},
                )
            return pie_chart, cyto

        self.img_graph.register_callbacks()

    def get_widget(self):
        return self.charts_grid
