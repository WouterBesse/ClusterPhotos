from widgets.image_cluster import ImageGraph
from dash import dcc, html, callback, Output, Input, State


class Tabs:
    def __init__(self, graphs: list[ImageGraph], canvas_id: str) -> None:
        self.graphs = graphs
        self.tabs = dcc.Tabs(
            id="tabs",
            value=f"{graphs[0].graph_id}",
            children=[
                dcc.Tab(
                    label=f"{graph.graph_id}",
                    value=f"{graph.graph_id}",
                    children=graph.graph,
                )
                for graph in graphs
            ],
        )
        self.canvas_id = canvas_id
        pass

    # def register_callback(self) -> None:
    #    @callback(Output(self.canvas_id, "children"), Input("tabs", "value"))
    #    def render_content(tab):
    #        for graph in self.graphs:
    #            if graph.graph_id == tab:
    #                graph.graph.style
    #        return html.Div(children="Error with tab :(")
