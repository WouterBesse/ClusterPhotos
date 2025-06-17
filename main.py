from dash import Dash, html
from widgets.image_cluster import ImageGraph
from widgets.tabs import Tabs
from pathlib import Path
from invLearning import ImageSI


def main(imagePath: Path) -> None:
    canvasID = "thecanvas"
    imageSI = ImageSI(imagePath, 40)
    print("Shape of projection:", imageSI.projection.shape)

    app = Dash(__name__)
    graph_component = ImageGraph(
        Path("./holidayimgs/archive/images/"), graph_id="image-graph", imageSi=imageSI
    )
    graph_component.register_callback()

    tabs = Tabs([graph_component], canvasID)

    app.layout = html.Div(
        [
            html.H2(
                "Clustered Image Graph",
                style={
                    "backgroundColor": "#FED4E7",
                    "margin": 0,
                    "height": "80px",
                    "padding": 20,
                },
            ),
            tabs.tabs,
            # html.Div(id=canvasID),
            # graph_component.graph,
        ],
        style={
            "backgroundColor": "#FEEBF4",
            "height": "100vh",
            "display": "flex",
            "flexDirection": "column",
            "padding": 0,
            "margin": 0,
        },
    )

    app.run(debug=True)


if __name__ == "__main__":
    imagePath = Path("./holidayimgs/archive/images/")

    assert imagePath.is_dir(), f"Path {imagePath} is not a directory"

    main(imagePath)
