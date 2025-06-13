from dash import Dash, html
from widgets.image_cluster import ImageGraph
from pathlib import Path
from invLearning import ImageSI


def main(imagePath: Path) -> None:
    imageSI = ImageSI(imagePath, 40)
    print("Shape of projection:", imageSI.projection.shape)

    app = Dash(__name__)
    graph_component = ImageGraph(
        Path("./holidayimgs/archive/images/"), graph_id="image-graph", imageSi=imageSI
    )
    graph_component.register_callback()

    app.layout = html.Div([html.H2("Clustered Image Graph"), graph_component.graph])

    app.run_server(debug=True)


if __name__ == "__main__":
    imagePath = Path("./holidayimgs/archive/images/")

    assert imagePath.is_dir(), f"Path {imagePath} is not a directory"

    main(imagePath)
