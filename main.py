from dash import Dash, html, Input, Output, State, dcc
import dash
from widgets import (
    image_cluster,
    chart_panel,
    img_grid,
    job_panel,
    stat_card,
    tabs,
    sel_img,
)
from utils.util import load_data
from utils.jobmanager import JobManager
from pathlib import Path
# from invLearning import ImageSI


def main(imagePath: Path) -> None:
    data_path = "./Text_clustering/data/stanford-40-actions/JPEGImages"
    prob_file = "./Text_clustering/ICTC/data/stanford-40-actions/gpt4/action_40_classes/name_your_experiment/step2a_result.txt"
    job_manager = JobManager()
    df = load_data(prob_file, data_path)

    jobPanel = job_panel.JobPanel(job_manager)
    statCards = stat_card.StatCards(df)
    chartPanel = chart_panel.ChartPanel(data_path)
    selImg = sel_img.SelectedImage()
    imgGrid = img_grid.ImageGrid()

    app = Dash(__name__)

    dataIds = ("data-store", "data")

    app.layout = html.Div(
        [
            dcc.Store(
                id=dataIds[0], data=df.to_dict("records") if not df.empty else []
            ),
            dcc.Store(id="selected-image-store"),
            jobPanel.get_store(),
            dcc.Interval(id="interval-component", interval=10000, n_intervals=0),
            html.Div(
                [
                    html.H1(
                        "🖼️ Interactive Image Classification Dashboard",
                        style={"text-align": "center", "color": "#1f2937"},
                    ),
                ],
                style={"margin-bottom": "30px"},
            ),
            jobPanel.get_widget()[1],
            jobPanel.get_widget()[0],
            statCards.get_widget(),
            chartPanel.get_widget(),
            selImg.get_widget(),
            imgGrid.get_widget(),
        ],
        style={
            "padding": "20px",
            "font-family": "system-ui, -apple-system",
            "background": "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
        },
    )

    jobPanel.register_callbacks(dataIds, prob_file, data_path)
    statCards.register_callbacks(dataIds)
    chartPanel.register_callbacks(dataIds)
    selImg.register_callbacks(dataIds)

    imgGrid.register_callbacks(dataIds)

    #    [
    #        html.H2(
    #            "Clustered Image Graph",
    #            style={
    #                "backgroundColor": "#FED4E7",
    #                "margin": 0,
    #                "height": "80px",
    #                "padding": 20,
    #            },
    #        ),
    #        tabs.tabs,
    #        # html.Div(id=canvasID),
    #        # graph_component.graph,
    #    ],
    #    style={
    #        "backgroundColor": "#FEEBF4",
    #        "height": "100vh",
    #        "display": "flex",
    #        "flexDirection": "column",
    #        "padding": 0,
    #        "margin": 0,
    #    },
    # )

    app.run(debug=True)


if __name__ == "__main__":
    imagePath = Path("./holidayimgs/archive/images/")

    #assert imagePath.is_dir(), f"Path {imagePath} is not a directory"

    main(imagePath)
