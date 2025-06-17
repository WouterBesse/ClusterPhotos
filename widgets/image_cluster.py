import plotly.graph_objs as go
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import base64
from tqdm import tqdm
from pathlib import Path
from dash import (
    dcc,
    Output,
    Input,
    State,
    callback,
    clientside_callback,
    html,
    no_update,
)
import dash_cytoscape as cyto
from PIL.ExifTags import TAGS
import io

from datetime import datetime
from functools import lru_cache


class ImageGraph:
    def __init__(
        self,
        img_folder: Path,
        graph_id: str,
        imageSi,
        data_folder: Path = Path("./data/"),
        size=(1600, 800),
    ):
        self.image_folder = img_folder
        self.graph_id = graph_id
        self.data_folder = data_folder
        self.si = imageSi
        self.size = size
        self.el_history: list[list[dict[str, dict | bool]]] = []

        # Make data folder if it doesn't exist yet, this saves us time so we don't have to load all images from disk each time
        if not data_folder.exists():
            data_folder.mkdir()

        self.thumbnail_dir = data_folder / "thumbnails"
        self.checkpoint_dir = data_folder / self.graph_id
        self.thumbnail_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self._progress_generator = None

        self.generate_thumbnails()
        self.set_elements()

        # Pre-generate thumbnails
        self.create_graph()

        self.graph = html.Div(
            [
                dcc.Store(id="progress-store", data=0),
                dcc.Interval(id="progress-interval", interval=6000, disabled=True),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    id="graph-loading-output",
                                    children=[
                                        self.cyto,
                                        html.Div(
                                            id="progress-bar-container",
                                            style={
                                                "position": "absolute",
                                                "top": "50%",
                                                "left": "50%",
                                                "transform": "translate(-50%, -50%)",
                                                "width": "50%",
                                                "display": "none",  # Hidden by default
                                                "backgroundColor": "#f3f3f3",
                                                "border": "1px solid #ccc",
                                                "borderRadius": "5px",
                                            },
                                            children=[
                                                html.Div(
                                                    id="progress-bar",
                                                    style={
                                                        "width": "0%",
                                                        "height": "30px",
                                                        "backgroundColor": "#4CAF50",
                                                        "textAlign": "center",
                                                        "lineHeight": "30px",
                                                        "color": "white",
                                                    },
                                                )
                                            ],
                                        ),
                                        dcc.Interval(
                                            id="interval",
                                            interval=1000,
                                            max_intervals=0,
                                        ),
                                    ],
                                ),
                            ],
                            style={"padding": 10, "width": "90%"},
                        ),
                        html.Div(
                            children=[
                                dcc.Loading(
                                    id="loading-1",
                                    type="dot",
                                    children=[
                                        html.Div(
                                            id="loading-output",
                                            children=[
                                                html.Button(
                                                    "Submit Changes",
                                                    id="submit-val",
                                                    n_clicks=0,
                                                    style={
                                                        "width": "100%",
                                                        "backgroundColor": "#EDC9FF",
                                                        "height": "40px",
                                                    },
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                            style={
                                "padding": 10,
                                "flex": 1,
                                "width": "10%",
                                "backgroundColor": "#FED4E7",
                            },
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "row"},
                ),
                html.Div(children="History Slider"),
                dcc.Slider(0, 0, 1, value=0, id="history_slider"),
                html.Div(id="dummy", style={"display": "none"}),
                html.Div(id="dummy2", style={"display": "none"}),
                html.Div(id="dummy3", style={"display": "none"}),
            ],
            style={"display": "flex", "flexDirection": "column"},
        )

        self.new_positions: dict[str, np.ndarray] = {}

    def create_graph(self) -> None:
        stylesheet = [
            {
                "selector": "#pca-graph .cy-node",
                "style": {
                    "width": "calc(50px / var(--cytoscape-zoom))",
                    "height": "calc(50px / var(--cytoscape-zoom))",
                },
            }
        ]

        self.cyto = cyto.Cytoscape(
            id="pca-graph",
            elements=self.elements,
            style={"width": "100%", "height": f"{self.size[1]}px"},
            stylesheet=stylesheet,
            layout={"name": "preset"},
            userZoomingEnabled=True,
            userPanningEnabled=True,
            boxSelectionEnabled=True,
            autoungrabify=False,
        )

    def pil_image_to_base64(self, img: Image.Image) -> str:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{encoded}"

    def generate_thumbnails(self, size=(100, 100)):
        """Generate and save thumbnails for all images"""
        for path in tqdm(self.si.image_paths, desc="Generating thumbnails"):
            thumbnail_path = self.thumbnail_dir / f"{Path(path).stem}_thumb.jpg"
            if not thumbnail_path.exists():
                try:
                    with Image.open(path) as img:
                        img.thumbnail(size)
                        img.save(thumbnail_path, "JPEG", quality=85)
                except Exception as e:
                    print(f"Error generating thumbnail for {path}: {e}")

    @lru_cache(maxsize=200)
    def get_thumbnail_base64(self, path) -> tuple[str, tuple[int, int]] | None:
        """Get cached base64 thumbnail"""
        thumbnail_path = self.thumbnail_dir / f"{Path(path).stem}_thumb.jpg"
        try:
            with Image.open(thumbnail_path) as img:
                b64 = self.pil_image_to_base64(img)
                return b64, img.size
        except Exception as e:
            print(f"Error loading thumbnail for {path}: {e}")
            return None

    def get_image_timestamp(self, path: Path):
        try:
            image = Image.open(path)
            exif = image._getexif()
            if not exif:
                return None
            for tag, value in exif.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "DateTimeOriginal":
                    return value
        except Exception:
            return None
        return None

    def set_elements(self) -> None:
        x_coords: np.ndarray = self.si.projection[:, 0]
        y_coords: np.ndarray = self.si.projection[:, 1]
        print(self.si.projection)

        self.min_x = np.min(x_coords)
        self.max_x = np.max(x_coords) - self.min_x
        self.min_y = np.min(y_coords)
        self.max_y = np.max(y_coords) - self.min_y

        x_coords_norm = ((x_coords - self.min_x) / self.max_x) * self.size[0]
        y_coords_norm = ((y_coords - self.min_y) / self.max_y) * self.size[1]

        self.elements: list[dict[str, dict | bool]] = []
        for i, (path, dims) in enumerate(
            zip(
                self.si.image_paths, zip(x_coords_norm.tolist(), y_coords_norm.tolist())
            )
        ):
            thumb, imgdims = self.get_thumbnail_base64(path)
            x, y = dims
            element = {
                "data": {"id": f"node{i}", "label": f"{i}", "bg": thumb, "path": path},
                "position": {"x": x, "y": y},
                "grabbable": True,
            }
            self.elements.append(element)

        self.el_history.append(self.elements)
        self.si.save_checkpoint(self.checkpoint_dir / f"{len(self.el_history) - 1}.pth")

    def register_callback(self):
        @callback(
            Output("dummy3", "children"),
            Input("pca-graph", "dragNode"),
            prevent_initial_call=True,
        )
        def save_new_node_pos(node):
            x = (node["position"]["x"] / self.size[0]) * self.max_x + self.min_x
            y = (node["position"]["y"] / self.size[1]) * self.max_y + self.min_y
            self.new_positions[node["data"]["path"]] = np.array([x, y])

            return ""

        @callback(
            Output("submit-val", "disabled"),
            Output("progress-interval", "disabled"),
            Output("progress-bar-container", "style"),
            Input("submit-val", "n_clicks"),
            State("history_slider", "value"),
            prevent_initial_call=True,
        )
        def start_processing(n_clicks, slider_val):
            self.si.load_checkpoint(self.checkpoint_dir / f"{slider_val}.pth")
            self._progress_generator = self.si.update_model(
                updated_positions=self.new_positions, epochs=5
            )
            style = {
                "position": "absolute",
                "top": "50%",
                "left": "50%",
                "transform": "translate(-50%, -50%)",
                "width": "50%",
                "display": "block",
                "backgroundColor": "#f3f3f3",
                "border": "1px solid #ccc",
                "borderRadius": "5px",
            }
            return True, False, style

        @callback(
            Output("progress-store", "data"),
            Output("progress-interval", "disabled", allow_duplicate=True),
            Output("progress-bar-container", "style", allow_duplicate=True),
            Output("pca-graph", "elements"),
            Output("submit-val", "disabled", allow_duplicate=True),
            Input("progress-interval", "n_intervals"),
            prevent_initial_call=True,
        )
        def update_progress(_):
            try:
                progress = next(self._progress_generator)
                return progress, False, no_update, no_update, True
            except StopIteration:
                self.new_positions = {}
                self.set_elements()
                style = {
                    "position": "absolute",
                    "top": "50%",
                    "left": "50%",
                    "transform": "translate(-50%, -50%)",
                    "width": "50%",
                    "display": "none",
                    "backgroundColor": "#f3f3f3",
                    "border": "1px solid #ccc",
                    "borderRadius": "5px",
                }
                return 100, True, style, self.elements, False

        clientside_callback(
            """
            function(progress) {
                const bar = document.getElementById('progress-bar');
                if (bar) {
                    bar.style.width = progress + '%';
                    bar.innerText = Math.round(progress) + '%';
                }
                return '';
            }
            """,
            Output("progress-bar", "children"),
            Input("progress-store", "data"),
        )

        # @callback(
        # Output("pca-graph", "elements"),
        #    Output("loading-output", "children"),
        # Output("graph-loading-output", "children"),
        # Input("submit-val", "n_clicks"),
        # State("history_slider", "value"),
        # prevent_initial_call=True,
        # )
        # def submit_change(_, slider_val) -> tuple[html.Button, list]:
        # print("Submitting Changes")
        # print(self.new_positions)
        # print(self.si.image_paths[0])
        # self.si.load_checkpoint(self.checkpoint_dir / f"{slider_val}.pth")
        # self.si.update_model(updated_positions=self.new_positions, epochs=5)
        # self.new_positions = {}
        # self.set_elements()
        # return (
        #    html.Button(
        #    "Submit Changes",
        #    id="submit-val",
        #    n_clicks=0,
        #    style={
        #        "width": "100%",
        #        "backgroundColor": "#EDC9FF",
        #        "height": "40px",
        #    },
        # ),
        #    [
        #        self.cyto,
        #        dcc.Interval(
        #        id="interval",
        #        interval=1000,
        #        max_intervals=0,
        #    ),
        #    ],
        # )

        #        @callback(
        #    Output("history_slider", "max"),
        #    Output("history_slider", "value"),
        #    Input("loading-output", "children"),
        #    State("history_slider", "value"),
        #    State("history_slider", "max"),
        #    prevent_initial_call=True,
        # )
        # def update_slider(_, value: int, max: int) -> tuple[int, int]:
        #    """If history depth is changed, change slider to match."""
        #    print("Updating Slider")
        #    new_max = value + 1
        #
        #    og_final_checkpoint = (
        #        self.checkpoint_dir / f"{len(self.el_history) - 1}.pth"
        #     )
        #
        #    if len(self.el_history) is not new_max:
        #        idcs_to_remove = range(value, len(self.el_history) - 2)
        #        print(idcs_to_remove)
        #        # Remove overwritten history in el_history and remove the checkpoints.
        #        self.el_history = [
        #            els
        #           for i, els in enumerate(self.el_history)
        #           if i not in idcs_to_remove
        #       ]
        #                for i in idcs_to_remove:
        #           old_chkpt = self.checkpoint_dir / f"{i}.pth"
        #            old_chkpt.unlink()

        # Make sure that the newly created checkpoint is correctly numbered
        #       og_final_checkpoint.rename(
        #        self.checkpoint_dir / f"{len(self.el_history) - 1}.pth"
        #    )

        #   return new_max, new_max

        # @callback(
        #    Output("pca-graph", "elements"),
        #    Input("history_slider", "value"),
        #    prevent_initial_call=True,
        # )
        # def set_elements_from_history(step: int) -> list[dict[str, dict | bool]]:
        #    """If history slider is changed, set graph to appropriate view."""
        #    print("Setting elements from history")
        #    elements = self.el_history[step]
        #    return elements

        @callback(
            Output("history_slider", "max"),
            Output("history_slider", "value"),
            Input("submit-val", "n_clicks"),
            State("history_slider", "value"),
            prevent_initial_call=True,
        )
        def update_slider(_, value):
            new_max = len(self.el_history) - 1
            return new_max, new_max

        @callback(
            Output("pca-graph", "elements", allow_duplicate=True),
            Input("history_slider", "value"),
            prevent_initial_call=True,
        )
        def set_elements_from_history(step: int):
            elements = self.el_history[step]
            return elements

        clientside_callback(
            """
            function(n_intervals) {
                
                const cy = document.querySelector('#pca-graph')._cyreg.cy;
                if (!cy) {
                    return window.dash_clientside.no_update;
                }
                const container = document.querySelector('#pca-graph');
                const baseSize = 50;
                if (cy._zoomListenerRegistered) {
                    return '';
                }

                let debounceTimeout;
                let lastZoom = cy.zoom();
                function updateNodeSizes() {
                    const zoom = cy.zoom()
                    if (Math.abs(zoom - lastZoom) > 0.5) {
                        console.log('zoomin boi');
                        lastZoom = zoom;
                        const newSize = baseSize / zoom;
                
                        //const newSize = baseSize / zoom;
                        //cy.nodes().forEach(node => {
                        //    node.style('width', newSize);
                        //    node.style('height', newSize);
                        //});

                        cy.style()
                            .selector('node')
                            .style({
                            'width': `${newSize}px`,
                            'height': `${newSize}px`,
                            })
                            .update();
                    }
                }

                cy.on('zoom', () => {
                    // Clear previous debounce timer
                    clearTimeout(debounceTimeout);

                    // Set a new debounce timer for 200ms after last zoom event
                    debounceTimeout = setTimeout(() => {
                        updateNodeSizes()
                        //container.style.setProperty('--cytoscape-zoom', zoom);
                    }, 200);
                });

               

                // Initial sizing on registration
                updateNodeSizes();

                cy._zoomListenerRegistered = true;

                return '';
            }
            """,
            Output("dummy", "children"),
            Input("interval", "n_intervals"),  # fires once at load
        )

        @callback(Output("pca-graph", "stylesheet"), Input("pca-graph", "zoom"))
        def update_stylesheet(zoom):
            return [
                {
                    "selector": "node",
                    "style": {
                        "shape": "rectangle",
                        "background-fit": "cover",
                        "background-image": "data(bg)",  # now referencing the data field
                        "background-opacity": 1,
                        "label": "data(label)",
                        "font-size": "5px",
                    },
                },
                {"selector": "edge", "style": {"width": 2, "line-color": "#ccc"}},
            ]
