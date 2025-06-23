from PIL import Image
import numpy as np
import base64
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dash import (
    dcc,
    Output,
    Input,
    State,
    callback,
    html,
)
import dash_cytoscape as cyto
from PIL.ExifTags import TAGS
import io
from utils.util import modify_image_description
from functools import lru_cache


class ImageGraph:
    def __init__(
        self,
        img_folder: Path,
        graph_id: str,
        data_folder: Path = Path("./data/"),
        size=(1200, 400),
    ):
        self.image_folder = img_folder
        self.graph_id = graph_id
        self.data_folder = data_folder
        self.size = size
        self.elements = []
        self.el_history: list[list[dict[str, dict | bool]]] = []

        # Make data folder if it doesn't exist yet, this saves us time so we don't have to load all images from disk each time
        if not data_folder.exists():
            data_folder.mkdir()

        self.thumbnail_dir = data_folder / "thumbnails"
        self.thumbnail_dir.mkdir(exist_ok=True)

        # self.generate_thumbnails()
        # self.set_elements()
        self.create_graph()

        # Pre-generate thumbnails

        self.graph = html.Div(
            [
                html.Div(
                    children=[
                        html.Div(
                            id="cyto_wrapper",
                            children=[
                                # Background layers
                                html.Div(
                                    style={
                                        "position": "absolute",
                                        "left": "0%",
                                        "width": "33.33%",
                                        "height": "100%",
                                        "backgroundColor": "#ffadad",
                                        "zIndex": 0,
                                    }
                                ),
                                html.Div(
                                    style={
                                        "position": "absolute",
                                        "left": "33.33%",
                                        "width": "33.33%",
                                        "height": "100%",
                                        "backgroundColor": "#fdffb6",
                                        "zIndex": 0,
                                    }
                                ),
                                html.Div(
                                    style={
                                        "position": "absolute",
                                        "left": "66.66%",
                                        "width": "33.33%",
                                        "height": "100%",
                                        "backgroundColor": "#caffbf",
                                        "zIndex": 0,
                                    }
                                ),
                                # Cytoscape graph
                                html.Div(
                                    id="cyto_container",
                                    children=[
                                        self.cyto
                                    ],  # the Cytoscape component is added later
                                    style={
                                        "width": "100%",
                                        "height": f"{self.size[1]}px",
                                        "position": "relative",
                                        "zIndex": 1,
                                    },
                                ),
                            ],
                            style={
                                "position": "relative",
                                "width": "90%",
                                "height": f"{self.size[1]}px",
                            },
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
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "width": "1200px",
                    },
                ),
                # html.Div(children="History Slider"),
                # dcc.Slider(0, 0, 1, value=0, id="history_slider"),
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
                "selector": "node",
                "style": {
                    "width": "50px",
                    "height": "50px",
                    "shape": "rectangle",
                    "background-fit": "cover",
                    "background-image": "data(bg)",  # now referencing the data field
                    "background-opacity": 1,
                    "label": "data(label)",
                    "font-size": "5px",
                },
            }
        ]

        self.cyto = cyto.Cytoscape(
            id="pca-graph",
            elements=self.elements,
            style={"width": "1100px", "height": f"{self.size[1]}px"},
            stylesheet=stylesheet,
            layout={"name": "preset"},
            userZoomingEnabled=False,
            userPanningEnabled=False,
            boxSelectionEnabled=True,
            autoungrabify=False,
        )

    def pil_image_to_base64(self, img: Image.Image) -> str:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{encoded}"

    def generate_thumbnails(self, img_paths: list[str], size=(100, 100)):
        """Generate and save thumbnails for all images"""
        for path in tqdm(img_paths, desc="Generating thumbnails"):
            path = self.image_folder / path
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
        path = "./" + path
        print(path)
        thumbnail_path = self.thumbnail_dir / f"{Path(path).stem}_thumb.jpg"
        print(thumbnail_path)
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

    def set_elements(self, data: pd.DataFrame) -> None:
        mapping = {"Zero probability": 0, "In between": 0.5, "Absolute probability": 1}
        data["cluster_numeric"] = data["cluster"].map(mapping)
        # ← use the full width here
        hor_size     = self.size[0]
        padding_x    = 50
        zone_width   = hor_size / 3
        cluster_idx  = (data["cluster_numeric"] * 2).to_numpy().astype(int)

        # random within each padded zone
        random_offsets = np.random.uniform(
            low=0,
            high=(zone_width - 2 * padding_x),
            size=len(cluster_idx)
        )

        x_coords: np.ndarray = (
            -hor_size / 2
            + cluster_idx * zone_width
            + padding_x
            + random_offsets
        )
        # Only on initialisation set random y values, else let them stay the same
        if self.elements:
            y_coords: np.ndarray = np.array(
                [el["position"]["y"] for el in self.elements]
            )
        else:
            y_coords: np.ndarray = np.random.uniform(
                low=50, high=self.size[1] - 50, size=len(x_coords)
            )

        image_paths = data["filename"]

        self.elements: list[dict[str, dict | bool]] = []
        for i, (path, dims) in enumerate(
            zip(
                image_paths.to_numpy().tolist(),
                zip(x_coords.tolist(), y_coords.tolist()),
            )
        ):
            img_path = self.image_folder / path
            thumb, imgdims = self.get_thumbnail_base64(str(img_path))
            x, y = dims
            element = {
                "data": {
                    "id": f"node{i}",
                    "label": f"{i}",
                    "bg": thumb,
                    "path": str(path),
                    "pl_idx": data["plot_index"][i],
                },
                "position": {"x": x, "y": y},
                "grabbable": True,
            }
            self.elements.append(element)

        self.el_history.append(self.elements)

    def update_images(
        self, data: pd.DataFrame, current_filter: str
    ) -> tuple[pd.DataFrame, bool]:
        """Update image positions and confidence values. Absolutely horrid code I made here but gotta go brrr"""
        changed = False
        # print(data)
        for i, row in data.iterrows():
            if (
                row["filename"] in self.new_positions.keys()
            ):  # if the list is not empty, there is at least one match
                filename = row["filename"]
                el = self.new_positions[filename]
                changed = True
                x: float = el[0]
                y: float = el[1]
                print(x, y)
                thresh_l = 0 - (self.size[0] / 3) / 2 + 50
                thresh_h = (self.size[0] / 3) / 2 - 50
                print(thresh_l)
                print(thresh_h)
                if x < thresh_l:
                    data.at[i, "cluster"] = "Zero probability"
                elif x > thresh_h:
                    data.at[i, "cluster"] = "Absolute probability"
                else:
                    print("INBETWEEN")
                    data.at[i, "cluster"] = "In between"

                data.at[i, "confidence_level"] = "High"

                print(data.iloc[i]["cluster"])
                print(data.iloc[i]["confidence_level"])

                description = f"{current_filter}: {data.iloc[i]['cluster']}"
                success, message = modify_image_description(
                    data.iloc[i]["filename"], description
                )

                for j, el_old in enumerate(self.elements):
                    if el_old["data"]["path"] == filename:
                        self.elements[j]["position"]["x"] = x
                        self.elements[j]["position"]["y"] = y
        if changed:
            self.el_history.append(self.elements)

        return data, changed

    def register_callbacks(self, dataStore: tuple[str, str] = ("data-store", "data")):
        @callback(
            Output("dummy3", "children"),
            Input("pca-graph", "dragNode"),
            prevent_initial_call=True,
        )
        def save_new_node_pos(node):
            if node is None:
                return ""
            x = node["position"]["x"]
            y = node["position"]["y"]
            print("posies")
            print(x, y)
            self.new_positions[node["data"]["path"]] = np.array([x, y])

            return ""

        @callback(
            # Output("history_slider", "max"),
            # Output("history_slider", "value"),
            Output(dataStore[0], dataStore[1], allow_duplicate=True),
            Input("submit-val", "n_clicks"),
            State(dataStore[0], dataStore[1]),
            State("current-filter-store", "data"),
            # State("history_slider", "value"),
            # State("history_slider", "max"),
            prevent_initial_call=True,
        )
        def update_slider(_, data: pd.DataFrame, current_filter: str) -> pd.DataFrame:
            """If history depth is changed, change slider to match."""
            data = pd.DataFrame(data)
            print(data)
            if not self.el_history:
                return data.to_dict("records")

            data, changed = self.update_images(data, current_filter)
            print("Updating Slider")
            # new_max = value + 1
            #
            # if len(self.el_history) is not new_max:
            #     idcs_to_remove = range(value, len(self.el_history) - 2)
            #     print(idcs_to_remove)
            #     # Remove overwritten history in el_history and remove the checkpoints.
            #     self.el_history = [
            #         els
            #         for i, els in enumerate(self.el_history)
            #         if i not in idcs_to_remove
            #     ]
            #     for i in idcs_to_remove:
            #         old_chkpt = self.checkpoint_dir / f"{i}.pth"
            #         old_chkpt.unlink()

            return data.to_dict("records")

        @callback(
            Output("pca-graph", "elements", allow_duplicate=True),
            Input("history_slider", "value"),
            prevent_initial_call=True,
        )
        def set_elements_from_history(step: int):
            elements = self.el_history[step]
            return elements

    def get_graph(self, data: pd.DataFrame):
        self.generate_thumbnails(data["filename"])
        self.set_elements(data)
        for el in self.elements:
            print(el["position"])
        self.create_graph()

        return html.Div(self.cyto)
