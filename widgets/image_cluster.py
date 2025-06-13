import plotly.graph_objs as go
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import base64
import os
from tqdm import tqdm
from pathlib import Path
from dash import dcc, Output, Input, State, callback, clientside_callback, html
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
        size=(1920, 800),
    ):
        self.image_folder = img_folder
        self.graph_id = graph_id
        self.si = imageSi
        self.size = size

        # Make data folder if it doesn't exist yet, this saves us time so we don't have to load all images from disk each time
        if not data_folder.exists():
            data_folder.mkdir()

        npy_path = data_folder / "data.npy.npz"
        if npy_path.exists():
            data = np.load(npy_path)
            self.labels = data["labels"]
            self.coords = data["coords"]
            self.paths = data["paths"]
            self.filenames = data["filenames"]
            self.timestamps = data["timestamps"]
        else:
            self.load_and_cluster_images(npy_path)
        print("X range:", np.min(self.coords[:, 0]), np.max(self.coords[:, 0]))
        print("Y range:", np.min(self.coords[:, 1]), np.max(self.coords[:, 1]))

        self.thumbnail_dir = data_folder / "thumbnails"
        self.thumbnail_dir.mkdir(exist_ok=True)

        self.set_elements()

        # Pre-generate thumbnails
        self.generate_thumbnails()
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
            style={"width": f"{size[0]}px", "height": f"{size[1]}px"},
            stylesheet=stylesheet,
            layout={"name": "preset"},
            userZoomingEnabled=True,
            userPanningEnabled=True,
            boxSelectionEnabled=True,
            autoungrabify=False,
        )

        self.graph = html.Div(
            [
                self.cyto,
                html.Div(id="output", style={"display": "none"}),
                html.Div(id="dummy", style={"display": "none"}),
                dcc.Interval(id="interval", interval=1000, max_intervals=0),
                html.Button("Submit", id="submit-val", n_clicks=0),
                html.Div(id="dummy2", style={"display": "none"}),
                dcc.Store(id="node-positions"),
                html.Div(id="dummy3", style={"display": "none"}),
            ]
        )

        self.new_positions: dict[str, np.ndarray] = {}

    def generate_thumbnails(self, size=(100, 100)):
        """Generate and save thumbnails for all images"""
        for path in tqdm(self.paths, desc="Generating thumbnails"):
            thumbnail_path = self.thumbnail_dir / f"{Path(path).stem}_thumb.jpg"
            if not thumbnail_path.exists():
                try:
                    with Image.open(path) as img:
                        img.thumbnail(size)
                        img.save(thumbnail_path, "JPEG", quality=85)
                except Exception as e:
                    print(f"Error generating thumbnail for {path}: {e}")

    @lru_cache(maxsize=200)
    def get_thumbnail_base64(self, path) -> tuple[str, tuple[int, int]]:
        """Get cached base64 thumbnail"""
        thumbnail_path = self.thumbnail_dir / f"{Path(path).stem}_thumb.jpg"
        try:
            with Image.open(thumbnail_path) as img:
                b64 = self.pil_image_to_base64(img)
                return b64, img.size
        except Exception as e:
            print(f"Error loading thumbnail for {path}: {e}")
            return None

    def get_image_timestamp(self, path):
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
        # print(x_coords)
        y_coords: np.ndarray = self.si.projection[:, 1]
        self.min_x = np.min(x_coords)
        self.max_x = np.max(x_coords) - self.min_x
        self.min_y = np.min(y_coords)
        self.max_y = np.max(y_coords) - self.min_y
        self.elements = []
        for i, (path, dims) in enumerate(
            zip(self.si.image_paths, zip(x_coords.tolist(), y_coords.tolist()))
        ):
            thumb, imgdims = self.get_thumbnail_base64(path)
            x, y = dims
            x_normalised = ((float(x) - self.min_x) / self.max_x) * self.size[0]
            y_normalised = ((float(y) - self.min_y) / self.max_y) * self.size[1]
            # print(f"{x} | {x_normalised}")
            # print(f"{y} | {y_normalised}")
            element = {
                "data": {"id": f"node{i}", "label": f"{i}", "bg": thumb, "path": path},
                "position": {"x": x_normalised, "y": y_normalised},
                "grabbable": True,
                # 'style': {
                #     'background-image': f'url("{thumb}")',
                #     'background-fit': 'cover',
                #     'background-zoom': 'false',
                #     'width': f'{dims[0]}px',
                #     'height': f'{dims[1]}px',
                #     'shape': 'rectangle'
                # }
            }
            self.elements.append(element)

    def load_and_cluster_images(self, data_path: Path, size=(32, 32), n_clusters=3):
        features, self.paths, self.filenames, self.timestamps = [], [], [], []

        for filename in tqdm(os.listdir(self.image_folder), desc="Loading files"):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(self.image_folder, filename)
                img = Image.open(path).convert("RGB").resize(size)
                features.append(np.array(img).flatten())
                self.paths.append(path)
                self.filenames.append(filename)

                timestamp_str = self.get_image_timestamp(path)
                if timestamp_str:
                    try:
                        # Convert EXIF date string to a sortable number (e.g., timestamp)
                        dt = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")
                        self.timestamps.append(dt.timestamp())
                    except:
                        self.timestamps.append(0)
                else:
                    self.timestamps.append(0)

        features = np.array(features)
        self.labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(
            features
        )
        y_coords = PCA(n_components=1).fit_transform(features)
        self.coords = np.hstack([np.array(self.timestamps).reshape(-1, 1), y_coords])

        np.savez(
            data_path,
            labels=self.labels,
            coords=self.coords,
            paths=self.paths,
            filenames=self.filenames,
            timestamps=self.timestamps,
        )
        # self.encoded_imgs = [self.encode_image(p) for p in self.paths]

    def pil_image_to_base64(self, img) -> str:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{encoded}"

    def register_callback(self):
        @callback(Output("dummy3", "children"), Input("pca-graph", "dragNode"))
        def print_node(node):
            print(node["position"])
            print(node["data"]["id"])
            x = (node["position"]["x"] / self.size[0]) * self.max_x + self.min_x
            y = (node["position"]["y"] / self.size[1]) * self.max_y + self.min_y
            self.new_positions[node["data"]["path"]] = np.array([x, y])

            return "kaas"

        @callback(
            Output("output", "children"),
            Input("node-positions", "data"),
            prevent_initial_call=True,
        )
        def update_positions(elements):
            # print(elements)
            # print(self.cyto.mouseoverNodeData())
            # for i, (e, e_new) in enumerate(zip(self.elements, elements)):
            #    # print(e["position"], end=" | ")
            #    or_pos = e["position"]
            #    new_pos = e_new["position"]
            #    x_or = or_pos["x"]
            #    y_or = or_pos["y"]

            #   x_new = new_pos["x"]
            #    y_new = new_pos["y"]
            #    # print(e_new["position"])
            #    if float(x_or) is not float(x_new) or float(y_or) is not float(y_new):
            #        pos = new_pos
            #        print(f"{float(x_or)}, {float(y_or)}", end=" | ")
            #        print(f"{float(x_new)}, {float(y_new)}", end=" | ")
            #        print(
            #        f"{float(x_new) == float(x_or)}, {float(y_new) == float(y_or)}"
            #   )
            #       x = (pos["x"] / self.size[0]) * self.max_x + self.min_x
            #       y = (pos["y"] / self.size[1]) * self.max_y + self.min_y
            #       self.new_positions[self.si.image_paths[i]] = np.array([x, y])
            # positions = {
            #    el["data"]["id"]: el["position"] for el in elements if "position" in el
            # }

            # self.elements = elements
            return "kaas3"

        @callback(
            Output("pca-graph", "elements"),
            Input("submit-val", "n_clicks"),
            prevent_initial_call=True,
        )
        def submit_change(_) -> list:
            # print(self.new_positions)
            self.si.update_model(updated_positions=self.new_positions, epochs=5)
            self.new_positions = {}
            # print(self.new_positions)
            self.set_elements()
            return self.elements

        clientside_callback(
            """
        function(n_intervals) {
            const cy = document.querySelector('#pca-graph')._cyreg.cy;
            if (!cy || cy._positionListenerRegistered) {
                return window.dash_clientside.no_update;
            }

            cy.on('dragfree', 'node', () => {
                const updated = cy.nodes().map(n => ({
                    data: { id: n.id() },
                    position: n.position()
                }));
                window.dash_clientside._lastDraggedPositions = updated;
                window.dispatchEvent(new CustomEvent("nodes-updated", { detail: updated }));
            });

            cy._positionListenerRegistered = true;
            return '';
        }
        """,
            Output("dummy2", "children"),
            Input("interval", "n_intervals"),
        )

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
                let lastZoom = null;
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

        @callback(
            Output("node-positions", "data"),
            Input("pca-graph", "elements"),
            State("node-positions", "data"),
        )
        def sync_positions(elements, current_data):
            updated_positions = {
                el["data"]["id"]: el["position"] for el in elements if "position" in el
            }
            return updated_positions

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
