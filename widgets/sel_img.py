from dash import html, callback, Output, Input, State, dcc
import dash
import pandas as pd
import json
import os
import re
from utils.util import CLUSTER_COLORS, modify_image_description, CUSTOM_COLORS


class SelectedImage:
    def __init__(self) -> None:
        self.selImg = html.Div(id="selected-image-area")
        self.descriptions_df = self._load_descriptions()
        pass

    def _load_descriptions(self):
        """Load image descriptions from the JSONL file"""
        jsonl_path = "./diff_descs.jsonl"

        try:
            if os.path.exists(jsonl_path):
                # Read JSONL file using pandas
                descriptions_df = pd.read_json(jsonl_path, lines=True)
                return descriptions_df
            else:
                print(f"Warning: JSONL file not found at {jsonl_path}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading descriptions: {e}")
            return pd.DataFrame()

    def _get_image_description(self, filename):
        """Get description for a specific image filename"""
        if self.descriptions_df.empty:
            return "No description available"

        # Find matching description by image_file
        matching_desc = self.descriptions_df[
            self.descriptions_df["image_file"] == filename
        ]
        if not matching_desc.empty:
            return matching_desc.iloc[0]["text"]
        else:
            return "Description not found for this image"

    def _parse_and_style_diff(self, diff_string: str) -> list:
        """
        Parses the diff string with <removed> and <added> tags
        and returns a list of Dash HTML components with styling.
        """
        elements = []
        # Split by <removed> and <added> tags, keeping the delimiters
        parts = re.split(
            r"(<removed>.*?</removed>|<added>.*?</added>)", diff_string, flags=re.DOTALL
        )

        for part in parts:
            if part.startswith("<removed>") and part.endswith("</removed>"):
                text = part[len("<removed>") : -len("</removed>")]
                # Split by newlines to apply strikethrough to each line
                for line in text.splitlines():
                    elements.append(
                        html.Span(
                            line,
                            style={
                                "color": "red",
                                "text-decoration": "line-through",
                                "display": "inline",
                                "white-space": "pre-wrap",  # Ensure each line takes its own space
                            },
                            className="removed-text",
                        )
                    )
            elif part.startswith("<added>") and part.endswith("</added>"):
                text = part[len("<added>") : -len("</added>")]
                # Split by newlines to apply green to each line
                for line in text.splitlines():
                    elements.append(
                        html.Span(
                            line,
                            style={
                                "color": "green",
                                "display": "inline",
                                "white-space": "pre-wrap",  # Ensure each line takes its own space
                            },
                            className="added-text",
                        )
                    )
            else:
                # Regular text, split by newlines for proper rendering
                for line in part.splitlines():
                    elements.append(html.Span(line, style={"display": "inline"}))
        return elements

    def register_callbacks(
        self, dataStore: tuple[str, str] = ("data-store", "data")
    ) -> None:
        @callback(
            Output("selected-image-area", "children"),
            Input(dataStore[0], dataStore[1]),
            prevent_initial_call=False,
        )
        def init_sel_image(data):
            return html.Div(
                "💡 Click on an image in the plot above or an image in the cluster tabs below to view and modify",
                style={
                    "text-align": "center",
                    "font-style": "italic",
                    "color": "#666",
                    "padding": "20px",
                    "background": "white",
                    "border-radius": "8px",
                    "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
                },
            )

        @callback(
            Output("selected-image-store", "data"),
            [
                Input("pca-graph", "tapNodeData"),
                Input(
                    {"type": "cluster-image", "index": dash.dependencies.ALL},
                    "n_clicks",
                ),
            ],
            State(dataStore[0], dataStore[1]),
            prevent_initial_call=True,
        )
        def sel_image_store(tap_node_data, cluster_clicks, data):
            ctx = dash.callback_context
            if not ctx.triggered:
                return None

            trigger_id = ctx.triggered[0]["prop_id"]
            print(trigger_id)

            # Handle scatter plot clicks
            # Handle Cytoscape node clicks
            if "pca-graph" in trigger_id and tap_node_data:
                # You can use any field from the node's `data` dict
                # e.g., label or path
                return tap_node_data.get(
                    "pl_idx"
                )  # or "path" or any identifier you use

            # Handle cluster image clicks
            if "cluster-image" in trigger_id:
                # Extract the index from the triggered component
                import json

                print(trigger_id.split(".")[0])
                trigger_info = json.loads(trigger_id.split(".")[0])
                plot_index = trigger_info["index"]
                return plot_index

            return None

        @callback(
            Output("selected-image-area", "children", allow_duplicate=True),
            Input("selected-image-store", "data"),
            State(dataStore[0], dataStore[1]),
            prevent_initial_call=True,
        )
        def display_sel_image(plot_idx, data):
            if plot_idx is None or not data:
                return init_sel_image(data)

            df = pd.DataFrame(data)
            # Find the row with the matching plot_index
            matching_rows = df[df["plot_index"] == plot_idx]
            if matching_rows.empty:
                return init_sel_image(data)

            row = matching_rows.iloc[0]

            # Determine display label for the cluster
            raw_cluster = row["cluster"]
            display_cluster = {
                "Absolute probability": "Definite Positive",
                "Zero probability": "Definite Negative",
                "In between": "Uncertain",
            }.get(raw_cluster, raw_cluster)

            # Get the image description
            image_description = self._get_image_description(row["filename"])
            styled_description_elements = self._parse_and_style_diff(image_description)

            return html.Div(
                [
                    html.H3(
                        "🖼️ Selected Image",
                        style={"color": "#1f2937", "margin-bottom": "20px"},
                    ),
                    # Two column layout
                    html.Div(
                        [
                            # Left column - Image and info
                            html.Div(
                                [
                                    # Image
                                    html.Img(
                                        src=row["uri"],
                                        style={
                                            "max-width": "100%",
                                            "border-radius": "12px",
                                            "box-shadow": "0 4px 12px rgba(0,0,0,0.15)",
                                        },
                                    ),
                                    # Image information below the image
                                    html.Div(
                                        [
                                            html.P(
                                                [
                                                    html.Strong("Filename: "),
                                                    row["filename"],
                                                ]
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Class: "),
                                                    html.Span(
                                                        display_cluster,
                                                        style={
                                                            "color": CUSTOM_COLORS.get(
                                                                display_cluster, "#666"
                                                            ),
                                                            "font-weight": "bold",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Probability: "),
                                                    f"{row['prob']:.2f}",
                                                ]
                                            ),
                                        ],
                                        style={
                                            "margin-top": "15px",
                                            "padding": "15px",
                                            "background": "#f8fafc",
                                            "border-radius": "8px",
                                            "border": "1px solid #e2e8f0",
                                        },
                                    ),
                                ],
                                style={
                                    "width": "48%",
                                    "display": "inline-block",
                                    "vertical-align": "top",
                                },
                            ),
                            # Right column - Description sections
                            html.Div(
                                [
                                    # Current Image Description Section
                                    html.Div(
                                        [
                                            html.H4(
                                                "📝 Current Image Description",
                                                style={
                                                    "color": "#1f2937",
                                                    "margin-bottom": "15px",
                                                },
                                            ),
                                            html.Div(
                                                styled_description_elements,
                                                style={
                                                    "background": "#f8fafc",
                                                    "padding": "15px",
                                                    "border-radius": "8px",
                                                    "border": "1px solid #e2e8f0",
                                                    "line-height": "1.6",
                                                    "color": "#374151",
                                                    "font-size": "14px",
                                                    "max-height": "200px",
                                                    "overflow-y": "auto",
                                                    "white-space": "pre-wrap",
                                                },
                                            ),
                                        ],
                                        style={
                                            "margin-bottom": "25px",
                                        },
                                    ),
                                    # Description modification section
                                    html.Div(
                                        [
                                            html.H4(
                                                "✏️ Modify Image Description",
                                                style={
                                                    "color": "#1f2937",
                                                    "margin-bottom": "15px",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Textarea(
                                                        id="description-input",
                                                        placeholder='Enter new description for this image (e.g., "there are only 7 people in the boat")',
                                                        style={
                                                            "width": "100%",
                                                            "height": "80px",
                                                            "padding": "12px",
                                                            "border-radius": "8px",
                                                            "border": "2px solid #e5e7eb",
                                                            "font-family": "system-ui, -apple-system",
                                                            "font-size": "14px",
                                                            "resize": "vertical",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Button(
                                                                "🔄 Modify Description",
                                                                id="modify-description-btn",
                                                                style={
                                                                    "background": "#8b5cf6",
                                                                    "color": "white",
                                                                    "padding": "10px 20px",
                                                                    "border-radius": "8px",
                                                                    "border": "none",
                                                                    "font-size": "14px",
                                                                    "cursor": "pointer",
                                                                    "margin-top": "10px",
                                                                },
                                                            )
                                                        ]
                                                    ),
                                                    html.Div(
                                                        id="modification-result",
                                                        style={"margin-top": "15px"},
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style={
                                            "background": "#f8fafc",
                                            "padding": "20px",
                                            "border-radius": "8px",
                                            "border": "1px solid #e2e8f0",
                                        },
                                    ),
                                ],
                                style={
                                    "width": "48%",
                                    "display": "inline-block",
                                    "vertical-align": "top",
                                    "margin-left": "4%",
                                },
                            ),
                        ],
                        style={
                            "width": "100%",
                            "display": "flex",
                            "justify-content": "space-between",
                        },
                    ),
                ],
                style={
                    "background": "white",
                    "padding": "25px",
                    "border-radius": "12px",
                    "box-shadow": "0 4px 6px rgba(0,0,0,0.1)",
                },
            )

        @callback(
            Output("modification-result", "children"),
            Input("modify-description-btn", "n_clicks"),
            [
                State("selected-image-store", "data"),
                State(dataStore[0], dataStore[1]),
                State("description-input", "value"),
            ],
            prevent_initial_call=True,
        )
        def modify_description(n_clicks, plot_idx, data, description):
            if not n_clicks or plot_idx is None or not data or not description:
                return html.Div()

            df = pd.DataFrame(data)
            matching_rows = df[df["plot_index"] == plot_idx]
            if matching_rows.empty:
                return html.Div(
                    "❌ Error: Image not found",
                    style={"color": "#dc2626", "font-weight": "bold"},
                )

            filename = matching_rows.iloc[0]["filename"]
            success, message = modify_image_description(filename, description)

            if success:
                return html.Div(
                    [
                        html.P(
                            "✅ " + message,
                            style={
                                "color": "#059669",
                                "font-weight": "bold",
                                "margin-bottom": "5px",
                            },
                        ),
                        html.P(
                            f'Command executed: python modify_description.py "{filename}" "{description}"',
                            style={
                                "color": "#666",
                                "font-size": "12px",
                                "font-family": "monospace",
                                "background": "#f1f5f9",
                                "padding": "8px",
                                "border-radius": "4px",
                            },
                        ),
                    ]
                )
            else:
                return html.Div(
                    [
                        html.P(
                            "❌ " + message,
                            style={"color": "#dc2626", "font-weight": "bold"},
                        )
                    ]
                )

    def get_widget(self):
        return self.selImg

