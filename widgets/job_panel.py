from dash import html, callback, Output, Input, State, dcc
from utils.jobmanager import JobManager
from utils.util import load_data
from datetime import datetime


class JobPanel:
    def __init__(self, manager: JobManager) -> None:
        self.panel = self.create_job_control_panel()
        self.filter_disp = html.Div(id="current-filter-display")
        self.manager = manager

        self.filter_store = dcc.Store(
            id="current-filter-store", data=manager.get_current_filter()
        )

    def create_job_control_panel(self):
        return html.Div(
            [
                html.H3(
                    "🚀 Keyword Control Panel",
                    style={"color": "#1f2937", "margin-bottom": "20px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Input Keyword:",
                                    style={
                                        "font-weight": "bold",
                                        "display": "block",
                                        "margin-bottom": "10px",
                                    },
                                ),
                                dcc.Input(
                                    id="filter-input",
                                    type="text",
                                    value="sunny day",
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "border-radius": "8px",
                                        "border": "2px solid #e5e7eb",
                                    },
                                ),
                            ],
                            style={"margin-bottom": "20px"},
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "▶️ Run Classification",
                                    id="run-job-btn",
                                    style={
                                        "background": "#3b82f6",
                                        "color": "white",
                                        "padding": "12px 24px",
                                        "border-radius": "8px",
                                        "border": "none",
                                        "font-size": "16px",
                                        "cursor": "pointer",
                                        "margin-right": "10px",
                                    },
                                ),
                                html.Button(
                                    "🔄 Refresh Data",
                                    id="refresh-btn",
                                    style={
                                        "background": "#10b981",
                                        "color": "white",
                                        "padding": "12px 24px",
                                        "border-radius": "8px",
                                        "border": "none",
                                        "font-size": "16px",
                                        "cursor": "pointer",
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={
                        "background": "white",
                        "padding": "25px",
                        "border-radius": "12px",
                        "box-shadow": "0 4px 6px rgba(0,0,0,0.1)",
                        "margin-bottom": "30px",
                    },
                ),
                html.Div(id="job-status", style={"margin-top": "20px"}),
                html.Div(id="job-output", style={"margin-top": "20px"}),
            ]
        )

    def create_current_filter_display(self, current_filter):
        """Create a display showing the current filter"""
        if not current_filter or current_filter == "No filter set":
            return html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                "🔍 Current Keyword",
                                style={"margin-bottom": "10px", "color": "#374151"},
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        "⚠️ No filter currently set",
                                        style={
                                            "background": "#fef3c7",
                                            "color": "#92400e",
                                            "padding": "8px 16px",
                                            "border-radius": "20px",
                                            "font-size": "14px",
                                            "font-weight": "bold",
                                        },
                                    )
                                ]
                            ),
                        ],
                        style={
                            "background": "white",
                            "padding": "20px",
                            "border-radius": "12px",
                            "box-shadow": "0 4px 6px rgba(0,0,0,0.1)",
                            "margin-bottom": "20px",
                        },
                    )
                ]
            )
        else:
            return html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                "🔍 Current Keyword",
                                style={"margin-bottom": "10px", "color": "#374151"},
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        f'"{current_filter}"',
                                        style={
                                            "background": "#dcfce7",
                                            "color": "#166534",
                                            "padding": "8px 16px",
                                            "border-radius": "20px",
                                            "font-size": "14px",
                                            "font-weight": "bold",
                                            "font-style": "italic",
                                        },
                                    )
                                ]
                            ),
                        ],
                        style={
                            "background": "white",
                            "padding": "20px",
                            "border-radius": "12px",
                            "box-shadow": "0 4px 6px rgba(0,0,0,0.1)",
                            "margin-bottom": "20px",
                        },
                    )
                ]
            )

    def register_callbacks(
        self,
        dataStore: tuple[str, str] = ("data-store", "data"),
        prob_file: str = "step2a_result.txt",
        dataPath: str = "/home/scur0274/Wouter_repo/ClusterPhotos/Text_clustering/data/stanford-40-actions/JPEGImages",
    ):
        @callback(
            [Output("job-status", "children"), Output("current-filter-store", "data")],
            Input("run-job-btn", "n_clicks"),
            State("filter-input", "value"),
            prevent_initial_call=True,
        )
        def run_job(n_clicks, filter_value):
            success, message = self.manager.submit_job(filter_value)
            if success:
                return (
                    html.Div(
                        [
                            html.P(
                                f"✅ {message}",
                                style={"color": "#059669", "font-weight": "bold"},
                            ),
                            html.P(
                                f"Filter: '{filter_value}'",
                                style={"color": "#666", "font-style": "italic"},
                            ),
                        ]
                    ),
                    filter_value,
                )
            else:
                return (
                    html.Div(
                        [
                            html.P(
                                f"❌ {message}",
                                style={"color": "#dc2626", "font-weight": "bold"},
                            )
                        ]
                    ),
                    self.manager.get_current_filter(),
                )

        @callback(
            Output("current-filter-display", "children"),
            Input("current-filter-store", "data"),
        )
        def update_filter_display(current_filter):
            return self.create_current_filter_display(current_filter)

        @callback(
            [
                Output(dataStore[0], dataStore[1], allow_duplicate=True),
                Output("current-filter-store", "data", allow_duplicate=True),
            ],
            Input("refresh-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def reload_data(n_clicks):
            new_df = load_data(prob_file, dataPath)
            current_filter = self.manager.get_current_filter()
            return (
                new_df.to_dict("records") if not new_df.empty else [],
                current_filter,
            )

        @callback(
            Output("job-output", "children"),
            [
                Input("interval-component", "n_intervals"),
                Input("refresh-btn", "n_clicks"),
            ],
            prevent_initial_call=True,
        )
        def refresh_job_panel(n_intervals, n_clicks):
            status = self.manager.check_job_status()
            return html.Div(
                [
                    html.P(
                        f"Job Status: {status.upper()}",
                        style={
                            "font-weight": "bold",
                            "color": (
                                "#059669"
                                if status == "completed"
                                else "#f59e0b"
                                if status == "running"
                                else "#6b7280"
                            ),
                        },
                    ),
                    html.P(
                        f"Last updated: {datetime.now().strftime('%H:%M:%S')}",
                        style={"color": "#666", "font-size": "12px"},
                    ),
                ]
            )

    def get_widget(self):
        """Returns widgets in a side-by-side layout"""
        # Create the side-by-side container
        side_by_side_container = html.Div(
            [
                # Left column - Job Control Panel
                html.Div(
                    [self.panel],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "vertical-align": "top",
                        "margin-right": "2%",
                    }
                ),
                # Right column - Current Filter Display
                html.Div(
                    [self.filter_disp],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "vertical-align": "top",
                        "margin-left": "1%",
                        "margin-top": "3.4%",
                    }
                ),
            ],
            style={
                "width": "100%",
                "display": "block",
            }
        )
        
        # Return the container and an empty div for backward compatibility
        # This avoids duplicate IDs since filter_disp is already in the container
        return side_by_side_container, html.Div()

    def get_store(self):
        return self.filter_store