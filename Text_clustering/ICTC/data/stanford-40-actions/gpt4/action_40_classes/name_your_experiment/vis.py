import os
import json
import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64
import subprocess
import glob
from datetime import datetime

home_path = os.getenv('HOME_PATH')

class JobManager:
    def __init__(self,
                 job_file_path='/home/scur0274/Wouter_repo/ClusterPhotos/Text_clustering/ICTC/step2a.job',
                 output_dir='outputs_train'):
        self.job_file_path = job_file_path
        self.output_dir = output_dir
        self.current_job_id = None
        self.job_status = "idle"
        self.current_filter = None  # Track current filter
        
    def get_current_filter(self):
        """Extract current filter from job file"""
        try:
            with open(self.job_file_path, 'r') as f:
                content = f.read()
            lines = content.split('\n')
            for line in lines:
                if line.strip().startswith('--filter'):
                    # Extract filter value from the line
                    filter_part = line.split('--filter')[1].strip()
                    # Remove quotes if present
                    if filter_part.startswith('"') and filter_part.endswith('"'):
                        filter_part = filter_part[1:-1]
                    elif filter_part.startswith("'") and filter_part.endswith("'"):
                        filter_part = filter_part[1:-1]
                    return filter_part
            return "No filter set"
        except Exception as e:
            print(f"Error reading current filter: {e}")
            return "Unknown"
        
    def update_job_file(self, filter_value):
        try:
            with open(self.job_file_path, 'r') as f:
                content = f.read()
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('--filter'):
                    lines[i] = f'  --filter "{filter_value}"'
                    break
            else:
                for i in range(len(lines)-1, -1, -1):
                    if lines[i].strip() and not lines[i].strip().startswith('#'):
                        lines.insert(i+1, f'  --filter "{filter_value}"')
                        break
            with open(self.job_file_path, 'w') as f:
                f.write('\n'.join(lines))
            self.current_filter = filter_value  # Update tracked filter
            return True
        except Exception as e:
            print(f"Error updating job file: {e}")
            return False
    
    def submit_job(self, filter_value):
        if not self.update_job_file(filter_value):
            return False, "Failed to update job file"
        result = subprocess.run(['sbatch', self.job_file_path],
                                capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if 'Submitted batch job' in line:
                    self.current_job_id = line.split()[-1]
                    break
            self.job_status = "running"
            return True, f"Job submitted successfully. Job ID: {self.current_job_id}"
        else:
            return False, f"Failed to submit job: {result.stderr}"
    
    def check_job_status(self):
        if not self.current_job_id:
            return "idle"
        try:
            result = subprocess.run(['squeue','-j',self.current_job_id],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    return "running"
                else:
                    self.job_status = "completed"
                    return "completed"
            else:
                return "unknown"
        except Exception as e:
            print(f"Error checking job status: {e}")
            return "unknown"

def load_data():
    PROB_FILE = 'step2a_result.txt'
    if not os.path.exists(PROB_FILE):
        return pd.DataFrame()
    
    image_files, probs = [], []
    with open(PROB_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                file_part, json_part = line.split(';', 1)
                name_raw = file_part.split(' ',1)[1].strip()
                filename = name_raw[5:] if name_raw.startswith('file-') else name_raw
                data = json.loads(json_part)
                prob = float(data.get('entailment_probability', 0.0))
            except:
                continue
            image_files.append(filename)
            probs.append(prob)
    
    df = pd.DataFrame({'filename': image_files, 'prob': probs})
    def assign_cluster(p):
        if p == 0.0:   return 'Zero probability'
        if p == 1.0:   return 'Absolute probability'
        return 'In between'
    df['cluster'] = df['prob'].apply(assign_cluster)
    df['confidence_level'] = df['prob'].apply(
        lambda x: 'High' if x >= 0.7 else ('Medium' if x > 0.3 else 'Low')
    )
    df['image_id'] = range(len(df))
    
    IMAGE_FOLDER = '/home/scur0274/Wouter_repo/ClusterPhotos/Text_clustering/data/stanford-40-actions/JPEGImages'
    def encode_image(path):
        try:
            if not os.path.isfile(path): return ''
            with open(path,'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{b64}"
        except:
            return ''
    df['uri'] = df['filename'].apply(
        lambda fn: encode_image(os.path.join(IMAGE_FOLDER, fn))
    )
    return df[df['uri']!=''].reset_index(drop=True)

def main():
    job_manager = JobManager()
    df = load_data()
    
    cluster_colors = {
        'Absolute probability': '#2E8B57',
        'In between':           '#FF8C00',
        'Zero probability':     '#DC143C'
    }
    
    def create_current_filter_display(current_filter):
        """Create a display showing the current filter"""
        if not current_filter or current_filter == "No filter set":
            return html.Div([
                html.Div([
                    html.H4('🔍 Current Filter', style={'margin-bottom': '10px', 'color': '#374151'}),
                    html.Div([
                        html.Span('⚠️ No filter currently set', style={
                            'background': '#fef3c7', 'color': '#92400e', 'padding': '8px 16px',
                            'border-radius': '20px', 'font-size': '14px', 'font-weight': 'bold'
                        })
                    ])
                ], style={
                    'background': 'white', 'padding': '20px', 'border-radius': '12px',
                    'box-shadow': '0 4px 6px rgba(0,0,0,0.1)', 'margin-bottom': '20px'
                })
            ])
        else:
            return html.Div([
                html.Div([
                    html.H4('🔍 Current Filter', style={'margin-bottom': '10px', 'color': '#374151'}),
                    html.Div([
                        html.Span(f'"{current_filter}"', style={
                            'background': '#dcfce7', 'color': '#166534', 'padding': '8px 16px',
                            'border-radius': '20px', 'font-size': '14px', 'font-weight': 'bold',
                            'font-style': 'italic'
                        })
                    ])
                ], style={
                    'background': 'white', 'padding': '20px', 'border-radius': '12px',
                    'box-shadow': '0 4px 6px rgba(0,0,0,0.1)', 'margin-bottom': '20px'
                })
            ])
    
    def create_scatter_plot(data):
        if data.empty:
            return px.scatter(title="No data available - Run a job first!")
        fig = px.scatter(
            data, x='image_id', y='prob', color='cluster',
            size=[15]*len(data),
            color_discrete_map=cluster_colors,
            title='📊 Probability Distribution Across Images',
            labels={'image_id':'Image Index','prob':'Entailment Probability'},
            custom_data=['filename','cluster']
        )
        fig.update_traces(
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Cluster: %{customdata[1]}<br>'
                'Probability: %{y:.2f}<br>'
                '<i>Click to view image</i><extra></extra>'
            )
        )
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
        )
        return fig
    
    def create_pie_chart(data):
        if data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        counts = data['cluster'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.4,
            marker_colors=[cluster_colors[l] for l in counts.index],
            textinfo='label+percent+value',
            textfont_size=12
        )])
        fig.update_layout(
            title='🥧 Cluster Distribution',
            height=400,
            annotations=[dict(text='Image<br>Clusters', x=0.5, y=0.5, showarrow=False, font_size=16)]
        )
        return fig
    
    def create_image_grid(data, cluster_name, max_images=12):
        subset = data[data['cluster']==cluster_name].head(max_images)
        if subset.empty:
            return html.Div("No images in this cluster", className="text-center text-gray-500")
        tiles = []
        for _, row in subset.iterrows():
            tiles.append(html.Div([
                html.Img(src=row['uri'], style={
                    'width':'120px','height':'120px','object-fit':'cover',
                    'border-radius':'8px','box-shadow':'0 2px 8px rgba(0,0,0,0.1)'
                }),
                html.P(
                    row['filename'][:15]+'...' if len(row['filename'])>15 else row['filename'],
                    style={'font-size':'10px','margin':'5px 0','text-align':'center','color':'#666'}
                ),
                html.P(
                    f"p={row['prob']:.2f}",
                    style={'font-size':'12px','font-weight':'bold','text-align':'center',
                           'color': cluster_colors[cluster_name]}
                )
            ], style={'margin':'10px','text-align':'center'}))
        return html.Div(
            tiles,
            style={
                'display':'grid',
                'grid-template-columns':'repeat(auto-fill,minmax(140px,1fr))',
                'gap':'15px','padding':'20px'
            }
        )
    
    def create_job_control_panel():
        return html.Div([
            html.H3('🚀 Job Control Panel', style={'color':'#1f2937','margin-bottom':'20px'}),
            html.Div([
                html.Div([
                    html.Label('Filter Value:', style={'font-weight':'bold','display':'block','margin-bottom':'10px'}),
                    dcc.Input(
                        id='filter-input', type='text', value='sunny day',
                        style={'width':'100%','padding':'10px','border-radius':'8px','border':'2px solid #e5e7eb'}
                    )
                ], style={'margin-bottom':'20px'}),
                html.Div([
                    html.Button('▶️ Run Job', id='run-job-btn', style={
                        'background':'#3b82f6','color':'white','padding':'12px 24px',
                        'border-radius':'8px','border':'none','font-size':'16px','cursor':'pointer',
                        'margin-right':'10px'
                    }),
                    html.Button('🔄 Refresh Data', id='refresh-btn', style={
                        'background':'#10b981','color':'white','padding':'12px 24px',
                        'border-radius':'8px','border':'none','font-size':'16px','cursor':'pointer'
                    })
                ])
            ], style={'background':'white','padding':'25px','border-radius':'12px','box-shadow':'0 4px 6px rgba(0,0,0,0.1)','margin-bottom':'30px'}),
            html.Div(id='job-status', style={'margin-top':'20px'}),
            html.Div(id='job-output', style={'margin-top':'20px'})
        ])
    
    def create_stats_cards(data):
        if data.empty:
            return html.Div([
                html.Div([
                    html.H3("0", style={'font-size':'2rem','color':'#6b7280'}),
                    html.P('No Data Available', style={'color':'#666'})
                ], style={'background':'white','padding':'20px','border-radius':'8px','box-shadow':'0 2px 4px rgba(0,0,0,0.1)','text-align':'center'})
            ])
        total = len(data)
        avg   = data['prob'].mean()
        high  = len(data[data['prob']>=0.7])
        return html.Div([
            html.Div([html.H3(str(total), style={'color':'#2563eb'}),html.P('Total Images', style={'color':'#666'})],
                     style={'background':'white','padding':'20px','border-radius':'8px','box-shadow':'0 2px 4px rgba(0,0,0,0.1)','text-align':'center'}),
            html.Div([html.H3(f"{avg:.2f}", style={'color':'#059669'}),html.P('Average Probability', style={'color':'#666'})],
                     style={'background':'white','padding':'20px','border-radius':'8px','box-shadow':'0 2px 4px rgba(0,0,0,0.1)','text-align':'center'}),
            html.Div([html.H3(str(high), style={'color':'#dc2626'}),html.P('High Confidence (≥0.7)', style={'color':'#666'})],
                     style={'background':'white','padding':'20px','border-radius':'8px','box-shadow':'0 2px 4px rgba(0,0,0,0.1)','text-align':'center'})
        ], style={'display':'grid','grid-template-columns':'repeat(auto-fit,minmax(200px,1fr))','gap':'20px','margin-bottom':'30px'})

    app = dash.Dash(__name__)
    app.layout = html.Div([
        dcc.Store(id='data-store', data=df.to_dict('records') if not df.empty else []),
        dcc.Store(id='selected-image-store'),
        dcc.Store(id='current-filter-store', data=job_manager.get_current_filter()),
        dcc.Interval(id='interval-component', interval=10000, n_intervals=0),
        html.Div([
            html.H1('🖼️ Interactive Image Analysis Dashboard', style={'text-align':'center','color':'#1f2937'}),
            html.P('Run jobs and analyze image entailment probabilities in real-time', style={'text-align':'center','color':'#6b7280'})
        ], style={'margin-bottom':'30px'}),
        create_job_control_panel(),
        html.Div(id='current-filter-display'),
        html.Div(id='stats-cards'),
        html.Div(id='charts-grid'),
        html.Div(id='selected-image-area'),
        html.Div(id='cluster-sections')
    ], style={'padding':'20px','font-family':'system-ui, -apple-system','background':'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)'})

    @app.callback(
        [Output('job-status','children'),
         Output('current-filter-store','data')],
        Input('run-job-btn','n_clicks'),
        State('filter-input','value'),
        prevent_initial_call=True
    )
    def run_job(n_clicks, filter_value):
        success, message = job_manager.submit_job(filter_value)
        if success:
            return (html.Div([
                html.P(f"✅ {message}", style={'color':'#059669','font-weight':'bold'}),
                html.P(f"Filter: '{filter_value}'", style={'color':'#666','font-style':'italic'})
            ]), filter_value)
        else:
            return (html.Div([html.P(f"❌ {message}", style={'color':'#dc2626','font-weight':'bold'})]), 
                    job_manager.get_current_filter())

    @app.callback(
        Output('current-filter-display','children'),
        Input('current-filter-store','data')
    )
    def update_filter_display(current_filter):
        return create_current_filter_display(current_filter)

    @app.callback(
        [Output('data-store','data'),
         Output('current-filter-store','data', allow_duplicate=True)],
        Input('refresh-btn','n_clicks'),
        prevent_initial_call=True
    )
    def reload_data(n_clicks):
        new_df = load_data()
        current_filter = job_manager.get_current_filter()
        return (new_df.to_dict('records') if not new_df.empty else [], current_filter)

    @app.callback(
        Output('job-output','children'),
        [Input('interval-component','n_intervals'),
         Input('refresh-btn','n_clicks')],
        prevent_initial_call=True
    )
    def refresh_job_panel(n_intervals, n_clicks):
        status = job_manager.check_job_status()
        return html.Div([
            html.P(f"Job Status: {status.upper()}",
                   style={'font-weight':'bold',
                          'color':('#059669' if status=='completed' else
                                   '#f59e0b' if status=='running' else
                                   '#6b7280')}),
            html.P(f"Last updated: {datetime.now().strftime('%H:%M:%S')}",
                   style={'color':'#666','font-size':'12px'})
        ])

    @app.callback(
        [Output('stats-cards','children'),
         Output('charts-grid','children'),
         Output('cluster-sections','children')],
        Input('data-store','data')
    )
    def update_visuals(data):
        df = pd.DataFrame(data)
        stats = create_stats_cards(df)
        if not df.empty:
            charts = html.Div([
                html.Div(dcc.Graph(figure=create_scatter_plot(df), id='scatter-plot'),
                         style={'grid-column':'span 2'}),
                html.Div(dcc.Graph(figure=create_pie_chart(df)))
            ], style={'display':'grid','grid-template-columns':'repeat(3,1fr)','gap':'20px','margin-bottom':'20px'})
            clusters = html.Div([
                html.H2('🖼️ Image Clusters', style={'text-align':'center','color':'#1f2937'}),
                dcc.Tabs(
                    id='cluster-tabs', value='Absolute probability', children=[
                        dcc.Tab(
                            label=f"✅ Absolute Probability ({len(df[df['cluster']=='Absolute probability'])})",
                            value='Absolute probability'
                        ),
                        dcc.Tab(
                            label=f"⚖️ In Between ({len(df[df['cluster']=='In between'])})",
                            value='In between'
                        ),
                        dcc.Tab(
                            label=f"❌ Zero Probability ({len(df[df['cluster']=='Zero probability'])})",
                            value='Zero probability'
                        )
                    ]
                ),
                html.Div(id='cluster-content', style={'margin-top':'20px'})
            ])
        else:
            charts = html.Div(
                "No data to display. Run a job to generate results!",
                style={'text-align':'center','padding':'40px','color':'#666'}
            )
            clusters = html.Div()
        return stats, charts, clusters

    @app.callback(
        Output('selected-image-area','children'),
        Input('data-store','data'),
        prevent_initial_call=False
    )
    def init_sel_image(data):
        return html.Div(
            "💡 Click on a point in the scatter plot to view the image",
            style={'text-align':'center','font-style':'italic','color':'#666',
                   'padding':'20px','background':'white','border-radius':'8px',
                   'box-shadow':'0 2px 4px rgba(0,0,0,0.1)'}
        )

    @app.callback(
        Output('selected-image-store','data'),
        Input('scatter-plot','clickData'),
        State('data-store','data'),
        prevent_initial_call=True
    )
    def sel_image_store(clickData, data):
        if not clickData:
            return None
        return clickData['points'][0]['pointIndex']

    @app.callback(
        Output('selected-image-area','children', allow_duplicate=True),
        Input('selected-image-store','data'),
        State('data-store','data'),
        prevent_initial_call=True
    )
    def display_sel_image(idx, data):
        if idx is None or not data:
            return init_sel_image(data)
        df = pd.DataFrame(data)
        row = df.iloc[idx]
        return html.Div([
            html.H3("🖼️ Selected Image", style={'color':'#1f2937'}),
            html.Div([
                html.Img(src=row['uri'], style={
                    'max-width':'300px','border-radius':'12px','box-shadow':'0 4px 12px rgba(0,0,0,0.15)'
                }),
                html.Div([
                    html.P([html.Strong("Filename: "), row['filename']]),
                    html.P([
                        html.Strong("Cluster: "),
                        html.Span(row['cluster'], style={
                            'color': cluster_colors[row['cluster']],
                            'font-weight': 'bold'
                        })
                    ]),
                    html.P([html.Strong("Probability: "), f"{row['prob']:.2f}"])
                ], style={'margin-left':'30px'})
            ], style={'display':'flex','align-items':'center'})
        ], style={'background':'white','padding':'25px','border-radius':'12px','box-shadow':'0 4px 6px rgba(0,0,0,0.1)'})

    @app.callback(
        Output('cluster-content','children'),
        Input('cluster-tabs','value'),
        State('data-store','data'),
        prevent_initial_call=True
    )
    def update_cluster_content(tab, data):
        if not data:
            return html.Div()
        df = pd.DataFrame(data)
        return create_image_grid(df, tab, max_images=24)

    app.run(debug=True, host='0.0.0.0', port=8052)

if __name__ == '__main__':
    main()