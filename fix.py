import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

file_paths = {
    "main": 'City_Pipe_Main_Changed.csv',
    "service": 'Service_Line_Table_Changed.csv',
    "corrosion": 'Corrosion_Changed.csv',
    "simulated_corrosion": 'Corrosion_evaluated.csv',
    "house": 'House_Information_Changed.csv',
    "pipe": 'Pipe_Information_Changed.csv',
    "water": 'State_of_Water_Changed.csv'
}

data = {key: pd.read_csv(path) for key, path in file_paths.items()}

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def reset_corrosion_data():
    original_data = pd.read_csv(file_paths['corrosion'])
    original_data.to_csv(file_paths['simulated_corrosion'], index=False)

reset_corrosion_data()
df_original = pd.read_csv(file_paths['simulated_corrosion'])

def calculate_corrosion(years):
    df_original = pd.read_csv(file_paths['corrosion'])
    corrosion_results = []

    for _, row in df_original.iterrows():
        segment_id = row['SegmentID']
        old_rate = row['CorrosionRate']

        pipe_data = data['pipe'][data['pipe']['SegmentID'] == segment_id].iloc[0]
        water_data = data['water'][data['water']['SegmentID'] == segment_id].iloc[0]

        diameter = pipe_data['Diameter (cm)']
        soil = pipe_data['Soil']
        ph_value = abs(water_data['PH-Value'])
        psi_level = abs(water_data['PSI Level'])
        water_temp = abs(water_data['Water Temperature'])

        diameter_params = {
            30.48: (100, 100, 4700, 4700, 4700, 5405),
            4.1: (45, 85, 36.3, 36.3, 36.3, 41.75),
            3.5: (43, 87, 31, 31, 31, 35.7),
            2.7: (40, 90, 23.9, 23.9, 23.9, 27.49)
        }

        if diameter not in diameter_params:
            continue

        mP, mPdev, F, mF, mFdev, fMax = diameter_params[diameter]

        ranges = {
            "A": (0, 14), "P": (0, 130), "mP": (0, mP), "mPdev": (0, mPdev),
            "T": (5, 15), "X": (2.7, 30.48), "F": (0, fMax), "mF": (0, mF),
            "mFdev": (0, mFdev), "S": (0, 1)
        }

        A = normalize(ph_value, *ranges["A"])
        P = normalize(psi_level, *ranges["P"])
        mP = normalize(mP, *ranges["mP"])
        T = normalize(water_temp, *ranges["T"])
        X = normalize(diameter, *ranges["X"])
        F = normalize(F, *ranges["F"])
        mF = normalize(mF, *ranges["mF"])
        mFdev = normalize(mFdev, *ranges["mFdev"])
        S = 0.7 if soil == "Clay" else 0.4
        S = normalize(S, *ranges["S"])
        I = 0.5

        if pipe_data['Pipe Type'] == 'City Pipe Main':
            term1 = 0.16 * (0.3 + (A * (1 + 0.15)))
            term2 = 0.08 * (0.3 + (abs((P - mP)) * (0.6 + (1 / mPdev))))
            term3 = 0.16 * (T * (1 + 0.1))
            term4 = 0.08 * (X * (1 + 0.05))
            term5 = 0.04 * (S * (1 + 0.1))
            term6 = 0.16 * (S * (1 + 0.1))
        else:
            term1 = 0.16 * (0.3 + (A * (1 + 0.15)))
            term2 = 0.08 * (0.3 + (abs((P - mP)) * (1 + (1 / mPdev))))
            term3 = 0.16 * (T * (1 + 0.1))
            term4 = 0.08 * (X * (1 + 0.5))
            term5 = 0.04 * (S * (1 + 0.1))
            term6 = 0.16 * (S * (1 + 0.1))

        adjustment = 2.2 * years * (term1 + term2 + term3 + term4 + term5 + term6)
        corrosion_rate = old_rate - adjustment

        if 80 <= corrosion_rate <= 100:
            corrosion_level = 1
        elif 60 <= corrosion_rate < 80:
            corrosion_level = 2
        elif 40 <= corrosion_rate < 60:
            corrosion_level = 3
        elif 20 <= corrosion_rate < 40:
            corrosion_level = 4
        elif 0 <= corrosion_rate < 20:
            corrosion_level = 5
        elif corrosion_rate > 100:
            corrosion_level = "Pipe Healed???"
        else:
            corrosion_level = "Pipe corroded."

        corrosion_results.append({
            "SegmentID": str(segment_id),
            "CorrosionRate": corrosion_rate,
            "CorrosionLevel": corrosion_level
        })

    return pd.DataFrame(corrosion_results)


CEval = pd.read_csv(file_paths['simulated_corrosion'])

corrosion_filtered = CEval[['SegmentID', 'CorrosionLevel', 'CorrosionRate']]

main_df = pd.read_csv(file_paths['main'])
service_df = pd.read_csv(file_paths['service'])

city_pipe_main_merged = pd.merge(main_df, corrosion_filtered, on='SegmentID', how='left', validate='1:1')
service_line_table_merged = pd.merge(service_df, corrosion_filtered, on='SegmentID', how='left')

city_pipe_main_valid = city_pipe_main_merged.dropna(subset=['Latitude', 'Longitude'])
service_line_table_valid = service_line_table_merged.dropna(subset=['Latitude', 'Longitude'])

corrosion_colors = {
    1: 'darkgreen',
    2: 'lightgreen',
    3: 'yellow',
    4: 'orange',
    5: 'red'
}

def get_corrosion_color(level):
    return corrosion_colors.get(level, 'gray')

def generate_edges(city_data, service_data):
    edges, hover_texts = [], []

    for _, row in city_data.iterrows():
        if pd.notna(row['Parent Pipe']):
            parent_row = city_data[city_data['SegmentID'] == row['Parent Pipe']]
            if not parent_row.empty:
                parent_lat, parent_lon = parent_row.iloc[0]['Latitude'], parent_row.iloc[0]['Longitude']
                node_lat, node_lon = row['Latitude'], row['Longitude']
                corrosion_color = corrosion_colors.get(row['CorrosionLevel'], 'gray')
                edges.append(((parent_lon, parent_lat), (node_lon, node_lat), corrosion_color))

                hover_text = f"Start:<br>Corrosion Level: {parent_row.iloc[0]['CorrosionLevel']}<br>" \
                             f"End:<br>Corrosion Level: {row['CorrosionLevel']}"
                hover_texts.append(hover_text)

    for _, row in service_data.iterrows():
        if pd.notna(row['ParentPipe']):
            parent_row = service_data[service_data['SegmentID'] == row['ParentPipe']]
            if not parent_row.empty:
                parent_lat, parent_lon = parent_row.iloc[0]['Latitude'], parent_row.iloc[0]['Longitude']
                node_lat, node_lon = row['Latitude'], row['Longitude']
                corrosion_color = corrosion_colors.get(row['CorrosionLevel'], 'gray')
                edges.append(((parent_lon, parent_lat), (node_lon, node_lat), corrosion_color))

                hover_text = f"Service Line:<br>Start Corrosion Level: {parent_row.iloc[0]['CorrosionLevel']}<br>" \
                             f"End Corrosion Level: {row['CorrosionLevel']}"
                hover_texts.append(hover_text)

    return edges, hover_texts

def create_figure(city_data, service_data):
    fig = go.Figure()
    edges, hover_texts = generate_edges(city_data, service_data)

    for edge, hover_text in zip(edges, hover_texts):
        x_start, x_end = edge[0][0], edge[1][0]
        y_start, y_end = edge[0][1], edge[1][1]

        fig.add_trace(go.Scattermapbox(
            lon=[x_start, x_end],
            lat=[y_start, y_end],
            mode='lines',
            line=dict(color=edge[2], width=5),
            text=hover_text, hoverinfo='text',
            showlegend=False
        ))

    for _, row in city_data.iterrows():
        corrosion_color = get_corrosion_color(row['CorrosionLevel'])
        fig.add_trace(go.Scattermapbox(
            lon=[row['Longitude']],
            lat=[row['Latitude']],
            mode='markers',
            marker=dict(size=10, color=corrosion_color),
            text=f"SegmentID: {row['SegmentID']}<br>Corrosion Level: {row['CorrosionLevel']}",
            hoverinfo='text'
        ))

    for _, row in service_data.iterrows():
        corrosion_color = get_corrosion_color(row['CorrosionLevel'])
        fig.add_trace(go.Scattermapbox(
            lon=[row['Longitude']],
            lat=[row['Latitude']],
            mode='markers',
            marker=dict(size=8, color=corrosion_color, symbol='triangle'),
            text=f"Service Line SegmentID: {row['SegmentID']}<br>Corrosion Level: {row['CorrosionLevel']}",
            hoverinfo='text'
        ))


    combined_latitudes = pd.concat([city_data['Latitude']])
    combined_longitudes = pd.concat([city_data['Longitude']])

    center_lon = (combined_longitudes.max() + combined_longitudes.min()) / 2
    center_lat = (combined_latitudes.max() + combined_latitudes.min()) / 2

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lon": center_lon, "lat": center_lat},
        mapbox_zoom=15,
        title="Interactive City Pipe Visualization",
        showlegend=False,
        template="plotly_white",
        autosize=True
    )

    return fig


@app.callback(
    [Output('slider-output-container', 'children'),
     Output('graph', 'figure')],
    [Input('my-slider', 'value')]
)
def update_graph(year):
    corrosion_results = calculate_corrosion(year)

    main_df = pd.read_csv(file_paths['main'])
    service_df = pd.read_csv(file_paths['service'])

    city_pipe_main_merged = pd.merge(main_df, corrosion_results, on='SegmentID', how='left', validate='1:1')
    service_line_table_merged = pd.merge(service_df, corrosion_results, on='SegmentID', how='left')

    city_pipe_main_valid = city_pipe_main_merged.dropna(subset=['Latitude', 'Longitude'])
    service_line_table_valid = service_line_table_merged.dropna(subset=['Latitude', 'Longitude'])

    slider_output = f"Selected corrosion years: {year}"
    figure = create_figure(city_pipe_main_valid, service_line_table_valid)

    return slider_output, figure

app.layout = html.Div([
    dcc.Slider(0, 80, 1, value=0, id='my-slider'), 
    html.Div(id='slider-output-container'),
    dcc.Graph(id='graph', figure=create_figure(city_pipe_main_merged, service_line_table_merged), config={'scrollZoom': True}) 
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)