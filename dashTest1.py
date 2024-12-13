# Libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Initialize the Dash app
app = Dash(__name__)

file_path_main = 'City_Pipe_Main.csv'
city_pipe_main = pd.read_csv(file_path_main)

file_path_service = 'Service_Line_Table.csv'
service_line_table = pd.read_csv(file_path_service)

file_path_corrosion = 'Corrosion.csv'
Corrosion = pd.read_csv(file_path_corrosion)

file_path_simulated_corrosion = 'Corrosion_evaluated.csv'
Corrosion_evaluated = pd.read_csv(file_path_simulated_corrosion)

file_path_house = 'House_Information.csv'
house_information = pd.read_csv(file_path_house)

file_path_pipe_information = 'Pipe_Information.csv'
pipe_information = pd.read_csv(file_path_pipe_information)

file_path_state_of_water = 'State_of_Water.csv'
state_of_water = pd.read_csv(file_path_state_of_water)

# Normalize function
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def reset_corrosion_data():
    df_original = pd.read_csv(file_path_corrosion)

    df_original.to_csv(file_path_simulated_corrosion, index=False)

reset_corrosion_data()

df_original = pd.read_csv(file_path_simulated_corrosion)

def calculate_corrosion(Years):
    corrosion_results = []

    for index, row in df_original.iterrows():
        segment_id = row['SegmentID']
        old_rate = row['CorrosionRate']

        pipe_data = pipe_information[pipe_information['SegmentID'] == segment_id].iloc[0]
        water_data = state_of_water[state_of_water['SegmentID'] == segment_id].iloc[0]

        diameter = pipe_data['Diameter (cm)']
        soil = pipe_data['Soil']
        ph_value = abs(water_data['PH-Value'])
        psi_level = abs(water_data['PSI Level'])
        water_temp = abs(water_data['Water Temperature'])

        # Define material properties based on diameter
        if diameter == 30.48:
            mP, mPdev, F, mF, mFdev, fMax = 100, 100, 4700, 4700, 4700, 5405
        elif diameter == 4.1:
            mP, mPdev, F, mF, mFdev, fMax = 45, 85, 36.3, 36.3, 36.3, 41.75
        elif diameter == 3.5:
            mP, mPdev, F, mF, mFdev, fMax = 43, 87, 31, 31, 31, 35.7
        elif diameter == 2.7:
            mP, mPdev, F, mF, mFdev, fMax = 40, 90, 23.9, 23.9, 23.9, 27.49
        else:
            continue

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
        else:
            term1 = 0.16 * (0.3 + (A * (1 + 0.15)))
            term2 = 0.08 * (0.3 + (abs((P - mP)) * (1 + (1 / mPdev))))
            term3 = 0.16 * (T * (1 + 0.1))
            term4 = 0.08 * (X * (1 + 0.5))

        adjustment = 2.2 * Years * (term1 + term2 + term3 + term4)
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

    # Convert the list of results to a DataFrame
    corrosion_results_df = pd.DataFrame(corrosion_results)

    # Set SegmentID as the index for both the Corrosion and the results DataFrame
    df_original.set_index('SegmentID', inplace=True)
    corrosion_results_df.reset_index(drop=True, inplace=True)

    # Update the original Corrosion_Evaluated DataFrame with the new results
    df_original.update(corrosion_results_df[['CorrosionRate', 'CorrosionLevel']])

    # Reset the index for both DataFrames
    df_original.reset_index(inplace=True)

    # Save the updated Corrosion_Evaluated DataFrame to a CSV file
    df_original.to_csv("Corrosion_evaluated.csv", index=False)

    #Display the first 100 rows of the updated corrosion results
    corrosion_results_df.head(100)
    return corrosion_results_df

CEval = pd.read_csv('Corrosion_evaluated.csv')
corrosion_map = CEval.set_index('SegmentID')['CorrosionLevel'].to_dict()

# Select only necessary columns from Corrosion
corrosion_filtered = CEval[['SegmentID', 'CorrosionLevel', 'CorrosionRate']]

# Merge Corrosion data into City_Pipe_Main
city_pipe_main_merged = pd.merge(city_pipe_main, corrosion_filtered, on='SegmentID', how='left')

# Merge Corrosion data into Service_Line_Table
service_line_table_merged = pd.merge(service_line_table, corrosion_filtered, on='SegmentID', how='left')

# Save the updated datasets
city_pipe_main_merged.to_csv('City_Pipe_Main_Updated.csv', index=False)
service_line_table_merged.to_csv('Service_Line_Table_Updated.csv', index=False)

# Filter rows with valid coordinates
city_pipe_main_valid = city_pipe_main_merged.dropna(subset=['Latitude', 'Longitude'])
service_line_table_valid = service_line_table_merged.dropna(subset=['Latitude', 'Longitude'])

# Define color mappings
corrosion_colors = {1: 'darkgreen', 2: 'lightgreen', 3: 'yellow', 4: 'orange', 5: 'red'}
direction_colors = {i: f"rgb({(i * 50) % 255}, {(i * 100) % 255}, {(i * 150) % 255})" for i in range(10)}

# Prepare city edges and nodes
city_edges, city_positions, city_edge_hover_texts = [], {}, []
for _, row in city_pipe_main_valid.iterrows():
    city_positions[row['SegmentID']] = (row['Longitude'], row['Latitude'])
    corrosion_level = row['CorrosionLevel']
    corrosion_color = corrosion_colors.get(corrosion_level, 'gray')  # Default to gray if no corrosion level

    if pd.notna(row['Parent Pipe']):
        parent_row = city_pipe_main_valid[city_pipe_main_valid['SegmentID'] == row['Parent Pipe']]
        if not parent_row.empty:
            parent_lat, parent_lon = parent_row.iloc[0]['Latitude'], parent_row.iloc[0]['Longitude']
            parent_hover_text = "<br>".join([f"{col}: {parent_row.iloc[0][col]}" for col in ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])
            node_hover_text = "<br>".join([f"{col}: {row[col]}" for col in ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])

            # Add edge with hover text that includes both parent (start) and current (end) node data
            city_edges.append(((parent_lon, parent_lat), (row['Longitude'], row['Latitude']), corrosion_color))
            city_edge_hover_texts.append(f"Start:<br>{parent_hover_text}<br><br>End:<br>{node_hover_text}")

# Prepare service edges and nodes
service_edges, service_positions, service_edge_hover_texts = [], {}, []

for _, row in service_line_table_valid.iterrows():
    service_positions[row['SegmentID']] = (row['Longitude'], row['Latitude'])
    corrosion_level = row['CorrosionLevel']
    corrosion_color = corrosion_colors.get(corrosion_level, 'gray')

    if pd.notna(row['ParentPipe']):
        parent_row_main = city_pipe_main_valid[city_pipe_main_valid['SegmentID'] == row['ParentPipe']]
        parent_row_service = service_line_table_valid[service_line_table_valid['SegmentID'] == row['ParentPipe']]
        parent_row = pd.concat([parent_row_main, parent_row_service])

        if not parent_row.empty:
            parent_lat, parent_lon = parent_row.iloc[0]['Latitude'], parent_row.iloc[0]['Longitude']
            parent_hover_text = "<br>".join([f"{col}: {parent_row.iloc[0][col]}" for col in ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])
            node_hover_text = "<br>".join([f"{col}: {row[col]}" for col in ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])

            # Add edge with hover text that includes both parent (start) and current (end) node data
            service_edges.append(((parent_lon, parent_lat), (row['Longitude'], row['Latitude']), corrosion_color))
            service_edge_hover_texts.append(f"Start:<br>{parent_hover_text}<br><br>End:<br>{node_hover_text}")

# Plot using Plotly
fig = go.Figure()

# Helper function for intermediate points
def generate_intermediate_points(x_start, y_start, x_end, y_end, num_points=10):
    x_points = np.linspace(x_start, x_end, num_points)
    y_points = np.linspace(y_start, y_end, num_points)
    return x_points, y_points

for edge, hover_text in zip(city_edges, city_edge_hover_texts):
    x_start, x_end = edge[0][0], edge[1][0]
    y_start, y_end = edge[0][1], edge[1][1]

    # Create intermediate points along the edge
    x_points, y_points = generate_intermediate_points(x_start, y_start, x_end, y_end, num_points=10)  # 10 intermediate points

    # Add the edge
    fig.add_trace(go.Scattermapbox(
        lon=[x_start, x_end],
        lat=[y_start, y_end],
        mode='lines',
        line=dict(color=edge[2], width=5),  # Thicker solid lines for city pipes, colored based on corrosion level
        text=hover_text, hoverinfo='text',  # Ensure text is shown on hover
        showlegend=False
    ))

    # Use the same hover_text for the midpoint label
    for x, y in zip(x_points, y_points):
        fig.add_trace(go.Scattermapbox(
            lon=[x],
            lat=[y],
            mode='text',
            text=[hover_text],  # Set the text of the midpoint as the hover_text for both start and end nodes
            textposition='top center',
            hoverinfo='text',
            hoverlabel=dict(bgcolor=edge[2]),
            showlegend=False
        ))


# Add edges for Service Line with mapbox and thinner lines
for edge, hover_text in zip(service_edges, service_edge_hover_texts):
    x_start, x_end = edge[0][0], edge[1][0]
    y_start, y_end = edge[0][1], edge[1][1]

    # Create intermediate points along the edge
    x_points, y_points = generate_intermediate_points(x_start, y_start, x_end, y_end, num_points=10)  # 10 intermediate points

    # Add the edge
    fig.add_trace(go.Scattermapbox(
        lon=[x_start, x_end],
        lat=[y_start, y_end],
        mode='lines',
        line=dict(color=edge[2], width=3),  # Thinner lines for service pipes, colored based on corrosion level
        text=hover_text, hoverinfo='text',  # Ensure text is shown on hover
        showlegend=False
    ))

    # Use the same hover_text for the midpoint label
    for x, y in zip(x_points, y_points):
        fig.add_trace(go.Scattermapbox(
            lon=[x],
            lat=[y],
            mode='text',
            text=[hover_text],  # Set the text of the midpoint as the hover_text for both start and end nodes
            textposition='top center',
            hoverinfo='text',
            hoverlabel=dict(bgcolor=edge[2]),
            showlegend=False
        ))

# City nodes
for segment_id, (lon, lat) in city_positions.items():
    fig.add_trace(go.Scattermapbox(
        lon=[lon],
        lat=[lat],
        mode='markers',
        marker=dict(size=8, color='blue'),
        text=f"SegmentID: {segment_id}",
        hoverinfo='text'
    ))

# Service nodes
for segment_id, (lon, lat) in city_positions.items():
    row = city_pipe_main_valid[city_pipe_main_valid['SegmentID'] == segment_id].iloc[0]
    corrosion_level = row['CorrosionLevel']
    corrosion_color = corrosion_colors.get(corrosion_level, 'gray')  # Default to gray if no corrosion level
    hover_text = "<br>".join([f"{col}: {row[col]}" for col in ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])
    fig.add_trace(go.Scattermapbox(
        lon=[lon], lat=[lat], mode='markers',
        marker=dict(size=10, color=corrosion_color),
        text=hover_text, hoverinfo='text',
        showlegend=False
    ))

# Add nodes for Service Line Table with mapbox
for segment_id, (lon, lat) in service_positions.items():
    row = service_line_table_valid[service_line_table_valid['SegmentID'] == segment_id].iloc[0]
    corrosion_level = row['CorrosionLevel']
    corrosion_color = corrosion_colors.get(corrosion_level, 'gray')  # Default to gray if no corrosion level
    hover_text = "<br>".join([f"{col}: {row[col]}" for col in ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])
    fig.add_trace(go.Scattermapbox(
        lon=[lon], lat=[lat], mode='markers',
        marker=dict(size=7, color=corrosion_color),
        text=hover_text, hoverinfo='text',
        showlegend=False
    ))

combined_latitudes = pd.concat([city_pipe_main['Latitude'], service_line_table['Latitude']])
combined_longitudes = pd.concat([city_pipe_main['Longitude'], service_line_table['Longitude']])

center_lon = (combined_longitudes.max() + combined_longitudes.min()) / 2
center_lat = (combined_latitudes.max() + combined_latitudes.min()) / 2

@app.callback(
    [Output('slider-output-container', 'children'),
     Output('graph', 'figure')],
    [Input('my-slider', 'value')]
)
def update_corrosion_and_graph(slider_value):
    # Update slider output text
    slider_output = f"Selected corrosion years: {slider_value}"

    corrosion_results_df = calculate_corrosion(slider_value)

    # Reload updated datasets
    city_pipe_main_valid = pd.read_csv('City_Pipe_Main_Updated.csv').dropna(subset=['Latitude', 'Longitude'])
    service_line_table_valid = pd.read_csv('Service_Line_Table_Updated.csv').dropna(subset=['Latitude', 'Longitude'])

    # Merge corrosion data for visualization
    city_pipe_main_valid = pd.merge(
        city_pipe_main_valid[['SegmentID', 'Latitude', 'Longitude', 'Parent Pipe']],
        corrosion_results_df[['SegmentID', 'CorrosionLevel', 'CorrosionRate']],
        on='SegmentID',
        how='left'
    )

    service_line_table_valid = pd.merge(
        service_line_table_valid[['SegmentID', 'Latitude', 'Longitude', 'ParentPipe']],
        corrosion_results_df[['SegmentID', 'CorrosionLevel', 'CorrosionRate']],
        on='SegmentID',
        how='left'
    )

    # Prepare data for visualization
    city_edges, city_positions, city_edge_hover_texts = [], {}, []
    service_edges, service_positions, service_edge_hover_texts = [], {}, []

    # Logic for preparing city_edges and service_edges goes here...
    for _, row in city_pipe_main_valid.iterrows():
        city_positions[row['SegmentID']] = (row['Longitude'], row['Latitude'])
        corrosion_level = row['CorrosionLevel']
        corrosion_color = corrosion_colors.get(corrosion_level, 'gray')

        if pd.notna(row['Parent Pipe']):
            parent_row = city_pipe_main_valid[city_pipe_main_valid['SegmentID'] == row['Parent Pipe']]
            # if not parent_row.empty:
            #     parent_lat, parent_lon = parent_row.iloc[0]['Latitude'], parent_row.iloc[0]['Longitude']
            #     parent_hover_text = "<br>".join([f"{col}: {parent_row.iloc[0][col]}" for col in
            #                                      ['Longitude', 'Latitude', city_pipe_main_valid['CorrosionLevel'], 'CorrosionRate']])
            #     node_hover_text = "<br>".join(
            #         [f"{col}: {row[col]}" for col in ['Longitude', 'Latitude', city_pipe_main_valid['CorrosionLevel'], 'CorrosionRate']])
            #
            #     # Add edge with hover text
            #     city_edges.append(((parent_lon, parent_lat), (row['Longitude'], row['Latitude']), corrosion_color))
            #     city_edge_hover_texts.append(f"Start:<br>{parent_hover_text}<br><br>End:<br>{node_hover_text}")

    for _, row in service_line_table_valid.iterrows():
        service_positions[row['SegmentID']] = (row['Longitude'], row['Latitude'])
        corrosion_level = row['CorrosionLevel']
        corrosion_color = corrosion_colors.get(corrosion_level, 'gray')

        if pd.notna(row['ParentPipe']):
            parent_row_main = city_pipe_main_valid[city_pipe_main_valid['SegmentID'] == row['ParentPipe']]
            parent_row_service = service_line_table_valid[service_line_table_valid['SegmentID'] == row['ParentPipe']]
            parent_row = pd.concat([parent_row_main, parent_row_service])

            # if not parent_row.empty:
            #     parent_lat, parent_lon = parent_row.iloc[0]['Latitude'], parent_row.iloc[0]['Longitude']
            #     parent_hover_text = "<br>".join([f"{col}: {parent_row.iloc[0][col]}" for col in
            #                                      ['Longitude', 'Latitude',service_line_table_valid['CorrosionLevel'], 'CorrosionRate']])
            #     node_hover_text = "<br>".join(
            #         [f"{col}: {row[col]}" for col in ['Longitude', 'Latitude', service_line_table_valid['CorrosionLevel'], 'CorrosionRate']])
            #
            #     # Add edge with hover text
            #     service_edges.append(((parent_lon, parent_lat), (row['Longitude'], row['Latitude']), corrosion_color))
            #     service_edge_hover_texts.append(f"Start:<br>{parent_hover_text}<br><br>End:<br>{node_hover_text}")

    # Create a new figure dynamically
    fig = go.Figure()

    # Add edges and nodes to the figure dynamically
    for edge, hover_text in zip(city_edges, city_edge_hover_texts):
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

    for edge, hover_text in zip(service_edges, service_edge_hover_texts):
        x_start, x_end = edge[0][0], edge[1][0]
        y_start, y_end = edge[0][1], edge[1][1]
        fig.add_trace(go.Scattermapbox(
            lon=[x_start, x_end],
            lat=[y_start, y_end],
            mode='lines',
            line=dict(color=edge[2], width=3),
            text=hover_text, hoverinfo='text',
            showlegend=False
        ))

    # Add nodes for city and service positions
    for segment_id, (lon, lat) in city_positions.items():
        row = city_pipe_main_valid[city_pipe_main_valid['SegmentID'] == segment_id]
        if not row.empty:
            if isinstance(corrosion_level, pd.Series):
                corrosion_level = corrosion_level.iloc[0]
            corrosion_color = corrosion_colors.get(corrosion_level, 'gray')  # Map to color
            print(row[['CorrosionRate']])
            print(corrosion_color)
            hover_text = "<br>".join(
                [f"{col}: {row.iloc[0][col]}" for col in ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])
            fig.add_trace(go.Scattermapbox(
                lon=[lon], lat=[lat], mode='markers',
                marker=dict(size=10, color=corrosion_color),
                text=hover_text, hoverinfo='text',
                showlegend=False
            ))

    for segment_id, (lon, lat) in service_positions.items():
        row = service_line_table_valid[service_line_table_valid['SegmentID'] == segment_id]
        if not row.empty:
            if isinstance(corrosion_level, pd.Series):
                corrosion_level = corrosion_level.iloc[0]
            corrosion_color = corrosion_colors.get(corrosion_level, 'gray')  # Map to color

            hover_text = "<br>".join(
                [f"{col}: {row.iloc[0][col]}" for col in ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])
            fig.add_trace(go.Scattermapbox(
                lon=[lon], lat=[lat], mode='markers',
                marker=dict(size=7, color=corrosion_color),
                text=hover_text, hoverinfo='text',
                showlegend=False
            ))

    # Update figure layout
    combined_latitudes = pd.concat([city_pipe_main['Latitude'], service_line_table['Latitude']])
    combined_longitudes = pd.concat([city_pipe_main['Longitude'], service_line_table['Longitude']])

    center_lon = (combined_longitudes.max() + combined_longitudes.min()) / 2
    center_lat = (combined_latitudes.max() + combined_latitudes.min()) / 2

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lon": center_lon, "lat": center_lat},
        mapbox_zoom=18,
        title="Interactive City Pipe Visualization",
        showlegend=False,
        template="plotly_white",
        autosize=True
    )

    return slider_output, fig

# Set map layout properties
fig.update_layout(
    mapbox_style="open-street-map",  # Use Open Street Map as the background
    mapbox_center={"lon": center_lon, "lat": center_lat},
    mapbox_zoom=18,
    title="Interactive City Pipe Main and Service Line Connections",
    showlegend=False,
    template="plotly_white",
    autosize=True
)

app.layout = html.Div([
    dcc.Slider(0, 80, 1, value=0, id='my-slider'),  # Slider to control corrosion
    html.Div(id='slider-output-container'),  # To display slider output
    dcc.Graph(id='graph', figure=fig)  # This graph will be updated dynamically
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)