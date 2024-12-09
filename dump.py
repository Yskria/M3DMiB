# Define the layout
app.layout = html.Div([
    dcc.Slider(
        0, 80, 1, value=0, id='my-slider',
        marks={i: str(i) for i in range(0, 81, 10)},  # Optional: Add marks for better UX
    ),
    html.Div(id='slider-output-container'),
    dcc.Graph(id='graph')  # Dynamically updated graph
])

@app.callback(
    [Output('slider-output-container', 'children'),
     Output('graph', 'figure')],
    [Input('my-slider', 'value')]
)
def update_corrosion_and_graph(slider_value):
    # Update slider output text
    slider_output = f"Selected corrosion years: {slider_value}"

    # Recalculate corrosion data
    calculate_corrosion(slider_value)

    # Reload updated datasets
    city_pipe_main_valid = pd.read_csv('City_Pipe_Main_Updated.csv').dropna(subset=['Latitude', 'Longitude'])
    service_line_table_valid = pd.read_csv('Service_Line_Table_Updated.csv').dropna(subset=['Latitude', 'Longitude'])

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
            if not parent_row.empty:
                parent_lat, parent_lon = parent_row.iloc[0]['Latitude'], parent_row.iloc[0]['Longitude']
                parent_hover_text = "<br>".join([f"{col}: {parent_row.iloc[0][col]}" for col in
                                                 ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])
                node_hover_text = "<br>".join(
                    [f"{col}: {row[col]}" for col in ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])

                # Add edge with hover text
                city_edges.append(((parent_lon, parent_lat), (row['Longitude'], row['Latitude']), corrosion_color))
                city_edge_hover_texts.append(f"Start:<br>{parent_hover_text}<br><br>End:<br>{node_hover_text}")

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
                parent_hover_text = "<br>".join([f"{col}: {parent_row.iloc[0][col]}" for col in
                                                 ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])
                node_hover_text = "<br>".join(
                    [f"{col}: {row[col]}" for col in ['Longitude', 'Latitude', 'CorrosionLevel', 'CorrosionRate']])

                # Add edge with hover text
                service_edges.append(((parent_lon, parent_lat), (row['Longitude'], row['Latitude']), corrosion_color))
                service_edge_hover_texts.append(f"Start:<br>{parent_hover_text}<br><br>End:<br>{node_hover_text}")

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
            corrosion_level = row.iloc[0]['CorrosionLevel']
            corrosion_color = corrosion_colors.get(corrosion_level, 'gray')
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
            corrosion_level = row.iloc[0]['CorrosionLevel']
            corrosion_color = corrosion_colors.get(corrosion_level, 'gray')
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