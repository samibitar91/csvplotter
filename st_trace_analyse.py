import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import plotly.express as px

# Use the streamlit-sortables component for ordering the variable list
try:
    import streamlit_sortables as st_sortables
except ImportError:
    st.warning("Please install streamlit-sortables via 'pip install streamlit-sortables'")
    st_sortables = None

# Enable wide-screen mode
st.set_page_config(
    page_title="Interactive Subplots & Main Plot",
    layout="wide"
)

def process_csv(df, time_col, measurement_col, output_col, quality_col, quality_threshold):
    # Reorder the columns: time_col, measurement_col, output_col, quality_col, then the rest
    reordered_columns = [time_col, measurement_col, output_col, quality_col] + [
        col for col in df.columns if col not in [time_col, measurement_col, output_col, quality_col]
    ]
    df = df[reordered_columns]

    # Replace bad quality readings in the measurement column with zeroes
    df[measurement_col] = df[measurement_col].apply(lambda x: 0 if x > quality_threshold else x)

    # Round numerical values to 3 decimal places
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col] = df[col].round(3)

    return df

def calculate_intersections(df, time_col, line1_x, line2_x, selected_columns):
    """Calculate the intersections (closest y values) for X cursors."""
    intersections = {}
    for col in selected_columns:
        closest_line1_idx = (df[time_col] - line1_x).abs().idxmin()
        closest_line2_idx = (df[time_col] - line2_x).abs().idxmin()
        val1 = df.at[closest_line1_idx, col]
        val2 = df.at[closest_line2_idx, col]
        diff = abs(val2 - val1)
        intersections[col] = {"val1": val1, "val2": val2, "diff": diff}
    return intersections

def plot_subplots(df, sel_time_col, x_range_min, x_range_max, theme, draw_lines,
                  line1_x, line2_x, plot_columns, line_y1=None, line_y2=None):
    """
    Create subplots for each selected variable. X cursors are drawn on all plots,
    while Y cursors are drawn only for non-binary variables.
    """
    # Filter the dataframe by the selected x-range
    df_filtered = df[(df[sel_time_col] >= x_range_min) & (df[sel_time_col] <= x_range_max)]
    
    # Determine subplot height: 100 pixels for binary, 200 for analogue variables
    pixel_heights = []
    for col in plot_columns:
        unique_vals = df_filtered[col].dropna().unique()
        pixel_heights.append(100 if len(unique_vals) == 2 else 500)
    total_height = sum(pixel_heights)
    row_heights = [h / total_height for h in pixel_heights]

    X1_COLOR = "#007FFF"  # Electric Blue
    X2_COLOR = "#FFD700"  # Gold
    Y1_COLOR = "#FF00FF"  # Magenta
    Y2_COLOR = "#00FFFF"  # Cyan
    
    # Create subplots
    fig = make_subplots(
        rows=len(plot_columns),
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights
    )

    colors = px.colors.qualitative.Set1

    # Calculate intersections for X cursors if enabled
    if draw_lines:
        intersections = calculate_intersections(df_filtered, sel_time_col, line1_x, line2_x, plot_columns)

    for idx, col in enumerate(plot_columns):
        # Add the main trace for the variable
        fig.add_trace(
            go.Scatter(
                x=df_filtered[sel_time_col],
                y=df_filtered[col],
                mode="lines",
                name=col,
                line=dict(color=colors[idx % len(colors)]),
            ),
            row=idx + 1,
            col=1,
        )

        # Draw vertical X cursors and markers if enabled
        if draw_lines and col in intersections:
            # Vertical line at X1
            fig.add_shape(
                dict(
                    type="line",
                    x0=line1_x,
                    x1=line1_x,
                    y0=df_filtered[col].min(),
                    y1=df_filtered[col].max(),
                    line=dict(color=X1_COLOR, width=2, dash="dot"),
                ),
                row=idx + 1,
                col=1,
            )
            # Vertical line at X2
            fig.add_shape(
                dict(
                    type="line",
                    x0=line2_x,
                    x1=line2_x,
                    y0=df_filtered[col].min(),
                    y1=df_filtered[col].max(),
                    line=dict(color=X2_COLOR, width=2, dash="dot"),
                ),
                row=idx + 1,
                col=1,
            )
            # Marker for X1 intersection
            fig.add_trace(
                go.Scatter(
                    x=[line1_x],
                    y=[intersections[col]["val1"]],
                    mode="markers",
                    name=f'{col} X1',
                    marker=dict(size=8, color=X1_COLOR),
                    hovertemplate=f"X1: x={line1_x:.2f}, y={intersections[col]['val1']:.2f}",
                ),
                row=idx + 1,
                col=1,
            )
            # Marker for X2 intersection
            fig.add_trace(
                go.Scatter(
                    x=[line2_x],
                    y=[intersections[col]["val2"]],
                    mode="markers",
                    name=f'{col} X2',
                    marker=dict(size=8, color=X2_COLOR),
                    hovertemplate=f"X2: x={line2_x:.2f}, y={intersections[col]['val2']:.2f}",
                ),
                row=idx + 1,
                col=1,
            )

        # Draw Y cursors only on non-binary variables
        unique_vals = df_filtered[col].dropna().unique()
        if len(unique_vals) != 2:
            if line_y1 is not None:
                closest_idx_y1 = (df_filtered[col] - line_y1).abs().idxmin()
                x_intersect_y1 = df_filtered.at[closest_idx_y1, sel_time_col]
                fig.add_shape(
                    dict(
                        type="line",
                        x0=df_filtered[sel_time_col].min(),
                        x1=df_filtered[sel_time_col].max(),
                        y0=line_y1,
                        y1=line_y1,
                        line=dict(color=Y1_COLOR, width=2, dash="dot"),
                    ),
                    row=idx + 1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x_intersect_y1],
                        y=[line_y1],
                        mode="markers",
                        name=f'{col} Y1',
                        marker=dict(size=8, color=Y1_COLOR),
                        hovertemplate=f"Y1: x={x_intersect_y1:.2f}, y={line_y1:.2f}",
                    ),
                    row=idx + 1,
                    col=1,
                )
            if line_y2 is not None:
                closest_idx_y2 = (df_filtered[col] - line_y2).abs().idxmin()
                x_intersect_y2 = df_filtered.at[closest_idx_y2, sel_time_col]
                fig.add_shape(
                    dict(
                        type="line",
                        x0=df_filtered[sel_time_col].min(),
                        x1=df_filtered[sel_time_col].max(),
                        y0=line_y2,
                        y1=line_y2,
                        line=dict(color=Y2_COLOR, width=2, dash="dot"),
                    ),
                    row=idx + 1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x_intersect_y2],
                        y=[line_y2],
                        mode="markers",
                        name=f'{col} Y2',
                        marker=dict(size=8, color=Y2_COLOR),
                        hovertemplate=f"Y2: x={x_intersect_y2:.2f}, y={line_y2:.2f}",
                    ),
                    row=idx + 1,
                    col=1,
                )

        # Update axes labels and fonts
        fig.update_yaxes(
            showticklabels=True,
            title_text=col,
            row=idx + 1,
            col=1,
            tickfont=dict(color=theme["font"]["color"])
        )
        fig.update_xaxes(
            showticklabels=True,
            row=idx + 1,
            col=1,
            title=sel_time_col,
            tickfont=dict(color=theme["font"]["color"])
        )

    fig.update_layout(
        height=total_height,
        uirevision='constant',  # Add this line to preserve zoom state
        **theme,
        dragmode="select",
        title="Subplots of Zoomed Area"
    )

    return fig

# ----------------------- Main Streamlit UI -----------------------

# Create two columns: left for controls and cursor info, right for plots and slider controls.
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Settings & Controls")
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file, delimiter=";")
        df = df.apply(lambda col: col.astype('float64') if col.dtype.kind in 'biufc' else col)

        # Select Time column (default is the first)
        Time_column_options = df.columns
        selected_Time_column = st.selectbox(
            "Select the Time column (default is the first column)",
            options=Time_column_options,
            index=0,
        )

        # Theme selection
        selected_theme = st.radio("Select Theme", ["dark", "light"])
        if selected_theme == "light":
            background_color = "#ffffff"
            text_color = "#000000"
        else:
            background_color = "#2c2f33"
            text_color = "#ffffff"
        theme_settings = {
            "title_font": dict(color=text_color),
            "plot_bgcolor": background_color,
            "paper_bgcolor": background_color,
            "font": dict(color=text_color),
        }

        # Select Measurement, Output, and Quality columns
        measurement_col = st.selectbox("Select Measurement Column", options=df.columns, index=4)
        output_col = st.selectbox("Select Output Column", options=df.columns, index=1)
        quality_col = st.selectbox("Select Quality Bit Column", options=df.columns, index=3)

        # Set threshold for bad quality readings
        max_output = df[measurement_col].max()
        quality_threshold = st.number_input(
            "Set Measurement Threshold (values above will be set to 0)",
            min_value=0.0,
            max_value=max_output,
            value=max_output - 1,
        )

        # Toggle for X and Y cursor display
        draw_lines_toggle = st.checkbox("Show X Cursor", value=True)
        show_y_cursor = st.checkbox("Show Y Cursor", value=True)

        # ----- Toggle & Order Variables to Plot -----
        all_columns = [col for col in df.columns if col != selected_Time_column]
        default_vars = [measurement_col, output_col, quality_col]
        selected_plot_vars = st.multiselect("Select Variables to Plot", options=all_columns, default=default_vars)

        if not selected_plot_vars:
            st.error("Please select at least one variable to plot.")
            st.stop()

        # Use streamlit-sortables for drag-and-drop ordering (if available)
        if st_sortables is not None:
            ordered_plot_vars = st_sortables.sort_items(selected_plot_vars)
        else:
            ordered_plot_vars = selected_plot_vars

        # Process the CSV based on selections
        processed_df = process_csv(df, selected_Time_column, measurement_col, output_col, quality_col, quality_threshold)

        
        non_binary_vars = [var for var in ordered_plot_vars if processed_df[var].dropna().nunique() != 2]
        if show_y_cursor and non_binary_vars:
            default_y_var = measurement_col if measurement_col in non_binary_vars else non_binary_vars[0]
            y_range_var = st.selectbox("Select Variable for Y Cursor", options=non_binary_vars,
                                       index=non_binary_vars.index(default_y_var))        

        # In col1 we reserve a placeholder to later display the cursor values.
        cursor_info_placeholder = st.empty()
        # (The cursor values will appear in the order: X1, X2, ΔX, Y1, Y2, ΔY)

with col2:
    if csv_file:
        st.subheader("Read CSV")
        st.dataframe(processed_df, height=200)

        st.subheader("Select Zoom Area using Box Select tool:")
        main_fig = go.Figure()
        for idx, col in enumerate(ordered_plot_vars):
            main_fig.add_trace(go.Scatter(
                x=processed_df[selected_Time_column],
                y=processed_df[col],
                mode="lines+markers",
                name=col,
                line=dict(color=px.colors.qualitative.Set1[idx % len(px.colors.qualitative.Set1)])
            ))
        main_fig.update_layout(
            **theme_settings,
            xaxis=dict(title=selected_Time_column),
        )
        event_data = plotly_events(
            main_fig,
            click_event=False,
            hover_event=False,
            select_event=True,
        )

        # Determine x-range from selection events (or use full range)
        if event_data:
            selected_x = [point["x"] for point in event_data]
            unique_selected_x = sorted(set(selected_x))
            if len(unique_selected_x) > 1:
                x_range_min = min(unique_selected_x)
                x_range_max = max(unique_selected_x)
            else:
                x_range_min = processed_df[selected_Time_column].min()
                x_range_max = processed_df[selected_Time_column].max()
        else:
            x_range_min = processed_df[selected_Time_column].min()
            x_range_max = processed_df[selected_Time_column].max()

        # X cursor sliders (using float values formatted to 5 decimal places)
        line1_x_slider = st.slider(
            "Position Cursor X1",
            min_value=float(x_range_min),
            max_value=float(x_range_max),
            value=float(x_range_min)
        )
        line2_x_slider = st.slider(
            "Position Cursor X2",
            min_value=float(x_range_min),
            max_value=float(x_range_max),
            value=float(x_range_max)
        )

        # Y cursor sliders 
        if show_y_cursor and non_binary_vars:
            y_min = float(processed_df[y_range_var].min())
            y_max = float(processed_df[y_range_var].max())
            line_y1_slider = st.slider(
                "Position Cursor Y1",
                min_value=y_min,
                max_value=y_max,
                value=y_min
            )
            line_y2_slider = st.slider(
                "Position Cursor Y2",
                min_value=y_min,
                max_value=y_max,
                value=y_max
            )
        else:
            line_y1_slider, line_y2_slider = None, None

        # Compute cursor deltas
        delta_x = line2_x_slider - line1_x_slider
        if line_y1_slider is not None and line_y2_slider is not None:
            delta_y = line_y2_slider - line_y1_slider
        else:
            delta_y = None

        # Update the left-column placeholder with cursor values (ordered as X1, X2, ΔX, Y1, Y2, ΔY)
        display_str = f"**X1:** {line1_x_slider} &nbsp;&nbsp; **X2:** {line2_x_slider} &nbsp;&nbsp; **ΔX:** {delta_x}"
        if line_y1_slider is not None and line_y2_slider is not None:
            display_str += f" &nbsp;&nbsp; **Y1:** {line_y1_slider} &nbsp;&nbsp; **Y2:** {line_y2_slider} &nbsp;&nbsp; **ΔY:** {delta_y}"
        else:
            display_str += " &nbsp;&nbsp; **Y1:** N/A &nbsp;&nbsp; **Y2:** N/A &nbsp;&nbsp; **ΔY:** N/A"
        cursor_info_placeholder.markdown(display_str)

        # Finally, display the subplots with the selected cursors
        st.plotly_chart(
            plot_subplots(
                processed_df,
                selected_Time_column,
                x_range_min,
                x_range_max,
                theme_settings,
                draw_lines_toggle,
                line1_x_slider,
                line2_x_slider,
                ordered_plot_vars,
                line_y1=line_y1_slider,
                line_y2=line_y2_slider,
            ),
            use_container_width=True,
        )
