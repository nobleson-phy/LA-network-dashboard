"""
Processor for Gaze Activity Data and click data using the integrated dataset
Combines time series plots and network graphs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
from netgraph import Graph
import matplotlib.cm as cm
import re
from datetime import datetime, timedelta
import os
import sys
import argparse
import warnings
import math
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments for the integrated processor"""
    parser = argparse.ArgumentParser(
        description='Process gaze activity data and create both time series plots and network graphs.'
    )
    parser.add_argument('csv_file', help='Path to the CSV file to process (single user data)')
    parser.add_argument('--output-dir', '-o', default='processed_results',
                       help='Base directory to save processed results (default: processed_results)')
    
    # Network graph specific options
    parser.add_argument('--modality', '-m', choices=['combined', 'separate', 'color'], default='separate',
                       help='How to handle modality for network graphs: combined (all together), separate (different graphs), color (different colors)')
    parser.add_argument('--min-duration', '-d', type=float, default=0,
                       help='Minimum duration threshold for network graph nodes (seconds)')
    parser.add_argument('--min-frequency', '-f', type=int, default=2,
                       help='Minimum frequency threshold for network graph edges')
    parser.add_argument('--user-filter', '-u', default='6',
                       help='Filter users by ID prefix (default: "6")')
    
    # Edge representation options
    parser.add_argument('--edge-weight', '-e', choices=['average', 'total'], default='total',
                       help='Edge weight calculation for time-based graphs: average (avg time before transition) or total (total time before transition) (default: average)')
    
    # NEW: Edge representation option
    parser.add_argument('--edge-representation', '-r', choices=['time', 'frequency', 'both'], default='time',
                       help='Edge representation: time (time spent before transition), frequency (edge frequency), or both (create both types) (default: time)')
    
    # Processing control options
    parser.add_argument('--skip-timeseries', action='store_true',
                       help='Skip time series plot generation')
    parser.add_argument('--skip-network', action='store_true',
                       help='Skip network graph generation')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing (use existing preprocessed file in user directory)')
    
    return parser.parse_args()

def create_user_directory_structure(base_output_dir, user_id):
    """
    Create comprehensive directory structure for user's results
    """
    user_dir = os.path.join(base_output_dir, f"user_{user_id}")
    timeseries_dir = os.path.join(user_dir, "timeseries")
    network_dir = os.path.join(user_dir, "network")
    
    # Create all directories if they don't exist
    os.makedirs(timeseries_dir, exist_ok=True)
    os.makedirs(network_dir, exist_ok=True)
    
    return user_dir, timeseries_dir, network_dir

def preprocess_data(csv_file, user_filter='6'):
    """
    Preprocess raw gaze activity CSV file (identical to original preprocessing)
    """
    print(f"\n{'='*80}")
    print(f"PREPROCESSING RAW CSV: {csv_file}")
    print(f"{'='*80}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file, skiprows=1, low_memory=False)
        print(f"Successfully loaded CSV file with {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Define column names
    column_names = [
        'User_ID', 'Timestamp', 'X', 'Y', 'Panel_Title', 'Task_ID', 
        'Activity_ID', 'Screen_ID', 'Activity', 'Task_Type', 'Element_ID', 
        'Element_Type', 'Verb', 'CourseID', 'Duration', 'Attempted'
    ]
    
    # Assign column names
    df.columns = column_names
    
    # Check if User_ID column exists
    if 'User_ID' not in df.columns:
        print("Error: User_ID column not found in the CSV file!")
        sys.exit(1)
    
    # Get the user ID from the data
    user_id = str(df['User_ID'].iloc[0])
    print(f"Processing data for user: {user_id}")
    
    # Check if all User_IDs are the same
    unique_user_ids = df['User_ID'].astype(str).unique()
    if len(unique_user_ids) > 1:
        print(f"Warning: Found multiple User_IDs in file: {unique_user_ids}")
        print(f"Using the first one: {user_id}")
    else:
        print(f"Confirmed single user in file: {user_id}")
    
    # Delete rows that do not start with specified prefix in User_ID
    initial_rows = len(df)
    df = df[df['User_ID'].astype(str).str.startswith(user_filter)]
    print(f"Filtered rows based on User_ID starting with '{user_filter}': {initial_rows} -> {len(df)} rows")
    
    # Create modality column
    df['modality'] = np.where(df['X'].isna() | (df['X'] == ''), 'mclick', 'eTrack')
    
    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M:%S:%f', errors='coerce')
    
    # Sort by Timestamp ascending (initial sort)
    df = df.sort_values('Timestamp')
    
    # Function to clean non-text characters from Panel_Title
    def clean_panel_title(title):
        if pd.isna(title) or title == '':
            return ''
        
        cleaned = re.sub(r'[^\x00-\x7F]+', '', str(title))
        cleaned = re.sub(r'[^\w\s\-()/.,]+', '', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    # Apply cleaning to Panel_Title
    df['Panel_Title'] = df['Panel_Title'].apply(clean_panel_title)
    
    # Fill "outside" for eTrack rows with empty Panel_Title
    print("\nApplying 'outside' fill for eTrack rows with empty Panel_Title...")
    
    df['Original_Panel_Title'] = df['Panel_Title'].copy()
    
    # Ensure X is numeric for comparison
    df['X_numeric'] = pd.to_numeric(df['X'], errors='coerce')
    
    # Apply the fill condition
    fill_condition = (
        df['X_numeric'].notna() &
        (df['X_numeric'] != 0) &
        (df['Panel_Title'] == '')
    )
    
    fill_count = fill_condition.sum()
    print(f"  Found {fill_count} eTrack rows with non-zero X and empty Panel_Title")
    
    df.loc[fill_condition, 'Panel_Title'] = 'outside'
    
    # Additional condition for string X values
    additional_fill_condition = (
        (df['modality'] == 'eTrack') &
        df['X'].notna() &
        (df['X'] != '') &
        (df['Panel_Title'] == '')
    )
    
    additional_fill_count = additional_fill_condition.sum()
    df.loc[additional_fill_condition, 'Panel_Title'] = 'outside'
    
    print(f"  Additional {additional_fill_count} eTrack rows with non-empty X and empty Panel_Title filled")
    print(f"  Total rows filled with 'outside': {fill_count + additional_fill_count}")
    
    # Clean up temporary column
    df = df.drop('X_numeric', axis=1)
    
    # Sort by Timestamp before returning
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    print(f"\nPreprocessing complete:")
    print(f"  First timestamp: {df['Timestamp'].iloc[0]}")
    print(f"  Last timestamp: {df['Timestamp'].iloc[-1]}")
    print(f"  Total rows: {len(df)}")
    
    return df, user_id

def create_timeseries_plots(df, user_id, timeseries_dir):
    """
    Create comprehensive time series plots with all original functionality
    Includes both main time series plots and stacked plots
    """
    print(f"\n{'='*80}")
    print(f"CREATING TIME SERIES PLOTS")
    print(f"{'='*80}")
    
    # Function to filter Panel_Title - keep only when different from previous row
    def filter_panel_titles(group):
        group = group.copy()
        prev_title = None
        for idx, row in group.iterrows():
            current_title = row['Panel_Title']
            if pd.isna(current_title) or current_title == '':
                group.at[idx, 'Panel_Title'] = ''
            elif current_title == prev_title:
                group.at[idx, 'Panel_Title'] = ''
            else:
                prev_title = current_title
        return group
    
    # Apply filtering by modality and Task_ID
    df_filtered = df.groupby(['modality', 'Task_ID']).apply(filter_panel_titles).reset_index(drop=True)
    
    # Sort by Timestamp
    df_filtered = df_filtered.sort_values('Timestamp').reset_index(drop=True)
    
    # Save filtered data
    filtered_csv_path = os.path.join(timeseries_dir, f'filtered_gaze_activity_user_{user_id}.csv')
    df_filtered.to_csv(filtered_csv_path, index=False)
    print(f"Filtered data saved to '{filtered_csv_path}'")
    
    # Read filtered CSV for plotting
    df_plot = pd.read_csv(filtered_csv_path)
    df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'], errors='coerce')
    df_plot = df_plot.sort_values('Timestamp').reset_index(drop=True)
    
    # Get unique elements for plotting
    unique_titles = df_plot['Panel_Title'].dropna().unique()
    unique_titles = [title for title in unique_titles if title != '']
    unique_task_ids = df_plot['Task_ID'].dropna().unique()
    unique_task_ids = [task_id for task_id in unique_task_ids if task_id != '']
    
    # Setup colors and markers (identical to original)
    high_contrast_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
                           '#FF8000', '#8000FF', '#008000', '#800000']
    
    # Task ID colors
    n_task_colors = len(unique_task_ids)
    if n_task_colors <= len(high_contrast_colors):
        task_id_colors = {}
        for i, task_id in enumerate(unique_task_ids):
            color_idx = i % len(high_contrast_colors)
            task_id_colors[task_id] = high_contrast_colors[color_idx]
    else:
        task_cmap = plt.cm.get_cmap('gist_rainbow', max(n_task_colors, 1))
        task_id_colors = {task_id: task_cmap(i % max(n_task_colors, 1)) for i, task_id in enumerate(unique_task_ids)}
    
    # Panel Title markers
    marker_shapes = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', 'H', 'd', 'P', 'X',
                     '8', '1', '2', '3', '4', '+', 'x', '|', '_', '.', ',']
    
    title_markers = {}
    for i, title in enumerate(unique_titles):
        title_markers[title] = marker_shapes[i % len(marker_shapes)]
    
    # Verb markers
    verb_markers = {
        'open': 'o',
        'clicked': 's',
        'timespend': '^',
        'answered': 'v',
        'default': 'D'
    }
    
    # Verb line styles
    verb_line_styles = {
        'open': {'color': '#FF0000', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.7},
        'clicked': {'color': '#00FF00', 'linestyle': '-.', 'linewidth': 1.5, 'alpha': 0.7},
        'timespend': {'color': '#0000FF', 'linestyle': ':', 'linewidth': 2.0, 'alpha': 0.7},
        'answered': {'color': '#FF00FF', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.7},
        'default': {'color': '#333333', 'linestyle': '-', 'linewidth': 1.0, 'alpha': 0.5}
    }
    
    # Create plots by date
    df_plot['Date'] = df_plot['Timestamp'].dt.date
    
    # Function to add x-tick labels for specific verbs (TIME ONLY)
    def add_verb_xticks(ax, verb_events):
        """Add x-tick labels for specific verbs like 'open' and 'answered' - TIME ONLY"""
        # Create a list to store x-tick positions for specific verbs
        xtick_positions = []
        
        for idx, row in verb_events.iterrows():
            verb = row['Verb'].lower()
            
            # Add x-tick for specific verbs
            if verb in ['open', 'answered']:
                x_val = mdates.date2num(row['Timestamp'])
                xtick_positions.append(x_val)
        
        # Add these as extra x-ticks
        if xtick_positions:
            current_ticks = list(ax.get_xticks())
            
            # Combine current ticks with verb ticks
            all_ticks = sorted(set(current_ticks + xtick_positions))
            
            ax.set_xticks(all_ticks)
            # Format all ticks as time only
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=8, ha='right')
    
    # Function to create main time series plot (Panel_Title vs Time)
    def create_main_timeseries_plot(date, date_group, task_id_colors, title_markers, 
                                   verb_markers, verb_line_styles, plots_dir, user_id):
        """Create main time series plot: Panel_Title vs Time"""
        print(f"\nCreating main time series plot for {date}")
        
        # Get data for plotting
        plot_data_main = date_group[
            (date_group['Panel_Title'].notna() & (date_group['Panel_Title'] != '')) |
            (date_group['Verb'].notna() & (date_group['Verb'] != ''))
        ].copy()
        
        if plot_data_main.empty:
            print(f"No events with non-empty Panel_Title OR Verb for main plot on date {date}")
            return
        
        plot_data_main = plot_data_main.sort_values('Timestamp').reset_index(drop=True)
        
        print(f"Main plot will include {len(plot_data_main)} events")
        
        # Calculate dynamic height
        base_height = 10
        legend_rows = len(unique_titles) + len(verb_markers) + 3
        extra_height = legend_rows * 0.3
        fig_height = base_height + min(extra_height, 8)
        
        fig1, ax1 = plt.subplots(figsize=(16, fig_height))
        
        # Create display title
        plot_data_main['Display_Title'] = plot_data_main.apply(
            lambda row: row['Panel_Title'] if pd.notna(row['Panel_Title']) and row['Panel_Title'] != ''
            else f"[Verb: {row['Verb']}]" if pd.notna(row['Verb']) and row['Verb'] != ''
            else '(unknown)',
            axis=1
        )
        
        # Create numeric representation for y-axis
        unique_titles_in_date = plot_data_main['Display_Title'].unique()
        title_to_num = {title: i for i, title in enumerate(unique_titles_in_date)}
        plot_data_main['Title_Num'] = plot_data_main['Display_Title'].map(title_to_num)
        
        # Get verb events for x-ticks
        verb_events = plot_data_main[plot_data_main['Verb'].notna() & (plot_data_main['Verb'] != '')]
        
        # Plot each point
        for idx, row in plot_data_main.iterrows():
            title = row['Panel_Title']
            verb = row['Verb']
            task_id = row['Task_ID']
            modality = row['modality']
            
            # Determine color
            if task_id and pd.notna(task_id) and task_id != '':
                color = task_id_colors.get(task_id, '#000000')
            else:
                color = '#000000'
            
            # Determine marker
            if pd.notna(title) and title != '':
                marker = title_markers.get(title, 'o')
            elif pd.notna(verb) and verb != '':
                marker = verb_markers.get(verb.lower(), verb_markers['default'])
                color = '#333333'
            else:
                marker = 'o'
            
            # Determine fill style based on modality
            if modality == 'mclick':
                facecolor = color
                edgecolor = '#000000'
                linewidth = 1.5
            else:  # eTrack
                facecolor = 'white'
                edgecolor = color
                linewidth = 2.0
            
            # Plot the point
            ax1.scatter(mdates.date2num(row['Timestamp']),
                      row['Title_Num'],
                      color=facecolor, s=100, marker=marker,
                      alpha=0.9, edgecolors=edgecolor, linewidth=linewidth)
        
        # Connect points in chronological order
        sorted_data = plot_data_main.sort_values('Timestamp')
        ax1.plot(mdates.date2num(sorted_data['Timestamp']),
                sorted_data['Title_Num'],
                'k-', alpha=0.5, linewidth=1.0, label='_nolegend_')
        
        # Customize plot
        ax1.set_xlabel('Timestamp', fontsize=20)
        ax1.set_ylabel('Panel Title / Verb', fontsize=20)
        ax1.set_title(f'User {user_id} - Time Series: Panel Title vs Time - {date}',
                     fontsize=24, fontweight='bold')
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        start_time = plot_data_main['Timestamp'].min()
        end_time = plot_data_main['Timestamp'].max()
        buffer = timedelta(seconds=30)
        x_min = mdates.date2num(start_time - buffer)
        x_max = mdates.date2num(end_time + buffer)
        ax1.set_xlim(x_min, x_max)
        
        # Set appropriate x-axis ticks based on total duration
        total_hours = (end_time - start_time).total_seconds() / 3600
        
        if total_hours < 0.5:
            ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        elif total_hours < 2:
            ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        elif total_hours < 6:
            ax1.xaxis.set_major_locator(mdates.HourLocator())
        else:
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Add x-ticks for important verbs (time only)
        add_verb_xticks(ax1, verb_events)
        
        # Set y-ticks
        if len(unique_titles_in_date) > 0:
            ax1.set_yticks(range(len(unique_titles_in_date)))
            ax1.set_yticklabels(unique_titles_in_date, fontsize=14)
            ax1.set_ylim(-0.5, len(unique_titles_in_date) - 0.5)
        
        # Create legend
        legend_items = []
        
        # Get unique Task_IDs in this date's data
        unique_task_ids_in_data = plot_data_main['Task_ID'].dropna().unique()
        unique_task_ids_in_data = [tid for tid in unique_task_ids_in_data if tid != '']
        
        # 1. Task_ID colors
        for task_id in unique_task_ids_in_data[:5]:
            color = task_id_colors.get(task_id, '#000000')
            legend_items.append(Patch(facecolor=color, edgecolor='black', 
                                    label=f'Task_ID: {task_id}'))
        
        # Get unique Panel_Titles in this data
        unique_titles_in_data = plot_data_main['Panel_Title'].dropna().unique()
        unique_titles_in_data = [t for t in unique_titles_in_data if t != '']
        
        # 2. Panel_Title markers
        for title in unique_titles_in_data[:10]:
            marker = title_markers.get(title, 'o')
            legend_items.append(Line2D([0], [0], color='black', marker=marker, 
                                      linestyle='None', markersize=10,
                                      markerfacecolor='black', markeredgecolor='black',
                                      label=title))
        
        # Get unique verbs in this data
        unique_verbs_in_data = plot_data_main['Verb'].dropna().unique()
        unique_verbs_in_data = [v for v in unique_verbs_in_data if v != '']
        
        # 3. Verb markers
        for verb in unique_verbs_in_data[:5]:
            verb_lower = verb.lower()
            if verb_lower in verb_markers:
                marker = verb_markers[verb_lower]
                legend_items.append(Line2D([0], [0], color='#333333', marker=marker, 
                                          linestyle='None', markersize=10,
                                          markerfacecolor='#333333', markeredgecolor='black',
                                          label=f'[Verb: {verb}]'))
            elif len(legend_items) < 25:
                marker = verb_markers['default']
                legend_items.append(Line2D([0], [0], color='#333333', marker=marker, 
                                          linestyle='None', markersize=10,
                                          markerfacecolor='#333333', markeredgecolor='black',
                                          label=f'[Verb: {verb}]'))
        
        # 4. Modality indicators
        legend_items.append(Line2D([0], [0], color='black', marker='o', 
                                  linestyle='None', markersize=10,
                                  markerfacecolor='black', markeredgecolor='black',
                                  label='mclick (filled)'))
        
        legend_items.append(Line2D([0], [0], color='black', marker='o', 
                                  linestyle='None', markersize=10,
                                  markerfacecolor='white', markeredgecolor='black', linewidth=1.5,
                                  label='eTrack (hollow)'))
        
        # 5. Chronological order line
        legend_items.append(Line2D([0], [0], color='black', marker=None, linestyle='-', 
                                  linewidth=1.5, label='Chronological order'))
        
        # Calculate optimal number of columns for legend
        n_items = len(legend_items)
        n_columns = min(4, max(2, int(math.sqrt(n_items))))
        
        # Place legend
        legend_y_offset = -0.02 - (0.008 * (n_items // n_columns))
        ax1.legend(handles=legend_items, loc='upper center', 
                 bbox_to_anchor=(0.5, legend_y_offset), fancybox=True, shadow=True, 
                 ncol=n_columns, fontsize=9)
        
        # Adjust layout
        fig1.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save plot
        plot_filename = os.path.join(plots_dir, f'timeseries_plot_{user_id}_{date}.pdf')
        fig1.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        print(f"Main time series plot saved: {plot_filename}")
    
    def create_stacked_timeseries_plot_before_split(date, date_group, task_id_colors, title_markers, 
                                                   verb_markers, verb_line_styles, plots_dir, user_id):
        """Create a stacked timeseries plot showing all events for the date before splitting into intervals"""
        print(f"\nCreating stacked timeseries plot for {date} (before splitting into intervals)")
        
        # Get data for plotting
        plot_data = date_group[
            (date_group['Panel_Title'].notna() & (date_group['Panel_Title'] != '')) |
            (date_group['Verb'].notna() & (date_group['Verb'] != ''))
        ].copy()
        
        if plot_data.empty:
            print(f"No events with non-empty Panel_Title OR Verb for stacked plot on date {date}")
            return
        
        # Sort by timestamp
        plot_data = plot_data.sort_values('Timestamp').reset_index(drop=True)
        
        print(f"Stacked plot will include {len(plot_data)} events")
        
        # Calculate dynamic height
        base_height = 12
        events_count = len(plot_data)
        extra_height = events_count * 0.03
        fig_height = base_height + min(extra_height, 12)
        
        fig, ax = plt.subplots(figsize=(18, fig_height))
        
        # Create a y-position for each event
        plot_data['Stack_Position'] = range(len(plot_data))
        
        # Get the actual start and end times
        start_time = plot_data['Timestamp'].min()
        end_time = plot_data['Timestamp'].max()
        total_duration = end_time - start_time
        
        print(f"  - Time range: {start_time.time()} to {end_time.time()}")
        print(f"  - Duration: {total_duration}")
        print(f"  - Plot height: {fig_height:.1f} inches")
        
        # ADD VERTICAL LINES FOR EACH VERB EVENT
        verb_events = plot_data[plot_data['Verb'].notna() & (plot_data['Verb'] != '')]
        print(f"  - Total verb events for vertical lines: {len(verb_events)}")
        
        # Draw vertical lines for each verb event
        for idx, row in verb_events.iterrows():
            verb = row['Verb'].lower()
            
            # Get line style for this verb
            if verb in verb_line_styles:
                line_style = verb_line_styles[verb]
            else:
                line_style = verb_line_styles['default']
            
            # Draw vertical line from bottom to top of plot
            x_val = mdates.date2num(row['Timestamp'])
            ax.axvline(x=x_val, 
                      color=line_style['color'], 
                      linestyle=line_style['linestyle'],
                      linewidth=line_style['linewidth'],
                      alpha=line_style['alpha'],
                      label=f'_nolegend_')
        
        # Plot each event with Task_ID colors, appropriate markers, and fill style based on modality
        for idx, row in plot_data.iterrows():
            title = row['Panel_Title']
            verb = row['Verb']
            task_id = row['Task_ID']
            modality = row['modality']
            
            # Determine color: use Task_ID color if available, otherwise use high contrast black
            if task_id and pd.notna(task_id) and task_id != '':
                color = task_id_colors.get(task_id, '#000000')
            else:
                color = '#000000'
            
            # Determine marker
            if pd.notna(title) and title != '':
                marker = title_markers.get(title, 'o')
            elif pd.notna(verb) and verb != '':
                marker = verb_markers.get(verb.lower(), verb_markers['default'])
                color = '#333333'
            else:
                marker = 'o'
            
            # Determine fill style based on modality
            if modality == 'mclick':
                facecolor = color
                edgecolor = '#000000'
                linewidth = 1.5
            else:  # eTrack
                facecolor = 'white'
                edgecolor = color
                linewidth = 2.0
            
            # Plot the point with larger size for better visibility
            ax.scatter(mdates.date2num(row['Timestamp']), 
                      row['Stack_Position'], 
                      color=facecolor, s=120, marker=marker,
                      alpha=0.9, edgecolors=edgecolor, linewidth=linewidth)
        
        # Connect all points in chronological order with a thicker line
        ax.plot(mdates.date2num(plot_data['Timestamp']), 
               plot_data['Stack_Position'], 
               'k-', alpha=0.3, linewidth=1.0, label='_nolegend_')
        
        # Customize plot
        ax.set_xlabel('Timestamp', fontsize=20)
        ax.set_ylabel('Event Sequence Number', fontsize=20)
        ax.set_title(f'User {user_id} - Stacked Time Series: All Events - {date}\n'
                    f'({len(plot_data)} events, all with Panel_Title OR Verb)\n'
                    f'Vertical lines show verb events', 
                    fontsize=24, fontweight='bold', pad=20)
        
        # Format x-axis to show time
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Set x-axis limits with buffer
        buffer = timedelta(seconds=30)
        x_min = mdates.date2num(start_time - buffer)
        x_max = mdates.date2num(end_time + buffer)
        ax.set_xlim(x_min, x_max)
        
        # Set appropriate x-axis ticks based on total duration
        total_hours = total_duration.total_seconds() / 3600
        
        if total_hours < 0.5:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        elif total_hours < 2:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        elif total_hours < 6:
            ax.xaxis.set_major_locator(mdates.HourLocator())
        else:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Add x-ticks for important verbs (time only)
        add_verb_xticks(ax, verb_events)
        
        # Adjust y-axis limits
        y_min = plot_data['Stack_Position'].min() - 1
        y_max = plot_data['Stack_Position'].max() + 1
        ax.set_ylim(y_min, y_max)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.1, linestyle='--', which='both')
        
        # Create comprehensive legend
        legend_items = []
        
        # Get unique Task_IDs in this data
        unique_task_ids_in_data = plot_data['Task_ID'].dropna().unique()
        unique_task_ids_in_data = [tid for tid in unique_task_ids_in_data if tid != '']
        
        # 1. Task_ID colors (limit to first 5)
        for task_id in unique_task_ids_in_data[:5]:
            color = task_id_colors.get(task_id, '#000000')
            legend_items.append(Patch(facecolor=color, edgecolor='black', 
                                    label=f'Task_ID: {task_id}'))
        
        # Get unique Panel_Titles in this data
        unique_titles_in_data = plot_data['Panel_Title'].dropna().unique()
        unique_titles_in_data = [t for t in unique_titles_in_data if t != '']
        
        # 2. Panel_Title markers (limit to first 10)
        for title in unique_titles_in_data[:10]:
            marker = title_markers.get(title, 'o')
            legend_items.append(Line2D([0], [0], color='black', marker=marker, 
                                      linestyle='None', markersize=10,
                                      markerfacecolor='black', markeredgecolor='black',
                                      label=title))
        
        # Get unique verbs in this data
        unique_verbs_in_data = plot_data['Verb'].dropna().unique()
        unique_verbs_in_data = [v for v in unique_verbs_in_data if v != '']
        
        # 3. Verb markers (dark gray markers, limit to first 5)
        for verb in unique_verbs_in_data[:5]:
            verb_lower = verb.lower()
            if verb_lower in verb_markers:
                marker = verb_markers[verb_lower]
                legend_items.append(Line2D([0], [0], color='#333333', marker=marker, 
                                          linestyle='None', markersize=10,
                                          markerfacecolor='#333333', markeredgecolor='black',
                                          label=f'[Verb: {verb}]'))
            elif len(legend_items) < 25:
                marker = verb_markers['default']
                legend_items.append(Line2D([0], [0], color='#333333', marker=marker, 
                                          linestyle='None', markersize=10,
                                          markerfacecolor='#333333', markeredgecolor='black',
                                          label=f'[Verb: {verb}]'))
        
        # 4. Modality indicators
        legend_items.append(Line2D([0], [0], color='black', marker='o', 
                                  linestyle='None', markersize=10,
                                  markerfacecolor='black', markeredgecolor='black',
                                  label='mclick (filled)'))
        
        legend_items.append(Line2D([0], [0], color='black', marker='o', 
                                  linestyle='None', markersize=10,
                                  markerfacecolor='white', markeredgecolor='black', linewidth=1.5,
                                  label='eTrack (hollow)'))
        
        # 5. Chronological order line
        legend_items.append(Line2D([0], [0], color='black', marker=None, linestyle='-', 
                                  linewidth=1.5, label='Chronological order'))
        
        # 6. Add legend items for verb vertical lines
        unique_verbs_for_lines = verb_events['Verb'].str.lower().unique()
        verb_counts = verb_events['Verb'].str.lower().value_counts()
        top_verbs = verb_counts.head(5).index.tolist()
        
        for verb in top_verbs:
            if verb in verb_line_styles:
                line_style = verb_line_styles[verb]
                legend_items.append(Line2D([0], [0], color=line_style['color'], 
                                          marker=None, linestyle=line_style['linestyle'],
                                          linewidth=line_style['linewidth'],
                                          label=f'Verb: {verb} (vertical line)'))
            else:
                line_style = verb_line_styles['default']
                legend_items.append(Line2D([0], [0], color=line_style['color'], 
                                          marker=None, linestyle=line_style['linestyle'],
                                          linewidth=line_style['linewidth'],
                                          label=f'Verb: {verb} (vertical line)'))
        
        # Calculate optimal number of columns for legend
        n_items = len(legend_items)
        n_columns = min(4, max(2, int(math.sqrt(n_items))))
        
        # Place legend at the bottom outside the plot
        legend_y_offset = -0.02 - (0.008 * (n_items // n_columns))
        ax.legend(handles=legend_items, loc='upper center', 
                 bbox_to_anchor=(0.5, legend_y_offset), fancybox=True, shadow=True, 
                 ncol=n_columns, fontsize=12)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the plot
        plot_filename = os.path.join(plots_dir, f'stacked_timeseries_all_events_{user_id}_{date}.pdf')
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Stacked timeseries plot saved: {plot_filename}")
        print(f"  Total events: {len(plot_data)}")
        print(f"  Verb events with vertical lines: {len(verb_events)}")
        print(f"  Task_IDs present: {len(unique_task_ids_in_data)}")
        print(f"  Panel_Titles present: {len(unique_titles_in_data)}")
        print(f"  Verbs present: {len(unique_verbs_in_data)}")
    
    def create_split_stacked_timeseries_plots(date, plot_data, task_id_colors, title_markers, 
                                             verb_markers, verb_line_styles, plots_dir, user_id):
        """Create multiple stacked time series plots, dividing the timeline into intervals"""
        # Sort data by timestamp
        sorted_data = plot_data.sort_values('Timestamp').reset_index(drop=True)
        
        print(f"\nCreating split stacked plots for {date} with {len(sorted_data)} events")
        
        # Get the overall time range for this date
        start_time = sorted_data['Timestamp'].min()
        end_time = sorted_data['Timestamp'].max()
        total_duration = end_time - start_time
        
        # Divide timeline into 10 equal intervals
        n_intervals = 10
        interval_duration = total_duration / n_intervals
        
        # Create intervals
        intervals = []
        for interval_num in range(n_intervals):
            interval_start = start_time + interval_num * interval_duration
            interval_end = start_time + (interval_num + 1) * interval_duration
            
            intervals.append({
                'interval_num': interval_num + 1,
                'start_time': interval_start,
                'end_time': interval_end,
                'duration': interval_duration
            })
        
        # Merge intervals with very few events
        merged_intervals = []
        i = 0
        while i < len(intervals):
            current_interval = intervals[i]
            
            interval_mask = (sorted_data['Timestamp'] >= current_interval['start_time']) & (sorted_data['Timestamp'] <= current_interval['end_time'])
            current_data = sorted_data[interval_mask]
            
            if len(current_data) < 3 and i < len(intervals) - 1:
                next_interval = intervals[i + 1]
                merged_interval = {
                    'interval_num': len(merged_intervals) + 1,
                    'start_time': current_interval['start_time'],
                    'end_time': next_interval['end_time'],
                    'duration': next_interval['end_time'] - current_interval['start_time'],
                    'is_merged': True,
                    'merged_from': [current_interval['interval_num'], next_interval['interval_num']]
                }
                merged_intervals.append(merged_interval)
                i += 2
            else:
                current_interval['is_merged'] = False
                current_interval['merged_from'] = [current_interval['interval_num']]
                merged_intervals.append(current_interval)
                i += 1
        
        final_intervals = merged_intervals
        total_final_intervals = len(final_intervals)
        
        print(f"\nAfter merging small intervals, we have {total_final_intervals} intervals to plot")
        
        for interval_info in final_intervals:
            interval_num = interval_info['interval_num']
            interval_start_time = interval_info['start_time']
            interval_end_time = interval_info['end_time']
            interval_duration = interval_info['duration']
            is_merged = interval_info.get('is_merged', False)
            merged_from = interval_info.get('merged_from', [interval_num])
            
            # Get data for this interval
            interval_mask = (sorted_data['Timestamp'] >= interval_start_time) & (sorted_data['Timestamp'] <= interval_end_time)
            interval_data = sorted_data[interval_mask].copy()
            
            if interval_data.empty:
                continue
            
            # Sort each interval by timestamp
            interval_data = interval_data.sort_values('Timestamp').reset_index(drop=True)
            
            # Calculate dynamic height for this interval plot
            base_height = 12
            unique_titles_in_interval = interval_data['Panel_Title'].dropna().unique()
            unique_titles_in_interval = [t for t in unique_titles_in_interval if t != '']
            unique_verbs_in_interval = interval_data['Verb'].dropna().unique()
            unique_verbs_in_interval = [v for v in unique_verbs_in_interval if v != '']
            unique_task_ids_in_interval = interval_data['Task_ID'].dropna().unique()
            unique_task_ids_in_interval = [tid for tid in unique_task_ids_in_interval if tid != '']
            
            legend_item_count = len(unique_titles_in_interval) + len(unique_verbs_in_interval) + len(unique_task_ids_in_interval) + 3
            extra_height = legend_item_count * 0.25
            fig_height = base_height + min(extra_height, 6)
            
            # Create figure with dynamic size
            fig_width = 18
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Create a y-position for each event (stacked vertically)
            interval_data['Stack_Position'] = range(len(interval_data))
            
            # ADD VERTICAL LINES FOR EACH VERB EVENT IN THIS INTERVAL
            verb_events_in_interval = interval_data[interval_data['Verb'].notna() & (interval_data['Verb'] != '')]
            
            # Draw vertical lines for each verb event
            for idx, row in verb_events_in_interval.iterrows():
                verb = row['Verb'].lower()
                
                if verb in verb_line_styles:
                    line_style = verb_line_styles[verb]
                else:
                    line_style = verb_line_styles['default']
                
                x_val = mdates.date2num(row['Timestamp'])
                ax.axvline(x=x_val, 
                          color=line_style['color'], 
                          linestyle=line_style['linestyle'],
                          linewidth=line_style['linewidth'],
                          alpha=line_style['alpha'],
                          label=f'_nolegend_')
            
            # Plot each event
            for idx, row in interval_data.iterrows():
                title = row['Panel_Title']
                verb = row['Verb']
                task_id = row['Task_ID']
                modality = row['modality']
                
                # Determine color
                if task_id and pd.notna(task_id) and task_id != '':
                    color = task_id_colors.get(task_id, '#000000')
                else:
                    color = '#000000'
                
                # Determine marker
                if pd.notna(title) and title != '':
                    marker = title_markers.get(title, 'o')
                elif pd.notna(verb) and verb != '':
                    verb_lower = verb.lower()
                    marker = verb_markers.get(verb_lower, verb_markers['default'])
                    color = '#333333'
                else:
                    marker = 'o'
                
                # Determine fill style based on modality
                if modality == 'mclick':
                    facecolor = color
                    edgecolor = '#000000'
                    linewidth = 1.5
                else:  # eTrack
                    facecolor = 'white'
                    edgecolor = color
                    linewidth = 2.0
                
                # Plot the point
                ax.scatter(mdates.date2num(row['Timestamp']), 
                          row['Stack_Position'], 
                          color=facecolor, s=120, marker=marker,
                          alpha=0.9, edgecolors=edgecolor, linewidth=linewidth)
            
            # Connect all points in chronological order
            ax.plot(mdates.date2num(interval_data['Timestamp']), 
                   interval_data['Stack_Position'], 
                   'k-', alpha=0.3, linewidth=1.0, label='_nolegend_')
            
            # Customize plot
            events_in_interval = len(interval_data)
            
            # Create title based on whether this is a merged interval
            if is_merged:
                title_text = f'User {user_id} - Stacked Time Series: Interval {interval_num}/{total_final_intervals} - {date}\n'
                title_text += f'Time: {interval_start_time.time()} to {interval_end_time.time()} (Merged intervals {merged_from})\n'
                title_text += f'({events_in_interval} events in this interval)\n'
                title_text += f'Vertical lines show verb events'
            else:
                title_text = f'User {user_id} - Stacked Time Series: Interval {interval_num}/{total_final_intervals} - {date}\n'
                title_text += f'Time: {interval_start_time.time()} to {interval_end_time.time()}\n'
                title_text += f'({events_in_interval} events in this interval)\n'
                title_text += f'Vertical lines show verb events'
            
            ax.set_xlabel('Timestamp', fontsize=20)
            ax.set_ylabel('Event Sequence Number (within interval)', fontsize=20)
            ax.set_title(title_text, fontsize=24, fontweight='bold', pad=20)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            
            # Set x-axis limits
            buffer = timedelta(seconds=15)
            x_min = mdates.date2num(interval_start_time - buffer)
            x_max = mdates.date2num(interval_end_time + buffer)
            ax.set_xlim(x_min, x_max)
            
            # Set appropriate x-axis ticks
            interval_hours = interval_duration.total_seconds() / 3600
            
            if interval_hours < 0.0833:
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
            elif interval_hours < 0.1667:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            elif interval_hours < 0.5:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
            elif interval_hours < 1:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            elif interval_hours < 2:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            elif interval_hours < 4:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            elif interval_hours < 8:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            else:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            
            # Add x-ticks for important verbs (time only)
            add_verb_xticks(ax, verb_events_in_interval)
            
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=9, ha='right')
            
            # Adjust y-axis
            if not interval_data.empty:
                y_min = interval_data['Stack_Position'].min() - 1
                y_max = interval_data['Stack_Position'].max() + 1
                ax.set_ylim(y_min, y_max)
            
            # Add subtle grid
            ax.grid(True, alpha=0.1, linestyle='--', which='both')
            
            # Create comprehensive legend
            legend_items = []
            
            # 1. Task_ID colors
            for task_id in unique_task_ids_in_interval[:5]:
                color = task_id_colors.get(task_id, '#000000')
                legend_items.append(Patch(facecolor=color, edgecolor='black', 
                                        label=f'Task_ID: {task_id}'))
            
            # 2. Panel_Title markers
            for title in unique_titles_in_interval[:10]:
                marker = title_markers.get(title, 'o')
                legend_items.append(Line2D([0], [0], color='black', marker=marker, 
                                          linestyle='None', markersize=10,
                                          markerfacecolor='black', markeredgecolor='black',
                                          label=title))
            
            # 3. Verb markers
            for verb in unique_verbs_in_interval:
                verb_lower = verb.lower()
                if verb_lower in verb_markers:
                    marker = verb_markers[verb_lower]
                    legend_items.append(Line2D([0], [0], color='#333333', marker=marker, 
                                              linestyle='None', markersize=10,
                                              markerfacecolor='#333333', markeredgecolor='black',
                                              label=f'[Verb: {verb}]'))
                elif len(legend_items) < 25:
                    marker = verb_markers['default']
                    legend_items.append(Line2D([0], [0], color='#333333', marker=marker, 
                                              linestyle='None', markersize=10,
                                              markerfacecolor='#333333', markeredgecolor='black',
                                              label=f'[Verb: {verb}]'))
            
            # 4. Modality indicators
            legend_items.append(Line2D([0], [0], color='black', marker='o', 
                                      linestyle='None', markersize=10,
                                      markerfacecolor='black', markeredgecolor='black',
                                      label='mclick (filled)'))
        
            legend_items.append(Line2D([0], [0], color='black', marker='o', 
                                      linestyle='None', markersize=10,
                                      markerfacecolor='white', markeredgecolor='black', linewidth=1.5,
                                      label='eTrack (hollow)'))
            
            # 5. Chronological order line
            legend_items.append(Line2D([0], [0], color='black', marker=None, linestyle='-', 
                                      linewidth=1.5, label='Chronological order'))
            
            # 6. Verb vertical lines legend
            if len(verb_events_in_interval) > 0:
                verb_counts_in_interval = verb_events_in_interval['Verb'].str.lower().value_counts()
                top_verbs_in_interval = verb_counts_in_interval.head(3).index.tolist()
                
                for verb in top_verbs_in_interval:
                    if verb in verb_line_styles:
                        line_style = verb_line_styles[verb]
                        legend_items.append(Line2D([0], [0], color=line_style['color'], 
                                                  marker=None, linestyle=line_style['linestyle'],
                                                  linewidth=line_style['linewidth'],
                                                  label=f'Verb: {verb} (vertical line)'))
                    else:
                        line_style = verb_line_styles['default']
                        legend_items.append(Line2D([0], [0], color=line_style['color'], 
                                                  marker=None, linestyle=line_style['linestyle'],
                                                  linewidth=line_style['linewidth'],
                                                  label=f'Verb: {verb} (vertical line)'))
            
            # Calculate optimal number of columns for legend
            n_items = len(legend_items)
            n_columns = min(4, max(2, int(math.sqrt(n_items))))
            
            # Place legend
            legend_y_offset = -0.02 - (0.008 * (n_items // n_columns))
            ax.legend(handles=legend_items, loc='upper center', 
                     bbox_to_anchor=(0.5, legend_y_offset), fancybox=True, shadow=True, 
                     ncol=n_columns, fontsize=12)
            
            # Adjust layout
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            # Save the plot
            plot_filename = os.path.join(plots_dir, f'stacked_timeseries_{user_id}_{date}_interval_{interval_num}_of_{total_final_intervals}.pdf')
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"\nInterval {interval_num}/{total_final_intervals} saved: {plot_filename}")
            print(f"  Time range: {interval_start_time.time()} to {interval_end_time.time()}")
            print(f"  Events in this interval: {len(interval_data)}")
    
    # Process each date
    timeseries_created = 0
    for date, date_group in df_plot.groupby('Date'):
        if date_group.empty:
            continue
            
        print(f"\nProcessing date: {date}")
        
        # Create main time series plot
        create_main_timeseries_plot(date, date_group, task_id_colors, title_markers, 
                                   verb_markers, verb_line_styles, timeseries_dir, user_id)
        
        # Create stacked timeseries plot before splitting
        create_stacked_timeseries_plot_before_split(date, date_group, task_id_colors, title_markers, 
                                                   verb_markers, verb_line_styles, timeseries_dir, user_id)
        
        # Get data for split plots
        stacked_plot_data = date_group[
            (date_group['Panel_Title'].notna() & (date_group['Panel_Title'] != '')) |
            (date_group['Verb'].notna() & (date_group['Verb'] != ''))
        ].copy()
        
        if not stacked_plot_data.empty:
            create_split_stacked_timeseries_plots(date, stacked_plot_data, task_id_colors, title_markers, 
                                                 verb_markers, verb_line_styles, timeseries_dir, user_id)
            timeseries_created += 1
    
    print(f"\nTime series plots completed.")
    print(f"Created plots for {timeseries_created} dates")
    print(f"Saved to: {timeseries_dir}")
    
    return filtered_csv_path, timeseries_created

def create_network_graphs(df, user_id, network_dir, modality_handling='combined',
                         min_duration=0, min_frequency=1, edge_weight_method='average',
                         edge_representation='time'):
    """
    Create network graphs with FIXED node positions and sizes
    All nodes from master layout, grayed out when not present in task
    Fixed node size = 10 (not scaling)
    Nodes colored by FIXED CLASSIFICATION (picture/text/instruction/outside)
    Fixed positions: outside at 12 o'clock, text at 6 o'clock, others in between
    EDGE REPRESENTATION: Can be time (width proportional to average/total time spent before transition)
                        OR frequency (width proportional to edge frequency)
    """
    print(f"\n{'='*80}")
    print(f"CREATING NETWORK GRAPHS WITH FIXED LAYOUT")
    print(f"Modality handling: {modality_handling}")
    print(f"Node coloring: Fixed classification (picture/text/instruction/outside)")
    print(f"Node arrangement: Fixed positions (outside at 12, text at 6)")
    print(f"Edge representation: {edge_representation}")
    if edge_representation == 'time':
        print(f"Edge weight calculation: {edge_weight_method} time")
    elif edge_representation == 'frequency':
        print(f"Edge width: Proportional to transition frequency")
    else:  # both
        print(f"Creating both time-based and frequency-based graphs")
    print(f"{'='*80}")
    
    print(f"Creating network graphs for user: {user_id}")
    print(f"Minimum duration threshold: {min_duration} seconds")
    print(f"Minimum frequency threshold: {min_frequency}")
    print(f"Edge weight method: {edge_weight_method}")
    print(f"Edge representation: {edge_representation}")
    
    # Filter out rows with empty Panel_Title
    initial_rows = len(df)
    df_network = df[df['Panel_Title'].notna() & (df['Panel_Title'] != '')].copy()
    print(f"Rows with non-empty Panel_Title: {initial_rows} -> {len(df_network)}")
    
    # FIXED CLASSIFICATION DICTIONARY BASED ON YOUR LIST
    FIXED_CLASSIFICATION = {
        # Picture elements
        'Mobile phones breakup details': 'picture',
        'Mobile phones for teaching': 'picture',
        'Desktop breakup details': 'picture',
        'Laptop breakup details': 'picture',
        'Laptop / Notebook': 'picture',
        'Distribution of schools': 'picture',
        'Dropout - secondary': 'picture',
        'Desktop': 'picture',
        'Projector': 'picture',
        'Digital Library': 'picture',
        'Pupil-teacher ratio (PTR)': 'picture',
        'Dropout - middle': 'picture',
        'Dropout - preparatory': 'picture',
        'Distribution of teachers': 'picture',
        'ICT labs': 'picture',
        'Infrastructure': 'picture',
        'Distribution of students': 'picture',
        'Tablet breakup details': 'picture',
        'Tablet': 'picture',
        
        # Text elements
        '1st chart in the report': 'text',
        '2nd chart in the report': 'text',
        '3rd chart in the report': 'text',
        
        # Instruction element
        'Ordering three charts as evidence': 'instruction',
        
        # Special element
        'outside': 'outside'
    }
    
    # FIXED COLOR SCHEME
    FIXED_COLORS = {
        'picture': '#2ca02c',      # Green for picture elements
        'text': '#C227F5',         # Violet for text elements
        'instruction': '#FFA500',  # Orange for instruction
        'outside': '#cccccc',      # light gray for outside
        'other': '#7f7f7f'         # Gray for other/unknown elements
    }
    
    def calculate_time_spent(task_data):
        """Calculate total time spent on each Panel_Title"""
        task_sorted = task_data.sort_values('Timestamp').reset_index(drop=True)
        time_spent = {}
        
        for i in range(len(task_sorted) - 1):
            current_row = task_sorted.iloc[i]
            next_row = task_sorted.iloc[i + 1]
            
            panel_title = current_row['Panel_Title']
            if pd.isna(panel_title) or panel_title == '':
                continue
                
            time_diff = (next_row['Timestamp'] - current_row['Timestamp']).total_seconds()
            if time_diff > 0:
                if panel_title not in time_spent:
                    time_spent[panel_title] = 0
                time_spent[panel_title] += time_diff
        
        return time_spent

    def calculate_edge_weights(task_data, edge_weight_method='average'):
        """
        Calculate edge frequencies and time weights based on selected method
        Returns: (edge_frequencies, edge_time_weights)
        edge_time_weights can be either average or total time depending on edge_weight_method
        """
        task_sorted = task_data.sort_values('Timestamp').reset_index(drop=True)
        edge_frequencies = {}  # For storing edge frequencies (for filtering)
        edge_times = {}        # For storing total time spent before transitions
        edge_counts = {}       # For storing count of transitions
        
        for i in range(len(task_sorted) - 1):
            current_row = task_sorted.iloc[i]
            next_row = task_sorted.iloc[i + 1]
            
            current_panel = current_row['Panel_Title']
            next_panel = next_row['Panel_Title']
            
            # Skip if either panel title is empty/NaN
            if (pd.isna(current_panel) or current_panel == '' or
                pd.isna(next_panel) or next_panel == ''):
                continue
            
            # Skip self-transitions (element to itself)
            if current_panel == next_panel:
                continue

            # EXCLUDE EDGES TO/FROM 'outside'
            if current_panel == 'outside' or next_panel == 'outside':
                continue
                
            edge_key = tuple(sorted([current_panel, next_panel]))
            
            # Calculate time difference
            time_diff = (next_row['Timestamp'] - current_row['Timestamp']).total_seconds()
            
            # Store frequency and time
            if edge_key not in edge_times:
                edge_times[edge_key] = 0
                edge_counts[edge_key] = 0
                edge_frequencies[edge_key] = 0
            
            edge_times[edge_key] += time_diff
            edge_counts[edge_key] += 1
            edge_frequencies[edge_key] += 1
        
        # Calculate time weights based on selected method
        edge_time_weights = {}
        for edge_key in edge_times.keys():
            if edge_weight_method == 'average':
                # Calculate average time for each edge
                if edge_counts[edge_key] > 0:
                    edge_time_weights[edge_key] = edge_times[edge_key] / edge_counts[edge_key]
                else:
                    edge_time_weights[edge_key] = 0
            elif edge_weight_method == 'total':
                # Use total time for each edge
                edge_time_weights[edge_key] = edge_times[edge_key]
        
        return edge_frequencies, edge_time_weights

    # Create master layout with FIXED positions in circular layout
    def create_master_layout(df_all):
        print("\nCreating master layout with FIXED positions in circular layout...")
        print("Outside at 12 o'clock, Text at 6 o'clock, Pictures in circular arrangement...")
        
        # Get all unique Panel_Titles from the entire dataset
        all_panels = sorted(df_all['Panel_Title'].dropna().unique())
        
        # Classify each panel
        panel_classification = {}
        classified_panels = {}
        
        for panel in all_panels:
            # Use fixed classification if available, otherwise classify as 'other'
            if panel in FIXED_CLASSIFICATION:
                classification = FIXED_CLASSIFICATION[panel]
            else:
                # Try to infer based on content
                panel_lower = panel.lower()
                if any(keyword in panel_lower for keyword in ['chart', 'text', 'report']):
                    classification = 'text'
                elif any(keyword in panel_lower for keyword in ['instruction', 'order', 'evidence']):
                    classification = 'instruction'
                elif panel == 'outside':
                    classification = 'outside'
                else:
                    classification = 'other'
            
            panel_classification[panel] = classification
            
            if classification not in classified_panels:
                classified_panels[classification] = []
            classified_panels[classification].append(panel)
        
        print(f"  Found {len(all_panels)} unique Panel_Titles")
        print(f"  Classification breakdown:")
        for cls in sorted(classified_panels.keys()):
            print(f"    {cls}: {len(classified_panels[cls])} panels (color: {FIXED_COLORS.get(cls, '#cccccc')})")
        
        # Create initial circular layout using networkx
        temp_G = nx.Graph()
        for panel in all_panels:
            temp_G.add_node(panel)
        
        # Get basic circular layout from networkx
        base_pos = nx.circular_layout(temp_G, scale=1.0)
        
        # Now we need to rotate the circular layout so that specific nodes are at specific positions
        # We want: outside at 12 o'clock (angle = -90 degrees or -π/2 radians)
        #          text elements at 6 o'clock (angle = 90 degrees or π/2 radians)
        
        # Convert positions to polar coordinates for rotation
        import math
        
        # Find the current angle of 'outside' in the circular layout
        outside_x, outside_y = 0, 0
        if 'outside' in base_pos:
            outside_x, outside_y = base_pos['outside']
        else:
            # If 'outside' doesn't exist, use the top-most node
            top_node = min(base_pos.items(), key=lambda x: x[1][1])  # Node with smallest y (top)
            outside_x, outside_y = top_node[1]
        
        # Calculate current angle of 'outside' (atan2 returns angle in radians from x-axis)
        current_angle = math.atan2(outside_y, outside_x)
        # We want 'outside' at -π/2 (12 o'clock)
        target_angle = -math.pi/2
        rotation_angle = target_angle - current_angle
        
        # Apply rotation to ALL nodes to keep the circular structure
        master_pos = {}
        for panel, (x, y) in base_pos.items():
            # Convert to polar, rotate, convert back to cartesian
            r = math.sqrt(x*x + y*y)
            angle = math.atan2(y, x) + rotation_angle
            master_pos[panel] = (r * math.cos(angle), r * math.sin(angle))
        
        # Now ensure text elements are near 6 o'clock (π/2 radians)
        # Find all text elements
        text_elements = classified_panels.get('text', [])
        # Ensure our 3 text elements are included
        for text_elem in ['1st chart in the report', '2nd chart in the report', '3rd chart in the report']:
            if text_elem in all_panels and text_elem not in text_elements:
                text_elements.append(text_elem)
        
        if text_elements:
            # Find the text element closest to 6 o'clock
            text_positions = {panel: master_pos[panel] for panel in text_elements if panel in master_pos}
            
            # Calculate angles of text elements
            text_angles = {}
            for panel, (x, y) in text_positions.items():
                angle = math.atan2(y, x)
                text_angles[panel] = angle
            
            # Target angle for text elements (6 o'clock = π/2 radians)
            text_target_angle = math.pi/2
            
            # Find the average angle of text elements
            if text_angles:
                avg_text_angle = sum(text_angles.values()) / len(text_angles)
                # Calculate how much we need to rotate to get text at 6 o'clock
                text_rotation_angle = text_target_angle - avg_text_angle
                
                # Apply additional rotation for text alignment
                # But only rotate half as much to not disrupt outside position too much
                partial_rotation = text_rotation_angle * 0.5
                
                for panel in master_pos:
                    x, y = master_pos[panel]
                    r = math.sqrt(x*x + y*y)
                    angle = math.atan2(y, x) + partial_rotation
                    master_pos[panel] = (r * math.cos(angle), r * math.sin(angle))
        
        # For better visualization, let's sort panels by classification
        # and ensure similar classifications are grouped together in the circle
        panel_order = []
        classification_order = ['outside', 'text', 'instruction', 'picture', 'other']
        
        # Count panels in each classification
        cls_counts = {}
        for cls in classification_order:
            panels_in_cls = [p for p in all_panels if panel_classification.get(p) == cls]
            cls_counts[cls] = len(panels_in_cls)
        
        # Create positions that group similar classifications
        # We'll create a new circular layout with grouped nodes
        grouped_panels = []
        for cls in classification_order:
            panels_in_cls = sorted([p for p in all_panels if panel_classification.get(p) == cls])
            grouped_panels.extend(panels_in_cls)
        
        # Create new graph with grouped panels
        grouped_G = nx.Graph()
        for panel in grouped_panels:
            grouped_G.add_node(panel)
        
        # Get circular layout for grouped nodes
        grouped_pos = nx.circular_layout(grouped_G, scale=1.0)
        
        # Rotate to put 'outside' at 12 o'clock
        if 'outside' in grouped_pos:
            outside_x, outside_y = grouped_pos['outside']
            current_angle = math.atan2(outside_y, outside_x)
            target_angle = -math.pi/2
            rotation_angle = target_angle - current_angle
            
            for panel in grouped_panels:
                x, y = grouped_pos[panel]
                r = math.sqrt(x*x + y*y)
                angle = math.atan2(y, x) + rotation_angle
                grouped_pos[panel] = (r * math.cos(angle), r * math.sin(angle))
        
        # Ensure text elements are near 6 o'clock
        text_elements_in_group = [p for p in grouped_panels if panel_classification.get(p) == 'text']
        if text_elements_in_group:
            # Calculate center of text elements
            text_x = sum(grouped_pos[p][0] for p in text_elements_in_group) / len(text_elements_in_group)
            text_y = sum(grouped_pos[p][1] for p in text_elements_in_group) / len(text_elements_in_group)
            current_text_angle = math.atan2(text_y, text_x)
            target_text_angle = math.pi/2  # 6 o'clock
            
            # Calculate rotation needed
            text_rotation = target_text_angle - current_text_angle
            
            # Apply rotation (partial to maintain outside at top)
            partial_text_rotation = text_rotation * 0.3
            
            for panel in grouped_panels:
                x, y = grouped_pos[panel]
                r = math.sqrt(x*x + y*y)
                angle = math.atan2(y, x) + partial_text_rotation
                grouped_pos[panel] = (r * math.cos(angle), r * math.sin(angle))
        
        # Use the grouped positions as master positions
        master_pos = grouped_pos
        
        print(f"  Circular layout created with {len(master_pos)} node positions")
        print(f"  Node arrangement in circle:")
        print(f"    • 'outside' at 12 o'clock (top)")
        print(f"    • text elements near 6 o'clock (bottom)")
        print(f"    • instruction elements grouped together")
        print(f"    • picture elements grouped together")
        print(f"    • similar classifications are adjacent in the circle")
        
        # Create ordered list of panels for consistent layout
        panel_order = grouped_panels
        
        return master_pos, panel_order, panel_classification

    # Create a single network graph
    def create_single_network_graph(task_id, task_data, time_spent, edge_frequencies, edge_time_weights, 
                                  output_dir, user_id, master_pos, all_panel_titles, panel_classification,
                                  graph_type='combined',
                                  modality=None, modality_handling='combined',
                                  edge_weight_method='average',
                                  edge_representation='time'):
        
        print(f"\nCreating {graph_type} network graph for Task_ID: {task_id}" + 
              (f", Modality: {modality}" if modality else ""))
        print(f"  Number of unique Panel_Titles in this {'modality' if modality else 'task'}: {len(time_spent)}")
        print(f"  Number of edges in this {'modality' if modality else 'task'}: {len(edge_frequencies)}")
        print(f"  Total nodes in master layout: {len(all_panel_titles)}")
        print(f"  Edge representation: {edge_representation}")
        if edge_representation == 'time':
            print(f"  Edge weight method: {edge_weight_method}")
        
        # Filter nodes by minimum duration
        filtered_time_spent = {k: v for k, v in time_spent.items() if v >= min_duration}
        if len(filtered_time_spent) == 0:
            print(f"  No nodes meet minimum duration threshold of {min_duration} seconds. Skipping...")
            return None
        
        # Filter edges by minimum frequency and nodes by duration
        filtered_edge_frequencies = {}
        filtered_edge_time_weights = {}
        
        for edge_key, frequency in edge_frequencies.items():
            node1, node2 = edge_key
            if (frequency >= min_frequency and
                node1 in filtered_time_spent and
                node2 in filtered_time_spent):
                filtered_edge_frequencies[edge_key] = frequency
                filtered_edge_time_weights[edge_key] = edge_time_weights[edge_key]
        
        print(f"  After filtering - Active nodes: {len(filtered_time_spent)}, Edges: {len(filtered_edge_frequencies)}")
        
        # NEW: Collect "Attempted" values for text nodes with verb "answered"
        attempted_values = {}
        text_nodes = ['1st chart in the report', '2nd chart in the report', '3rd chart in the report']
        
        for text_node in text_nodes:
            if text_node in task_data['Panel_Title'].values:
                # Find rows for this text node with verb "answered"
                answered_rows = task_data[(task_data['Panel_Title'] == text_node) & 
                                         (task_data['Verb'].str.lower() == 'answered')]
                
                if not answered_rows.empty:
                    # Take the first "answered" occurrence for this node
                    attempted_val = answered_rows.iloc[0]['Attempted']
                    if pd.notna(attempted_val):
                        attempted_values[text_node] = str(attempted_val)
                        print(f"  Found 'answered' for {text_node}: Attempted = {attempted_val}")
        
        # Create graph with ALL nodes from master layout
        G = nx.Graph()
        
        # Add ALL nodes from master layout
        for panel_title in all_panel_titles:
            classification = panel_classification.get(panel_title, 'other')
            if panel_title in filtered_time_spent:
                # Active node - has time spent data
                duration = filtered_time_spent[panel_title]
                G.add_node(panel_title, duration=duration, active=True, 
                          classification=classification)
            else:
                # Inactive node - exists in master layout but not in this task/modality
                # DO NOT FILL WITH COLOR - keep hollow/transparent
                G.add_node(panel_title, duration=0, active=False, 
                          classification=classification)
        
        # Add edges with frequency and time weights as attributes
        for edge_key, frequency in filtered_edge_frequencies.items():
            node1, node2 = edge_key
            if node1 in G.nodes() and node2 in G.nodes():
                time_weight = filtered_edge_time_weights[edge_key]
                G.add_edge(node1, node2, frequency=frequency, time_weight=time_weight)
        
        # Use master layout positions for ALL nodes
        pos = master_pos
        
        # Prepare node sizes and colors: FIXED SIZE = 10, COLOR BASED ON CLASSIFICATION
        node_sizes_dict = {}
        node_colors_dict = {}
        node_edge_colors_dict = {}
        node_alpha_dict = {}
        
        for node in G.nodes():
            classification = G.nodes[node].get('classification', 'other')
            classification_color = FIXED_COLORS.get(classification, '#cccccc')
            
            if G.nodes[node].get('active', False):
                # Active node - use classification color with full opacity
                node_sizes_dict[node] = 10  # FIXED SIZE
                node_colors_dict[node] = classification_color
                node_edge_colors_dict[node] = classification_color  # edge for active nodes
                node_alpha_dict[node] = 1.0
            else:
                # Inactive node - HOLLOW with classification border color
                node_sizes_dict[node] = 10  # FIXED SIZE (same size)
                node_colors_dict[node] = 'none'  # No fill color
                node_edge_colors_dict[node] = classification_color  # Border color matches classification
                node_alpha_dict[node] = 0.5  # Semi-transparent
        
        # Prepare edge widths based on selected edge representation
        edge_widths_dict = {}
        edge_colors_dict = {}
        edge_alpha_dict = {}
        
        # Define the special nodes that should have Violet connecting edges
        SPECIAL_NODES = {
            '2nd chart in the report',
            '3rd chart in the report', 
            '1st chart in the report'
        }
        
        # Choose which weights to use based on edge representation
        if edge_representation == 'frequency':
            edge_weights = filtered_edge_frequencies
        else:  # time
            edge_weights = filtered_edge_time_weights
        
        if edge_weights:
            weights_list = list(edge_weights.values())
            min_weight, max_weight = min(weights_list), max(weights_list)
            
            if max_weight > min_weight:
                for edge in G.edges():
                    # Get weight for this edge (sorted tuple)
                    edge_key = tuple(sorted(edge))
                    weight = edge_weights.get(edge_key, 0)
                    if weight > 0:
                        # Scale edge width based on weight
                        edge_widths_dict[edge] = 0.5 + 4.5 * (weight - min_weight) / (max_weight - min_weight)
                        
                        # Check if edge connects to any special node
                        if edge[0] in SPECIAL_NODES or edge[1] in SPECIAL_NODES:
                            # Edge connects to a special node - use Violet color
                            edge_colors_dict[edge] = '#C227F5'  # Violet
                        else:
                            # Regular edge - use gray color
                            edge_colors_dict[edge] = '#cccccc'
                            
                        edge_alpha_dict[edge] = 0.7
                    else:
                        # No edge in this task/modality
                        edge_widths_dict[edge] = 0
                        edge_colors_dict[edge] = '#cccccc'  # Light gray for no edges
                        edge_alpha_dict[edge] = 0
            else:
                for edge in G.edges():
                    edge_key = tuple(sorted(edge))
                    weight = edge_weights.get(edge_key, 0)
                    if weight > 0:
                        edge_widths_dict[edge] = 2.5  # Default width for edges with same weight
                        
                        # Check if edge connects to any special node
                        if edge[0] in SPECIAL_NODES or edge[1] in SPECIAL_NODES:
                            # Edge connects to a special node - use Violet color
                            edge_colors_dict[edge] = '#C227F5'  # Violet
                        else:
                            # Regular edge - use gray color
                            edge_colors_dict[edge] = '#666666'
                            
                        edge_alpha_dict[edge] = 0.7
                    else:
                        edge_widths_dict[edge] = 0
                        edge_colors_dict[edge] = '#cccccc'
                        edge_alpha_dict[edge] = 0
        else:
            # No edges at all
            for edge in G.edges():
                edge_widths_dict[edge] = 0
                edge_colors_dict[edge] = '#cccccc'
                edge_alpha_dict[edge] = 0
        
        # Create edge labels dictionary - show appropriate label based on representation
        edge_labels_dict = {}
        for edge in G.edges():
            edge_key = tuple(sorted(edge))
            if edge_key in filtered_edge_frequencies:
                frequency = filtered_edge_frequencies[edge_key]
                if edge_representation == 'frequency':
                    edge_labels_dict[edge] = f"{frequency}"  # Just show frequency count
                else:  # time
                    time_weight = filtered_edge_time_weights.get(edge_key, 0)
                    if edge_weight_method == 'average':
                        edge_labels_dict[edge] = f"{time_weight:.1f}s"  # Format as seconds with 1 decimal
                    else:  # total
                        edge_labels_dict[edge] = f"{time_weight:.1f}s"  # Format as seconds with 1 decimal
        
        # Create figure with adjusted size for better spacing
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Draw graph with netgraph
        plot = Graph(G,
                    node_layout=pos,
                    node_size=node_sizes_dict,
                    node_color=node_colors_dict,
                    node_edge_width=1.5,
                    node_edge_color=node_edge_colors_dict,
                    node_alpha=node_alpha_dict,
                    edge_width=edge_widths_dict,
                    edge_color=edge_colors_dict,
                    edge_alpha=edge_alpha_dict,
                    node_labels=True,
                    node_label_fontdict=dict(size=9, weight='bold'),
                    edge_labels=edge_labels_dict,
                    edge_label_fontdict=dict(size=8),
                    ax=ax)
        
        # NEW: Add "Attempted" value boxes for text nodes
        for text_node in text_nodes:
            if text_node in attempted_values and text_node in pos:
                attempted_val = attempted_values[text_node]
                x, y = pos[text_node]
                
                # Calculate offset position (right of the node)
                offset_x = 0.15  # Offset in x-direction
                offset_y = 0.05   # Offset in y-direction
                
                # Create a text box with the attempted value
                box_text = f"Attempted: {attempted_val}"
                box_props = dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                edgecolor='gold', alpha=0.9, linewidth=1.5)
                
                # Add the text box
                ax.text(x + offset_x, y + offset_y, box_text, 
                       transform=ax.transData, fontsize=8, fontweight='bold',
                       verticalalignment='center', bbox=box_props)
                
                # Add a connecting line from node to box (optional)
                ax.plot([x, x + offset_x * 0.7], [y, y + offset_y], 
                       'k-', alpha=0.3, linewidth=0.8, linestyle='--')
        
        # Add title based on edge representation
        if modality:
            title = f'Network Graph - User {user_id}, Task_ID: {task_id}, Modality: {modality}\n'
        else:
            title = f'Network Graph - User {user_id}, Task_ID: {task_id}\n'
        
        if edge_representation == 'frequency':
            title += f'Nodes colored by fixed classification, fixed positions\n'
            title += f'Active Nodes: {len(filtered_time_spent)}/{len(all_panel_titles)} (colored), '
            title += f'Inactive Nodes: {len(all_panel_titles) - len(filtered_time_spent)} (hollow with category border), '
            title += f'Edges: {len(filtered_edge_frequencies)} (width ∝ transition frequency)'
        else:  # time
            time_label = "avg. time" if edge_weight_method == 'average' else "total time"
            title += f'Nodes colored by fixed classification, fixed positions\n'
            title += f'Active Nodes: {len(filtered_time_spent)}/{len(all_panel_titles)} (colored), '
            title += f'Inactive Nodes: {len(all_panel_titles) - len(filtered_time_spent)} (hollow with category border), '
            title += f'Edges: {len(filtered_edge_frequencies)} (width ∝ {time_label} before transition)'
        
        # Add note about attempted values if any exist
        if attempted_values:
            title += f'\nText nodes with "answered" verb show Attempted value in yellow box'
        
        ax.set_title(title, fontsize=24, fontweight='bold', pad=25)
        
        # Set axis limits to give more space for boxes
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        ax.set_xlim(x_lim[0] - 0.2 * x_range, x_lim[1] + 0.2 * x_range)  # Increased for boxes
        ax.set_ylim(y_lim[0] - 0.2 * y_range, y_lim[1] + 0.2 * y_range)
        
        # Add legend
        legend_elements = []
        
        # Add classification color patches
        classification_counts = {}
        classification_durations = {}
        for node in G.nodes():
            if G.nodes[node].get('active', False):
                classification = G.nodes[node].get('classification', 'other')
                classification_counts[classification] = classification_counts.get(classification, 0) + 1
                classification_durations[classification] = classification_durations.get(classification, 0) + G.nodes[node].get('duration', 0)
        
        # Show all classifications in fixed order
        classification_order = ['outside', 'text', 'instruction', 'picture', 'other']
        for cls in classification_order:
            if cls in classification_counts:
                color = FIXED_COLORS.get(cls, '#cccccc')
                count = classification_counts[cls]
                avg_duration = classification_durations[cls] / count if count > 0 else 0
                legend_elements.append(Patch(facecolor=color, edgecolor='black',
                                          label=f'{cls}: {count} nodes ({avg_duration:.1f}s avg)'))
        
        # Add inactive nodes legend (hollow with colored border)
        # Use 'picture' color as an example for the legend
        example_classification_color = FIXED_COLORS.get('picture', '#cccccc')
        legend_elements.append(Line2D([0], [0], color='black', marker='o', 
                                    linestyle='None', markersize=10,
                                    markerfacecolor='none', markeredgecolor=example_classification_color, 
                                    markeredgewidth=1.5, alpha=0.7,
                                    label=f'Inactive node (hollow, border = classification color)'))
        
        # Add edge legend based on representation
        if edge_representation == 'frequency':
            legend_elements.append(Line2D([0], [0], color='#666666', linewidth=2.5,
                       label=f'Transition (Width ∝ Frequency)'))
        else:  # time
            time_label_legend = "Avg. Time" if edge_weight_method == 'average' else "Total Time"
            legend_elements.append(Line2D([0], [0], color='#666666', linewidth=2.5,
                       label=f'Transition (Width ∝ {time_label_legend})'))
        
        # Add attempted value legend if applicable
        if attempted_values:
            legend_elements.append(Patch(facecolor='lightyellow', edgecolor='gold',
                                       label=f'Attempted value (when verb="answered")'))
        
        # Create legend
        #legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9,
        #                  bbox_to_anchor=(0.98, 0.98), borderaxespad=0.5)
        #legend.set_title('Classification Legend', prop={'size': 14, 'weight': 'bold'})
        
        # # Add stats text box
        # stats_text = f'Statistics for Task {task_id}'
        # if modality:
        #     stats_text += f' ({modality})'
        # stats_text += ':\n'
        # stats_text += f'Total time tracked: {sum(filtered_time_spent.values()):.1f}s\n'
        # stats_text += f'Avg time per element: {np.mean(list(filtered_time_spent.values())):.1f}s\n'
        # stats_text += f'Max time on element: {max(filtered_time_spent.values()):.1f}s\n'
        # stats_text += f'Total transitions: {sum(filtered_edge_frequencies.values())}\n'
        
        # if edge_representation == 'frequency':
        #     if filtered_edge_frequencies:
        #         stats_text += f'Avg transitions per edge: {np.mean(list(filtered_edge_frequencies.values())):.1f}\n'
        #         stats_text += f'Max transitions on edge: {max(filtered_edge_frequencies.values())}\n'
        #     else:
        #         stats_text += f'Avg transitions per edge: 0.0\n'
        # else:  # time
        #     if filtered_edge_time_weights:
        #         time_label_stats = "Average" if edge_weight_method == 'average' else "Total"
        #         stats_text += f'{time_label_stats} time before transition: {np.mean(list(filtered_edge_time_weights.values())):.1f}s\n'
        #     else:
        #         stats_text += f'Time before transition: 0.0s\n'
        
        # # Add attempted values to stats
        # if attempted_values:
        #     stats_text += f'\nAttempted values for text nodes:\n'
        #     for text_node in text_nodes:
        #         if text_node in attempted_values:
        #             stats_text += f'  {text_node}: {attempted_values[text_node]}\n'
        
        # stats_text += f'\nClassification in this {"modality" if modality else "task"}:\n'
        # for cls in classification_order:
        #     if cls in classification_counts:
        #         count = classification_counts[cls]
        #         total_time = classification_durations.get(cls, 0)
        #         stats_text += f'{cls}: {count} nodes ({total_time:.1f}s total)\n'
        
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='brown', linewidth=1.5)
        # ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=11,
        #         verticalalignment='bottom', bbox=props)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        plt.tight_layout()
        
        # Save the plot with appropriate filename based on edge representation
        if modality:
            if edge_representation == 'frequency':
                plot_filename = os.path.join(output_dir, f'network_user_{user_id}_task_{task_id}_{modality}_frequency.pdf')
            else:
                plot_filename = os.path.join(output_dir, f'network_user_{user_id}_task_{task_id}_{modality}_{edge_weight_method}.pdf')
        else:
            modality_suffix = f"_{modality_handling}" if modality_handling != 'combined' else ""
            if edge_representation == 'frequency':
                plot_filename = os.path.join(output_dir, f'network_user_{user_id}_task_{task_id}{modality_suffix}_frequency.pdf')
            else:
                plot_filename = os.path.join(output_dir, f'network_user_{user_id}_task_{task_id}{modality_suffix}_{edge_weight_method}.pdf')
        
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Graph saved: {plot_filename}")
        
        # Save graph data with appropriate filename
        if modality:
            if edge_representation == 'frequency':
                text_filename = os.path.join(output_dir, f'network_user_{user_id}_task_{task_id}_{modality}_frequency.txt')
            else:
                text_filename = os.path.join(output_dir, f'network_user_{user_id}_task_{task_id}_{modality}_{edge_weight_method}.txt')
        else:
            modality_suffix = f"_{modality_handling}" if modality_handling != 'combined' else ""
            if edge_representation == 'frequency':
                text_filename = os.path.join(output_dir, f'network_user_{user_id}_task_{task_id}{modality_suffix}_frequency.txt')
            else:
                text_filename = os.path.join(output_dir, f'network_user_{user_id}_task_{task_id}{modality_suffix}_{edge_weight_method}.txt')
        
        # Save graph data function
        with open(text_filename, 'w') as f:
            f.write(f"Network Graph Data\n")
            f.write(f"==================\n\n")
            f.write(f"User ID: {user_id}\n")
            f.write(f"Task ID: {task_id}\n")
            if modality:
                f.write(f"Modality: {modality}\n")
            f.write(f"Edge representation: {edge_representation}\n")
            if edge_representation == 'time':
                f.write(f"Edge weight calculation: {edge_weight_method} time\n")
            f.write(f"Node Coloring: Fixed classification\n")
            f.write(f"Node Positions: Fixed layout (outside at 12, text at 6)\n")
            f.write(f"Inactive nodes: Hollow with border color matching classification\n")
            f.write(f"\n")
            
            f.write(f"Fixed Classification Scheme:\n")
            f.write(f"{'='*60}\n")
            for panel, cls in FIXED_CLASSIFICATION.items():
                color = FIXED_COLORS.get(cls, '#cccccc')
                f.write(f"{panel:40s} : {cls:12s} (color: {color})\n")
            
            f.write(f"\n")
            f.write(f"Active Nodes by Classification:\n")
            f.write(f"{'='*60}\n")
            
            # Group nodes by classification
            nodes_by_classification = {}
            for panel_title in filtered_time_spent.keys():
                cls = panel_classification.get(panel_title, 'other')
                if cls not in nodes_by_classification:
                    nodes_by_classification[cls] = []
                nodes_by_classification[cls].append(panel_title)
            
            for cls in ['outside', 'text', 'instruction', 'picture', 'other']:
                if cls in nodes_by_classification:
                    f.write(f"\n{cls}:\n")
                    f.write(f"{'-'*len(cls)}\n")
                    for panel_title in sorted(nodes_by_classification[cls]):
                        duration = filtered_time_spent[panel_title]
                        f.write(f"  {panel_title:40s} : {duration:8.2f} seconds\n")
            
            # Add attempted values section
            if attempted_values:
                f.write(f"\n")
                f.write(f"Attempted Values for Text Nodes (when verb='answered'):\n")
                f.write(f"{'='*60}\n")
                for text_node in text_nodes:
                    if text_node in attempted_values:
                        f.write(f"  {text_node:40s} : Attempted = {attempted_values[text_node]}\n")
            
            f.write(f"\n")
            if edge_representation == 'frequency':
                f.write(f"Edges (Transitions with frequency):\n")
            else:
                time_label = "average time" if edge_weight_method == 'average' else "total time"
                f.write(f"Edges (Transitions with frequency and {time_label} spent before transition):\n")
            f.write(f"{'='*60}\n")
            
            # Sort by appropriate weight
            if edge_representation == 'frequency':
                sorted_edges = sorted(filtered_edge_frequencies.items(), key=lambda x: x[1], reverse=True)
            else:
                sorted_edges = sorted(filtered_edge_time_weights.items(), key=lambda x: x[1], reverse=True)
            
            for (node1, node2), weight in sorted_edges:
                frequency = filtered_edge_frequencies.get((node1, node2), 0)
                cls1 = panel_classification.get(node1, 'other')
                cls2 = panel_classification.get(node2, 'other')
                if edge_representation == 'frequency':
                    f.write(f"{node1:30s} ({cls1}) <-> {node2:30s} ({cls2}) : {weight:4d} times\n")
                else:
                    time_label_display = "avg" if edge_weight_method == 'average' else "total"
                    f.write(f"{node1:30s} ({cls1}) <-> {node2:30s} ({cls2}) : {weight:8.2f}s {time_label_display} ({frequency:4d} times)\n")
            
            f.write(f"\n")
            f.write(f"Summary Statistics:\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total active nodes: {len(filtered_time_spent)}\n")
            f.write(f"Total edges: {len(filtered_edge_frequencies)}\n")
            f.write(f"Total time spent: {sum(filtered_time_spent.values()):.2f} seconds\n")
            f.write(f"Total transitions: {sum(filtered_edge_frequencies.values())}\n")
            f.write(f"Average time per node: {np.mean(list(filtered_time_spent.values())):.2f} seconds\n")
            if edge_representation == 'frequency':
                if filtered_edge_frequencies:
                    f.write(f"Average transitions per edge: {np.mean(list(filtered_edge_frequencies.values())):.2f}\n")
                else:
                    f.write(f"Average transitions per edge: 0.0\n")
            else:
                if filtered_edge_time_weights:
                    time_label_stats = "Average" if edge_weight_method == 'average' else "Total"
                    f.write(f"{time_label_stats} time before transition: {np.mean(list(filtered_edge_time_weights.values())):.2f} seconds\n")
                else:
                    f.write(f"Time before transition: 0.0 seconds\n")
            
            # Add attempted values to summary
            if attempted_values:
                f.write(f"\nAttempted Values Summary:\n")
                for text_node in text_nodes:
                    if text_node in attempted_values:
                        f.write(f"  {text_node}: {attempted_values[text_node]}\n")
            
            f.write(f"\nClassification Distribution:\n")
            for cls in ['outside', 'text', 'instruction', 'picture', 'other']:
                if cls in classification_counts:
                    count = classification_counts[cls]
                    total_time = classification_durations.get(cls, 0)
                    avg_time = total_time / count if count > 0 else 0
                    f.write(f"  {cls}: {count} nodes, {total_time:.1f}s total, {avg_time:.1f}s avg\n")
        
        return G
    
    # Create master layout with fixed positions
    master_pos, all_panel_titles, panel_classification = create_master_layout(df_network)
    
    # Save master layout info with classification information
    master_layout_file = os.path.join(network_dir, f'master_layout_user_{user_id}.txt')
    with open(master_layout_file, 'w') as f:
        f.write(f"Master Layout - User {user_id}\n")
        f.write(f"===========================\n\n")
        f.write(f"Fixed Classification Scheme:\n")
        f.write(f"{'='*60}\n")
        for panel, cls in sorted(FIXED_CLASSIFICATION.items()):
            color = FIXED_COLORS.get(cls, '#cccccc')
            f.write(f"{panel:40s} : {cls:12s} (color: {color})\n")
        
        f.write(f"\n")
        f.write(f"Total unique Panel_Titles: {len(all_panel_titles)}\n")
        f.write(f"Classification breakdown:\n")
        
        classification_counts = {}
        for panel in all_panel_titles:
            cls = panel_classification.get(panel, 'other')
            classification_counts[cls] = classification_counts.get(cls, 0) + 1
        
        for cls in sorted(classification_counts.keys()):
            count = classification_counts[cls]
            color = FIXED_COLORS.get(cls, '#cccccc')
            f.write(f"  {cls}: {count} panels (color: {color})\n")
        
        f.write(f"\n")
        f.write(f"Fixed Positions:\n")
        f.write(f"{'='*60}\n")
        f.write(f"• outside at 12 o'clock (top)\n")
        f.write(f"• text elements at 6 o'clock (bottom)\n")
        f.write(f"• instruction elements near text\n")
        f.write(f"• picture elements fill remaining space\n")
        f.write(f"• inactive nodes are hollow with border color matching classification\n")
    
    print(f"\nMaster layout info saved: {master_layout_file}")
    
    # Get unique Task_IDs
    unique_task_ids = df_network['Task_ID'].dropna().unique()
    unique_task_ids = [tid for tid in unique_task_ids if tid != '']
    
    print(f"\nFound {len(unique_task_ids)} unique Task_IDs")
    
    # Process each Task_ID
    graphs_created = 0
    for task_id in unique_task_ids:
        task_data = df_network[df_network['Task_ID'] == task_id].copy()
        
        if len(task_data) < 2:
            print(f"\nTask_ID {task_id}: Only {len(task_data)} rows. Skipping...")
            continue
        
        print(f"\nProcessing Task_ID: {task_id}")
        print(f"  Rows: {len(task_data)}")
        print(f"  Unique Panel_Titles in this task: {task_data['Panel_Title'].nunique()}")
        
        # Determine which representations to create
        if edge_representation == 'both':
            representations = ['time', 'frequency']
        else:
            representations = [edge_representation]
        
        for representation in representations:
            if modality_handling == 'separate':
                modalities = ['mclick', 'eTrack']
                for modality in modalities:
                    modality_data = task_data[task_data['modality'] == modality].copy()
                    
                    if len(modality_data) < 2:
                        print(f"  Modality '{modality}': Only {len(modality_data)} rows. Skipping...")
                        continue
                    
                    modality_time_spent = calculate_time_spent(modality_data)
                    modality_edge_frequencies, modality_edge_time_weights = calculate_edge_weights(modality_data, edge_weight_method)
                    
                    G = create_single_network_graph(task_id, modality_data, modality_time_spent, 
                                                  modality_edge_frequencies, modality_edge_time_weights,
                                                  network_dir, user_id, master_pos, all_panel_titles,
                                                  panel_classification,
                                                  graph_type='modality', modality=modality,
                                                  modality_handling=modality_handling,
                                                  edge_weight_method=edge_weight_method,
                                                  edge_representation=representation)
                    
                    if G:
                        graphs_created += 1
            else:
                time_spent = calculate_time_spent(task_data)
                edge_frequencies, edge_time_weights = calculate_edge_weights(task_data, edge_weight_method)
                
                G = create_single_network_graph(task_id, task_data, time_spent, 
                                              edge_frequencies, edge_time_weights,
                                              network_dir, user_id, master_pos, all_panel_titles,
                                              panel_classification,
                                              graph_type='combined', modality=None,
                                              modality_handling=modality_handling,
                                              edge_weight_method=edge_weight_method,
                                              edge_representation=representation)
                
                if G:
                    graphs_created += 1
    
    print(f"\nNetwork graphs completed.")
    print(f"Created {graphs_created} graphs for {len(unique_task_ids)} Task_IDs")
    print(f"All graphs use the same fixed node positions for comparison.")
    print(f"Nodes colored by fixed classification:")
    print(f"  • picture: Green (#2ca02c)")
    print(f"  • text: Violet (#C227F5)")
    print(f"  • instruction: Orange (#FFA500)")
    print(f"  • outside: Gray (#cccccc)")
    print(f"  • other: Gray (#7f7f7f)")
    print(f"Fixed positions: outside at 12 o'clock, text at 6 o'clock")
    print(f"Inactive nodes are hollow with border color matching their classification")
    
    if edge_representation == 'frequency':
        print(f"Edge width = proportional to transition frequency.")
    elif edge_representation == 'time':
        print(f"Edge width = proportional to {edge_weight_method} time spent in node before transition.")
    else:  # both
        print(f"Created both time-based and frequency-based graphs.")
    
    print(f"Saved to: {network_dir}")
    
    return graphs_created

def create_summary_report(df, user_id, user_dir, timeseries_created, network_created, args):
    """Create a comprehensive summary report"""
    summary_filename = os.path.join(user_dir, f'summary_user_{user_id}.txt')
    
    with open(summary_filename, 'w') as f:
        f.write(f"User Data Processing Summary\n")
        f.write(f"=============================\n\n")
        f.write(f"User ID: {user_id}\n")
        f.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Input File: {args.csv_file}\n")
        f.write(f"Output Directory: {user_dir}\n\n")
        
        f.write(f"Processing Parameters:\n")
        f.write(f"----------------------\n")
        f.write(f"User filter prefix: '{args.user_filter}'\n")
        f.write(f"Skip time series: {args.skip_timeseries}\n")
        f.write(f"Skip network graphs: {args.skip_network}\n")
        if not args.skip_network:
            f.write(f"Network modality handling: {args.modality}\n")
            f.write(f"Minimum duration threshold: {args.min_duration} seconds\n")
            f.write(f"Minimum frequency threshold: {args.min_frequency}\n")
            f.write(f"Edge weight calculation: {args.edge_weight} time\n")
            f.write(f"Edge representation: {args.edge_representation}\n")
        
        f.write(f"\nData Statistics:\n")
        f.write(f"---------------\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}\n")
        f.write(f"Unique dates: {df['Timestamp'].dt.date.nunique()}\n")
        f.write(f"Unique Task_IDs: {df['Task_ID'].nunique()}\n")
        f.write(f"Unique Panel_Titles: {df['Panel_Title'].nunique()}\n")
        
        f.write(f"\nModality distribution:\n")
        modality_counts = df['modality'].value_counts()
        for modality, count in modality_counts.items():
            f.write(f"  {modality}: {count} ({count/len(df)*100:.1f}%)\n")
        
        f.write(f"\nProcessing Results:\n")
        f.write(f"------------------\n")
        if not args.skip_timeseries:
            f.write(f"Time series plots created: {timeseries_created}\n")
            f.write(f"  • Main time series plots (Panel_Title vs Time)\n")
            f.write(f"  • Stacked time series plots (all events)\n")
            f.write(f"  • Split stacked plots (10 intervals)\n")
            f.write(f"  • Additional x-ticks for 'open' and 'answered' events\n")
        if not args.skip_network:
            f.write(f"Network graphs created: {network_created}\n")
            if args.modality == 'separate':
                f.write(f"  (Separate graphs for mclick and eTrack)\n")
            elif args.modality == 'color':
                f.write(f"  (Color-coded by dominant modality)\n")
            else:
                f.write(f"  (Combined modality)\n")
            if args.edge_representation == 'frequency':
                f.write(f"  Edge representation: Frequency-based (width ∝ transition frequency)\n")
            elif args.edge_representation == 'time':
                f.write(f"  Edge representation: Time-based (width ∝ {args.edge_weight} time)\n")
            else:
                f.write(f"  Edge representation: Both time-based and frequency-based graphs\n")
        
        f.write(f"\nDirectory Structure:\n")
        f.write(f"-------------------\n")
        f.write(f"User directory: {user_dir}\n")
        if not args.skip_timeseries:
            f.write(f"Time series directory: {os.path.join(user_dir, 'timeseries')}\n")
        if not args.skip_network:
            f.write(f"Network directory: {os.path.join(user_dir, 'network')}\n")
        
        f.write(f"\nGenerated Files:\n")
        f.write(f"---------------\n")
        
        # List files in user directory
        for root, dirs, files in os.walk(user_dir):
            level = root.replace(user_dir, '').count(os.sep)
            indent = '  ' * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = '  ' * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")
    
    print(f"Summary report saved: {summary_filename}")

def perform_psychometric_analysis(df, user_id, network_dir, task_data_dict=None):
    """
    Perform psychometric analysis on user interaction data
    Calculates cognitive, behavioral, and learning metrics
    Saves results as JSON files
    """
    print(f"\n{'='*80}")
    print(f"PERFORMING PSYCHOMETRIC ANALYSIS FOR USER {user_id}")
    print(f"{'='*80}")
    
    # Define classification dictionary (same as in network graphs)
    PSYCHOMETRIC_CLASSIFICATION = {
        # Picture elements
        'Mobile phones breakup details': 'picture',
        'Mobile phones for teaching': 'picture',
        'Desktop breakup details': 'picture',
        'Laptop breakup details': 'picture',
        'Laptop / Notebook': 'picture',
        'Distribution of schools': 'picture',
        'Dropout - secondary': 'picture',
        'Desktop': 'picture',
        'Projector': 'picture',
        'Digital Library': 'picture',
        'Pupil-teacher ratio (PTR)': 'picture',
        'Dropout - middle': 'picture',
        'Dropout - preparatory': 'picture',
        'Distribution of teachers': 'picture',
        'ICT labs': 'picture',
        'Infrastructure': 'picture',
        'Distribution of students': 'picture',
        'Tablet breakup details': 'picture',
        'Tablet': 'picture',
        
        # Text elements
        '1st chart in the report': 'text',
        '2nd chart in the report': 'text',
        '3rd chart in the report': 'text',
        
        # Instruction element
        'Ordering three charts as evidence': 'instruction',
        
        # Special element
        'outside': 'outside'
    }
    
    # Define classification helper function
    def classify_panel(panel_title):
        """Classify panel title into content types"""
        if pd.isna(panel_title) or panel_title == '':
            return 'unknown'
        
        # Use the psychometric classification if available
        if panel_title in PSYCHOMETRIC_CLASSIFICATION:
            return PSYCHOMETRIC_CLASSIFICATION[panel_title]
        
        # Fallback classification based on content
        panel_lower = panel_title.lower()
        
        if 'outside' in panel_lower:
            return 'outside'
        elif any(keyword in panel_lower for keyword in ['chart', 'text', 'report', 'question']):
            return 'text'
        elif any(keyword in panel_lower for keyword in ['instruction', 'order', 'evidence', 'guide', 'help']):
            return 'instruction'
        elif any(keyword in panel_lower for keyword in ['picture', 'image', 'graph', 'diagram', 'photo', 'visual']):
            return 'picture'
        elif any(keyword in panel_lower for keyword in ['button', 'menu', 'tab', 'link', 'icon']):
            return 'interface'
        else:
            return 'other'
    
    def calculate_task_psychometrics(task_id, task_df):
        """Calculate psychometric metrics for a specific task"""
        
        if task_df.empty or len(task_df) < 2:
            return None
        
        # Sort by timestamp
        task_df = task_df.sort_values('Timestamp').reset_index(drop=True)
        
        # 1. Basic temporal metrics
        total_duration = (task_df['Timestamp'].iloc[-1] - task_df['Timestamp'].iloc[0]).total_seconds()
        
        # 2. Content type analysis
        content_time = {}
        content_counts = {}
        
        for idx, row in task_df.iterrows():
            panel = row['Panel_Title']
            if pd.isna(panel) or panel == '':
                continue
            
            content_type = classify_panel(panel)
            
            # Calculate time spent (except for last row)
            if idx < len(task_df) - 1:
                time_spent = (task_df['Timestamp'].iloc[idx + 1] - row['Timestamp']).total_seconds()
                if time_spent > 0:
                    content_time[content_type] = content_time.get(content_type, 0) + time_spent
            
            content_counts[content_type] = content_counts.get(content_type, 0) + 1
        
        # 3. Transition analysis
        transitions = []
        transition_types = []
        
        for i in range(len(task_df) - 1):
            current = task_df.iloc[i]['Panel_Title']
            next_panel = task_df.iloc[i + 1]['Panel_Title']
            
            if pd.isna(current) or pd.isna(next_panel) or current == '' or next_panel == '':
                continue
            
            # Skip self-transitions
            if current == next_panel:
                continue
            
            current_type = classify_panel(current)
            next_type = classify_panel(next_panel)
            
            transitions.append((current, next_panel))
            transition_types.append((current_type, next_type))
        
        # 4. Verb analysis
        verb_patterns = {}
        if 'Verb' in task_df.columns:
            verb_counts = task_df['Verb'].value_counts().to_dict()
            verb_patterns = verb_counts
        
        # 5. Calculate psychometric metrics
        metrics = {
            # Task completion metrics
            'task_id': task_id,
            'total_duration_seconds': total_duration,
            'total_events': len(task_df),
            'unique_panels': task_df['Panel_Title'].nunique(),
            
            # Content engagement metrics
            'content_time_distribution': content_time,
            'content_event_distribution': content_counts,
            'text_picture_ratio': content_time.get('text', 0) / max(content_time.get('picture', 1), 1),
            'instruction_engagement_ratio': content_time.get('instruction', 0) / max(total_duration, 1),
            
            # Transition metrics
            'total_transitions': len(transitions),
            'transition_rate': len(transitions) / max(total_duration, 1),  # transitions per second
            'unique_transitions': len(set(transitions)),
            'exploration_index': task_df['Panel_Title'].nunique() / max(len(set(task_df['Panel_Title'].dropna())), 1),
            
            # Cognitive style indicators
            'focus_index': 0,
            'switching_tendency': 0,
            'integration_score': 0,
            'systematicity_score': 0,
            
            # Behavioral patterns
            'outside_transitions': sum(1 for t in transition_types if 'outside' in t),
            'cross_content_transitions': sum(1 for t in transition_types if t[0] != t[1]),
            'revisits': len(transitions) - len(set(transitions)),
            
            # Verb/action patterns
            'verb_diversity': len(verb_patterns),
            'verb_patterns': verb_patterns,
            
            # Performance indicators (if available)
            'performance_indicators': {}
        }
        
        # Calculate focus index (time concentration)
        if content_time:
            max_time = max(content_time.values())
            total_content_time = sum(content_time.values())
            metrics['focus_index'] = max_time / max(total_content_time, 1)
        
        # Calculate switching tendency
        if transitions:
            metrics['switching_tendency'] = len(transitions) / max(len(task_df), 1)
        
        # Calculate integration score (cross-content transitions)
        if transitions:
            metrics['integration_score'] = metrics['cross_content_transitions'] / max(len(transitions), 1)
        
        # Calculate systematicity (pattern consistency)
        if transitions:
            unique_ratio = len(set(transitions)) / max(len(transitions), 1)
            metrics['systematicity_score'] = 1 - unique_ratio  # Lower = more systematic/repetitive
        
        # 6. Network-based metrics (if network data is available)
        # Note: This would require passing network data to the function
        # For now, we'll skip this or you can implement it later
        
        # 7. Time series analysis
        if len(task_df) > 2:
            time_diffs = task_df['Timestamp'].diff().dt.total_seconds().dropna()
            if len(time_diffs) > 1:
                metrics['time_std'] = float(time_diffs.std())
                metrics['time_cv'] = float(time_diffs.std() / max(time_diffs.mean(), 1))  # Coefficient of variation
                metrics['pacing_consistency'] = float(1 / (metrics['time_std'] + 1))
        
        # 8. Attempted values analysis (performance)
        if 'Attempted' in task_df.columns:
            attempted_values = task_df[task_df['Attempted'].notna()]['Attempted'].unique()
            if len(attempted_values) > 0:
                metrics['performance_indicators']['attempted_values'] = attempted_values.tolist()
                metrics['performance_indicators']['unique_attempts'] = len(attempted_values)
        
        if 'Verb' in task_df.columns:
            answered_count = task_df[task_df['Verb'].str.lower() == 'answered'].shape[0]
            metrics['performance_indicators']['answered_count'] = answered_count
        
        # Convert numpy values to Python native types
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
        
        return metrics
    
    def interpret_score(score):
        """Interpret psychometric scores"""
        if score >= 80:
            return "High"
        elif score >= 60:
            return "Above Average"
        elif score >= 40:
            return "Average"
        elif score >= 20:
            return "Below Average"
        else:
            return "Low"
    
    def calculate_user_level_psychometrics(all_task_metrics):
        """Aggregate task-level metrics to user-level psychometric profile"""
        
        if not all_task_metrics:
            return None
        
        # Initialize aggregates
        user_profile = {
            'user_id': user_id,
            'total_tasks': len(all_task_metrics),
            'overall_metrics': {},
            'task_by_task': {},
            'psychometric_dimensions': {}
        }
        
        # Aggregate across all tasks
        for task_id, metrics in all_task_metrics.items():
            user_profile['task_by_task'][task_id] = metrics
            
            # Aggregate numerical metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in user_profile['overall_metrics']:
                        user_profile['overall_metrics'][key] = []
                    user_profile['overall_metrics'][key].append(value)
        
        # Calculate averages - create a new dictionary for results
        overall_metrics_results = {}
        
        # First, copy the existing metrics
        for key, values in user_profile['overall_metrics'].items():
            overall_metrics_results[key] = values  # Keep the original values
        
        # Then calculate averages in a separate step
        for key, values in user_profile['overall_metrics'].items():
            if values:
                overall_metrics_results[f'avg_{key}'] = float(np.mean(values))
                if len(values) > 1:
                    overall_metrics_results[f'std_{key}'] = float(np.std(values))
                else:
                    overall_metrics_results[f'std_{key}'] = 0.0
        
        # Replace the original with the new dictionary that includes averages
        user_profile['overall_metrics'] = overall_metrics_results
        
        # Define psychometric dimensions
        user_profile['psychometric_dimensions'] = {
            'cognitive_engagement': {
                'score': min(max(user_profile['overall_metrics'].get('avg_focus_index', 0) * 100, 0), 100),
                'indicators': ['focus_index', 'total_duration_seconds', 'exploration_index']
            },
            'learning_strategy': {
                'score': min(max(user_profile['overall_metrics'].get('avg_integration_score', 0) * 100, 0), 100),
                'indicators': ['integration_score', 'systematicity_score', 'cross_content_transitions']
            },
            'cognitive_flexibility': {
                'score': min(max(user_profile['overall_metrics'].get('avg_switching_tendency', 0) * 100, 0), 100),
                'indicators': ['switching_tendency', 'transition_rate', 'unique_transitions']
            },
            'task_persistence': {
                'score': min(max(user_profile['overall_metrics'].get('avg_total_duration_seconds', 0) / 600 * 100, 0), 100),
                'indicators': ['total_duration_seconds', 'total_events', 'revisits']
            },
            'metacognitive_awareness': {
                'score': min(max(user_profile['overall_metrics'].get('avg_instruction_engagement_ratio', 0) * 200, 0), 100),
                'indicators': ['instruction_engagement_ratio', 'verb_diversity', 'outside_transitions']
            }
        }
        
        # Calculate dimension scores (normalized 0-100)
        for dim, data in user_profile['psychometric_dimensions'].items():
            # Cap scores at 100
            data['score'] = min(max(data['score'], 0), 100)
            data['interpretation'] = interpret_score(data['score'])
        
        return user_profile
    
    def generate_recommendations(profile):
        """Generate learning recommendations based on psychometric profile"""
        
        recommendations = []
        
        # Cognitive Engagement recommendations
        engagement = profile['psychometric_dimensions']['cognitive_engagement']['score']
        if engagement < 40:
            recommendations.append({
                'area': 'Cognitive Engagement',
                'issue': 'Low focus and engagement detected',
                'suggestion': 'Try breaking tasks into smaller chunks with clear goals',
                'strategy': 'Use Pomodoro technique (25 min focus, 5 min break)'
            })
        elif engagement > 80:
            recommendations.append({
                'area': 'Cognitive Engagement',
                'issue': 'Very high focus, risk of cognitive fatigue',
                'suggestion': 'Schedule regular breaks to maintain optimal performance',
                'strategy': 'Follow 52-17 rule (52 min work, 17 min break)'
            })
        
        # Learning Strategy recommendations
        strategy = profile['psychometric_dimensions']['learning_strategy']['score']
        if strategy < 40:
            recommendations.append({
                'area': 'Learning Strategy',
                'issue': 'Minimal integration between different content types',
                'suggestion': 'Practice connecting visual and textual information',
                'strategy': 'Use concept mapping to link different content elements'
            })
        
        # Cognitive Flexibility recommendations
        flexibility = profile['psychometric_dimensions']['cognitive_flexibility']['score']
        if flexibility < 30:
            recommendations.append({
                'area': 'Cognitive Flexibility',
                'issue': 'Low switching between content elements',
                'suggestion': 'Practice alternating between different types of information',
                'strategy': 'Set timers to switch content types every 5-10 minutes'
            })
        elif flexibility > 80:
            recommendations.append({
                'area': 'Cognitive Flexibility',
                'issue': 'Very high switching, may indicate distraction',
                'suggestion': 'Practice sustained attention on single elements',
                'strategy': 'Use attention training exercises like mindfulness'
            })
        
        # Task Persistence recommendations
        persistence = profile['psychometric_dimensions']['task_persistence']['score']
        if persistence < 40:
            recommendations.append({
                'area': 'Task Persistence',
                'issue': 'Short task durations detected',
                'suggestion': 'Build endurance with gradually increasing task times',
                'strategy': 'Use gamification with progress tracking'
            })
        
        # Metacognitive Awareness recommendations
        metacognition = profile['psychometric_dimensions']['metacognitive_awareness']['score']
        if metacognition < 40:
            recommendations.append({
                'area': 'Metacognitive Awareness',
                'issue': 'Low engagement with instructional elements',
                'suggestion': 'Spend more time reviewing instructions and guidance',
                'strategy': 'Use think-aloud protocol to verbalize thought process'
            })
        
        return recommendations
    
    # Main analysis logic
    print(f"Analyzing psychometric patterns for user {user_id}...")
    
    # Get unique tasks
    unique_tasks = df['Task_ID'].dropna().unique()
    unique_tasks = [t for t in unique_tasks if t != '']
    
    print(f"Found {len(unique_tasks)} unique tasks for analysis")
    
    # Calculate task-level metrics
    all_task_metrics = {}
    
    for task_id in unique_tasks:
        task_df = df[df['Task_ID'] == task_id].copy()
        
        if len(task_df) < 5:  # Skip tasks with too few events
            print(f"  Skipping Task {task_id}: Only {len(task_df)} events")
            continue
        
        print(f"  Analyzing Task {task_id} ({len(task_df)} events)")
        
        metrics = calculate_task_psychometrics(task_id, task_df)
        if metrics:
            all_task_metrics[task_id] = metrics
    
    if not all_task_metrics:
        print(f"No valid tasks found for psychometric analysis")
        return None
    
    # Calculate user-level profile
    user_profile = calculate_user_level_psychometrics(all_task_metrics)
    
    if not user_profile:
        print(f"Could not calculate user profile")
        return None
    
    # Generate recommendations
    recommendations = generate_recommendations(user_profile)
    user_profile['recommendations'] = recommendations
    
    # Save detailed psychometric report (JSON)
    psychometric_file_json = os.path.join(network_dir, f'psychometric_profile_{user_id}.json')
    with open(psychometric_file_json, 'w') as f:
        import json
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                return super().default(obj)
        
        json.dump(user_profile, f, indent=2, cls=NumpyEncoder)
    
    # Save human-readable summary (TXT)
    psychometric_file_txt = os.path.join(network_dir, f'psychometric_summary_{user_id}.txt')
    with open(psychometric_file_txt, 'w') as f:
        f.write(f"PSYCHOMETRIC PROFILE - USER {user_id}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Tasks Analyzed: {user_profile['total_tasks']}\n")
        f.write(f"Total Events: {sum(m['total_events'] for m in all_task_metrics.values())}\n")
        f.write(f"Total Duration: {sum(m['total_duration_seconds'] for m in all_task_metrics.values()):.0f} seconds\n")
        f.write(f"Average Task Duration: {user_profile['overall_metrics'].get('avg_total_duration_seconds', 0):.0f} seconds\n\n")
        
        f.write("PSYCHOMETRIC DIMENSIONS\n")
        f.write("-" * 30 + "\n")
        for dim_name, dim_data in user_profile['psychometric_dimensions'].items():
            dim_name_formatted = dim_name.replace('_', ' ').title()
            f.write(f"{dim_name_formatted}:\n")
            f.write(f"  Score: {dim_data['score']:.1f}/100 ({dim_data['interpretation']})\n")
            f.write(f"  Key Indicators: {', '.join(dim_data['indicators'])}\n\n")
        
        f.write("LEARNING RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec['area']}:\n")
                f.write(f"   Issue: {rec['issue']}\n")
                f.write(f"   Suggestion: {rec['suggestion']}\n")
                f.write(f"   Strategy: {rec['strategy']}\n\n")
        else:
            f.write("No specific recommendations - profile appears balanced\n\n")
        
        f.write("DETAILED TASK ANALYSIS\n")
        f.write("-" * 30 + "\n")
        for task_id, metrics in all_task_metrics.items():
            f.write(f"\nTask {task_id}:\n")
            f.write(f"  Duration: {metrics['total_duration_seconds']:.0f}s\n")
            f.write(f"  Events: {metrics['total_events']}\n")
            f.write(f"  Transitions: {metrics['total_transitions']}\n")
            f.write(f"  Focus Index: {metrics['focus_index']:.2f}\n")
            f.write(f"  Integration Score: {metrics['integration_score']:.2f}\n")
            
            if metrics.get('performance_indicators'):
                f.write(f"  Performance Indicators:\n")
                for key, val in metrics['performance_indicators'].items():
                    if isinstance(val, list):
                        f.write(f"    {key}: {', '.join(map(str, val))}\n")
                    else:
                        f.write(f"    {key}: {val}\n")
        
        f.write(f"\n\nDATA QUALITY INDICATORS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Tasks with sufficient data: {len(all_task_metrics)}/{len(unique_tasks)}\n")
        f.write(f"Average events per task: {user_profile['overall_metrics'].get('avg_total_events', 0):.0f}\n")
        f.write(f"Data consistency (time CV): {user_profile['overall_metrics'].get('avg_time_cv', 0):.3f}\n")
        
        f.write(f"\n\nGENERATED ON: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nPsychometric analysis completed:")
    print(f"  JSON profile saved: {psychometric_file_json}")
    print(f"  Text summary saved: {psychometric_file_txt}")
    
    # Print summary to console
    print(f"\n{'='*50}")
    print(f"PSYCHOMETRIC SUMMARY FOR USER {user_id}")
    print(f"{'='*50}")
    for dim_name, dim_data in user_profile['psychometric_dimensions'].items():
        dim_name_formatted = dim_name.replace('_', ' ').title()
        print(f"{dim_name_formatted:25s}: {dim_data['score']:5.1f}/100 ({dim_data['interpretation']})")
    
    if recommendations:
        print(f"\nKey Recommendations:")
        for rec in recommendations[:3]:  # Show top 3
            print(f"  • {rec['suggestion']}")
    
    return user_profile


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"\n{'='*80}")
    print(f"INTEGRATED PROCESSOR FOR GAZE ACTIVITY DATA")
    print(f"{'='*80}")
    
    if args.edge_representation == 'time':
        print(f"EDGE WIDTH PROPORTIONAL TO {args.edge_weight.upper()} TIME SPENT BEFORE TRANSITION")
    elif args.edge_representation == 'frequency':
        print(f"EDGE WIDTH PROPORTIONAL TO TRANSITION FREQUENCY")
    else:
        print(f"CREATING BOTH TIME-BASED AND FREQUENCY-BASED GRAPHS")
    
    # Preprocess data (or load existing preprocessed file)
    if args.skip_preprocessing:
        # Try to find existing preprocessed file
        user_id = os.path.splitext(os.path.basename(args.csv_file))[0].replace('user_', '')
        user_dir = os.path.join(args.output_dir, f"user_{user_id}")
        preprocessed_csv_path = os.path.join(user_dir, f'preprocessed_data_user_{user_id}.csv')
        
        if os.path.exists(preprocessed_csv_path):
            print(f"Loading existing preprocessed data: {preprocessed_csv_path}")
            df = pd.read_csv(preprocessed_csv_path)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df = df.sort_values('Timestamp')
        else:
            print(f"Preprocessed file not found. Running preprocessing...")
            df, user_id = preprocess_data(args.csv_file, args.user_filter)
    else:
        df, user_id = preprocess_data(args.csv_file, args.user_filter)
    
    # Create directory structure
    user_dir, timeseries_dir, network_dir = create_user_directory_structure(args.output_dir, user_id)
    
    # Save preprocessed data
    preprocessed_csv_path = os.path.join(user_dir, f'preprocessed_data_user_{user_id}.csv')
    df.to_csv(preprocessed_csv_path, index=False)
    print(f"\nPreprocessed data saved: {preprocessed_csv_path}")
    
    # Initialize counters
    timeseries_created = 0
    network_created = 0
    
    # Create time series plots
    if not args.skip_timeseries:
        try:
            print(f"\n{'='*80}")
            print(f"STARTING TIME SERIES PLOT GENERATION")
            filtered_csv_path, timeseries_created = create_timeseries_plots(df, user_id, timeseries_dir)
            print(f"\n✓ Time series plots completed successfully")
        except Exception as e:
            print(f"\n✗ Error creating time series plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nSkipping time series plot generation as requested.")
    
    # Create network graphs
    if not args.skip_network:
        try:
            print(f"\n{'='*80}")
            print(f"STARTING NETWORK GRAPH GENERATION")
            network_created = create_network_graphs(df, user_id, network_dir,
                                                   args.modality, args.min_duration,
                                                   args.min_frequency, args.edge_weight,
                                                   args.edge_representation)
            print(f"\n✓ Network graphs completed successfully")
        except Exception as e:
            print(f"\n✗ Error creating network graphs: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nSkipping network graph generation as requested.")

        # ADD THIS NEW SECTION FOR PSYCHOMETRIC ANALYSIS
    print(f"\n{'='*80}")
    print(f"PERFORMING PSYCHOMETRIC ANALYSIS")
    print(f"{'='*80}")
    
    try:
        # You might want to collect some network data to pass to the psychometric function
        # For simplicity, we'll just pass the dataframe and let the function calculate everything
        psychometric_profile = perform_psychometric_analysis(df, user_id, network_dir)
        
        if psychometric_profile:
            print(f"\n✓ Psychometric analysis completed successfully")
            # You could also add the psychometric profile to your summary report
        else:
            print(f"\n⚠ Psychometric analysis completed with limited data")
    except Exception as e:
        print(f"\n✗ Error in psychometric analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Create summary report
    create_summary_report(df, user_id, user_dir, timeseries_created, network_created, args)
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"User ID: {user_id}")
    print(f"All outputs saved in: {user_dir}")
    print(f"\nSummary:")
    print(f"  - Preprocessed data: {preprocessed_csv_path}")
    if not args.skip_timeseries:
        print(f"  - Time series plots: {timeseries_dir}")
        print(f"    • Filtered CSV: {os.path.join(timeseries_dir, f'filtered_gaze_activity_user_{user_id}.csv')}")
        print(f"    • Main time series plots (Panel_Title vs Time)")
        print(f"    • Stacked time series plots (all events)")
        print(f"    • Split stacked plots (10 intervals)")
        print(f"    • Additional x-ticks for 'open' and 'answered' events (time only)")
    if not args.skip_network:
        print(f"  - Network graphs: {network_dir}")
        print(f"    • Master layout info: {os.path.join(network_dir, f'master_layout_user_{user_id}.txt')}")
        print(f"    • Graphs for {network_created} Task_IDs")
        if args.modality == 'separate':
            print(f"    • Separate graphs for mclick (orange) and eTrack (green)")
        elif args.modality == 'color':
            print(f"    • Color-coded by dominant modality")
        else:
            print(f"    • Combined modality (Violet)")
        print(f"    • All graphs use fixed node positions (circular layout)")
        print(f"    • Fixed node size = 10")
        print(f"    • Hollow nodes = elements not used in that specific task/modality")
        if args.edge_representation == 'frequency':
            print(f"    • Edge width = proportional to transition frequency")
        elif args.edge_representation == 'time':
            print(f"    • Edge width = proportional to {args.edge_weight} time spent in node before transition")
        else:  # both
            print(f"    • Created both time-based and frequency-based graphs")
    print(f"  - Summary report: {os.path.join(user_dir, f'summary_user_{user_id}.txt')}")

if __name__ == "__main__":
    main()