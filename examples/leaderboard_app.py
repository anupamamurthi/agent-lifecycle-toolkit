"""
Agent Leaderboard Streamlit App

A simple Streamlit UI to display agent performance scores in a leaderboard format.
Data is loaded from a configurable JSON file.

Usage:
    streamlit run leaderboard_app.py
"""

import json
from pathlib import Path

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Agent Leaderboard",
    page_icon="🏆",
    layout="wide",
)

# Configuration - change this path to use a different data file
CONFIG_FILE = "leaderboard_data.json"


def load_config(config_path: Path) -> dict:
    """Load leaderboard configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def get_score_color(score: float, score_ranges: dict) -> str:
    """Get the color for a score based on defined ranges."""
    for range_info in score_ranges.values():
        if range_info["min"] <= score <= range_info["max"]:
            return range_info["color"]
    return "#6b7280"  # Default gray


def calculate_overall_score(scores: dict) -> float:
    """Calculate the overall score as the mean of all scores."""
    if not scores:
        return 0.0
    return round(sum(scores.values()) / len(scores), 2)


def get_all_metrics(categories: dict) -> list:
    """Get flat list of all metric keys in order."""
    metrics = []
    for cat_key in ["single_turn", "multi_turn"]:
        for sub in categories[cat_key]["subcategories"].values():
            metrics.extend(sub["metrics"])
    return metrics


# Resolve config path relative to script location
script_dir = Path(__file__).parent
config_path = script_dir / CONFIG_FILE

# Load configuration
try:
    config = load_config(config_path)
except FileNotFoundError:
    st.error(f"Configuration file not found: {config_path}")
    st.info("Please create a leaderboard_data.json file in the same directory")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"Invalid JSON in configuration file: {e}")
    st.stop()

# Get data from config
categories = config["categories"]
metrics_labels = config["metrics"]
agents = config["agents"]
score_ranges = config["score_ranges"]

# Calculate overall scores
for agent in agents:
    agent["overall"] = calculate_overall_score(agent["scores"])

# Initialize session state for sorting
if "sort_column" not in st.session_state:
    st.session_state.sort_column = "overall"
if "sort_ascending" not in st.session_state:
    st.session_state.sort_ascending = False


def toggle_sort(column: str):
    """Toggle sort direction or change sort column."""
    if st.session_state.sort_column == column:
        st.session_state.sort_ascending = not st.session_state.sort_ascending
    else:
        st.session_state.sort_column = column
        st.session_state.sort_ascending = False  # Default to descending for scores


def get_sort_value(agent: dict, column: str) -> float:
    """Get the sortable value for an agent and column."""
    if column == "overall":
        return agent["overall"]
    elif column == "name":
        return agent["name"].lower()
    else:
        return agent["scores"].get(column, 0)


# Sort agents based on current sort state
agents_sorted = sorted(
    agents,
    key=lambda x: get_sort_value(x, st.session_state.sort_column),
    reverse=not st.session_state.sort_ascending,
)

# Title
st.markdown(
    f"<h1 style='text-align: center; margin-bottom: 2rem;'>{config.get('title', 'Agent Leaderboard')}</h1>",
    unsafe_allow_html=True,
)

# Calculate column spans
single_turn_cols = sum(
    len(sub["metrics"]) for sub in categories["single_turn"]["subcategories"].values()
)
multi_turn_cols = sum(
    len(sub["metrics"]) for sub in categories["multi_turn"]["subcategories"].values()
)

# Get all metrics for column ordering
all_metrics = get_all_metrics(categories)

# Build sortable column list for display
sortable_columns = [("name", "Agent"), ("overall", "Overall")]
for metric_key in all_metrics:
    sortable_columns.append((metric_key, metrics_labels.get(metric_key, metric_key)))


def get_sort_icon(column: str) -> str:
    """Get the sort icon for a column header."""
    if st.session_state.sort_column == column:
        return "↓" if not st.session_state.sort_ascending else "↑"
    return "↕"


def get_sort_icon_style(column: str) -> str:
    """Get the style for sort icon."""
    if st.session_state.sort_column == column:
        return "color: #ef4444; font-weight: bold;"
    return "color: #9ca3af;"


# CSS Styles - White/Light Theme with Table Borders
st.markdown(
    """
<style>
    /* Force white background for entire app */
    .stApp {
        background-color: #ffffff;
    }
    .main .block-container {
        background-color: #ffffff;
        padding-top: 2rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    /* Style the container border */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color: #e2e8f0 !important;
    }

    .category-pill {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 24px;
        font-size: 14px;
        font-weight: 600;
    }
    .single-turn-pill {
        background: #dbeafe;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }
    .multi-turn-pill {
        background: #f3e8ff;
        color: #7c3aed;
        border: 1px solid #e9d5ff;
    }
    .subcategory-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 500;
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    .rank-badge {
        font-weight: 700;
        font-size: 14px;
        width: 32px;
        height: 32px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
    }
    .rank-1 { background: #fef3c7; color: #b45309; border: 2px solid #fcd34d; }
    .rank-2 { background: #f1f5f9; color: #475569; border: 2px solid #cbd5e1; }
    .rank-3 { background: #fed7aa; color: #c2410c; border: 2px solid #fdba74; }
    .rank-other { background: #ffffff; color: #64748b; border: 1px solid #e2e8f0; }
    .agent-name {
        font-weight: 600;
        color: #1e293b;
        text-align: left !important;
    }
    .score-cell { font-weight: 600; }
    .overall-cell { font-weight: 700; font-size: 16px; }
    .legend-container {
        display: flex;
        gap: 24px;
        margin-top: 24px;
        padding-top: 16px;
        flex-wrap: wrap;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: #475569;
    }
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        border: 1px solid rgba(0,0,0,0.1);
    }

    /* Sortable header button styles */
    .stButton > button {
        background: transparent !important;
        border: none !important;
        padding: 4px 8px !important;
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 14px !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        background: #e2e8f0 !important;
        color: #1e293b !important;
    }
    .stButton > button:focus {
        box-shadow: none !important;
    }

    /* Title styling */
    h1 {
        color: #1e293b !important;
    }

    /* Row divider line */
    .row-divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 8px 0;
    }

    /* Header divider - thicker */
    .header-divider {
        border: none;
        border-top: 2px solid #cbd5e1;
        margin: 8px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Create the bordered table using Streamlit container
with st.container(border=True):
    # Category header row
    header_cols = st.columns([1, 1.5] + [1] * single_turn_cols + [1] * multi_turn_cols + [1])

    with header_cols[0]:
        st.write("")  # Rank
    with header_cols[1]:
        st.write("")  # Agent

    # Single Turn pill spans multiple columns
    single_turn_start = 2
    single_turn_end = single_turn_start + single_turn_cols
    with header_cols[single_turn_start + single_turn_cols // 2]:
        st.markdown(
            '<span class="category-pill single-turn-pill">Single Turn</span>',
            unsafe_allow_html=True,
        )

    # Multi Turn pill
    multi_turn_start = single_turn_end
    with header_cols[multi_turn_start + multi_turn_cols // 2]:
        st.markdown(
            '<span class="category-pill multi-turn-pill">Multi-turn</span>',
            unsafe_allow_html=True,
        )

    # Subcategory header row
    sub_cols = st.columns([1, 1.5] + [1] * single_turn_cols + [1] * multi_turn_cols + [1])

    col_idx = 2
    for cat_key in ["single_turn", "multi_turn"]:
        for sub in categories[cat_key]["subcategories"].values():
            num_metrics = len(sub["metrics"])
            mid_idx = col_idx + num_metrics // 2
            with sub_cols[mid_idx]:
                st.markdown(
                    f'<span class="subcategory-pill">{sub["label"]}</span>',
                    unsafe_allow_html=True,
                )
            col_idx += num_metrics

    # Sortable column headers
    header_row = st.columns([1, 1.5] + [1] * len(all_metrics) + [1])

    with header_row[0]:
        st.markdown("**Rank**")

    with header_row[1]:
        icon = get_sort_icon("name")
        icon_style = get_sort_icon_style("name")
        if st.button(f"Agent {icon}", key="sort_name", help="Click to sort by Agent"):
            toggle_sort("name")
            st.rerun()

    for i, metric in enumerate(all_metrics):
        with header_row[i + 2]:
            label = metrics_labels.get(metric, metric)
            icon = get_sort_icon(metric)
            icon_style = get_sort_icon_style(metric)
            if st.button(f"{label} {icon}", key=f"sort_{metric}", help=f"Click to sort by {label}"):
                toggle_sort(metric)
                st.rerun()

    with header_row[-1]:
        icon = get_sort_icon("overall")
        icon_style = get_sort_icon_style("overall")
        if st.button(f"Overall {icon}", key="sort_overall", help="Click to sort by Overall"):
            toggle_sort("overall")
            st.rerun()

    # Header divider
    st.markdown('<hr class="header-divider">', unsafe_allow_html=True)

    # Data rows (inside the container)
    for rank, agent in enumerate(agents_sorted, 1):
        row = st.columns([1, 1.5] + [1] * len(all_metrics) + [1])

        # Rank badge
        with row[0]:
            rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
            st.markdown(
                f'<span class="rank-badge {rank_class}">{rank}</span>',
                unsafe_allow_html=True,
            )

        # Agent name
        with row[1]:
            st.markdown(f'<span class="agent-name">{agent["name"]}</span>', unsafe_allow_html=True)

        # Metric scores
        for i, metric in enumerate(all_metrics):
            with row[i + 2]:
                score = agent["scores"].get(metric, 0)
                color = get_score_color(score, score_ranges)
                st.markdown(
                    f'<span class="score-cell" style="color: {color};">{score}</span>',
                    unsafe_allow_html=True,
                )

        # Overall score
        with row[-1]:
            overall_color = get_score_color(agent["overall"], score_ranges)
            st.markdown(
                f'<span class="overall-cell" style="color: {overall_color};">{agent["overall"]}</span>',
                unsafe_allow_html=True,
            )

        # Row divider (except for last row)
        if rank < len(agents_sorted):
            st.markdown('<hr class="row-divider">', unsafe_allow_html=True)

# Legend (outside the container)
st.markdown("<br>", unsafe_allow_html=True)
legend_html = '<div class="legend-container">'
for range_info in score_ranges.values():
    legend_html += f'''<div class="legend-item">
        <span class="legend-dot" style="background: {range_info['color']};"></span>
        <span>{range_info['min']}-{range_info['max']}: {range_info['label']}</span>
    </div>'''
legend_html += "</div>"
st.markdown(legend_html, unsafe_allow_html=True)

# Sidebar with configuration info
with st.sidebar:
    st.header("Configuration")
    st.write(f"**Data source:** `{CONFIG_FILE}`")
    st.write(f"**Agents:** {len(agents)}")
    st.write(f"**Metrics:** {len(metrics_labels)}")

    st.divider()

    st.subheader("Current Sort")
    sort_col_label = "Overall" if st.session_state.sort_column == "overall" else (
        "Agent" if st.session_state.sort_column == "name" else
        metrics_labels.get(st.session_state.sort_column, st.session_state.sort_column)
    )
    sort_dir = "Ascending" if st.session_state.sort_ascending else "Descending"
    st.write(f"**Column:** {sort_col_label}")
    st.write(f"**Direction:** {sort_dir}")

    st.divider()

    st.subheader("Score Ranges")
    for range_info in score_ranges.values():
        st.markdown(
            f'<span style="color: {range_info["color"]};">●</span> '
            f'**{range_info["label"]}**: {range_info["min"]}-{range_info["max"]}',
            unsafe_allow_html=True,
        )
