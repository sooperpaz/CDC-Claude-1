"""
CDC Provisional Natality Dashboard — 2025

Single-file version (all modules combined) for manual upload. Sections below
mirror what would otherwise be separate files: constants, data loading,
filters, KPIs, charts, then the page layout at the bottom.

To run: place this file and `Provisional_Natality_2025_CDC.csv` in the same
folder, then `streamlit run app.py`.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# Constants
# ============================================================================

# Chronological month order. Used to build a pandas Categorical dtype so
# charts and tables sort Jan -> Dec instead of alphabetically.
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# State (and DC) name -> USPS abbreviation. Hardcoded rather than looked up
# from a third-party package so the choropleth's `locations` field never
# depends on an external mapping matching the CDC's exact naming.
STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT",
    "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
    "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}

# Colorblind-safe categorical palette (Okabe-Ito), reused across every chart
# so Female/Male and other categorical encodings stay consistent dashboard-wide.
COLOR_FEMALE = "#E69F00"   # orange
COLOR_MALE = "#0072B2"     # blue
CATEGORICAL_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
]
# Sequential palette for the choropleth and heatmap (perceptually uniform,
# colorblind-safe).
SEQUENTIAL_SCALE = "Viridis"

REQUIRED_COLUMNS = [
    "state_of_residence", "month", "month_code", "year_code",
    "sex_of_infant", "births",
]
VALID_SEXES = {"Female", "Male"}

# Resolved relative to this file, not the working directory, so the app finds
# the CSV whether it's run locally or deployed on Streamlit Community Cloud.
DATA_PATH = Path(__file__).resolve().parent / "Provisional_Natality_2025_CDC.csv"

SEX_OPTIONS = ["Female", "Male"]
STATE_KEY = "filter_states"
MONTH_KEY = "filter_months"
SEX_KEY = "filter_sex"


# ============================================================================
# Data loading and validation
# ============================================================================

class DataValidationError(Exception):
    """Raised when the source CSV fails a sanity check."""


def _validate(df: pd.DataFrame) -> None:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing expected column(s): {missing_cols}")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise DataValidationError("Found null values in one or more required columns.")

    if not (df["births"] >= 0).all():
        raise DataValidationError("Found negative birth counts.")

    bad_sex = set(df["sex_of_infant"].unique()) - VALID_SEXES
    if bad_sex:
        raise DataValidationError(f"Unexpected sex_of_infant value(s): {bad_sex}")

    unknown_states = set(df["state_of_residence"].unique()) - set(STATE_ABBREV.keys())
    if unknown_states:
        raise DataValidationError(
            f"State name(s) not found in abbreviation mapping: {unknown_states}"
        )

    months_present = set(df["month"].unique())
    if not months_present.issubset(set(MONTH_ORDER)):
        raise DataValidationError(f"Unrecognized month value(s): {months_present - set(MONTH_ORDER)}")


@st.cache_data
def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load, validate, and lightly enrich the natality CSV.

    Returns a DataFrame with `month` as an ordered Categorical (Jan -> Dec)
    and a `state_abbrev` column added for map plotting.
    """
    df = pd.read_csv(path)
    _validate(df)

    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)
    df["state_abbrev"] = df["state_of_residence"].map(STATE_ABBREV)

    return df


# ============================================================================
# Sidebar filters
# ============================================================================

def _init_defaults(all_states: list[str]) -> None:
    st.session_state.setdefault(STATE_KEY, all_states)
    st.session_state.setdefault(MONTH_KEY, MONTH_ORDER.copy())
    st.session_state.setdefault(SEX_KEY, SEX_OPTIONS.copy())


def _select_all(all_states: list[str]) -> None:
    st.session_state[STATE_KEY] = all_states
    st.session_state[MONTH_KEY] = MONTH_ORDER.copy()
    st.session_state[SEX_KEY] = SEX_OPTIONS.copy()


def _reset(all_states: list[str]) -> None:
    # Reset currently means "back to everything selected," matching the
    # dashboard's default (unfiltered) view.
    _select_all(all_states)


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar widgets and return the filtered DataFrame."""
    all_states = sorted(df["state_of_residence"].unique())
    _init_defaults(all_states)

    st.sidebar.header("Filters")

    col_a, col_b = st.sidebar.columns(2)
    col_a.button("Select All", on_click=_select_all, args=(all_states,), use_container_width=True)
    col_b.button("Reset Filters", on_click=_reset, args=(all_states,), use_container_width=True)

    st.sidebar.multiselect("State / geography", options=all_states, key=STATE_KEY)
    st.sidebar.multiselect("Month", options=MONTH_ORDER, key=MONTH_KEY)
    st.sidebar.multiselect("Infant sex", options=SEX_OPTIONS, key=SEX_KEY)

    selected_states = st.session_state[STATE_KEY]
    selected_months = st.session_state[MONTH_KEY]
    selected_sexes = st.session_state[SEX_KEY]

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**Active filters:** {len(selected_states)} of {len(all_states)} geographies · "
        f"{len(selected_months)} of 12 months · "
        f"sex: {', '.join(selected_sexes) if selected_sexes else 'none selected'}"
    )

    filtered = df[
        df["state_of_residence"].isin(selected_states)
        & df["month"].isin(selected_months)
        & df["sex_of_infant"].isin(selected_sexes)
    ]
    return filtered


# ============================================================================
# KPI calculations
# ============================================================================

def total_births(df: pd.DataFrame) -> int:
    return int(df["births"].sum())


def geography_count(df: pd.DataFrame) -> int:
    return df["state_of_residence"].nunique()


def avg_births_per_month(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    by_month = df.groupby("month", observed=True)["births"].sum()
    return float(by_month.mean()) if not by_month.empty else 0.0


def top_geography(df: pd.DataFrame) -> tuple[str, int] | None:
    if df.empty:
        return None
    by_state = df.groupby("state_of_residence")["births"].sum()
    top = by_state.idxmax()
    return top, int(by_state.loc[top])


def top_month(df: pd.DataFrame) -> tuple[str, int] | None:
    if df.empty:
        return None
    by_month = df.groupby("month", observed=True)["births"].sum()
    top = by_month.idxmax()
    return str(top), int(by_month.loc[top])


# ============================================================================
# Chart builders
# ============================================================================
# Every function takes the filtered DataFrame and returns a Plotly Figure,
# or None if there's nothing to plot (empty selection) so the layout code
# can show a consistent "no data" message instead of a blank chart. Axes
# are never truncated to a non-zero baseline for bar/choropleth values.

def monthly_trend(df: pd.DataFrame) -> go.Figure | None:
    """Total births by month, summed across current selection."""
    if df.empty:
        return None
    by_month = (
        df.groupby("month", observed=False)["births"].sum().reindex(MONTH_ORDER).reset_index()
    )
    fig = px.line(
        by_month, x="month", y="births", markers=True,
        title="Monthly Birth Trend",
        labels={"month": "Month", "births": "Total Births"},
        color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
    )
    fig.update_traces(hovertemplate="%{x}: %{y:,} births<extra></extra>")
    fig.update_yaxes(rangemode="tozero", tickformat=",")
    return fig


def sex_comparison(df: pd.DataFrame) -> go.Figure | None:
    """Female vs. male births by month, grouped bars."""
    if df.empty:
        return None
    by_month_sex = (
        df.groupby(["month", "sex_of_infant"], observed=False)["births"]
        .sum()
        .reset_index()
    )
    fig = px.bar(
        by_month_sex, x="month", y="births", color="sex_of_infant",
        barmode="group",
        category_orders={"month": MONTH_ORDER},
        title="Female vs. Male Births by Month",
        labels={"month": "Month", "births": "Total Births", "sex_of_infant": "Sex"},
        color_discrete_map={"Female": COLOR_FEMALE, "Male": COLOR_MALE},
    )
    fig.update_traces(hovertemplate="%{x}, %{fullData.name}: %{y:,} births<extra></extra>")
    fig.update_yaxes(rangemode="tozero", tickformat=",")
    return fig


def state_ranking(df: pd.DataFrame, top_n: int | None = None) -> go.Figure | None:
    """Horizontal bar of total births by geography, descending."""
    if df.empty:
        return None
    by_state = df.groupby("state_of_residence")["births"].sum().sort_values(ascending=False)
    if top_n:
        by_state = by_state.head(top_n)
    fig = px.bar(
        by_state.sort_values().reset_index(),
        x="births", y="state_of_residence", orientation="h",
        title="Births by Geography" if not top_n else f"Top {top_n} Geographies by Births",
        labels={"births": "Total Births", "state_of_residence": "Geography"},
        color_discrete_sequence=[CATEGORICAL_PALETTE[0]],
    )
    fig.update_traces(hovertemplate="%{y}: %{x:,} births<extra></extra>")
    fig.update_xaxes(rangemode="tozero", tickformat=",")
    fig.update_layout(height=max(400, 18 * len(by_state)))
    return fig


def choropleth_map(df: pd.DataFrame) -> go.Figure | None:
    """US choropleth of total births by state."""
    if df.empty:
        return None
    by_state = df.groupby(["state_of_residence", "state_abbrev"])["births"].sum().reset_index()
    fig = px.choropleth(
        by_state, locations="state_abbrev", locationmode="USA-states",
        color="births", scope="usa", color_continuous_scale=SEQUENTIAL_SCALE,
        title="Total Births by State",
        hover_name="state_of_residence",
        labels={"births": "Total Births"},
    )
    fig.update_traces(hovertemplate="%{hovertext}: %{z:,} births<extra></extra>")
    return fig


def state_month_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """State x month heatmap of total births."""
    if df.empty:
        return None
    pivot = (
        df.groupby(["state_of_residence", "month"], observed=False)["births"]
        .sum()
        .unstack("month")
        .reindex(columns=MONTH_ORDER)
    )
    # Order states by total descending so the heatmap reads top-to-bottom
    # from highest to lowest volume.
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(
        pivot, aspect="auto", color_continuous_scale=SEQUENTIAL_SCALE,
        title="Births by State and Month",
        labels={"x": "Month", "y": "Geography", "color": "Births"},
    )
    fig.update_traces(hovertemplate="%{y}, %{x}: %{z:,} births<extra></extra>")
    fig.update_layout(height=max(500, 16 * len(pivot)))
    return fig


def top_bottom_comparison(df: pd.DataFrame, n: int = 5) -> go.Figure | None:
    """Side-by-side comparison of the top-N and bottom-N geographies."""
    if df.empty:
        return None
    by_state = df.groupby("state_of_residence")["births"].sum().sort_values(ascending=False)
    if len(by_state) < 2:
        return None
    n = min(n, len(by_state) // 2) or 1
    top = by_state.head(n).reset_index()
    top["group"] = f"Top {n}"
    bottom = by_state.tail(n).reset_index()
    bottom["group"] = f"Bottom {n}"
    combined = pd.concat([top, bottom]).sort_values("births")
    fig = px.bar(
        combined, x="births", y="state_of_residence", color="group", orientation="h",
        title=f"Top {n} vs. Bottom {n} Geographies",
        labels={"births": "Total Births", "state_of_residence": "Geography", "group": ""},
        color_discrete_map={f"Top {n}": CATEGORICAL_PALETTE[0], f"Bottom {n}": CATEGORICAL_PALETTE[1]},
    )
    fig.update_traces(hovertemplate="%{y}: %{x:,} births<extra></extra>")
    fig.update_xaxes(rangemode="tozero", tickformat=",")
    return fig


# ============================================================================
# Page layout
# ============================================================================

st.set_page_config(
    page_title="CDC Provisional Natality Dashboard 2025",
    page_icon="👶",
    layout="wide",
)


def render_header() -> None:
    st.title("CDC Provisional Natality Dashboard — 2025")
    st.markdown(
        "Explore 2025 U.S. birth counts by state, month, and infant sex using "
        "provisional CDC data. Use the sidebar filters to narrow the selection."
    )
    st.warning(
        "**Provisional data:** These 2025 figures are preliminary and subject to "
        "revision as more complete records become available.",
        icon="⚠️",
    )
    st.info(
        "**Counts, not rates:** All figures shown are raw birth *counts*, not "
        "birth *rates*. They are not adjusted for population size, so larger "
        "states will naturally show higher counts.",
        icon="ℹ️",
    )
    st.caption("Source: CDC National Center for Health Statistics, Provisional Natality data, 2025.")


def render_kpis(df: pd.DataFrame) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Births", f"{total_births(df):,}")
    col2.metric("Geographies Selected", f"{geography_count(df):,}")
    col3.metric("Avg. Births / Month", f"{avg_births_per_month(df):,.0f}")

    top_geo = top_geography(df)
    col4.metric(
        "Top Geography",
        top_geo[0] if top_geo else "—",
        f"{top_geo[1]:,} births" if top_geo else None,
    )

    top_mo = top_month(df)
    col5.metric(
        "Top Month",
        top_mo[0] if top_mo else "—",
        f"{top_mo[1]:,} births" if top_mo else None,
    )


def _show_or_empty_message(fig: go.Figure | None) -> None:
    if fig is None:
        st.info("No data for the current filter selection. Try adjusting the filters in the sidebar.")
    else:
        st.plotly_chart(fig, use_container_width=True)


def render_overview_tab(df: pd.DataFrame) -> None:
    _show_or_empty_message(monthly_trend(df))
    _show_or_empty_message(sex_comparison(df))


def render_geographic_tab(df: pd.DataFrame) -> None:
    _show_or_empty_message(choropleth_map(df))
    _show_or_empty_message(state_ranking(df))


def render_monthly_sex_tab(df: pd.DataFrame) -> None:
    _show_or_empty_message(state_month_heatmap(df))
    _show_or_empty_message(top_bottom_comparison(df))


def render_data_tab(df: pd.DataFrame) -> None:
    st.subheader("Filtered Data")
    search = st.text_input("Search (matches any column)", "")
    display_df = df.drop(columns=["state_abbrev"])
    if search:
        mask = display_df.apply(
            lambda col: col.astype(str).str.contains(search, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(f"{len(display_df):,} rows shown.")

    st.download_button(
        "Download filtered data as CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_natality_data.csv",
        mime="text/csv",
    )


def render_about_tab() -> None:
    st.subheader("About the Data")
    st.markdown(
        """
- **Source:** CDC National Center for Health Statistics (NCHS), Provisional Natality data for 2025.
- **Provisional status:** These figures are preliminary. Provisional data are released ahead of
  final data to give a timely picture, but counts can change — sometimes materially for the most
  recent months — once late-reported and corrected birth certificates are incorporated.
- **Counts vs. rates:** This dashboard shows birth *counts* (raw numbers of births), not birth
  *rates* (births relative to population). A state with a high count may still have a low rate if
  its population is large; comparing counts across states of very different sizes should be done
  with that in mind.
- **Geography:** "State" here means state (or D.C.) of maternal residence, not the state where the
  birth occurred.
- **Granularity:** Data are reported by state, month, and infant sex for 2025.
        """
    )


def main() -> None:
    try:
        df = load_data()
    except DataValidationError as e:
        st.error(f"Data failed validation and could not be loaded: {e}")
        st.stop()

    render_header()
    filtered_df = render_filters(df)

    st.markdown("---")
    render_kpis(filtered_df)
    st.markdown("---")

    tab_overview, tab_geo, tab_monthly_sex, tab_data, tab_about = st.tabs(
        ["Overview", "Geographic Analysis", "Monthly and Sex Analysis", "Data Table and Download", "About the Data"]
    )
    with tab_overview:
        render_overview_tab(filtered_df)
    with tab_geo:
        render_geographic_tab(filtered_df)
    with tab_monthly_sex:
        render_monthly_sex_tab(filtered_df)
    with tab_data:
        render_data_tab(filtered_df)
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
