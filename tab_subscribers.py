"""
Subscribers Tab - CRM Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shared import get_gspread_client, show_metric, format_number


@st.cache_data(ttl=300, show_spinner=False)  # Cache i 5 minutter
def load_subscribers_data():
    """Henter Subscribers data fra Google Sheet"""
    try:
        gc = get_gspread_client()
        
        # Tjek om subscribers_spreadsheet er konfigureret
        if "subscribers_spreadsheet" not in st.secrets["connections"]["gsheets"]:
            st.error("⚠️ Mangler 'subscribers_spreadsheet' i secrets. Tilføj: subscribers_spreadsheet = 'URL'")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        subscribers_url = st.secrets["connections"]["gsheets"]["subscribers_spreadsheet"]
        spreadsheet = gc.open_by_url(subscribers_url)
        
        # Hent worksheets
        full_subs = spreadsheet.worksheet("Full_Subscribers").get_all_values()
        light_subs = spreadsheet.worksheet("Light_Subscribers").get_all_values()
        sub_events = spreadsheet.worksheet("Full_Sub_Events").get_all_values()
        
        # Konverter til DataFrames
        full_df = pd.DataFrame(full_subs[1:], columns=full_subs[0]) if len(full_subs) > 1 else pd.DataFrame()
        light_df = pd.DataFrame(light_subs[1:], columns=light_subs[0]) if len(light_subs) > 1 else pd.DataFrame()
        events_df = pd.DataFrame(sub_events[1:], columns=sub_events[0]) if len(sub_events) > 1 else pd.DataFrame()
        
        # Konverter numeriske kolonner
        country_cols = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK', 'DE', 'AT', 'NL', 'BE', 'CH', 'Total']
        
        for df in [full_df, light_df]:
            if not df.empty:
                for col in country_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('"', ''), errors='coerce').fillna(0).astype(int)
                if 'Month' in df.columns:
                    df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m', errors='coerce')
        
        # Events har flere kolonner
        if not events_df.empty:
            for col in country_cols:
                if col in events_df.columns:
                    events_df[col] = pd.to_numeric(events_df[col].astype(str).str.replace(',', '').str.replace('"', ''), errors='coerce').fillna(0).astype(int)
            if 'Month' in events_df.columns:
                events_df['Month'] = pd.to_datetime(events_df['Month'], format='%Y-%m', errors='coerce')
        
        return full_df, light_df, events_df
        
    except Exception as e:
        st.error(f"Fejl ved indlæsning af Subscribers data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def format_month_short(month_dt):
    """Formater måned til kort dansk format (Jan 25, Feb 25, osv.)"""
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Maj', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Dec'
    }
    return f"{month_names[month_dt.month]} {str(month_dt.year)[2:]}"


def render_overview_tab(full_df, light_df):
    """Render Oversigt sub-tab med scorecards og graf"""
    
    # Sorter efter dato
    if not full_df.empty:
        full_df = full_df.sort_values('Month', ascending=False)
    if not light_df.empty:
        light_df = light_df.sort_values('Month', ascending=False)
    
    # --- FILTRE: Land og Periode ---
    all_countries = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK', 'DE', 'AT', 'NL', 'BE', 'CH']
    
    # Session state for land selection
    if 'sub_selected_countries' not in st.session_state:
        st.session_state.sub_selected_countries = list(all_countries)
    if 'sub_cb_reset_land' not in st.session_state:
        st.session_state.sub_cb_reset_land = 0
    
    # Find tilgængelige måneder fra data
    all_months = set()
    if not full_df.empty:
        all_months.update(full_df['Month'].dropna().tolist())
    if not light_df.empty:
        all_months.update(light_df['Month'].dropna().tolist())
    
    available_months = sorted(list(all_months))
    
    if not available_months:
        st.warning("Ingen data tilgængelig.")
        return
    
    # Seneste måned som default
    current_month = available_months[-1]
    
    # Filter row
    col_land, col_slider = st.columns([1, 5])
    
    # Land filter med popover
    with col_land:
        land_count = len(st.session_state.sub_selected_countries)
        land_label = f"Land ({land_count})" if land_count < len(all_countries) else "Land"
        with st.popover(land_label, use_container_width=True):
            reset_land = st.session_state.sub_cb_reset_land
            all_land_selected = len(st.session_state.sub_selected_countries) == len(all_countries)
            select_all_land = st.checkbox("Vælg alle", value=all_land_selected, key=f"sub_sel_all_land_{reset_land}")
            
            new_selected = []
            only_clicked_land = None
            for country in all_countries:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    checked = country in st.session_state.sub_selected_countries
                    if st.checkbox(country, value=checked, key=f"sub_cb_land_{country}_{reset_land}"):
                        new_selected.append(country)
                with only_col:
                    if st.button("Kun", key=f"sub_only_{country}_{reset_land}"):
                        only_clicked_land = country
            
            # Handle Kun button click first
            if only_clicked_land:
                st.session_state.sub_selected_countries = [only_clicked_land]
                st.session_state.sub_cb_reset_land += 1
                st.rerun()
            elif select_all_land and not all_land_selected:
                st.session_state.sub_selected_countries = list(all_countries)
                st.session_state.sub_cb_reset_land += 1
                st.rerun()
            elif not select_all_land and all_land_selected:
                st.session_state.sub_selected_countries = []
                st.session_state.sub_cb_reset_land += 1
                st.rerun()
            elif set(new_selected) != set(st.session_state.sub_selected_countries):
                st.session_state.sub_selected_countries = new_selected
                st.session_state.sub_cb_reset_land += 1
                st.rerun()
    
    selected_countries = st.session_state.sub_selected_countries
    
    with col_slider:
        if len(available_months) > 1:
            # Range slider med to bobler - begge på denne måned som default
            month_range = st.select_slider(
                "Periode",
                options=available_months,
                value=(current_month, current_month),  # Begge på denne måned
                format_func=format_month_short,
                key="sub_overview_period",
                label_visibility="collapsed"
            )
            start_month, end_month = month_range
        else:
            start_month = available_months[0]
            end_month = available_months[0]
            st.write(f"Periode: {format_month_short(start_month)}")
    
    # Filtrer måneder i range
    selected_months = [m for m in available_months if start_month <= m <= end_month]
    
    # For sammenligning: find måneden før start_month
    start_idx = available_months.index(start_month)
    prev_month = available_months[start_idx - 1] if start_idx > 0 else None

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Check at mindst ét land er valgt
    if not selected_countries:
        st.warning("Vælg mindst ét land.")
        return

    # --- KPI CARDS ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Hent data for seneste valgte måned (end_month) og summér over valgte lande
    if not full_df.empty:
        current_full_row = full_df[full_df['Month'] == end_month]
        current_full = current_full_row[selected_countries].sum(axis=1).values[0] if not current_full_row.empty else 0
        
        if prev_month is not None:
            prev_full_row = full_df[full_df['Month'] == prev_month]
            prev_full = prev_full_row[selected_countries].sum(axis=1).values[0] if not prev_full_row.empty else None
        else:
            prev_full = None
        
        full_growth = (current_full - prev_full) if prev_full is not None else 0
    else:
        current_full = 0
        prev_full = None
        full_growth = 0

    if not light_df.empty:
        current_light_row = light_df[light_df['Month'] == end_month]
        current_light = current_light_row[selected_countries].sum(axis=1).values[0] if not current_light_row.empty else 0
        
        if prev_month is not None:
            prev_light_row = light_df[light_df['Month'] == prev_month]
            prev_light = prev_light_row[selected_countries].sum(axis=1).values[0] if not prev_light_row.empty else None
        else:
            prev_light = None
        
        light_growth = (current_light - prev_light) if prev_light is not None else 0
    else:
        current_light = 0
        prev_light = None
        light_growth = 0

    total_subscribers = current_full + current_light
    prev_total = (prev_full or 0) + (prev_light or 0) if (prev_full is not None or prev_light is not None) else None
    total_growth = full_growth + light_growth

    show_metric(col1, "Full Subscribers", current_full, prev_full)
    show_metric(col2, "Light Subscribers", current_light, prev_light)
    show_metric(col3, "Total Subscribers", total_subscribers, prev_total)
    
    # Nye subscribers - vis med +/- prefix
    col4.metric("Nye Full", f"+{format_number(full_growth)}" if full_growth >= 0 else format_number(full_growth))
    col5.metric("Nye Light", f"+{format_number(light_growth)}" if light_growth >= 0 else format_number(light_growth))
    col6.metric("Nye Total", f"+{format_number(total_growth)}" if total_growth >= 0 else format_number(total_growth))

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # --- GRAF: Subscriber vækst over tid ---
    if not full_df.empty or not light_df.empty:
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Sorter kronologisk for graf og summér over valgte lande
        if not full_df.empty:
            full_chart = full_df.sort_values('Month').copy()
            full_chart['Selected_Total'] = full_chart[selected_countries].sum(axis=1)
            fig.add_trace(
                go.Scatter(
                    x=full_chart['Month'], y=full_chart['Selected_Total'],
                    name='Full Subscribers', mode='lines+markers',
                    line=dict(color='#9B7EBD', width=3),
                    marker=dict(size=8)
                )
            )
        
        if not light_df.empty:
            light_chart = light_df.sort_values('Month').copy()
            light_chart['Selected_Total'] = light_chart[selected_countries].sum(axis=1)
            fig.add_trace(
                go.Scatter(
                    x=light_chart['Month'], y=light_chart['Selected_Total'],
                    name='Light Subscribers', mode='lines+markers',
                    line=dict(color='#E8B4CB', width=3),
                    marker=dict(size=8)
                )
            )
        
        # Tilføj markering for valgt periode
        if start_month == end_month:
            # Enkelt måned - vis vertikal linje
            fig.add_vline(
                x=end_month.to_pydatetime(),
                line_dash="dash",
                line_color="#9B7EBD",
                opacity=0.5
            )
        else:
            # Range - vis skraveret område
            fig.add_vrect(
                x0=start_month.to_pydatetime(),
                x1=end_month.to_pydatetime(),
                fillcolor="#D4BFFF",
                opacity=0.2,
                line_width=0
            )
        
        fig.update_layout(
            title="", showlegend=True, height=400,
            margin=dict(l=50, r=50, t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(250,245,255,0.5)', paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', tickformat='%Y-%m', automargin=True, ticklabelstandoff=10)
        fig.update_yaxes(gridcolor='rgba(212,191,255,0.3)', tickformat=',', automargin=True, ticklabelstandoff=10)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_full_subscribers_tab(full_df):
    """Render Full Subscribers sub-tab med tabel"""
    country_cols = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK', 'DE', 'AT', 'NL', 'BE', 'CH', 'Total']
    
    if not full_df.empty:
        full_df = full_df.sort_values('Month', ascending=False)
        display_full = full_df.copy()
        display_full['Month'] = display_full['Month'].dt.strftime('%Y-%m')
        cols_to_show = ['Month'] + [c for c in country_cols if c in display_full.columns]
        
        st.dataframe(
            display_full[cols_to_show],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Month": st.column_config.TextColumn("Måned", width="small"),
                **{col: st.column_config.NumberColumn(col, format="localized", width="small") for col in country_cols if col in display_full.columns}
            }
        )
    else:
        st.info("Ingen Full Subscribers data.")


def render_light_subscribers_tab(light_df):
    """Render Light Subscribers sub-tab med tabel"""
    country_cols = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK', 'DE', 'AT', 'NL', 'BE', 'CH', 'Total']
    
    if not light_df.empty:
        light_df = light_df.sort_values('Month', ascending=False)
        display_light = light_df.copy()
        display_light['Month'] = display_light['Month'].dt.strftime('%Y-%m')
        cols_to_show = ['Month'] + [c for c in country_cols if c in display_light.columns]
        
        st.dataframe(
            display_light[cols_to_show],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Month": st.column_config.TextColumn("Måned", width="small"),
                **{col: st.column_config.NumberColumn(col, format="localized", width="small") for col in country_cols if col in display_light.columns}
            }
        )
    else:
        st.info("Ingen Light Subscribers data.")


def render_nye_subscribers_tab(events_df):
    """Render Nye Subscribers per Kilde sub-tab med tabel og filtre"""
    country_cols = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK', 'DE', 'AT', 'NL', 'BE', 'CH', 'Total']
    
    if not events_df.empty:
        display_events = events_df.copy()
        display_events['Month'] = display_events['Month'].dt.strftime('%Y-%m')
        
        # Filter muligheder
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            master_sources = ['Alle'] + sorted(display_events['Master Source'].unique().tolist())
            selected_master = st.selectbox("Master Source", master_sources, key="sub_master_source")
        
        with col_filter2:
            if selected_master != 'Alle':
                sources = ['Alle'] + sorted(display_events[display_events['Master Source'] == selected_master]['Source'].unique().tolist())
            else:
                sources = ['Alle'] + sorted(display_events['Source'].unique().tolist())
            selected_source = st.selectbox("Source", sources, key="sub_source")
        
        # Filtrer data
        filtered_events = display_events.copy()
        if selected_master != 'Alle':
            filtered_events = filtered_events[filtered_events['Master Source'] == selected_master]
        if selected_source != 'Alle':
            filtered_events = filtered_events[filtered_events['Source'] == selected_source]
        
        cols_to_show = ['Month', 'Master Source', 'Source'] + [c for c in country_cols if c in filtered_events.columns]
        
        st.dataframe(
            filtered_events[cols_to_show].sort_values('Month', ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Month": st.column_config.TextColumn("Måned", width="small"),
                "Master Source": st.column_config.TextColumn("Master Source", width="medium"),
                "Source": st.column_config.TextColumn("Source", width="medium"),
                **{col: st.column_config.NumberColumn(col, format="localized", width="small") for col in country_cols if col in filtered_events.columns}
            }
        )
    else:
        st.info("Ingen subscriber events data.")


def render_subscribers_tab():
    """Render Subscribers tab indhold"""
    
    # Load data
    try:
        with st.spinner('Henter subscriber data...'):
            full_df, light_df, events_df = load_subscribers_data()
        
        if full_df.empty and light_df.empty:
            st.error("Kunne ikke hente subscriber data.")
            return
    except Exception as e:
        st.error(f"Fejl: {e}")
        return

    # --- SUB-TABS ---
    sub_tab_oversigt, sub_tab_full, sub_tab_light, sub_tab_kilder = st.tabs([
        "Oversigt", "Full Subscribers", "Light Subscribers", "Nye Subscribers per Kilde"
    ])
    
    with sub_tab_oversigt:
        render_overview_tab(full_df.copy(), light_df.copy())
    
    with sub_tab_full:
        render_full_subscribers_tab(full_df.copy())
    
    with sub_tab_light:
        render_light_subscribers_tab(light_df.copy())
    
    with sub_tab_kilder:
        render_nye_subscribers_tab(events_df.copy())

    if st.button('Opdater Data', key="sub_refresh"):
        st.rerun()
