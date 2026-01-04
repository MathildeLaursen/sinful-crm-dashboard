"""
Nye Subscribers Tab - CRM Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from shared import (get_gspread_client, format_number, style_graph, 
                    COUNTRY_ORDER, COUNTRY_ROW1, COUNTRY_ROW2, 
                    get_colors_for_categories, get_color_for_category,
                    format_month_short, get_month_progress, is_current_month)


@st.cache_data(ttl=300, show_spinner=False)
def load_nye_subscribers_data():
    """Henter Subscriber Events data fra Google Sheet"""
    try:
        gc = get_gspread_client()
        
        if "subscribers_spreadsheet" not in st.secrets["connections"]["gsheets"]:
            st.error("Mangler 'subscribers_spreadsheet' i secrets.")
            return pd.DataFrame()
        
        subscribers_url = st.secrets["connections"]["gsheets"]["subscribers_spreadsheet"]
        spreadsheet = gc.open_by_url(subscribers_url)
        
        sub_events = spreadsheet.worksheet("Full_Sub_Events").get_all_values()
        events_df = pd.DataFrame(sub_events[1:], columns=sub_events[0]) if len(sub_events) > 1 else pd.DataFrame()
        
        if not events_df.empty:
            country_cols = COUNTRY_ORDER + ['Total']
            for col in country_cols:
                if col in events_df.columns:
                    events_df[col] = pd.to_numeric(events_df[col].astype(str).str.replace(',', '').str.replace('"', ''), errors='coerce').fillna(0).astype(int)
            if 'Month' in events_df.columns:
                events_df['Month'] = pd.to_datetime(events_df['Month'], format='%Y-%m', errors='coerce')
        
        return events_df
        
    except Exception as e:
        st.error(f"Fejl ved hentning af data: {e}")
        return pd.DataFrame()


def render_nye_subscribers_tab():
    """Render Nye Subscribers per Kilde fane"""
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    try:
        with st.spinner('Henter subscriber events data...'):
            events_df = load_nye_subscribers_data()
        
        if events_df.empty:
            st.error("Kunne ikke hente subscriber events data.")
            return
    except Exception as e:
        st.error(f"Fejl: {e}")
        return
    
    render_nye_subscribers_content(events_df.copy())
    
    if st.button('Opdater Data', key="nye_sub_refresh"):
        st.rerun()


def render_nye_subscribers_content(events_df):
    """Render Nye Subscribers per Kilde indhold med underfaner per Master Source"""
    country_cols = COUNTRY_ORDER + ['Total']
    
    if not events_df.empty:
        display_events = events_df.copy()
        display_events['Month'] = pd.to_datetime(display_events['Month'])
        
        # Fjern LightPermission fra data
        display_events = display_events[display_events['Master Source'] != 'LightPermission']
        
        # Definér rækkefølge for Master Sources
        master_source_order = ['On Site', 'Game', 'Lead Ad', 'Sleeknote', 'Other']
        available_sources = display_events['Master Source'].unique().tolist()
        master_sources = [ms for ms in master_source_order if ms in available_sources]
        
        # Generer farver dynamisk for Master Sources
        master_colors = get_colors_for_categories(master_sources)
        
        # Opret sub-tabs: Oversigt + en per Master Source
        tab_names = ["Oversigt"] + master_sources
        source_tabs = st.tabs(tab_names)
        
        # --- OVERSIGT TAB ---
        with source_tabs[0]:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            
            # --- FILTRE: Land og Periode ---
            all_countries = COUNTRY_ORDER
            
            # Session state for land selection
            if 'kilder_selected_countries' not in st.session_state:
                st.session_state.kilder_selected_countries = list(all_countries)
            if 'kilder_cb_reset_land' not in st.session_state:
                st.session_state.kilder_cb_reset_land = 0
            
            # Find tilgængelige måneder
            available_months = sorted(display_events['Month'].unique())
            current_month = available_months[-1] if available_months else None
            
            # Filter row: Land + Slider
            col_land, col_slider = st.columns([1, 5])
            
            # Land filter med popover
            with col_land:
                land_count = len(st.session_state.kilder_selected_countries)
                land_label = f"Land ({land_count})" if land_count < len(all_countries) else "Land"
                with st.popover(land_label, use_container_width=True):
                    reset_land = st.session_state.kilder_cb_reset_land
                    all_land_selected = len(st.session_state.kilder_selected_countries) == len(all_countries)
                    select_all_land = st.checkbox("Vælg alle", value=all_land_selected, key=f"kilder_sel_all_land_{reset_land}")
                    
                    new_selected = []
                    only_clicked_land = None
                    for country in all_countries:
                        cb_col, only_col = st.columns([4, 1])
                        with cb_col:
                            checked = country in st.session_state.kilder_selected_countries
                            if st.checkbox(country, value=checked, key=f"kilder_cb_land_{country}_{reset_land}"):
                                new_selected.append(country)
                        with only_col:
                            if st.button("Kun", key=f"kilder_only_{country}_{reset_land}"):
                                only_clicked_land = country
                    
                    # Handle selections
                    if only_clicked_land:
                        st.session_state.kilder_selected_countries = [only_clicked_land]
                        st.session_state.kilder_cb_reset_land += 1
                        st.rerun()
                    elif select_all_land and not all_land_selected:
                        st.session_state.kilder_selected_countries = list(all_countries)
                        st.session_state.kilder_cb_reset_land += 1
                        st.rerun()
                    elif not select_all_land and all_land_selected:
                        st.session_state.kilder_selected_countries = []
                        st.session_state.kilder_cb_reset_land += 1
                        st.rerun()
                    elif set(new_selected) != set(st.session_state.kilder_selected_countries):
                        st.session_state.kilder_selected_countries = new_selected
                        st.session_state.kilder_cb_reset_land += 1
                        st.rerun()
            
            selected_countries = st.session_state.kilder_selected_countries
            
            # Slider
            with col_slider:
                if len(available_months) > 1:
                    month_range = st.select_slider(
                        "Periode",
                        options=available_months,
                        value=(current_month, current_month),
                        format_func=format_month_short,
                        key="kilder_oversigt_period",
                        label_visibility="collapsed"
                    )
                    start_month, end_month = month_range
                else:
                    start_month = available_months[0] if available_months else None
                    end_month = start_month
            
            # Check at mindst ét land er valgt
            if not selected_countries:
                st.warning("Vælg mindst ét land.")
            else:
                # Aggregér data per måned og Master Source (sum over valgte lande)
                filtered_events = display_events.copy()
                filtered_events['Selected_Total'] = filtered_events[selected_countries].sum(axis=1)
                agg_df = filtered_events.groupby(['Month', 'Master Source'])['Selected_Total'].sum().reset_index()
                agg_df = agg_df.rename(columns={'Selected_Total': 'Total'})
                agg_df = agg_df.sort_values('Month')
                
                # Filtrer data til valgt periode
                period_df = agg_df[(agg_df['Month'] >= start_month) & (agg_df['Month'] <= end_month)]
                
                # Find valgte måneder
                selected_months = sorted(period_df['Month'].unique().tolist())
                num_months = len(selected_months)
                
                # Beregn måned progress
                month_progress = get_month_progress()
                
                # Tjek om nuværende måned er valgt
                current_month_selected = any(is_current_month(m) for m in selected_months)
                
                # Find sammenligningsperiode
                if selected_months and num_months > 0:
                    oldest_selected = selected_months[0]
                    oldest_idx = available_months.index(oldest_selected) if oldest_selected in available_months else -1
                    
                    prev_months = []
                    for i in range(num_months):
                        prev_idx = oldest_idx - 1 - i
                        if prev_idx >= 0:
                            prev_months.append(available_months[prev_idx])
                    
                    oldest_prev_month = min(prev_months) if prev_months else None
                else:
                    prev_months = []
                    oldest_prev_month = None
                
                # --- SCORECARDS ---
                num_sources = len(master_sources)
                cols = st.columns(num_sources + 1)
                
                grand_total = 0
                grand_prev_total = 0
                
                for i, master in enumerate(master_sources):
                    master_total = period_df[period_df['Master Source'] == master]['Total'].sum()
                    grand_total += master_total
                    
                    prev_total = 0
                    if prev_months:
                        for pm in prev_months:
                            pm_total = agg_df[(agg_df['Month'] == pm) & (agg_df['Master Source'] == master)]['Total'].sum()
                            
                            if pm == oldest_prev_month and current_month_selected:
                                prev_total += pm_total * month_progress
                            else:
                                prev_total += pm_total
                    
                    grand_prev_total += prev_total
                    
                    if prev_total > 0:
                        pct = ((master_total - prev_total) / prev_total * 100)
                        growth = int(master_total - prev_total)
                        growth_str = f"+{growth:,}" if growth >= 0 else f"{growth:,}"
                        cols[i].metric(master, format_number(master_total), delta=f"{pct:+.1f}% ({growth_str})")
                    else:
                        cols[i].metric(master, format_number(master_total))
                
                # Total scorecard
                if grand_prev_total > 0:
                    grand_pct = ((grand_total - grand_prev_total) / grand_prev_total * 100)
                    grand_growth = int(grand_total - grand_prev_total)
                    grand_growth_str = f"+{grand_growth:,}" if grand_growth >= 0 else f"{grand_growth:,}"
                    cols[num_sources].metric("Total", format_number(grand_total), delta=f"{grand_pct:+.1f}% ({grand_growth_str})")
                else:
                    cols[num_sources].metric("Total", format_number(grand_total))
                
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                
                # --- GRAF ---
                fig = go.Figure()
                for master in master_sources:
                    master_data = agg_df[agg_df['Master Source'] == master]
                    fig.add_trace(
                        go.Scatter(
                            x=master_data['Month'],
                            y=master_data['Total'],
                            name=master,
                            mode='lines+markers',
                            line=dict(color=master_colors.get(master, '#9B7EBD'), width=2),
                            marker=dict(size=6)
                        )
                    )
                
                style_graph(fig, height=500, legend_position='center',
                            x_tickformat='%Y-%m', y_tickformat=',')
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # --- MASTER SOURCE TABS ---
        # Generer farver dynamisk for lande
        country_colors = get_colors_for_categories(country_cols)
        
        masters_with_source_tabs = ['On Site', 'Game', 'Lead Ad']
        
        excluded_game_sources = ['LF_Birthday24', 'LF_Paaske_2021', 'LF_Paaske_2022']
        excluded_onsite_sources = ['Black_Friday_SignUp', 'Sinful_Live_Newsletter_SignUp']
        
        for i, master in enumerate(master_sources):
            with source_tabs[i + 1]:
                st.markdown(f"<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                master_df = display_events[display_events['Master Source'] == master].copy()
                
                if master == 'Game':
                    master_df = master_df[~master_df['Source'].isin(excluded_game_sources)]
                elif master == 'On Site':
                    master_df = master_df[~master_df['Source'].isin(excluded_onsite_sources)]
                
                if master in masters_with_source_tabs:
                    sources_list = sorted(master_df['Source'].unique().tolist())
                    
                    def clean_source_name(name):
                        return name.replace('LF_', '') if name.startswith('LF_') else name
                    
                    source_display_names = [clean_source_name(s) for s in sources_list]
                    
                    source_tab_names = ["Oversigt"] + source_display_names
                    source_sub_tabs = st.tabs(source_tab_names)
                    
                    with source_sub_tabs[0]:
                        render_source_content(master_df, country_cols, country_colors, f"kilder_{master}_oversigt", is_overview=True)
                    
                    for src_idx, source_name in enumerate(sources_list):
                        with source_sub_tabs[src_idx + 1]:
                            source_df = master_df[master_df['Source'] == source_name].copy()
                            render_source_content(source_df, country_cols, country_colors, f"kilder_{master}_{source_name}", is_overview=False)
                else:
                    render_source_content(master_df, country_cols, country_colors, f"kilder_{master}", is_overview=True)
    else:
        st.info("Ingen subscriber events data.")


def render_source_content(df, country_cols, country_colors, key_prefix, is_overview=True):
    """Render indhold for en source/master source med slider, scorecards, graf og tabel"""
    if df.empty:
        st.info("Ingen data.")
        return
    
    # Aggreger per måned og land
    agg_df = df.groupby('Month')[country_cols].sum().reset_index()
    agg_df = agg_df.sort_values('Month')
    
    # Beregn Total kolonne
    agg_df['Total'] = agg_df[country_cols].sum(axis=1)
    
    # --- SLIDER ---
    available_months = sorted(agg_df['Month'].unique().tolist())
    current_month_val = available_months[-1] if available_months else None
    
    if len(available_months) > 1:
        month_range = st.select_slider(
            "Periode",
            options=available_months,
            value=(current_month_val, current_month_val),
            format_func=format_month_short,
            key=f"{key_prefix}_period",
            label_visibility="collapsed"
        )
        start_month, end_month = month_range
    else:
        start_month = available_months[0] if available_months else None
        end_month = start_month
    
    # Filtrer data til valgt periode
    period_df = agg_df[(agg_df['Month'] >= start_month) & (agg_df['Month'] <= end_month)]
    
    # Find valgte måneder
    selected_months = sorted(period_df['Month'].unique().tolist())
    num_months = len(selected_months)
    
    # Beregn måned progress
    month_progress = get_month_progress()
    
    # Tjek om nuværende måned er valgt
    current_month_selected = any(is_current_month(m) for m in selected_months)
    
    # Find sammenligningsperiode
    if selected_months and num_months > 0:
        oldest_selected = selected_months[0]
        oldest_idx = available_months.index(oldest_selected) if oldest_selected in available_months else -1
        
        prev_months = []
        for idx in range(num_months):
            prev_idx = oldest_idx - 1 - idx
            if prev_idx >= 0:
                prev_months.append(available_months[prev_idx])
        
        oldest_prev_month = min(prev_months) if prev_months else None
    else:
        prev_months = []
        oldest_prev_month = None
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # --- SCORECARDS PER LAND ---
    row1_countries = COUNTRY_ROW1
    cols_row1 = st.columns(6)
    for j, country in enumerate(row1_countries):
        if country in agg_df.columns:
            current = period_df[country].sum()
            
            prev = 0
            if prev_months:
                for pm in prev_months:
                    pm_row = agg_df[agg_df['Month'] == pm]
                    if not pm_row.empty:
                        pm_val = pm_row.iloc[0][country]
                        if pm == oldest_prev_month and current_month_selected:
                            prev += pm_val * month_progress
                        else:
                            prev += pm_val
            
            if prev > 0:
                growth = current - prev
                pct = ((current - prev) / prev * 100)
                growth_str = f"+{growth:,.0f}" if growth >= 0 else f"{growth:,.0f}"
                cols_row1[j].metric(f"{country}", format_number(current), delta=f"{pct:+.1f}% ({growth_str})")
            else:
                cols_row1[j].metric(f"{country}", format_number(current))
        else:
            cols_row1[j].metric(f"{country}", "—")
    
    # Række 2: DE, AT, NL, BE, CH + Total
    row2_countries = COUNTRY_ROW2
    cols_row2 = st.columns(6)
    for j, country in enumerate(row2_countries):
        if country in agg_df.columns:
            current = period_df[country].sum()
            
            prev = 0
            if prev_months:
                for pm in prev_months:
                    pm_row = agg_df[agg_df['Month'] == pm]
                    if not pm_row.empty:
                        pm_val = pm_row.iloc[0][country]
                        if pm == oldest_prev_month and current_month_selected:
                            prev += pm_val * month_progress
                        else:
                            prev += pm_val
            
            if prev > 0:
                growth = current - prev
                pct = ((current - prev) / prev * 100)
                growth_str = f"+{growth:,.0f}" if growth >= 0 else f"{growth:,.0f}"
                cols_row2[j].metric(f"{country}", format_number(current), delta=f"{pct:+.1f}% ({growth_str})")
            else:
                cols_row2[j].metric(f"{country}", format_number(current))
        else:
            cols_row2[j].metric(f"{country}", "—")
    
    # Total kort
    if 'Total' in agg_df.columns:
        current_total = period_df['Total'].sum()
        
        prev_total = 0
        if prev_months:
            for pm in prev_months:
                pm_row = agg_df[agg_df['Month'] == pm]
                if not pm_row.empty:
                    pm_val = pm_row.iloc[0]['Total']
                    if pm == oldest_prev_month and current_month_selected:
                        prev_total += pm_val * month_progress
                    else:
                        prev_total += pm_val
        
        if prev_total > 0:
            total_growth = current_total - prev_total
            total_pct = ((current_total - prev_total) / prev_total * 100)
            total_growth_str = f"+{total_growth:,.0f}" if total_growth >= 0 else f"{total_growth:,.0f}"
            cols_row2[5].metric("Total", format_number(current_total), delta=f"{total_pct:+.1f}% ({total_growth_str})")
        else:
            cols_row2[5].metric("Total", format_number(current_total))
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # --- GRAF ---
    fig = go.Figure()
    for country in country_cols:
        if country in agg_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=agg_df['Month'],
                    y=agg_df[country],
                    name=country,
                    mode='lines+markers',
                    line=dict(color=country_colors.get(country, '#9B7EBD'), width=2),
                    marker=dict(size=6)
                )
            )
    
    style_graph(fig, height=500, legend_position='center', margin_top=60,
                x_tickformat='%Y-%m', y_tickformat=',')
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # --- TABEL ---
    if is_overview:
        df['Month_str'] = df['Month'].dt.strftime('%Y-%m')
        cols_to_show = ['Month_str', 'Source'] + [c for c in country_cols if c in df.columns]
        
        table_height = min(len(df) * 38 + 60, 1200)
        
        st.dataframe(
            df[cols_to_show].sort_values('Month_str', ascending=False),
            use_container_width=True,
            hide_index=True,
            height=table_height,
            column_config={
                "Month_str": st.column_config.TextColumn("Måned", width="small"),
                "Source": st.column_config.TextColumn("Source", width="medium"),
                **{col: st.column_config.NumberColumn(col, format="localized", width="small") for col in country_cols if col in df.columns}
            }
        )
    else:
        agg_df['Month_str'] = agg_df['Month'].dt.strftime('%Y-%m')
        cols_to_show = ['Month_str'] + [c for c in country_cols if c in agg_df.columns]
        
        table_height = min(len(agg_df) * 38 + 60, 800)
        
        st.dataframe(
            agg_df[cols_to_show].sort_values('Month_str', ascending=False),
            use_container_width=True,
            hide_index=True,
            height=table_height,
            column_config={
                "Month_str": st.column_config.TextColumn("Måned", width="small"),
                **{col: st.column_config.NumberColumn(col, format="localized", width="small") for col in country_cols if col in agg_df.columns}
            }
        )

