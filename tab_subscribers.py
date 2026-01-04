"""
Subscribers Tab - CRM Dashboard
"""
import streamlit as st
import pandas as pd
import datetime
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
    num_months = len(selected_months)
    
    # Beregn måned progress (hvor langt vi er i nuværende måned)
    import calendar
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    days_with_data = yesterday.day
    total_days_in_month = calendar.monthrange(today.year, today.month)[1]
    month_progress = days_with_data / total_days_in_month
    
    # Tjek om nuværende måned er valgt
    current_month_dt = pd.Timestamp(today.year, today.month, 1)
    current_month_selected = any(
        m.year == current_month_dt.year and m.month == current_month_dt.month 
        for m in selected_months
    )
    
    # Find sammenligningsperiode: N måneder FØR den ældste valgte måned
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

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Check at mindst ét land er valgt
    if not selected_countries:
        st.warning("Vælg mindst ét land.")
        return

    # --- KPI CARDS (3 kort med % og absolut ændring, samme bredde som før) ---
    col1, col2, col3, _, _, _ = st.columns(6)
    
    # Beregn current og previous for Full Subscribers
    current_full = 0
    prev_full = 0
    if not full_df.empty:
        # Sum over valgte måneder og lande
        for m in selected_months:
            m_row = full_df[full_df['Month'] == m]
            if not m_row.empty:
                current_full += m_row[selected_countries].sum(axis=1).values[0]
        
        # Sum over sammenligningsperiode med skalering
        for pm in prev_months:
            pm_row = full_df[full_df['Month'] == pm]
            if not pm_row.empty:
                pm_val = pm_row[selected_countries].sum(axis=1).values[0]
                if pm == oldest_prev_month and current_month_selected:
                    prev_full += pm_val * month_progress
                else:
                    prev_full += pm_val
    
    # Beregn current og previous for Light Subscribers
    current_light = 0
    prev_light = 0
    if not light_df.empty:
        # Sum over valgte måneder og lande
        for m in selected_months:
            m_row = light_df[light_df['Month'] == m]
            if not m_row.empty:
                current_light += m_row[selected_countries].sum(axis=1).values[0]
        
        # Sum over sammenligningsperiode med skalering
        for pm in prev_months:
            pm_row = light_df[light_df['Month'] == pm]
            if not pm_row.empty:
                pm_val = pm_row[selected_countries].sum(axis=1).values[0]
                if pm == oldest_prev_month and current_month_selected:
                    prev_light += pm_val * month_progress
                else:
                    prev_light += pm_val
    
    # Totaler
    total_subscribers = current_full + current_light
    prev_total = prev_full + prev_light
    
    # Full Subscribers med % og absolut
    if prev_full > 0:
        full_growth = current_full - prev_full
        full_pct = (full_growth / prev_full * 100)
        full_growth_str = f"+{full_growth:,.0f}" if full_growth >= 0 else f"{full_growth:,.0f}"
        col1.metric("Full Subscribers", format_number(current_full), delta=f"{full_pct:+.1f}% ({full_growth_str})")
    else:
        col1.metric("Full Subscribers", format_number(current_full))
    
    # Light Subscribers med % og absolut
    if prev_light > 0:
        light_growth = current_light - prev_light
        light_pct = (light_growth / prev_light * 100)
        light_growth_str = f"+{light_growth:,.0f}" if light_growth >= 0 else f"{light_growth:,.0f}"
        col2.metric("Light Subscribers", format_number(current_light), delta=f"{light_pct:+.1f}% ({light_growth_str})")
    else:
        col2.metric("Light Subscribers", format_number(current_light))
    
    # Total Subscribers med % og absolut
    if prev_total > 0:
        total_growth = total_subscribers - prev_total
        total_pct = (total_growth / prev_total * 100)
        total_growth_str = f"+{total_growth:,.0f}" if total_growth >= 0 else f"{total_growth:,.0f}"
        col3.metric("Total Subscribers", format_number(total_subscribers), delta=f"{total_pct:+.1f}% ({total_growth_str})")
    else:
        col3.metric("Total Subscribers", format_number(total_subscribers))

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
    """Render Full Subscribers sub-tab med scorecards, graf og tabel"""
    country_cols = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK', 'DE', 'AT', 'NL', 'BE', 'CH']
    
    # Farver per land
    country_colors = {
        'DK': '#9B7EBD', 'SE': '#E8B4CB', 'NO': '#A8E6CF', 'FI': '#FFD3B6',
        'FR': '#D4BFFF', 'UK': '#F0B4D4', 'DE': '#B4E0F0', 'AT': '#E0D4B4',
        'NL': '#F0D4B4', 'BE': '#D4F0B4', 'CH': '#B4D4F0'
    }
    
    if not full_df.empty:
        # Find tilgængelige måneder
        available_months = sorted(full_df['Month'].dropna().unique().tolist())
        current_month = available_months[-1] if available_months else None
        
        # --- SLIDER ---
        if len(available_months) > 1:
            month_range = st.select_slider(
                "Periode",
                options=available_months,
                value=(current_month, current_month),
                format_func=format_month_short,
                key="sub_full_period",
                label_visibility="collapsed"
            )
            start_month, end_month = month_range
        else:
            start_month = available_months[0] if available_months else None
            end_month = start_month
        
        # Find data for valgt periode
        if end_month:
            current_row = full_df[full_df['Month'] == end_month]
            # Find forrige måned
            end_idx = available_months.index(end_month)
            prev_month = available_months[end_idx - 1] if end_idx > 0 else None
            prev_row = full_df[full_df['Month'] == prev_month] if prev_month else None
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # --- SCORECARDS PER LAND (med % og absolut ændring) ---
        # Række 1: DK, SE, NO, FI, FR, UK
        row1_countries = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK']
        cols_row1 = st.columns(6)
        for i, country in enumerate(row1_countries):
            if country in full_df.columns and not current_row.empty:
                current = current_row.iloc[0][country]
                if prev_row is not None and not prev_row.empty:
                    prev = prev_row.iloc[0][country]
                    growth = current - prev
                    pct = ((current - prev) / prev * 100) if prev > 0 else 0
                    growth_str = f"+{growth:,.0f}" if growth >= 0 else f"{growth:,.0f}"
                    delta_str = f"{pct:+.1f}% ({growth_str})"
                    cols_row1[i].metric(f"{country}", format_number(current), delta=delta_str)
                else:
                    cols_row1[i].metric(f"{country}", format_number(current))
            else:
                cols_row1[i].metric(f"{country}", "—")
        
        # Række 2: DE, AT, NL, BE, CH + Total
        row2_countries = ['DE', 'AT', 'NL', 'BE', 'CH']
        cols_row2 = st.columns(6)
        for i, country in enumerate(row2_countries):
            if country in full_df.columns and not current_row.empty:
                current = current_row.iloc[0][country]
                if prev_row is not None and not prev_row.empty:
                    prev = prev_row.iloc[0][country]
                    growth = current - prev
                    pct = ((current - prev) / prev * 100) if prev > 0 else 0
                    growth_str = f"+{growth:,.0f}" if growth >= 0 else f"{growth:,.0f}"
                    delta_str = f"{pct:+.1f}% ({growth_str})"
                    cols_row2[i].metric(f"{country}", format_number(current), delta=delta_str)
                else:
                    cols_row2[i].metric(f"{country}", format_number(current))
            else:
                cols_row2[i].metric(f"{country}", "—")
        
        # Total kort i position 6
        if 'Total' in full_df.columns and not current_row.empty:
            current_total = current_row.iloc[0]['Total']
            if prev_row is not None and not prev_row.empty:
                prev_total = prev_row.iloc[0]['Total']
                total_growth = current_total - prev_total
                total_pct = ((current_total - prev_total) / prev_total * 100) if prev_total > 0 else 0
                total_growth_str = f"+{total_growth:,.0f}" if total_growth >= 0 else f"{total_growth:,.0f}"
                cols_row2[5].metric("Total", format_number(current_total), delta=f"{total_pct:+.1f}% ({total_growth_str})")
            else:
                cols_row2[5].metric("Total", format_number(current_total))
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # --- GRAF: Udvikling per land ---
        chart_df = full_df.sort_values('Month')
        
        fig = go.Figure()
        for country in country_cols:
            if country in chart_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_df['Month'],
                        y=chart_df[country],
                        name=country,
                        mode='lines+markers',
                        line=dict(color=country_colors.get(country, '#9B7EBD'), width=2),
                        marker=dict(size=6)
                    )
                )
        
        fig.update_layout(
            title="", showlegend=True, height=600,
            margin=dict(l=50, r=50, t=60, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            plot_bgcolor='rgba(250,245,255,0.5)', paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', tickformat='%Y-%m', automargin=True, ticklabelstandoff=10)
        fig.update_yaxes(gridcolor='rgba(212,191,255,0.3)', tickformat=',', automargin=True, ticklabelstandoff=10)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # --- TABEL ---
        full_df = full_df.sort_values('Month', ascending=False)
        display_full = full_df.copy()
        display_full['Month'] = display_full['Month'].dt.strftime('%Y-%m')
        cols_to_show = ['Month'] + [c for c in country_cols + ['Total'] if c in display_full.columns]
        
        # Beregn højde baseret på antal rækker (38px per række + 60px header)
        table_height = len(display_full) * 38 + 60
        
        st.dataframe(
            display_full[cols_to_show],
            use_container_width=True,
            hide_index=True,
            height=table_height,
            column_config={
                "Month": st.column_config.TextColumn("Måned", width="small"),
                **{col: st.column_config.NumberColumn(col, format="localized", width="small") for col in country_cols + ['Total'] if col in display_full.columns}
            }
        )
    else:
        st.info("Ingen Full Subscribers data.")


def render_light_subscribers_tab(light_df):
    """Render Light Subscribers sub-tab med scorecards, graf og tabel"""
    country_cols = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK', 'DE', 'AT', 'NL', 'BE', 'CH']
    
    # Farver per land
    country_colors = {
        'DK': '#9B7EBD', 'SE': '#E8B4CB', 'NO': '#A8E6CF', 'FI': '#FFD3B6',
        'FR': '#D4BFFF', 'UK': '#F0B4D4', 'DE': '#B4E0F0', 'AT': '#E0D4B4',
        'NL': '#F0D4B4', 'BE': '#D4F0B4', 'CH': '#B4D4F0'
    }
    
    if not light_df.empty:
        # Find tilgængelige måneder
        available_months = sorted(light_df['Month'].dropna().unique().tolist())
        current_month = available_months[-1] if available_months else None
        
        # --- SLIDER ---
        if len(available_months) > 1:
            month_range = st.select_slider(
                "Periode",
                options=available_months,
                value=(current_month, current_month),
                format_func=format_month_short,
                key="sub_light_period",
                label_visibility="collapsed"
            )
            start_month, end_month = month_range
        else:
            start_month = available_months[0] if available_months else None
            end_month = start_month
        
        # Find data for valgt periode
        if end_month:
            current_row = light_df[light_df['Month'] == end_month]
            # Find forrige måned
            end_idx = available_months.index(end_month)
            prev_month = available_months[end_idx - 1] if end_idx > 0 else None
            prev_row = light_df[light_df['Month'] == prev_month] if prev_month else None
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # --- SCORECARDS PER LAND (med % og absolut ændring) ---
        # Række 1: DK, SE, NO, FI, FR, UK
        row1_countries = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK']
        cols_row1 = st.columns(6)
        for i, country in enumerate(row1_countries):
            if country in light_df.columns and not current_row.empty:
                current = current_row.iloc[0][country]
                if prev_row is not None and not prev_row.empty:
                    prev = prev_row.iloc[0][country]
                    growth = current - prev
                    pct = ((current - prev) / prev * 100) if prev > 0 else 0
                    growth_str = f"+{growth:,.0f}" if growth >= 0 else f"{growth:,.0f}"
                    delta_str = f"{pct:+.1f}% ({growth_str})"
                    cols_row1[i].metric(f"{country}", format_number(current), delta=delta_str)
                else:
                    cols_row1[i].metric(f"{country}", format_number(current))
            else:
                cols_row1[i].metric(f"{country}", "—")
        
        # Række 2: DE, AT, NL, BE, CH + Total
        row2_countries = ['DE', 'AT', 'NL', 'BE', 'CH']
        cols_row2 = st.columns(6)
        for i, country in enumerate(row2_countries):
            if country in light_df.columns and not current_row.empty:
                current = current_row.iloc[0][country]
                if prev_row is not None and not prev_row.empty:
                    prev = prev_row.iloc[0][country]
                    growth = current - prev
                    pct = ((current - prev) / prev * 100) if prev > 0 else 0
                    growth_str = f"+{growth:,.0f}" if growth >= 0 else f"{growth:,.0f}"
                    delta_str = f"{pct:+.1f}% ({growth_str})"
                    cols_row2[i].metric(f"{country}", format_number(current), delta=delta_str)
                else:
                    cols_row2[i].metric(f"{country}", format_number(current))
            else:
                cols_row2[i].metric(f"{country}", "—")
        
        # Total kort i position 6
        if 'Total' in light_df.columns and not current_row.empty:
            current_total = current_row.iloc[0]['Total']
            if prev_row is not None and not prev_row.empty:
                prev_total = prev_row.iloc[0]['Total']
                total_growth = current_total - prev_total
                total_pct = ((current_total - prev_total) / prev_total * 100) if prev_total > 0 else 0
                total_growth_str = f"+{total_growth:,.0f}" if total_growth >= 0 else f"{total_growth:,.0f}"
                cols_row2[5].metric("Total", format_number(current_total), delta=f"{total_pct:+.1f}% ({total_growth_str})")
            else:
                cols_row2[5].metric("Total", format_number(current_total))
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # --- GRAF: Udvikling per land ---
        chart_df = light_df.sort_values('Month')
        
        fig = go.Figure()
        for country in country_cols:
            if country in chart_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_df['Month'],
                        y=chart_df[country],
                        name=country,
                        mode='lines+markers',
                        line=dict(color=country_colors.get(country, '#9B7EBD'), width=2),
                        marker=dict(size=6)
                    )
                )
        
        fig.update_layout(
            title="", showlegend=True, height=600,
            margin=dict(l=50, r=50, t=60, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            plot_bgcolor='rgba(250,245,255,0.5)', paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', tickformat='%Y-%m', automargin=True, ticklabelstandoff=10)
        fig.update_yaxes(gridcolor='rgba(212,191,255,0.3)', tickformat=',', automargin=True, ticklabelstandoff=10)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # --- TABEL ---
        light_df = light_df.sort_values('Month', ascending=False)
        display_light = light_df.copy()
        display_light['Month'] = display_light['Month'].dt.strftime('%Y-%m')
        cols_to_show = ['Month'] + [c for c in country_cols + ['Total'] if c in display_light.columns]
        
        # Beregn højde baseret på antal rækker (38px per række + 60px header)
        table_height = len(display_light) * 38 + 60
        
        st.dataframe(
            display_light[cols_to_show],
            use_container_width=True,
            hide_index=True,
            height=table_height,
            column_config={
                "Month": st.column_config.TextColumn("Måned", width="small"),
                **{col: st.column_config.NumberColumn(col, format="localized", width="small") for col in country_cols + ['Total'] if col in display_light.columns}
            }
        )
    else:
        st.info("Ingen Light Subscribers data.")


def render_nye_subscribers_tab(events_df):
    """Render Nye Subscribers per Kilde sub-tab med underfaner per Master Source"""
    country_cols = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK', 'DE', 'AT', 'NL', 'BE', 'CH', 'Total']
    
    # Farver for Master Sources - unikke farver til hver
    master_colors = {
        'Game': '#9B7EBD',           # Lilla
        'Lead Ad': '#E8B4CB',        # Pink
        'LightPermission': '#A8E6CF', # Mint
        'On Site': '#FFD3B6',        # Fersken
        'Other': '#F0E68C',          # Gul
        'Sleeknote': '#87CEEB',      # Lyseblå
        'Paid': '#FF6B6B',           # Koral
        'Organic': '#4ECDC4',        # Turkis
        'Referral': '#45B7D1',       # Blå
        'Direct': '#96CEB4',         # Sage
        'Social': '#DDA0DD',         # Plum
        'Email': '#F7DC6F',          # Guld
    }
    
    if not events_df.empty:
        display_events = events_df.copy()
        display_events['Month'] = pd.to_datetime(display_events['Month'])
        
        # Fjern LightPermission fra data
        display_events = display_events[display_events['Master Source'] != 'LightPermission']
        
        # Definér rækkefølge for Master Sources
        master_source_order = ['On Site', 'Game', 'Lead Ad', 'Sleeknote', 'Other']
        available_sources = display_events['Master Source'].unique().tolist()
        master_sources = [ms for ms in master_source_order if ms in available_sources]
        
        # Opret sub-tabs: Oversigt + en per Master Source
        tab_names = ["Oversigt"] + master_sources
        source_tabs = st.tabs(tab_names)
        
        # --- OVERSIGT TAB ---
        with source_tabs[0]:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            
            # Aggregér data per måned og Master Source (sum Total)
            agg_df = display_events.groupby(['Month', 'Master Source'])['Total'].sum().reset_index()
            agg_df = agg_df.sort_values('Month')
            
            # --- SLIDER ---
            available_months = sorted(agg_df['Month'].unique())
            current_month = available_months[-1] if available_months else None
            
            def format_month_short_kilder(m):
                return m.strftime('%b %y') if hasattr(m, 'strftime') else str(m)
            
            if len(available_months) > 1:
                month_range = st.select_slider(
                    "Periode",
                    options=available_months,
                    value=(current_month, current_month),
                    format_func=format_month_short_kilder,
                    key="kilder_oversigt_period",
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
            
            # Beregn måned progress (hvor langt vi er i nuværende måned)
            import calendar
            today = datetime.date.today()
            yesterday = today - datetime.timedelta(days=1)
            days_with_data = yesterday.day
            total_days_in_month = calendar.monthrange(today.year, today.month)[1]
            month_progress = days_with_data / total_days_in_month
            
            # Tjek om nuværende måned er valgt
            current_month_dt = pd.Timestamp(today.year, today.month, 1)
            current_month_selected = any(
                m.year == current_month_dt.year and m.month == current_month_dt.month 
                for m in selected_months
            )
            
            # Find sammenligningsperiode: N måneder FØR den ældste valgte måned
            if selected_months and num_months > 0:
                oldest_selected = selected_months[0]
                oldest_idx = available_months.index(oldest_selected) if oldest_selected in available_months else -1
                
                # Find N måneder før oldest_selected
                prev_months = []
                for i in range(num_months):
                    prev_idx = oldest_idx - 1 - i
                    if prev_idx >= 0:
                        prev_months.append(available_months[prev_idx])
                
                # Den ældste sammenligningsmåned (skal skaleres hvis current month er valgt)
                oldest_prev_month = min(prev_months) if prev_months else None
            else:
                prev_months = []
                oldest_prev_month = None
            
            # --- SCORECARDS ---
            num_sources = len(master_sources)
            cols = st.columns(num_sources + 1)  # +1 for Total
            
            # Total for alle Master Sources
            grand_total = 0
            grand_prev_total = 0
            
            for i, master in enumerate(master_sources):
                master_total = period_df[period_df['Master Source'] == master]['Total'].sum()
                grand_total += master_total
                
                # Beregn sammenligning med forrige periode (N måneder)
                prev_total = 0
                if prev_months:
                    for pm in prev_months:
                        pm_total = agg_df[(agg_df['Month'] == pm) & (agg_df['Master Source'] == master)]['Total'].sum()
                        
                        # Skaler kun den ældste sammenligningsmåned hvis nuværende måned er valgt
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
            
            fig.update_layout(
                title="",
                showlegend=True,
                height=500,
                margin=dict(l=50, r=50, t=30, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                plot_bgcolor='rgba(250,245,255,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified'
            )
            fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', tickformat='%Y-%m', automargin=True, ticklabelstandoff=10)
            fig.update_yaxes(gridcolor='rgba(212,191,255,0.3)', tickformat=',', automargin=True, ticklabelstandoff=10)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # --- MASTER SOURCE TABS ---
        # Farver per land (Total får en mørk farve der skiller sig ud)
        country_colors = {
            'DK': '#9B7EBD', 'SE': '#E8B4CB', 'NO': '#A8E6CF', 'FI': '#FFD3B6',
            'FR': '#D4BFFF', 'UK': '#F0B4D4', 'DE': '#B4E0F0', 'AT': '#E0D4B4',
            'NL': '#F0D4B4', 'BE': '#D4F0B4', 'CH': '#B4D4F0', 'Total': '#4A3F55'
        }
        
        # Master sources der skal have Source sub-tabs
        masters_with_source_tabs = ['On Site', 'Game', 'Lead Ad']
        
        for i, master in enumerate(master_sources):
            with source_tabs[i + 1]:
                st.markdown(f"<div style='height: 10px;'></div>", unsafe_allow_html=True)
                
                # Filtrer til denne Master Source
                master_df = display_events[display_events['Master Source'] == master].copy()
                
                # Tjek om denne Master Source skal have Source sub-tabs
                if master in masters_with_source_tabs:
                    # Hent unikke Sources under denne Master Source
                    sources_list = sorted(master_df['Source'].unique().tolist())
                    
                    # Opret sub-tabs: Oversigt + hver Source
                    source_tab_names = ["Oversigt"] + sources_list
                    source_sub_tabs = st.tabs(source_tab_names)
                    
                    # --- OVERSIGT TAB (aggregeret for hele Master Source) ---
                    with source_sub_tabs[0]:
                        render_source_content(master_df, country_cols, country_colors, f"kilder_{master}_oversigt", is_overview=True)
                    
                    # --- INDIVIDUELLE SOURCE TABS ---
                    for src_idx, source_name in enumerate(sources_list):
                        with source_sub_tabs[src_idx + 1]:
                            source_df = master_df[master_df['Source'] == source_name].copy()
                            render_source_content(source_df, country_cols, country_colors, f"kilder_{master}_{source_name}", is_overview=False)
                else:
                    # For Sleeknote og Other: vis direkte uden sub-tabs
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
    
    def format_month_short_src(m):
        return m.strftime('%b %y') if hasattr(m, 'strftime') else str(m)
    
    if len(available_months) > 1:
        month_range = st.select_slider(
            "Periode",
            options=available_months,
            value=(current_month_val, current_month_val),
            format_func=format_month_short_src,
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
    import calendar
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    days_with_data = yesterday.day
    total_days_in_month = calendar.monthrange(today.year, today.month)[1]
    month_progress = days_with_data / total_days_in_month
    
    # Tjek om nuværende måned er valgt
    current_month_dt = pd.Timestamp(today.year, today.month, 1)
    current_month_selected = any(
        m.year == current_month_dt.year and m.month == current_month_dt.month 
        for m in selected_months
    )
    
    # Find sammenligningsperiode: N måneder FØR den ældste valgte måned
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
    # Række 1: DK, SE, NO, FI, FR, UK
    row1_countries = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK']
    cols_row1 = st.columns(6)
    for j, country in enumerate(row1_countries):
        if country in agg_df.columns:
            current = period_df[country].sum()
            
            # Beregn sammenligning
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
    row2_countries = ['DE', 'AT', 'NL', 'BE', 'CH']
    cols_row2 = st.columns(6)
    for j, country in enumerate(row2_countries):
        if country in agg_df.columns:
            current = period_df[country].sum()
            
            # Beregn sammenligning
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
    
    # Total kort i position 6
    if 'Total' in agg_df.columns:
        current_total = period_df['Total'].sum()
        
        # Beregn sammenligning for Total
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
    
    # --- GRAF: Udvikling per land ---
    fig = go.Figure()
    graph_cols = country_cols + ['Total']
    for country in graph_cols:
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
    
    fig.update_layout(
        title="",
        showlegend=True,
        height=500,
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor='rgba(250,245,255,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', tickformat='%Y-%m', automargin=True, ticklabelstandoff=10)
    fig.update_yaxes(gridcolor='rgba(212,191,255,0.3)', tickformat=',', automargin=True, ticklabelstandoff=10)
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # --- TABEL ---
    if is_overview:
        # For oversigt: vis alle rækker med Source kolonne
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
        # For individuel source: vis aggregeret per måned
        agg_df['Month_str'] = agg_df['Month'].dt.strftime('%Y-%m')
        cols_to_show = ['Month_str'] + [c for c in country_cols if c in agg_df.columns] + ['Total']
        
        table_height = min(len(agg_df) * 38 + 60, 800)
        
        st.dataframe(
            agg_df[cols_to_show].sort_values('Month_str', ascending=False),
            use_container_width=True,
            hide_index=True,
            height=table_height,
            column_config={
                "Month_str": st.column_config.TextColumn("Måned", width="small"),
                **{col: st.column_config.NumberColumn(col, format="localized", width="small") for col in country_cols + ['Total'] if col in agg_df.columns}
            }
        )


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
