"""
Subscribers Tab - CRM Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shared import get_gspread_client, show_metric, format_number


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

    # Sorter efter dato
    if not full_df.empty:
        full_df = full_df.sort_values('Month', ascending=False)
    if not light_df.empty:
        light_df = light_df.sort_values('Month', ascending=False)

    # --- KPI CARDS ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Seneste data
    if not full_df.empty and len(full_df) >= 2:
        current_full = full_df.iloc[0]['Total']
        prev_full = full_df.iloc[1]['Total']
        full_growth = current_full - prev_full
        full_growth_pct = ((current_full - prev_full) / prev_full * 100) if prev_full > 0 else 0
    else:
        current_full = full_df.iloc[0]['Total'] if not full_df.empty else 0
        prev_full = None
        full_growth = 0
        full_growth_pct = 0

    if not light_df.empty and len(light_df) >= 2:
        current_light = light_df.iloc[0]['Total']
        prev_light = light_df.iloc[1]['Total']
        light_growth = current_light - prev_light
        light_growth_pct = ((current_light - prev_light) / prev_light * 100) if prev_light > 0 else 0
    else:
        current_light = light_df.iloc[0]['Total'] if not light_df.empty else 0
        prev_light = None
        light_growth = 0
        light_growth_pct = 0

    total_subscribers = current_full + current_light
    total_growth = full_growth + light_growth

    show_metric(col1, "Full Subscribers", current_full, prev_full)
    show_metric(col2, "Light Subscribers", current_light, prev_light)
    show_metric(col3, "Total Subscribers", total_subscribers)
    
    # Nye subscribers denne maned
    col4.metric("Nye Full", f"+{format_number(full_growth)}" if full_growth >= 0 else format_number(full_growth))
    col5.metric("Nye Light", f"+{format_number(light_growth)}" if light_growth >= 0 else format_number(light_growth))
    col6.metric("Nye Total", f"+{format_number(total_growth)}" if total_growth >= 0 else format_number(total_growth))

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # --- GRAF: Subscriber vaekst over tid ---
    if not full_df.empty or not light_df.empty:
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Sorter kronologisk for graf
        if not full_df.empty:
            full_chart = full_df.sort_values('Month')
            fig.add_trace(
                go.Scatter(
                    x=full_chart['Month'], y=full_chart['Total'],
                    name='Full Subscribers', mode='lines+markers',
                    line=dict(color='#9B7EBD', width=3),
                    marker=dict(size=8)
                )
            )
        
        if not light_df.empty:
            light_chart = light_df.sort_values('Month')
            fig.add_trace(
                go.Scatter(
                    x=light_chart['Month'], y=light_chart['Total'],
                    name='Light Subscribers', mode='lines+markers',
                    line=dict(color='#E8B4CB', width=3),
                    marker=dict(size=8)
                )
            )
        
        fig.update_layout(
            title="", showlegend=True, height=400,
            margin=dict(l=50, r=50, t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(250,245,255,0.5)', paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', tickformat='%Y-%m')
        fig.update_yaxes(gridcolor='rgba(212,191,255,0.3)', tickformat=',')
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # --- TABS FOR DETALJERET DATA ---
    detail_tab1, detail_tab2, detail_tab3 = st.tabs(["Full Subscribers", "Light Subscribers", "Nye Subscribers per Kilde"])
    
    country_cols = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK', 'DE', 'AT', 'NL', 'BE', 'CH', 'Total']
    
    with detail_tab1:
        if not full_df.empty:
            display_full = full_df.copy()
            display_full['Month'] = display_full['Month'].dt.strftime('%Y-%m')
            cols_to_show = ['Month'] + [c for c in country_cols if c in display_full.columns]
            
            st.dataframe(
                display_full[cols_to_show],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Month": st.column_config.TextColumn("Maned", width="small"),
                    **{col: st.column_config.NumberColumn(col, format="localized", width="small") for col in country_cols if col in display_full.columns}
                }
            )
        else:
            st.info("Ingen Full Subscribers data.")
    
    with detail_tab2:
        if not light_df.empty:
            display_light = light_df.copy()
            display_light['Month'] = display_light['Month'].dt.strftime('%Y-%m')
            cols_to_show = ['Month'] + [c for c in country_cols if c in display_light.columns]
            
            st.dataframe(
                display_light[cols_to_show],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Month": st.column_config.TextColumn("Maned", width="small"),
                    **{col: st.column_config.NumberColumn(col, format="localized", width="small") for col in country_cols if col in display_light.columns}
                }
            )
        else:
            st.info("Ingen Light Subscribers data.")
    
    with detail_tab3:
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
                    "Month": st.column_config.TextColumn("Maned", width="small"),
                    "Master Source": st.column_config.TextColumn("Master Source", width="medium"),
                    "Source": st.column_config.TextColumn("Source", width="medium"),
                    **{col: st.column_config.NumberColumn(col, format="localized", width="small") for col in country_cols if col in filtered_events.columns}
                }
            )
        else:
            st.info("Ingen subscriber events data.")

    if st.button('Opdater Data', key="sub_refresh"):
        st.rerun()

