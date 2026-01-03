"""
Flows Tab - CRM Dashboard
Med sub-tabs for hvert flow
"""
import streamlit as st
import pandas as pd
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shared import get_gspread_client, show_metric

def col_letter_to_index(col_str):
    """Konverter kolonnebogstav til 0-baseret indeks (A=0, B=1, ..., Z=25, AA=26, ...)"""
    result = 0
    for char in col_str.upper():
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result - 1


@st.cache_data(ttl=300, show_spinner=False)  # Cache i 5 minutter
def load_flows_data():
    """Henter Flows data fra Google Sheet"""
    try:
        gc = get_gspread_client()
        
        # Tjek om flows_spreadsheet er konfigureret
        if "flows_spreadsheet" not in st.secrets["connections"]["gsheets"]:
            st.error("Mangler 'flows_spreadsheet' i secrets. Tilføj: flows_spreadsheet = 'URL'")
            return pd.DataFrame()
        
        flows_url = st.secrets["connections"]["gsheets"]["flows_spreadsheet"]
        spreadsheet = gc.open_by_url(flows_url)
        
        worksheet = spreadsheet.worksheet("All_Flow")
        all_values = worksheet.get_all_values()
        
        if len(all_values) > 2:
            # Skip header rows (row 1-2 contains headers)
            data = all_values[2:]
            raw_df = pd.DataFrame(data)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Fejl ved indlæsning fra Google Sheets: {type(e).__name__}: {e}")
        st.info(f"URL brugt: {st.secrets['connections']['gsheets'].get('flows_spreadsheet', 'IKKE SAT')}")
        return pd.DataFrame()
    
    # Landekonfiguration med startkolonner (0-indexed)
    # P=15, W=22, AD=29, AK=36, AR=43, AY=50, BF=57, BM=64, BT=71, CA=78, CH=85
    country_configs = [
        ('DK', col_letter_to_index('P')),   # 15
        ('SE', col_letter_to_index('W')),   # 22
        ('NO', col_letter_to_index('AD')),  # 29
        ('FI', col_letter_to_index('AK')),  # 36
        ('FR', col_letter_to_index('AR')),  # 43
        ('UK', col_letter_to_index('AY')),  # 50
        ('DE', col_letter_to_index('BF')),  # 57
        ('AT', col_letter_to_index('BM')),  # 64
        ('NL', col_letter_to_index('BT')),  # 71
        ('BE', col_letter_to_index('CA')),  # 78
        ('CH', col_letter_to_index('CH')),  # 85
    ]
    
    # Metrics offset fra startkolonne
    METRIC_OFFSETS = {
        'Received_Email': 0,
        'Total_Opens': 1,
        'Unique_Opens': 2,
        'Total_Clicks': 3,
        'Unique_Clicks': 4,
        'Unsubscribed': 5,
        'Bounced': 6,
    }
    
    all_country_data = []
    
    for country_code, start_col in country_configs:
        try:
            country_df = pd.DataFrame()
            
            # Info kolonner (A-H, index 0-7)
            country_df['Send_Date'] = raw_df.iloc[:, 0]      # A: Send Date (2025-12)
            country_df['Tags'] = raw_df.iloc[:, 1]           # B: Tags
            country_df['Flow'] = raw_df.iloc[:, 2]           # C: Flow
            country_df['Trigger'] = raw_df.iloc[:, 3]        # D: Trigger
            country_df['Group'] = raw_df.iloc[:, 4]          # E: Group
            country_df['Mail'] = raw_df.iloc[:, 5]           # F: Mail
            country_df['Message'] = raw_df.iloc[:, 6]        # G: Message
            country_df['AB'] = raw_df.iloc[:, 7]             # H: A/B
            
            # Metrics med offset
            for metric_name, offset in METRIC_OFFSETS.items():
                col_idx = start_col + offset
                if col_idx < len(raw_df.columns):
                    country_df[metric_name] = raw_df.iloc[:, col_idx]
                else:
                    country_df[metric_name] = 0
            
            country_df['Country'] = country_code
            all_country_data.append(country_df)
            
        except Exception as e:
            continue
    
    if not all_country_data:
        return pd.DataFrame()

    df = pd.concat(all_country_data, ignore_index=True)
    
    # Parse Send_Date (format: 2025-12 = Ar-Maned)
    df['Year_Month'] = df['Send_Date'].astype(str).str.strip()
    df = df[df['Year_Month'].str.match(r'^\d{4}-\d{1,2}$', na=False)]
    
    # Konverter numeriske kolonner
    numeric_cols = ['Received_Email', 'Total_Opens', 'Unique_Opens', 'Total_Clicks', 'Unique_Clicks', 'Unsubscribed', 'Bounced']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Beregn rater
    df['Open_Rate'] = df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    df['Click_Rate'] = df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    df['CTR'] = df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    # Opret Flow-Trigger identifier
    df['Flow_Trigger'] = df['Flow'].astype(str).str.strip() + ' - ' + df['Trigger'].astype(str).str.strip()
    
    return df


def get_available_months(df):
    """Returner liste af tilgængelige måneder sorteret faldende (nyeste først)"""
    months = df['Year_Month'].unique()
    # Sorter som datoer, ikke tekst (2025-12 skal komme før 2025-9)
    def month_sort_key(m):
        try:
            parts = m.split('-')
            return (int(parts[0]), int(parts[1]))
        except:
            return (0, 0)
    return sorted(months, key=month_sort_key, reverse=True)


def aggregate_to_flow_level(df):
    """Aggreger data til flow niveau (summer alle mails under samme flow)"""
    agg_df = df.groupby(['Year_Month', 'Flow_Trigger', 'Country'], as_index=False).agg({
        'Received_Email': 'sum',
        'Total_Opens': 'sum',
        'Unique_Opens': 'sum',
        'Total_Clicks': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })
    
    # Genberegn rater efter aggregering
    agg_df['Open_Rate'] = agg_df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    agg_df['Click_Rate'] = agg_df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    agg_df['CTR'] = agg_df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    return agg_df


def get_unique_flows(df):
    """Hent unikke flows sorteret efter flow nummer (ekskluderer ikke-aktive flows)"""
    # Flows der ikke længere er aktive
    INACTIVE_FLOWS = {0, 1, 2, 3, 9}
    
    def flow_sort_key(f):
        match = re.search(r'Flow\s*(\d+)', f)
        return int(match.group(1)) if match else 999
    
    def is_active_flow(f):
        match = re.search(r'Flow\s*(\d+)', f)
        if match:
            flow_num = int(match.group(1))
            return flow_num not in INACTIVE_FLOWS
        return True
    
    flows = df['Flow_Trigger'].unique()
    active_flows = [f for f in flows if is_active_flow(f)]
    return sorted(active_flows, key=flow_sort_key)


def render_overview_content(flow_df, sel_countries, sel_flows):
    """Render oversigt (alle flows aggregeret)"""
    current_df = flow_df[
        (flow_df['Country'].isin(sel_countries)) &
        (flow_df['Flow_Trigger'].isin(sel_flows))
    ].copy()

    if current_df.empty:
        st.warning("Ingen data matcher de valgte filtre.")
        return

    # Aggreger til visning (sum over alle lande og måneder)
    display_df = current_df.groupby(['Year_Month', 'Flow_Trigger'], as_index=False).agg({
        'Received_Email': 'sum',
        'Total_Opens': 'sum',
        'Unique_Opens': 'sum',
        'Total_Clicks': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })
    
    # Genberegn rater
    display_df['Open_Rate'] = display_df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    display_df['Click_Rate'] = display_df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    display_df['CTR'] = display_df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)

    # KPI totaler
    total_received = display_df['Received_Email'].sum()
    total_opens = display_df['Unique_Opens'].sum()
    total_clicks = display_df['Unique_Clicks'].sum()
    
    open_rate = (total_opens / total_received * 100) if total_received > 0 else 0
    click_rate = (total_clicks / total_received * 100) if total_received > 0 else 0
    ctr = (total_clicks / total_opens * 100) if total_opens > 0 else 0

    # KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    show_metric(col1, "Emails Sendt", total_received)
    show_metric(col2, "Unikke Opens", total_opens)
    show_metric(col3, "Unikke Clicks", total_clicks)
    show_metric(col4, "Open Rate", open_rate, is_percent=True)
    show_metric(col5, "Click Rate", click_rate, is_percent=True)
    show_metric(col6, "Click Through Rate", ctr, is_percent=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Chart - aggregeret per flow
    chart_df = display_df.groupby('Flow_Trigger', as_index=False).agg({
        'Received_Email': 'sum',
        'Unique_Opens': 'sum',
        'Unique_Clicks': 'sum',
    })
    chart_df['Open_Rate'] = (chart_df['Unique_Opens'] / chart_df['Received_Email'] * 100).round(1)
    chart_df['Click_Rate'] = (chart_df['Unique_Clicks'] / chart_df['Received_Email'] * 100).round(2)
    
    # Sorter efter flow nummer (Flow 1, Flow 2, ...)
    def chart_flow_sort_key(f):
        match = re.search(r'Flow\s*(\d+)', f)
        return int(match.group(1)) if match else 999
    chart_df = chart_df.iloc[chart_df['Flow_Trigger'].map(chart_flow_sort_key).argsort()]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=chart_df['Flow_Trigger'], y=chart_df['Open_Rate'],
            name='Open Rate', marker_color='#9B7EBD',
            text=chart_df['Open_Rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside', textfont=dict(size=14), offsetgroup=0
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=chart_df['Flow_Trigger'], y=chart_df['Click_Rate'],
            name='Click Rate', marker_color='#E8B4CB',
            text=chart_df['Click_Rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside', textfont=dict(size=12), offsetgroup=1
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="", showlegend=True, height=455,
        margin=dict(l=50, r=50, t=50, b=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(250,245,255,0.5)', paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified', barmode='group', bargap=0.3, bargroupgap=0.1
    )
    
    max_open = chart_df['Open_Rate'].max() if not chart_df.empty else 50
    max_click = chart_df['Click_Rate'].max() if not chart_df.empty else 5
    
    fig.update_yaxes(title_text="Open Rate %", secondary_y=False, gridcolor='rgba(212,191,255,0.3)', ticksuffix='%', range=[0, max_open * 1.2])
    fig.update_yaxes(title_text="Click Rate %", secondary_y=True, gridcolor='rgba(232,180,203,0.3)', ticksuffix='%', showgrid=False, range=[0, max_click * 1.2])
    fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', tickangle=-45, type='category', tickfont=dict(size=12))
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Tabel - aggregeret per flow (ikke per måned)
    table_df = display_df.groupby('Flow_Trigger', as_index=False).agg({
        'Received_Email': 'sum',
        'Unique_Opens': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })
    
    # Genberegn rater efter aggregering
    table_df['Open_Rate'] = table_df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    table_df['Click_Rate'] = table_df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    table_df['CTR'] = table_df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    # Sorter efter flow nummer
    table_df['_flow_num'] = table_df['Flow_Trigger'].apply(lambda f: int(re.search(r'Flow\s*(\d+)', f).group(1)) if re.search(r'Flow\s*(\d+)', f) else 999)
    table_df = table_df.sort_values('_flow_num', ascending=True)
    table_df = table_df.drop(columns=['_flow_num'])
    
    # Vælg og sorter kolonner
    table_df = table_df[['Flow_Trigger', 'Received_Email', 'Unique_Opens', 'Unique_Clicks', 'Open_Rate', 'Click_Rate', 'CTR', 'Unsubscribed', 'Bounced']]
    
    # Beregn højde så alle rækker vises (35px per række + 38px header)
    table_height = (len(table_df) + 1) * 35 + 3
    
    st.dataframe(
        table_df, use_container_width=True, hide_index=True, height=table_height,
        column_config={
            "Flow_Trigger": st.column_config.TextColumn("Flow", width="large"),
            "Received_Email": st.column_config.NumberColumn("Sendt", format="localized", width="small"),
            "Unique_Opens": st.column_config.NumberColumn("Opens", format="localized", width="small"),
            "Unique_Clicks": st.column_config.NumberColumn("Clicks", format="localized", width="small"),
            "Open_Rate": st.column_config.NumberColumn("Open Rate", format="%.1f%%", width="small"),
            "Click_Rate": st.column_config.NumberColumn("Click Rate", format="%.1f%%", width="small"),
            "CTR": st.column_config.NumberColumn("CTR", format="%.1f%%", width="small"),
            "Unsubscribed": st.column_config.NumberColumn("Unsub", format="localized", width="small"),
            "Bounced": st.column_config.NumberColumn("Bounced", format="localized", width="small"),
        }
    )


def render_single_flow_content(raw_df, flow_trigger, sel_countries, sel_mails=None):
    """Render indhold for et enkelt flow med mail-niveau breakdown"""
    # Filtrer til dette specifikke flow og valgte lande
    flow_data = raw_df[
        (raw_df['Flow_Trigger'] == flow_trigger) &
        (raw_df['Country'].isin(sel_countries))
    ].copy()
    
    # Filtrer også på valgte mails hvis angivet
    if sel_mails is not None:
        flow_data = flow_data[flow_data['Mail'].isin(sel_mails)]
    
    if flow_data.empty:
        st.warning(f"Ingen data for {flow_trigger} med de valgte filtre.")
        return
    
    # Aggreger per mail (sum over alle lande og måneder)
    mail_df = flow_data.groupby(['Year_Month', 'Mail', 'Message'], as_index=False).agg({
        'Received_Email': 'sum',
        'Total_Opens': 'sum',
        'Unique_Opens': 'sum',
        'Total_Clicks': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })
    
    # Genberegn rater
    mail_df['Open_Rate'] = mail_df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    mail_df['Click_Rate'] = mail_df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    mail_df['CTR'] = mail_df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)

    # KPI totaler for dette flow
    total_received = mail_df['Received_Email'].sum()
    total_opens = mail_df['Unique_Opens'].sum()
    total_clicks = mail_df['Unique_Clicks'].sum()
    total_unsub = mail_df['Unsubscribed'].sum()
    total_bounced = mail_df['Bounced'].sum()
    
    open_rate = (total_opens / total_received * 100) if total_received > 0 else 0
    click_rate = (total_clicks / total_received * 100) if total_received > 0 else 0
    ctr = (total_clicks / total_opens * 100) if total_opens > 0 else 0

    # KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    show_metric(col1, "Emails Sendt", total_received)
    show_metric(col2, "Unikke Opens", total_opens)
    show_metric(col3, "Unikke Clicks", total_clicks)
    show_metric(col4, "Open Rate", open_rate, is_percent=True)
    show_metric(col5, "Click Rate", click_rate, is_percent=True)
    show_metric(col6, "Click Through Rate", ctr, is_percent=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Chart - per mail i flowet
    chart_df = mail_df.groupby(['Mail', 'Message'], as_index=False).agg({
        'Received_Email': 'sum',
        'Unique_Opens': 'sum',
        'Unique_Clicks': 'sum',
    })
    chart_df['Open_Rate'] = (chart_df['Unique_Opens'] / chart_df['Received_Email'] * 100).round(1)
    chart_df['Click_Rate'] = (chart_df['Unique_Clicks'] / chart_df['Received_Email'] * 100).round(2)
    
    # Opret label til chart (Mail + Message)
    chart_df['Label'] = chart_df.apply(
        lambda x: f"{x['Mail']}" if pd.isna(x['Message']) or x['Message'] == '' else f"{x['Mail']}: {x['Message']}", 
        axis=1
    )
    
    # Sorter efter mail nummer
    def mail_sort_key(m):
        match = re.search(r'Mail\s*(\d+)', str(m))
        return int(match.group(1)) if match else 999
    chart_df = chart_df.iloc[chart_df['Mail'].map(mail_sort_key).argsort()]

    if len(chart_df) > 0:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=chart_df['Label'], y=chart_df['Open_Rate'],
                name='Open Rate', marker_color='#9B7EBD',
                text=chart_df['Open_Rate'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside', textfont=dict(size=14), offsetgroup=0
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(
                x=chart_df['Label'], y=chart_df['Click_Rate'],
                name='Click Rate', marker_color='#E8B4CB',
                text=chart_df['Click_Rate'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside', textfont=dict(size=12), offsetgroup=1
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title="", showlegend=True, height=455,
            margin=dict(l=50, r=50, t=50, b=120),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(250,245,255,0.5)', paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified', barmode='group', bargap=0.3, bargroupgap=0.1
        )
        
        max_open = chart_df['Open_Rate'].max() if not chart_df.empty else 50
        max_click = chart_df['Click_Rate'].max() if not chart_df.empty else 5
        
        fig.update_yaxes(title_text="Open Rate %", secondary_y=False, gridcolor='rgba(212,191,255,0.3)', ticksuffix='%', range=[0, max_open * 1.2])
        fig.update_yaxes(title_text="Click Rate %", secondary_y=True, gridcolor='rgba(232,180,203,0.3)', ticksuffix='%', showgrid=False, range=[0, max_click * 1.2])
        fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', tickangle=-45, type='category', tickfont=dict(size=12))
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Tabel med mail-niveau data
    table_df = mail_df[['Year_Month', 'Mail', 'Message', 'Received_Email', 'Unique_Opens', 'Unique_Clicks', 'Open_Rate', 'Click_Rate', 'CTR', 'Unsubscribed', 'Bounced']].copy()
    
    # Sorter: nyeste måned først, derefter laveste mail nummer
    def month_to_sortable(m):
        try:
            parts = m.split('-')
            return int(parts[0]) * 100 + int(parts[1])
        except:
            return 0
    table_df['_month_num'] = table_df['Year_Month'].apply(month_to_sortable)
    table_df['_mail_num'] = table_df['Mail'].apply(lambda m: int(re.search(r'Mail\s*(\d+)', str(m)).group(1)) if re.search(r'Mail\s*(\d+)', str(m)) else 999)
    table_df = table_df.sort_values(['_month_num', '_mail_num'], ascending=[False, True])
    table_df = table_df.drop(columns=['_month_num', '_mail_num'])
    # Beregn højde så alle rækker vises (ingen max begrænsning)
    table_height = (len(table_df) + 1) * 35 + 3
    
    st.dataframe(
        table_df, use_container_width=True, hide_index=True, height=table_height,
        column_config={
            "Year_Month": st.column_config.TextColumn("Måned", width="small"),
            "Mail": st.column_config.TextColumn("Mail", width="small"),
            "Message": st.column_config.TextColumn("Message", width="medium"),
            "Received_Email": st.column_config.NumberColumn("Sendt", format="localized", width="small"),
            "Unique_Opens": st.column_config.NumberColumn("Opens", format="localized", width="small"),
            "Unique_Clicks": st.column_config.NumberColumn("Clicks", format="localized", width="small"),
            "Open_Rate": st.column_config.NumberColumn("Open Rate", format="%.1f%%", width="small"),
            "Click_Rate": st.column_config.NumberColumn("Click Rate", format="%.1f%%", width="small"),
            "CTR": st.column_config.NumberColumn("CTR", format="%.1f%%", width="small"),
            "Unsubscribed": st.column_config.NumberColumn("Unsub", format="localized", width="small"),
            "Bounced": st.column_config.NumberColumn("Bounced", format="localized", width="small"),
        }
    )


def get_short_flow_name(flow_trigger):
    """Udtræk kun 'Flow X' fra 'Flow X - Trigger Name'"""
    match = re.search(r'(Flow\s*\d+)', flow_trigger)
    return match.group(1) if match else flow_trigger


def render_flows_tab():
    """Render Flows tab indhold med sub-tabs for hvert flow"""
    
    # Load data
    try:
        with st.spinner('Henter flow data...'):
            df = load_flows_data()
        if df.empty:
            st.error("Kunne ikke hente flow data. Tjek Google Sheets konfiguration.")
            return
    except Exception as e:
        st.error(f"Fejl: {e}")
        return

    # Få tilgængelige måneder
    available_months = get_available_months(df)
    
    if not available_months:
        st.warning("Ingen måneder tilgængelige i data.")
        return

    # Session state
    if 'fl_selected_months' not in st.session_state:
        st.session_state.fl_selected_months = [available_months[0]] if available_months else []
    if 'fl_selected_countries' not in st.session_state:
        st.session_state.fl_selected_countries = None
    if 'fl_selected_flows' not in st.session_state:
        st.session_state.fl_selected_flows = None
    if 'fl_cb_reset_month' not in st.session_state:
        st.session_state.fl_cb_reset_month = 0
    if 'fl_cb_reset_land' not in st.session_state:
        st.session_state.fl_cb_reset_land = 0
    if 'fl_cb_reset_flow' not in st.session_state:
        st.session_state.fl_cb_reset_flow = 0

    # Hent alle flows først (før filtrering) for at kunne vise subtabs
    all_flow_triggers = get_unique_flows(df)
    
    # Sub-tabs: Oversigt + et tab per flow (med korte navne)
    tab_labels = ["Oversigt"] + [get_short_flow_name(f) for f in all_flow_triggers]
    sub_tabs = st.tabs(tab_labels)
    
    # Oversigt tab
    with sub_tabs[0]:
        render_overview_tab_content(df, available_months)
    
    # Individuelle flow tabs
    for i, flow_trigger in enumerate(all_flow_triggers):
        with sub_tabs[i + 1]:
            render_single_flow_tab_content(df, flow_trigger, available_months)


def render_overview_tab_content(df, available_months):
    """Render oversigt tab med filtre og indhold"""
    
    # Layout - filters
    col_month, col_land, col_flow, col_spacer = st.columns([1, 1, 1, 3])

    # Måned vælger (dropdown med multiselect)
    with col_month:
        with st.popover("Måned", use_container_width=True):
            reset_month = st.session_state.fl_cb_reset_month
            all_months_selected = set(st.session_state.fl_selected_months) == set(available_months)
            select_all_months = st.checkbox("Vælg alle", value=all_months_selected, key=f"fl_sel_all_month_{reset_month}")
            
            new_selected_months = []
            for month in available_months:
                checked = month in st.session_state.fl_selected_months
                if st.checkbox(month, value=checked, key=f"fl_cb_month_{month}_{reset_month}"):
                    new_selected_months.append(month)
            
            if select_all_months and not all_months_selected:
                st.session_state.fl_selected_months = list(available_months)
                st.session_state.fl_cb_reset_month += 1
                st.rerun()
            elif not select_all_months and all_months_selected:
                st.session_state.fl_selected_months = []
                st.session_state.fl_cb_reset_month += 1
                st.rerun()
            elif set(new_selected_months) != set(st.session_state.fl_selected_months):
                st.session_state.fl_selected_months = new_selected_months
                st.session_state.fl_cb_reset_month += 1
                st.rerun()

    # Filtrer data efter valgte måneder
    sel_months = st.session_state.fl_selected_months
    if not sel_months:
        st.warning("Vælg mindst én måned.")
        return
    
    df_month_filtered = df[df['Year_Month'].isin(sel_months)]

    # Aggreger til flow niveau for oversigten
    flow_df = aggregate_to_flow_level(df_month_filtered)

    # Filter options
    all_countries = sorted(flow_df['Country'].unique())
    all_flows = get_unique_flows(flow_df)

    # Initialize selections
    if st.session_state.fl_selected_countries is None:
        st.session_state.fl_selected_countries = list(all_countries)
    else:
        st.session_state.fl_selected_countries = [c for c in st.session_state.fl_selected_countries if c in all_countries]

    if st.session_state.fl_selected_flows is None:
        st.session_state.fl_selected_flows = list(all_flows)
    else:
        st.session_state.fl_selected_flows = [f for f in st.session_state.fl_selected_flows if f in all_flows]

    # Land filter
    with col_land:
        land_count = len(st.session_state.fl_selected_countries)
        land_label = f"Land ({land_count})" if land_count < len(all_countries) else "Land"
        with st.popover(land_label, use_container_width=True):
            reset_land = st.session_state.fl_cb_reset_land
            all_land_selected = len(st.session_state.fl_selected_countries) == len(all_countries)
            select_all_land = st.checkbox("Vælg alle", value=all_land_selected, key=f"fl_sel_all_land_{reset_land}")
            
            new_selected = []
            for country in all_countries:
                checked = country in st.session_state.fl_selected_countries
                if st.checkbox(country, value=checked, key=f"fl_cb_land_{country}_{reset_land}"):
                    new_selected.append(country)
            
            if select_all_land and not all_land_selected:
                st.session_state.fl_selected_countries = list(all_countries)
                st.session_state.fl_cb_reset_land += 1
                st.rerun()
            elif not select_all_land and all_land_selected:
                st.session_state.fl_selected_countries = []
                st.session_state.fl_cb_reset_land += 1
                st.rerun()
            elif set(new_selected) != set(st.session_state.fl_selected_countries):
                st.session_state.fl_selected_countries = new_selected
                st.session_state.fl_cb_reset_land += 1
                st.rerun()

    # Flow filter
    with col_flow:
        flow_count = len(st.session_state.fl_selected_flows)
        flow_label = f"Flow ({flow_count})" if flow_count < len(all_flows) else "Flow"
        with st.popover(flow_label, use_container_width=True):
            reset_flow = st.session_state.fl_cb_reset_flow
            all_flow_selected = len(st.session_state.fl_selected_flows) == len(all_flows)
            select_all_flow = st.checkbox("Vælg alle", value=all_flow_selected, key=f"fl_sel_all_flow_{reset_flow}")
            
            new_selected_flows = []
            for flow in all_flows:
                # Vis kun kort flow-navn i checkboxen
                short_name = get_short_flow_name(flow)
                checked = flow in st.session_state.fl_selected_flows
                if st.checkbox(short_name, value=checked, key=f"fl_cb_flow_{flow}_{reset_flow}"):
                    new_selected_flows.append(flow)
            
            if select_all_flow and not all_flow_selected:
                st.session_state.fl_selected_flows = list(all_flows)
                st.session_state.fl_cb_reset_flow += 1
                st.rerun()
            elif not select_all_flow and all_flow_selected:
                st.session_state.fl_selected_flows = []
                st.session_state.fl_cb_reset_flow += 1
                st.rerun()
            elif set(new_selected_flows) != set(st.session_state.fl_selected_flows):
                st.session_state.fl_selected_flows = new_selected_flows
                st.session_state.fl_cb_reset_flow += 1
                st.rerun()

    # Check selections
    sel_countries = st.session_state.fl_selected_countries
    sel_flows = st.session_state.fl_selected_flows

    if not sel_countries:
        st.warning("Vælg mindst ét land.")
        return
    
    if not sel_flows:
        st.warning("Vælg mindst ét flow.")
        return

    # Render indhold
    render_overview_content(flow_df, sel_countries, sel_flows)

    if st.button('Opdater Data', key="fl_refresh_overview"):
        st.rerun()


def render_single_flow_tab_content(df, flow_trigger, available_months):
    """Render enkelt flow tab med filtre og indhold"""
    
    # Layout - filters
    col_month, col_land, col_mail, col_spacer = st.columns([1, 1, 1, 3])

    # Måned vælger
    with col_month:
        with st.popover("Måned", use_container_width=True):
            reset_month = st.session_state.fl_cb_reset_month
            all_months_selected = set(st.session_state.fl_selected_months) == set(available_months)
            select_all_months = st.checkbox("Vælg alle", value=all_months_selected, key=f"fl_sel_all_month_sf_{flow_trigger}_{reset_month}")
            
            new_selected_months = []
            for month in available_months:
                checked = month in st.session_state.fl_selected_months
                if st.checkbox(month, value=checked, key=f"fl_cb_month_sf_{flow_trigger}_{month}_{reset_month}"):
                    new_selected_months.append(month)
            
            if select_all_months and not all_months_selected:
                st.session_state.fl_selected_months = list(available_months)
                st.session_state.fl_cb_reset_month += 1
                st.rerun()
            elif not select_all_months and all_months_selected:
                st.session_state.fl_selected_months = []
                st.session_state.fl_cb_reset_month += 1
                st.rerun()
            elif set(new_selected_months) != set(st.session_state.fl_selected_months):
                st.session_state.fl_selected_months = new_selected_months
                st.session_state.fl_cb_reset_month += 1
                st.rerun()

    # Filtrer data efter valgte måneder
    sel_months = st.session_state.fl_selected_months
    if not sel_months:
        st.warning("Vælg mindst én måned.")
        return
    
    df_month_filtered = df[df['Year_Month'].isin(sel_months)]

    # Filter options - kun lande der har data for dette flow
    flow_data = df_month_filtered[df_month_filtered['Flow_Trigger'] == flow_trigger]
    all_countries = sorted(flow_data['Country'].unique()) if not flow_data.empty else []

    if not all_countries:
        st.warning(f"Ingen data for dette flow i de valgte måneder.")
        return

    # Hent alle mails for dette flow
    def mail_sort_key(m):
        match = re.search(r'Mail\s*(\d+)', str(m))
        return int(match.group(1)) if match else 999
    all_mails = sorted(flow_data['Mail'].unique(), key=mail_sort_key)

    # Initialize country selection
    if st.session_state.fl_selected_countries is None:
        st.session_state.fl_selected_countries = list(all_countries)
    else:
        st.session_state.fl_selected_countries = [c for c in st.session_state.fl_selected_countries if c in all_countries]
        if not st.session_state.fl_selected_countries:
            st.session_state.fl_selected_countries = list(all_countries)

    # Initialize mail selection (per flow)
    mail_state_key = f'fl_selected_mails_{flow_trigger}'
    mail_reset_key = f'fl_cb_reset_mail_{flow_trigger}'
    if mail_state_key not in st.session_state:
        st.session_state[mail_state_key] = list(all_mails)
    else:
        st.session_state[mail_state_key] = [m for m in st.session_state[mail_state_key] if m in all_mails]
        if not st.session_state[mail_state_key]:
            st.session_state[mail_state_key] = list(all_mails)
    if mail_reset_key not in st.session_state:
        st.session_state[mail_reset_key] = 0

    # Land filter
    with col_land:
        land_count = len(st.session_state.fl_selected_countries)
        land_label = f"Land ({land_count})" if land_count < len(all_countries) else "Land"
        with st.popover(land_label, use_container_width=True):
            reset_land = st.session_state.fl_cb_reset_land
            all_land_selected = len(st.session_state.fl_selected_countries) == len(all_countries)
            select_all_land = st.checkbox("Vælg alle", value=all_land_selected, key=f"fl_sel_all_land_sf_{flow_trigger}_{reset_land}")
            
            new_selected = []
            for country in all_countries:
                checked = country in st.session_state.fl_selected_countries
                if st.checkbox(country, value=checked, key=f"fl_cb_land_sf_{flow_trigger}_{country}_{reset_land}"):
                    new_selected.append(country)
            
            if select_all_land and not all_land_selected:
                st.session_state.fl_selected_countries = list(all_countries)
                st.session_state.fl_cb_reset_land += 1
                st.rerun()
            elif not select_all_land and all_land_selected:
                st.session_state.fl_selected_countries = []
                st.session_state.fl_cb_reset_land += 1
                st.rerun()
            elif set(new_selected) != set(st.session_state.fl_selected_countries):
                st.session_state.fl_selected_countries = new_selected
                st.session_state.fl_cb_reset_land += 1
                st.rerun()

    # Mail filter
    with col_mail:
        mail_count = len(st.session_state[mail_state_key])
        mail_label = f"Mail ({mail_count})" if mail_count < len(all_mails) else "Mail"
        with st.popover(mail_label, use_container_width=True):
            reset_mail = st.session_state[mail_reset_key]
            all_mail_selected = len(st.session_state[mail_state_key]) == len(all_mails)
            select_all_mail = st.checkbox("Vælg alle", value=all_mail_selected, key=f"fl_sel_all_mail_{flow_trigger}_{reset_mail}")
            
            new_selected_mails = []
            for mail in all_mails:
                checked = mail in st.session_state[mail_state_key]
                if st.checkbox(str(mail), value=checked, key=f"fl_cb_mail_{flow_trigger}_{mail}_{reset_mail}"):
                    new_selected_mails.append(mail)
            
            if select_all_mail and not all_mail_selected:
                st.session_state[mail_state_key] = list(all_mails)
                st.session_state[mail_reset_key] += 1
                st.rerun()
            elif not select_all_mail and all_mail_selected:
                st.session_state[mail_state_key] = []
                st.session_state[mail_reset_key] += 1
                st.rerun()
            elif set(new_selected_mails) != set(st.session_state[mail_state_key]):
                st.session_state[mail_state_key] = new_selected_mails
                st.session_state[mail_reset_key] += 1
                st.rerun()

    # Check selections
    sel_countries = st.session_state.fl_selected_countries
    sel_mails = st.session_state[mail_state_key]

    if not sel_countries:
        st.warning("Vælg mindst ét land.")
        return
    
    if not sel_mails:
        st.warning("Vælg mindst én mail.")
        return

    # Render indhold
    render_single_flow_content(df_month_filtered, flow_trigger, sel_countries, sel_mails)

    if st.button('Opdater Data', key=f"fl_refresh_{flow_trigger}"):
        st.rerun()
