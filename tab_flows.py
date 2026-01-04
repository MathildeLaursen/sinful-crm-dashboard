"""
Flows Tab - CRM Dashboard
Med sub-tabs for hvert flow
"""
import streamlit as st
import pandas as pd
import re
import datetime
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


def get_previous_month(year_month):
    """Returnerer forrige måned i format YYYY-M"""
    parts = year_month.split('-')
    year, month = int(parts[0]), int(parts[1])
    if month == 1:
        return f"{year - 1}-12"
    else:
        return f"{year}-{month - 1}"


def calculate_month_progress():
    """Beregn hvor langt vi er i den nuværende måned (data til og med i går)"""
    import calendar
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    
    # Dage med data i denne måned (fra 1. til i går)
    days_with_data = yesterday.day
    
    # Totale dage i måneden
    total_days_in_month = calendar.monthrange(today.year, today.month)[1]
    
    # Procent af måneden
    return days_with_data / total_days_in_month


def get_current_year_month():
    """Returnerer nuværende måned i format YYYY-M"""
    today = datetime.date.today()
    return f"{today.year}-{today.month}"


def get_months_in_range(start_month, end_month, all_months):
    """Returnerer alle måneder mellem start og slut (inklusiv) fra listen af tilgængelige måneder"""
    def month_to_tuple(m):
        parts = m.split('-')
        return (int(parts[0]), int(parts[1]))
    
    start_tuple = month_to_tuple(start_month)
    end_tuple = month_to_tuple(end_month)
    
    # Sørg for at start er før end
    if start_tuple > end_tuple:
        start_tuple, end_tuple = end_tuple, start_tuple
    
    # Filtrer måneder inden for range
    result = []
    for m in all_months:
        m_tuple = month_to_tuple(m)
        if start_tuple <= m_tuple <= end_tuple:
            result.append(m)
    
    return sorted(result, key=month_to_tuple)


def format_month_short(year_month):
    """Formater måned til kort dansk format (Jan 25, Feb 25, osv.)"""
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Maj', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Dec'
    }
    parts = year_month.split('-')
    year = parts[0][2:]  # Sidste 2 cifre af året
    month = int(parts[1])
    return f"{month_names[month]} {year}"


def render_overview_content(flow_df, sel_countries, sel_flows, full_df=None, all_months_df=None):
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

    # KPI totaler for nuværende periode
    total_received = display_df['Received_Email'].sum()
    total_opens = display_df['Unique_Opens'].sum()
    total_clicks = display_df['Unique_Clicks'].sum()
    
    open_rate = (total_opens / total_received * 100) if total_received > 0 else 0
    click_rate = (total_clicks / total_received * 100) if total_received > 0 else 0
    ctr = (total_clicks / total_opens * 100) if total_opens > 0 else 0

    # Beregn sammenligning med forrige periode
    prev_received = None
    prev_opens = None
    prev_clicks = None
    prev_or = None
    prev_cr = None
    prev_ctr = None
    
    if full_df is not None:
        # Find de valgte måneder og sorter dem
        selected_months = display_df['Year_Month'].unique().tolist()
        current_month = get_current_year_month()
        
        # Sorter måneder (ældste først) for at finde den ældste
        def month_to_tuple(m):
            parts = m.split('-')
            return (int(parts[0]), int(parts[1]))
        
        sorted_months = sorted(selected_months, key=month_to_tuple)
        oldest_selected = sorted_months[0]
        num_months = len(selected_months)
        
        # Find sammenligningsperioden: N måneder FØR den ældste valgte måned
        prev_months = []
        temp_month = oldest_selected
        for _ in range(num_months):
            temp_month = get_previous_month(temp_month)
            prev_months.append(temp_month)
        
        # Den ældste måned i sammenligningsperioden (skal evt. skaleres)
        oldest_prev_month = min(prev_months, key=month_to_tuple)
        
        # Hent data for forrige periode
        prev_df = full_df[
            (full_df['Year_Month'].isin(prev_months)) &
            (full_df['Country'].isin(sel_countries)) &
            (full_df['Flow_Trigger'].isin(sel_flows))
        ].copy()
        
        if not prev_df.empty:
            # Aggreger forrige periode PER MÅNED
            prev_agg = prev_df.groupby(['Year_Month'], as_index=False).agg({
                'Received_Email': 'sum',
                'Unique_Opens': 'sum',
                'Unique_Clicks': 'sum',
            })
            
            # Beregn totaler med korrekt skalering
            prev_received = 0
            prev_opens = 0
            prev_clicks = 0
            
            # Tjek om nuværende måned er valgt (så skal den ældste i sammenligning skaleres)
            current_month_selected = current_month in selected_months
            month_progress = calculate_month_progress() if current_month_selected else 1.0
            
            for _, row in prev_agg.iterrows():
                month = row['Year_Month']
                
                if month == oldest_prev_month and current_month_selected:
                    # Denne er den ældste i sammenligningen og nuværende måned er valgt - skaler
                    prev_received += row['Received_Email'] * month_progress
                    prev_opens += row['Unique_Opens'] * month_progress
                    prev_clicks += row['Unique_Clicks'] * month_progress
                else:
                    # Fuld måned sammenligning
                    prev_received += row['Received_Email']
                    prev_opens += row['Unique_Opens']
                    prev_clicks += row['Unique_Clicks']
            
            # Beregn rater for forrige periode
            if prev_received > 0:
                prev_or = (prev_opens / prev_received * 100)
                prev_cr = (prev_clicks / prev_received * 100)
            if prev_opens > 0:
                prev_ctr = (prev_clicks / prev_opens * 100)

    # KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    show_metric(col1, "Emails Sendt", total_received, prev_received)
    show_metric(col2, "Unikke Opens", total_opens, prev_opens)
    show_metric(col3, "Unikke Clicks", total_clicks, prev_clicks)
    show_metric(col4, "Open Rate", open_rate, prev_or, is_percent=True)
    show_metric(col5, "Click Rate", click_rate, prev_cr, is_percent=True)
    show_metric(col6, "Click Through Rate", ctr, prev_ctr, is_percent=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Chart - tidslinje: Antal sendt per flow over tid (ALLE måneder, uanset slider)
    # Unicorn farvepalette - dynamisk tildeling til flows
    # Farver gentages hvis der er flere flows end farver
    UNICORN_COLORS = [
        '#9B7EBD',   # Lilla
        '#E8B4CB',   # Pink
        '#A8E6CF',   # Mint
        '#7EC8E3',   # Lyseblå
        '#F7DC6F',   # Gul
        '#BB8FCE',   # Violet
        '#F1948A',   # Koral
        '#85C1E9',   # Himmelblå
        '#82E0AA',   # Lysegrøn
        '#D7BDE2',   # Lavendel
        '#F5B7B1',   # Fersken
        '#FAD7A0',   # Abrikos
        '#AED6F1',   # Isblå
        '#D5F5E3',   # Mintgrøn
        '#FADBD8',   # Lyserød
    ]
    
    # Brug alle måneder til grafen (ikke filtreret af slider)
    chart_source_df = all_months_df if all_months_df is not None else display_df
    chart_source_df = chart_source_df[
        (chart_source_df['Country'].isin(sel_countries)) &
        (chart_source_df['Flow_Trigger'].isin(sel_flows))
    ].copy()
    
    # Aggreger per flow og måned
    chart_df = chart_source_df.groupby(['Year_Month', 'Flow_Trigger'], as_index=False).agg({
        'Received_Email': 'sum',
    })
    
    # Formater måned til visning
    chart_df['Month_Label'] = chart_df['Year_Month'].apply(format_month_short)
    
    # Opret sorteringsnøgle (YYYYMM som tal for korrekt kronologisk sortering)
    def month_sort_key(ym):
        parts = ym.split('-')
        return int(parts[0]) * 100 + int(parts[1])
    
    chart_df['Sort_Key'] = chart_df['Year_Month'].apply(month_sort_key)
    chart_df = chart_df.sort_values('Sort_Key')
    
    # Få kronologisk sorterede måneder til x-aksen
    months_sorted = chart_df.drop_duplicates('Year_Month').sort_values('Sort_Key')['Month_Label'].tolist()
    
    # Få unikke flows sorteret efter nummer
    def chart_flow_sort_key(f):
        match = re.search(r'Flow\s*(\d+)', f)
        return int(match.group(1)) if match else 999
    unique_flows = sorted(chart_df['Flow_Trigger'].unique(), key=chart_flow_sort_key)

    fig = go.Figure()
    
    # Tilføj linje for hvert flow
    for idx, flow in enumerate(unique_flows):
        flow_data = chart_df[chart_df['Flow_Trigger'] == flow].sort_values('Sort_Key')
        short_name = get_short_flow_name(flow)
        
        # Hent farve dynamisk baseret på index (gentager hvis flere flows end farver)
        color = UNICORN_COLORS[idx % len(UNICORN_COLORS)]
        
        fig.add_trace(
            go.Scatter(
                x=flow_data['Month_Label'], 
                y=flow_data['Received_Email'],
                name=short_name,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f'{short_name}<br>%{{x}}: %{{y:,.0f}} sendt<extra></extra>'
            )
        )
    
    fig.update_layout(
        title="",
        showlegend=True, 
        height=520,
        margin=dict(l=80, r=30, t=60, b=50),
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=1.02, 
            xanchor="center", x=0.5,
            font=dict(size=10)
        ),
        plot_bgcolor='rgba(250,245,255,0.5)', 
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    # Formater y-aksen med kompakte tal (K for tusinder)
    def format_y_tick(val):
        if val >= 1000000:
            return f'{val/1000000:.0f}M'
        elif val >= 1000:
            return f'{val/1000:.0f}K'
        return f'{val:.0f}'
    
    # Beregn passende tick values
    max_val = chart_df['Received_Email'].max() if not chart_df.empty else 100000
    tick_step = 50000 if max_val > 100000 else 25000
    tick_vals = list(range(0, int(max_val * 1.2), tick_step))
    tick_text = [format_y_tick(v) for v in tick_vals]
    
    fig.update_yaxes(
        title_text="Antal sendt",
        title_font=dict(size=12, color='#7B5EA5'),
        title_standoff=15,
        gridcolor='rgba(212,191,255,0.3)',
        tickvals=tick_vals,
        ticktext=tick_text,
        tickfont=dict(size=10)
    )
    fig.update_xaxes(
        gridcolor='rgba(212,191,255,0.2)', 
        tickfont=dict(size=11),
        categoryorder='array',
        categoryarray=months_sorted
    )
    
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


def render_single_flow_content(raw_df, flow_trigger, sel_countries, sel_mails=None, full_df=None):
    """Render indhold for et enkelt flow med mail-niveau breakdown"""
    # Filtrer til dette specifikke flow og valgte lande (UDEN mail filter - til scorecards)
    flow_data_all = raw_df[
        (raw_df['Flow_Trigger'] == flow_trigger) &
        (raw_df['Country'].isin(sel_countries))
    ].copy()
    
    if flow_data_all.empty:
        st.warning(f"Ingen data for {flow_trigger} med de valgte filtre.")
        return
    
    # Aggreger ALLE mails for scorecards (ingen mail filter)
    scorecard_df = flow_data_all.groupby(['Year_Month'], as_index=False).agg({
        'Received_Email': 'sum',
        'Unique_Opens': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })

    # KPI totaler for dette flow (ALLE mails)
    total_received = scorecard_df['Received_Email'].sum()
    total_opens = scorecard_df['Unique_Opens'].sum()
    total_clicks = scorecard_df['Unique_Clicks'].sum()
    total_unsub = scorecard_df['Unsubscribed'].sum()
    total_bounced = scorecard_df['Bounced'].sum()
    
    open_rate = (total_opens / total_received * 100) if total_received > 0 else 0
    click_rate = (total_clicks / total_received * 100) if total_received > 0 else 0
    ctr = (total_clicks / total_opens * 100) if total_opens > 0 else 0

    # Sammenligning med forrige periode (ALLE mails - ingen mail filter)
    prev_received = None
    prev_opens = None
    prev_clicks = None
    prev_or = None
    prev_cr = None
    prev_ctr = None
    
    if full_df is not None:
        # Find de valgte måneder
        selected_months = scorecard_df['Year_Month'].unique().tolist()
        current_month = get_current_year_month()
        
        # Sorter måneder for at finde den ældste
        def month_to_tuple(m):
            parts = m.split('-')
            return (int(parts[0]), int(parts[1]))
        
        sorted_months = sorted(selected_months, key=month_to_tuple)
        oldest_selected = sorted_months[0]
        num_months = len(selected_months)
        
        # Find sammenligningsperioden: N måneder FØR den ældste valgte måned
        prev_months = []
        temp_month = oldest_selected
        for _ in range(num_months):
            temp_month = get_previous_month(temp_month)
            prev_months.append(temp_month)
        
        # Den ældste måned i sammenligningsperioden
        oldest_prev_month = min(prev_months, key=month_to_tuple)
        
        # Hent data for forrige periode (samme flow og lande - INGEN mail filter for scorecards)
        prev_filter = (
            (full_df['Year_Month'].isin(prev_months)) &
            (full_df['Flow_Trigger'] == flow_trigger) &
            (full_df['Country'].isin(sel_countries))
        )
        
        prev_data = full_df[prev_filter].copy()
        
        if not prev_data.empty:
            # Aggreger forrige periode PER MÅNED
            prev_agg = prev_data.groupby(['Year_Month'], as_index=False).agg({
                'Received_Email': 'sum',
                'Unique_Opens': 'sum',
                'Unique_Clicks': 'sum',
            })
            
            # Beregn totaler med korrekt skalering
            prev_received = 0
            prev_opens = 0
            prev_clicks = 0
            
            # Tjek om nuværende måned er valgt
            current_month_selected = current_month in selected_months
            month_progress = calculate_month_progress() if current_month_selected else 1.0
            
            for _, row in prev_agg.iterrows():
                month = row['Year_Month']
                
                if month == oldest_prev_month and current_month_selected:
                    # Skaler den ældste måned i sammenligningen
                    prev_received += row['Received_Email'] * month_progress
                    prev_opens += row['Unique_Opens'] * month_progress
                    prev_clicks += row['Unique_Clicks'] * month_progress
                else:
                    # Fuld måned
                    prev_received += row['Received_Email']
                    prev_opens += row['Unique_Opens']
                    prev_clicks += row['Unique_Clicks']
            
            # Beregn rater for forrige periode
            if prev_received > 0:
                prev_or = (prev_opens / prev_received * 100)
                prev_cr = (prev_clicks / prev_received * 100)
            if prev_opens > 0:
                prev_ctr = (prev_clicks / prev_opens * 100)

    # KPI Cards (viser ALLE mails - ikke påvirket af Ignorer Inaktive)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    show_metric(col1, "Emails Sendt", total_received, prev_received)
    show_metric(col2, "Unikke Opens", total_opens, prev_opens)
    show_metric(col3, "Unikke Clicks", total_clicks, prev_clicks)
    show_metric(col4, "Open Rate", open_rate, prev_or, is_percent=True)
    show_metric(col5, "Click Rate", click_rate, prev_cr, is_percent=True)
    show_metric(col6, "Click Through Rate", ctr, prev_ctr, is_percent=True)
    
    # --- Filtreret data til tabel og grafer (påvirkes af Ignorer Inaktive) ---
    flow_data = flow_data_all.copy()
    if sel_mails is not None and len(sel_mails) > 0:
        # Opret filter for hver (Mail, Message) kombination
        mail_filter = pd.Series([False] * len(flow_data), index=flow_data.index)
        for mail, msg in sel_mails:
            if pd.notna(msg) and msg != '':
                mail_filter |= (flow_data['Mail'] == mail) & (flow_data['Message'] == msg)
            else:
                mail_filter |= (flow_data['Mail'] == mail) & (flow_data['Message'].isna() | (flow_data['Message'] == ''))
        flow_data = flow_data[mail_filter]
    
    # Aggreger per mail for tabel og grafer
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

    # Fast afstand mellem scorecards og grafer (30px)
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    # Chart - stacked grafer, en per mail (viser ALLE måneder, uanset slider)
    # Sorter måneder kronologisk
    def month_to_sortkey(m):
        parts = m.split('-')
        return int(parts[0]) * 100 + int(parts[1])
    
    # Brug ALLE måneder fra full_df (ikke filtreret af slider)
    all_months_for_chart = full_df[full_df['Flow_Trigger'] == flow_trigger]['Year_Month'].unique()
    sorted_chart_months = sorted(all_months_for_chart, key=month_to_sortkey)
    
    # Hent ufiltreret data til grafen
    chart_base_df = full_df[
        (full_df['Flow_Trigger'] == flow_trigger) &
        (full_df['Country'].isin(sel_countries))
    ].copy()
    if sel_mails is not None and len(sel_mails) > 0:
        # Opret filter for hver (Mail, Message) kombination
        mail_filter_chart = pd.Series([False] * len(chart_base_df), index=chart_base_df.index)
        for mail, msg in sel_mails:
            if pd.notna(msg) and msg != '':
                mail_filter_chart |= (chart_base_df['Mail'] == mail) & (chart_base_df['Message'] == msg)
            else:
                mail_filter_chart |= (chart_base_df['Mail'] == mail) & (chart_base_df['Message'].isna() | (chart_base_df['Message'] == ''))
        chart_base_df = chart_base_df[mail_filter_chart]
    
    # Formatér måneder til visning
    def format_month_label(m):
        parts = m.split('-')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
        return f"{month_names[int(parts[1])-1]} {parts[0][2:]}"
    
    # Hent unikke Mail+Message kombinationer sorteret (matcher tabellen)
    def mail_sort_key(row):
        match = re.search(r'Mail\s*(\d+)', str(row['Mail']))
        mail_num = int(match.group(1)) if match else 999
        msg = str(row['Message']) if pd.notna(row['Message']) else ''
        return (mail_num, msg)
    
    # Opret unikke kombinationer af Mail+Message
    unique_combos = chart_base_df.groupby(['Mail', 'Message'], as_index=False).first()[['Mail', 'Message']]
    unique_combos['_sort_key'] = unique_combos.apply(mail_sort_key, axis=1)
    unique_combos = unique_combos.sort_values('_sort_key').drop(columns=['_sort_key'])
    unique_combos_list = list(unique_combos.itertuples(index=False, name=None))
    
    if len(sorted_chart_months) > 0 and len(unique_combos_list) > 0:
        # Opret subplots - en raekke per mail+message kombination
        num_items = len(unique_combos_list)
        
        # Titel format: "Mail x - Message" hvis message findes, ellers bare "Mail x"
        subplot_titles = []
        for mail, message in unique_combos_list:
            if pd.notna(message) and message != '':
                subplot_titles.append(f"{mail} - {message}")
            else:
                subplot_titles.append(str(mail))
        
        # Fast pixel-afstand mellem grafer uanset antal
        # Beregn spacing saa pixel-afstanden er ens
        height_per_chart = 150  # pixels per graf
        gap_pixels = 70  # oensket afstand i pixels
        chart_height = num_items * height_per_chart + (num_items - 1) * gap_pixels
        
        # v_spacing er relativ til total hoejde, saa beregn den
        # Spacing = gap / (chart_height - margins) ca.
        v_spacing = gap_pixels / chart_height if num_items > 1 else 0.1
        
        # Sørg for at v_spacing er inden for Plotly's grænser
        max_allowed = 1.0 / (num_items - 1) if num_items > 1 else 0.5
        v_spacing = min(v_spacing, max_allowed * 0.95)
        
        fig = make_subplots(
            rows=num_items, cols=1,
            shared_xaxes=False,  # Vis x-akse labels paa alle grafer
            vertical_spacing=v_spacing,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True}] for _ in range(num_items)]
        )
        
        for i, (mail, message) in enumerate(unique_combos_list):
            row = i + 1
            # Filtrer paa baade Mail og Message
            if pd.notna(message) and message != '':
                mail_data = chart_base_df[(chart_base_df['Mail'] == mail) & (chart_base_df['Message'] == message)].copy()
            else:
                mail_data = chart_base_df[(chart_base_df['Mail'] == mail) & (chart_base_df['Message'].isna() | (chart_base_df['Message'] == ''))].copy()
            
            # Opret data for alle maaneder
            sent_values = []
            opens_values = []
            clicks_values = []
            for month in sorted_chart_months:
                month_data = mail_data[mail_data['Year_Month'] == month]
                if not month_data.empty:
                    sent_values.append(month_data['Received_Email'].sum())
                    opens_values.append(month_data['Unique_Opens'].sum())
                    clicks_values.append(month_data['Unique_Clicks'].sum())
                else:
                    sent_values.append(0)
                    opens_values.append(0)
                    clicks_values.append(0)
            
            x_labels = [format_month_label(m) for m in sorted_chart_months]
            
            # Linje for Antal Sendt (venstre y-akse)
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=sent_values,
                    name='Sendt',
                    mode='lines+markers',
                    line=dict(color='#9B7EBD', width=2),
                    marker=dict(size=6, color='#9B7EBD'),
                    showlegend=(i == 0),
                    legendgroup='sendt'
                ),
                row=row, col=1, secondary_y=False
            )
            
            # Linje for Opens (venstre y-akse - sammen med Sendt)
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=opens_values,
                    name='Opens',
                    mode='lines+markers',
                    line=dict(color='#A8E6CF', width=2),
                    marker=dict(size=6, color='#A8E6CF'),
                    showlegend=(i == 0),
                    legendgroup='opens'
                ),
                row=row, col=1, secondary_y=False
            )
            
            # Linje for Clicks (hoejre y-akse)
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=clicks_values,
                    name='Clicks',
                    mode='lines+markers',
                    line=dict(color='#E8B4CB', width=2),
                    marker=dict(size=6, color='#E8B4CB'),
                    showlegend=(i == 0),
                    legendgroup='clicks'
                ),
                row=row, col=1, secondary_y=True
            )
            
            # Beregn max for at skalere højre y-akse korrekt
            max_left = max(max(sent_values) if sent_values else 0, max(opens_values) if opens_values else 0)
            max_clicks = max(clicks_values) if clicks_values else 0
            
            # Skaler højre y-akse så clicks vises i nederste del (ca. 25% af plottet)
            if max_clicks > 0 and max_left > 0:
                clicks_range_max = max_clicks * 4  # Udvid range så max clicks er ved ca. 25%
                fig.update_yaxes(range=[0, clicks_range_max], row=row, col=1, secondary_y=True)
        
        # chart_height er allerede beregnet ovenfor - brug minimum 350
        chart_height = max(350, chart_height)
        
        # Fast top margin der inkluderer plads til legend
        top_margin = 80
        
        # Layout med faste pixel-margins
        fig.update_layout(
            showlegend=True,
            height=chart_height + top_margin,  # Tilfoej top margin til total hoejde
            margin=dict(l=60, r=60, t=top_margin, b=40),
            legend=dict(
                orientation="h", 
                yanchor="bottom",  # Ankrer bund af legend
                y=1.02,  # Over plottet
                xanchor="right", 
                x=1,
                bgcolor='rgba(0,0,0,0)'
            ),
            plot_bgcolor='rgba(250,245,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        # Style subplot titler - taettere paa grafen
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=13, color='#7B5EA5', family='sans-serif')
            annotation['xanchor'] = 'left'
            annotation['x'] = 0  # Venstrestillet
            annotation['yanchor'] = 'bottom'
            annotation['yshift'] = 10  # Taettere paa grafen
        
        # Opdater alle y-akser med titler (paa alle grafer)
        for i in range(num_items):
            # Venstre y-akse (Sendt + Opens) - starter altid ved 0
            fig.update_yaxes(
                title_text="Sendt / Opens",
                title_font=dict(size=10, color='#4A3F55'),
                gridcolor='rgba(212,191,255,0.3)',
                tickformat=',d',
                rangemode='tozero',
                row=i+1, col=1, secondary_y=False
            )
            # Hoejre y-akse (Clicks) - starter altid ved 0
            fig.update_yaxes(
                title_text="Clicks",
                title_font=dict(size=10, color='#4A3F55'),
                gridcolor='rgba(232,180,203,0.2)',
                showgrid=False,
                tickformat=',d',
                rangemode='tozero',
                row=i+1, col=1, secondary_y=True
            )
        
        # Opdater x-akser (kun vis labels paa nederste)
        fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', type='category', tickfont=dict(size=10))
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Tabel med mail-niveau data (altid aggregeret - uden måned kolonne)
    table_df = mail_df.groupby(['Mail', 'Message'], as_index=False).agg({
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
    
    # Kombiner Mail og Message til én kolonne
    table_df['Mail_Display'] = table_df.apply(
        lambda x: f"{x['Mail']} - {x['Message']}" if pd.notna(x['Message']) and x['Message'] != '' else str(x['Mail']), 
        axis=1
    )
    
    # Sorter efter mail nummer og message
    table_df['_mail_num'] = table_df['Mail'].apply(lambda m: int(re.search(r'Mail\s*(\d+)', str(m)).group(1)) if re.search(r'Mail\s*(\d+)', str(m)) else 999)
    table_df['_msg'] = table_df['Message'].apply(lambda m: str(m) if pd.notna(m) else '')
    table_df = table_df.sort_values(['_mail_num', '_msg'], ascending=True)
    table_df = table_df.drop(columns=['_mail_num', '_msg', 'Mail', 'Message'])
    
    # Kolonner med kombineret Mail kolonne
    table_df = table_df[['Mail_Display', 'Received_Email', 'Unique_Opens', 'Unique_Clicks', 'Open_Rate', 'Click_Rate', 'CTR', 'Unsubscribed', 'Bounced']]
    
    # Beregn hoejde saa alle raekker vises (35px per raekke + 38px header)
    table_height = (len(table_df) + 1) * 35 + 3
    
    st.dataframe(
        table_df, use_container_width=True, hide_index=True, height=table_height,
        column_config={
            "Mail_Display": st.column_config.TextColumn("Mail", width="medium"),
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

    # Session state (kun for lande og flows - måneder styres af slider)
    if 'fl_selected_countries' not in st.session_state:
        st.session_state.fl_selected_countries = None
    if 'fl_selected_flows' not in st.session_state:
        st.session_state.fl_selected_flows = None
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
    
    # Sorter måneder kronologisk (ældste først for slider)
    def month_to_tuple(m):
        parts = m.split('-')
        return (int(parts[0]), int(parts[1]))
    
    sorted_months = sorted(available_months, key=month_to_tuple)
    
    # Aggreger fuld df først for at få filter options
    full_flow_df = aggregate_to_flow_level(df)
    all_countries = sorted(full_flow_df['Country'].unique())
    all_flows_full = get_unique_flows(full_flow_df)
    
    # Find aktive flows (sendt denne eller sidste måned)
    current_month = get_current_year_month()
    today = datetime.date.today()
    if today.month == 1:
        prev_month = f"{today.year - 1}-12"
    else:
        prev_month = f"{today.year}-{today.month - 1}"
    
    recent_months = [current_month, prev_month]
    recent_data = full_flow_df[
        (full_flow_df['Year_Month'].isin(recent_months)) & 
        (full_flow_df['Received_Email'] > 0)
    ]
    active_flows = set(recent_data['Flow_Trigger'].unique())
    
    # Initialize ignore inactive state (default True)
    if 'fl_ignore_inactive_overview' not in st.session_state:
        st.session_state.fl_ignore_inactive_overview = True
    
    # Filtrer flows baseret på ignore_inactive
    all_flows = [f for f in all_flows_full if f in active_flows] if st.session_state.fl_ignore_inactive_overview else all_flows_full

    # Initialize selections
    if st.session_state.fl_selected_countries is None:
        st.session_state.fl_selected_countries = list(all_countries)
    else:
        st.session_state.fl_selected_countries = [c for c in st.session_state.fl_selected_countries if c in all_countries]

    if st.session_state.fl_selected_flows is None:
        st.session_state.fl_selected_flows = list(all_flows)
    else:
        st.session_state.fl_selected_flows = [f for f in st.session_state.fl_selected_flows if f in all_flows]
        if not st.session_state.fl_selected_flows:
            st.session_state.fl_selected_flows = list(all_flows)

    # Layout - dropdowns, checkbox og slider på samme linje
    col_land, col_flow, col_inaktive, col_slider = st.columns([1, 1, 0.8, 3.2])

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

    # Ignorer Inaktive checkbox
    with col_inaktive:
        ignore_inactive = st.checkbox(
            "Ignorer Inaktive", 
            value=st.session_state.fl_ignore_inactive_overview, 
            key="fl_ignore_inactive_overview_cb"
        )
        if ignore_inactive != st.session_state.fl_ignore_inactive_overview:
            st.session_state.fl_ignore_inactive_overview = ignore_inactive
            # Reset flow selection når toggle ændres
            new_all_flows = [f for f in all_flows_full if f in active_flows] if ignore_inactive else all_flows_full
            st.session_state.fl_selected_flows = list(new_all_flows)
            st.session_state.fl_cb_reset_flow += 1
            st.rerun()

    # Periode slider (på samme linje som dropdowns)
    with col_slider:
        if len(sorted_months) > 1:
            # Beregn default værdier
            default_end = sorted_months[-1]
            default_start_idx = max(0, len(sorted_months) - 3)
            default_start = sorted_months[default_start_idx]
            
            # Brug separat saved state (ikke widget key) for at overleve st.rerun()
            saved_key = "fl_month_range_overview_saved"
            if saved_key not in st.session_state:
                st.session_state[saved_key] = (default_start, default_end)
            
            # Valider at gemte værdier stadig er i sorted_months
            saved_range = st.session_state[saved_key]
            if saved_range[0] in sorted_months and saved_range[1] in sorted_months:
                initial_value = saved_range
            else:
                initial_value = (default_start, default_end)
                st.session_state[saved_key] = initial_value
            
            month_range = st.select_slider(
                "Periode",
                options=sorted_months,
                value=initial_value,
                format_func=format_month_short,
                key="fl_month_range_overview",
                label_visibility="collapsed"
            )
            
            # Gem værdien i separat state efter rendering
            st.session_state[saved_key] = month_range
            
            sel_months = get_months_in_range(month_range[0], month_range[1], sorted_months)
        else:
            # Kun én måned tilgængelig
            sel_months = sorted_months
    
    if not sel_months:
        st.warning("Vælg mindst én måned.")
        return

    # Filtrer data efter valgte måneder
    df_month_filtered = df[df['Year_Month'].isin(sel_months)]
    
    # Aggreger til flow niveau for oversigten
    flow_df = aggregate_to_flow_level(df_month_filtered)

    # Check selections
    sel_countries = st.session_state.fl_selected_countries
    sel_flows = st.session_state.fl_selected_flows

    if not sel_countries:
        st.warning("Vælg mindst ét land.")
        return

    if not sel_flows:
        st.warning("Vælg mindst ét flow.")
        return

    # Render indhold (send også alle måneder til grafen)
    all_months_flow_df = aggregate_to_flow_level(df)
    render_overview_content(flow_df, sel_countries, sel_flows, full_flow_df, all_months_flow_df)

    if st.button('Opdater Data', key="fl_refresh_overview"):
        st.rerun()


def render_single_flow_tab_content(df, flow_trigger, available_months):
    """Render enkelt flow tab med filtre og indhold"""
    
    # Sorter måneder kronologisk (ældste først for slider)
    def month_to_tuple(m):
        parts = m.split('-')
        return (int(parts[0]), int(parts[1]))
    
    sorted_months = sorted(available_months, key=month_to_tuple)
    
    # Hent data for dette flow for at få filter options
    flow_data_all = df[df['Flow_Trigger'] == flow_trigger]
    all_countries = sorted(flow_data_all['Country'].unique()) if not flow_data_all.empty else []
    
    if not all_countries:
        st.warning(f"Ingen data for dette flow.")
        return

    # Hent alle mail+message kombinationer for dette flow
    def mail_combo_sort_key(combo):
        mail, msg = combo
        match = re.search(r'Mail\s*(\d+)', str(mail))
        mail_num = int(match.group(1)) if match else 999
        msg_str = str(msg) if pd.notna(msg) else ''
        return (mail_num, msg_str)
    
    # Opret unikke Mail+Message kombinationer
    mail_combos = flow_data_all.groupby(['Mail', 'Message'], as_index=False).first()[['Mail', 'Message']]
    all_mails_full = sorted([tuple(x) for x in mail_combos.values], key=mail_combo_sort_key)
    
    # Find aktive mails (sendt denne eller sidste måned)
    current_month = get_current_year_month()
    today = datetime.date.today()
    if today.month == 1:
        prev_month = f"{today.year - 1}-12"
    else:
        prev_month = f"{today.year}-{today.month - 1}"
    
    recent_months = [current_month, prev_month]
    recent_data = flow_data_all[
        (flow_data_all['Year_Month'].isin(recent_months)) & 
        (flow_data_all['Received_Email'] > 0)
    ]
    active_mail_combos = recent_data.groupby(['Mail', 'Message'], as_index=False).first()[['Mail', 'Message']]
    active_mails = set([tuple(x) for x in active_mail_combos.values])
    
    # Initialize ignore inactive state (per flow, default True)
    ignore_inactive_key = f'fl_ignore_inactive_{flow_trigger}'
    if ignore_inactive_key not in st.session_state:
        st.session_state[ignore_inactive_key] = True
    
    # Filtrer mails baseret på ignore_inactive
    all_mails = [m for m in all_mails_full if m in active_mails] if st.session_state[ignore_inactive_key] else all_mails_full
    
    # Funktion til at formatere mail+message til visning
    def format_mail_label(combo):
        mail, msg = combo
        if pd.notna(msg) and msg != '':
            return f"{mail} - {msg}"
        return str(mail)

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

    # Layout - dropdowns, checkbox og slider på samme linje
    col_land, col_mail, col_inaktive, col_slider = st.columns([1, 1, 0.8, 3.2])

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
            for mail_combo in all_mails:
                checked = mail_combo in st.session_state[mail_state_key]
                label = format_mail_label(mail_combo)
                # Brug hash af combo til unik key
                combo_key = f"{mail_combo[0]}_{mail_combo[1] if pd.notna(mail_combo[1]) else 'none'}"
                if st.checkbox(label, value=checked, key=f"fl_cb_mail_{flow_trigger}_{combo_key}_{reset_mail}"):
                    new_selected_mails.append(mail_combo)
            
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

    # Ignorer Inaktive checkbox
    with col_inaktive:
        ignore_inactive = st.checkbox(
            "Ignorer Inaktive", 
            value=st.session_state[ignore_inactive_key], 
            key=f"fl_ignore_inactive_cb_{flow_trigger}"
        )
        if ignore_inactive != st.session_state[ignore_inactive_key]:
            st.session_state[ignore_inactive_key] = ignore_inactive
            # Reset mail selection når toggle ændres
            new_all_mails = [m for m in all_mails_full if m in active_mails] if ignore_inactive else all_mails_full
            st.session_state[mail_state_key] = list(new_all_mails)
            st.session_state[mail_reset_key] += 1
            st.rerun()

    # Periode slider (på samme linje som dropdowns)
    with col_slider:
        if len(sorted_months) > 1:
            # Beregn default værdier
            default_end = sorted_months[-1]
            default_start_idx = max(0, len(sorted_months) - 3)
            default_start = sorted_months[default_start_idx]
            
            # Brug separat saved state (ikke widget key) for at overleve st.rerun()
            saved_key = f"fl_month_range_{flow_trigger}_saved"
            if saved_key not in st.session_state:
                st.session_state[saved_key] = (default_start, default_end)
            
            # Valider at gemte værdier stadig er i sorted_months
            saved_range = st.session_state[saved_key]
            if saved_range[0] in sorted_months and saved_range[1] in sorted_months:
                initial_value = saved_range
            else:
                initial_value = (default_start, default_end)
                st.session_state[saved_key] = initial_value
            
            month_range = st.select_slider(
                "Periode",
                options=sorted_months,
                value=initial_value,
                format_func=format_month_short,
                key=f"fl_month_range_{flow_trigger}",
                label_visibility="collapsed"
            )
            
            # Gem værdien i separat state efter rendering
            st.session_state[saved_key] = month_range
            
            sel_months = get_months_in_range(month_range[0], month_range[1], sorted_months)
        else:
            # Kun én måned tilgængelig
            sel_months = sorted_months
    
    if not sel_months:
        st.warning("Vælg mindst én måned.")
        return

    # Filtrer data efter valgte måneder
    df_month_filtered = df[df['Year_Month'].isin(sel_months)]

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
    render_single_flow_content(df_month_filtered, flow_trigger, sel_countries, sel_mails, df)

    if st.button('Opdater Data', key=f"fl_refresh_{flow_trigger}"):
        st.rerun()
