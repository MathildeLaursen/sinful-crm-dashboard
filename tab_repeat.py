"""
Repeat Tab - CRM Dashboard
Med sub-tabs for hver repeat og group
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


@st.cache_data(ttl=300, show_spinner=False)
def load_repeat_data():
    """Henter Repeat data fra Google Sheet"""
    try:
        gc = get_gspread_client()
        
        if "flows_spreadsheet" not in st.secrets["connections"]["gsheets"]:
            st.error("Mangler 'flows_spreadsheet' i secrets. Tilføj: flows_spreadsheet = 'URL'")
            return pd.DataFrame()
        
        flows_url = st.secrets["connections"]["gsheets"]["flows_spreadsheet"]
        spreadsheet = gc.open_by_url(flows_url)
        
        worksheet = spreadsheet.worksheet("All_Repeat")
        all_values = worksheet.get_all_values()
        
        if len(all_values) > 2:
            data = all_values[2:]
            raw_df = pd.DataFrame(data)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Fejl ved indlæsning fra Google Sheets: {type(e).__name__}: {e}")
        st.info(f"URL brugt: {st.secrets['connections']['gsheets'].get('flows_spreadsheet', 'IKKE SAT')}")
        return pd.DataFrame()
    
    # Repeat har en kolonne mindre end Flow (ingen Trigger)
    # Flow: A-H = Send Date, Tags, Flow, Trigger, Group, Mail, Message, A/B
    # Repeat: A-G = Send Date, Tags, Repeat, Group, Mail, Message, A/B
    # Derfor er alle country kolonner forskudt med -1
    country_configs = [
        ('DK', col_letter_to_index('O')),   # 14 (var P=15)
        ('SE', col_letter_to_index('V')),   # 21 (var W=22)
        ('NO', col_letter_to_index('AC')),  # 28 (var AD=29)
        ('FI', col_letter_to_index('AJ')),  # 35 (var AK=36)
        ('FR', col_letter_to_index('AQ')),  # 42 (var AR=43)
        ('UK', col_letter_to_index('AX')),  # 49 (var AY=50)
        ('DE', col_letter_to_index('BE')),  # 56 (var BF=57)
        ('AT', col_letter_to_index('BL')),  # 63 (var BM=64)
        ('NL', col_letter_to_index('BS')),  # 70 (var BT=71)
        ('BE', col_letter_to_index('BZ')),  # 77 (var CA=78)
        ('CH', col_letter_to_index('CG')),  # 84 (var CH=85)
    ]
    
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
            
            # Info kolonner (A-G, index 0-6) - Repeat har ingen Trigger kolonne
            country_df['Send_Date'] = raw_df.iloc[:, 0]      # A: Send Date
            country_df['Tags'] = raw_df.iloc[:, 1]           # B: Tags
            country_df['Repeat'] = raw_df.iloc[:, 2]         # C: Repeat
            country_df['Group'] = raw_df.iloc[:, 3]          # D: Group (var E)
            country_df['Mail'] = raw_df.iloc[:, 4]           # E: Mail (var F)
            country_df['Message'] = raw_df.iloc[:, 5]        # F: Message (var G)
            country_df['AB'] = raw_df.iloc[:, 6]             # G: A/B (var H)
            
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
    
    df['Year_Month'] = df['Send_Date'].astype(str).str.strip()
    df = df[df['Year_Month'].str.match(r'^\d{4}-\d{1,2}$', na=False)]
    
    numeric_cols = ['Received_Email', 'Total_Opens', 'Unique_Opens', 'Total_Clicks', 'Unique_Clicks', 'Unsubscribed', 'Bounced']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    df['Open_Rate'] = df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    df['Click_Rate'] = df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    df['CTR'] = df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    return df


def get_available_months(df):
    """Returner liste af tilgængelige måneder sorteret faldende"""
    months = df['Year_Month'].unique()
    def month_sort_key(m):
        try:
            parts = m.split('-')
            return (int(parts[0]), int(parts[1]))
        except:
            return (0, 0)
    return sorted(months, key=month_sort_key, reverse=True)


def aggregate_to_repeat_level(df):
    """Aggreger data til repeat niveau"""
    agg_df = df.groupby(['Year_Month', 'Repeat', 'Country'], as_index=False).agg({
        'Received_Email': 'sum',
        'Total_Opens': 'sum',
        'Unique_Opens': 'sum',
        'Total_Clicks': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })
    
    agg_df['Open_Rate'] = agg_df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    agg_df['Click_Rate'] = agg_df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    agg_df['CTR'] = agg_df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    return agg_df


def aggregate_to_group_level(df):
    """Aggreger data til group niveau inden for en repeat"""
    agg_df = df.groupby(['Year_Month', 'Repeat', 'Group', 'Country'], as_index=False).agg({
        'Received_Email': 'sum',
        'Total_Opens': 'sum',
        'Unique_Opens': 'sum',
        'Total_Clicks': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })
    
    agg_df['Open_Rate'] = agg_df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    agg_df['Click_Rate'] = agg_df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    agg_df['CTR'] = agg_df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    return agg_df


def get_unique_repeats(df):
    """Hent unikke repeats sorteret"""
    def repeat_sort_key(r):
        match = re.search(r'Repeat\s*(\d+)', str(r))
        return int(match.group(1)) if match else 999
    
    repeats = df['Repeat'].unique()
    repeats = [r for r in repeats if pd.notna(r) and str(r).strip() != '']
    return sorted(repeats, key=repeat_sort_key)


def get_unique_groups_for_repeat(df, repeat_name):
    """Hent unikke groups for en specifik repeat"""
    repeat_data = df[df['Repeat'] == repeat_name]
    groups = repeat_data['Group'].unique()
    groups = [g for g in groups if pd.notna(g) and str(g).strip() != '']
    return sorted(groups)


def get_previous_month(year_month):
    """Returnerer forrige måned i format YYYY-M"""
    parts = year_month.split('-')
    year, month = int(parts[0]), int(parts[1])
    if month == 1:
        return f"{year - 1}-12"
    else:
        return f"{year}-{month - 1}"


def calculate_month_progress():
    """Beregn hvor langt vi er i den nuværende måned"""
    import calendar
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    days_with_data = yesterday.day
    total_days_in_month = calendar.monthrange(today.year, today.month)[1]
    return days_with_data / total_days_in_month


def get_current_year_month():
    """Returnerer nuværende måned i format YYYY-M"""
    today = datetime.date.today()
    return f"{today.year}-{today.month}"


def get_months_in_range(start_month, end_month, all_months):
    """Returnerer alle måneder mellem start og slut"""
    def month_to_tuple(m):
        parts = m.split('-')
        return (int(parts[0]), int(parts[1]))
    
    start_tuple = month_to_tuple(start_month)
    end_tuple = month_to_tuple(end_month)
    
    if start_tuple > end_tuple:
        start_tuple, end_tuple = end_tuple, start_tuple
    
    result = []
    for m in all_months:
        m_tuple = month_to_tuple(m)
        if start_tuple <= m_tuple <= end_tuple:
            result.append(m)
    
    return sorted(result, key=month_to_tuple)


def format_month_short(year_month):
    """Formater måned til kort dansk format"""
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Maj', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Dec'
    }
    parts = year_month.split('-')
    year = parts[0][2:]
    month = int(parts[1])
    return f"{month_names[month]} {year}"


def get_short_repeat_name(repeat_name):
    """Udtræk kun 'Repeat X' fra repeat navn"""
    match = re.search(r'(Repeat\s*\d+)', str(repeat_name))
    return match.group(1) if match else str(repeat_name)


UNICORN_COLORS = [
    '#9B7EBD', '#E8B4CB', '#A8E6CF', '#7EC8E3', '#F7DC6F',
    '#BB8FCE', '#F1948A', '#85C1E9', '#82E0AA', '#D7BDE2',
    '#F5B7B1', '#FAD7A0', '#AED6F1', '#D5F5E3', '#FADBD8',
]


def render_overview_content(repeat_df, sel_countries, sel_repeats, full_df=None, all_months_df=None, ignore_inactive=True, active_repeats=None):
    """Render oversigt (alle repeats aggregeret)"""
    current_df = repeat_df[
        (repeat_df['Country'].isin(sel_countries)) &
        (repeat_df['Repeat'].isin(sel_repeats))
    ].copy()

    if current_df.empty:
        st.warning("Ingen data matcher de valgte filtre.")
        return

    display_df = current_df.groupby(['Year_Month', 'Repeat'], as_index=False).agg({
        'Received_Email': 'sum',
        'Total_Opens': 'sum',
        'Unique_Opens': 'sum',
        'Total_Clicks': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })
    
    display_df['Open_Rate'] = display_df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    display_df['Click_Rate'] = display_df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    display_df['CTR'] = display_df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)

    total_received = display_df['Received_Email'].sum()
    total_opens = display_df['Unique_Opens'].sum()
    total_clicks = display_df['Unique_Clicks'].sum()
    
    open_rate = (total_opens / total_received * 100) if total_received > 0 else 0
    click_rate = (total_clicks / total_received * 100) if total_received > 0 else 0
    ctr = (total_clicks / total_opens * 100) if total_opens > 0 else 0

    prev_received = prev_opens = prev_clicks = prev_or = prev_cr = prev_ctr = None
    
    if full_df is not None:
        selected_months = display_df['Year_Month'].unique().tolist()
        current_month = get_current_year_month()
        
        def month_to_tuple(m):
            parts = m.split('-')
            return (int(parts[0]), int(parts[1]))
        
        sorted_months = sorted(selected_months, key=month_to_tuple)
        oldest_selected = sorted_months[0]
        num_months = len(selected_months)
        
        prev_months = []
        temp_month = oldest_selected
        for _ in range(num_months):
            temp_month = get_previous_month(temp_month)
            prev_months.append(temp_month)
        
        oldest_prev_month = min(prev_months, key=month_to_tuple)
        
        prev_df = full_df[
            (full_df['Year_Month'].isin(prev_months)) &
            (full_df['Country'].isin(sel_countries)) &
            (full_df['Repeat'].isin(sel_repeats))
        ].copy()
        
        if not prev_df.empty:
            prev_agg = prev_df.groupby(['Year_Month'], as_index=False).agg({
                'Received_Email': 'sum',
                'Unique_Opens': 'sum',
                'Unique_Clicks': 'sum',
            })
            
            prev_received = prev_opens = prev_clicks = 0
            
            current_month_selected = current_month in selected_months
            month_progress = calculate_month_progress() if current_month_selected else 1.0
            
            for _, row in prev_agg.iterrows():
                month = row['Year_Month']
                if month == oldest_prev_month and current_month_selected:
                    prev_received += row['Received_Email'] * month_progress
                    prev_opens += row['Unique_Opens'] * month_progress
                    prev_clicks += row['Unique_Clicks'] * month_progress
                else:
                    prev_received += row['Received_Email']
                    prev_opens += row['Unique_Opens']
                    prev_clicks += row['Unique_Clicks']
            
            if prev_received > 0:
                prev_or = (prev_opens / prev_received * 100)
                prev_cr = (prev_clicks / prev_received * 100)
            if prev_opens > 0:
                prev_ctr = (prev_clicks / prev_opens * 100)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    show_metric(col1, "Emails Sendt", total_received, prev_received)
    show_metric(col2, "Unikke Opens", total_opens, prev_opens)
    show_metric(col3, "Unikke Clicks", total_clicks, prev_clicks)
    show_metric(col4, "Open Rate", open_rate, prev_or, is_percent=True)
    show_metric(col5, "Click Rate", click_rate, prev_cr, is_percent=True)
    show_metric(col6, "Click Through Rate", ctr, prev_ctr, is_percent=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    chart_source_df = all_months_df if all_months_df is not None else display_df
    chart_source_df = chart_source_df[
        (chart_source_df['Country'].isin(sel_countries)) &
        (chart_source_df['Repeat'].isin(sel_repeats))
    ].copy()
    
    if ignore_inactive and active_repeats is not None:
        chart_source_df = chart_source_df[chart_source_df['Repeat'].isin(active_repeats)]
    
    chart_df = chart_source_df.groupby(['Year_Month', 'Repeat'], as_index=False).agg({
        'Received_Email': 'sum',
    })
    
    chart_df['Month_Label'] = chart_df['Year_Month'].apply(format_month_short)
    
    def month_sort_key(ym):
        parts = ym.split('-')
        return int(parts[0]) * 100 + int(parts[1])
    
    chart_df['Sort_Key'] = chart_df['Year_Month'].apply(month_sort_key)
    chart_df = chart_df.sort_values('Sort_Key')
    
    months_sorted = chart_df.drop_duplicates('Year_Month').sort_values('Sort_Key')['Month_Label'].tolist()
    
    def chart_repeat_sort_key(r):
        match = re.search(r'Repeat\s*(\d+)', str(r))
        return int(match.group(1)) if match else 999
    unique_repeats = sorted(chart_df['Repeat'].unique(), key=chart_repeat_sort_key)

    fig = go.Figure()
    
    for idx, repeat in enumerate(unique_repeats):
        repeat_data = chart_df[chart_df['Repeat'] == repeat].sort_values('Sort_Key')
        short_name = get_short_repeat_name(repeat)
        color = UNICORN_COLORS[idx % len(UNICORN_COLORS)]
        
        fig.add_trace(
            go.Scatter(
                x=repeat_data['Month_Label'], 
                y=repeat_data['Received_Email'],
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
    
    def format_y_tick(val):
        if val >= 1000000:
            return f'{val/1000000:.0f}M'
        elif val >= 1000:
            return f'{val/1000:.0f}K'
        return f'{val:.0f}'
    
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
        tickfont=dict(size=10),
        automargin=True,
        ticklabelstandoff=10
    )
    fig.update_xaxes(
        gridcolor='rgba(212,191,255,0.2)', 
        tickfont=dict(size=11),
        categoryorder='array',
        categoryarray=months_sorted,
        automargin=True
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    table_df = display_df.groupby('Repeat', as_index=False).agg({
        'Received_Email': 'sum',
        'Unique_Opens': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })
    
    if ignore_inactive and active_repeats is not None:
        table_df = table_df[table_df['Repeat'].isin(active_repeats)]
    
    table_df['Open_Rate'] = table_df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    table_df['Click_Rate'] = table_df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    table_df['CTR'] = table_df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    table_df['_repeat_num'] = table_df['Repeat'].apply(lambda r: int(re.search(r'Repeat\s*(\d+)', str(r)).group(1)) if re.search(r'Repeat\s*(\d+)', str(r)) else 999)
    table_df = table_df.sort_values('_repeat_num', ascending=True)
    table_df = table_df.drop(columns=['_repeat_num'])
    
    table_df = table_df[['Repeat', 'Received_Email', 'Unique_Opens', 'Unique_Clicks', 'Open_Rate', 'Click_Rate', 'CTR', 'Unsubscribed', 'Bounced']]
    
    table_height = (len(table_df) + 1) * 35 + 3
    
    st.dataframe(
        table_df, use_container_width=True, hide_index=True, height=table_height,
        column_config={
            "Repeat": st.column_config.TextColumn("Repeat", width="large"),
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

    # === GRAFER PER REPEAT (UNDER TABELLEN) ===
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Hent alle måneder til grafen
    all_months_for_chart = all_months_df['Year_Month'].unique() if all_months_df is not None else display_df['Year_Month'].unique()
    
    def month_to_sortkey(m):
        parts = m.split('-')
        return int(parts[0]) * 100 + int(parts[1])
    
    sorted_chart_months = sorted(all_months_for_chart, key=month_to_sortkey)
    
    # Hent data til grafen
    chart_base_df = all_months_df if all_months_df is not None else repeat_df
    chart_base_df = chart_base_df[
        (chart_base_df['Country'].isin(sel_countries)) &
        (chart_base_df['Repeat'].isin(sel_repeats))
    ].copy()
    
    if ignore_inactive and active_repeats is not None:
        chart_base_df = chart_base_df[chart_base_df['Repeat'].isin(active_repeats)]
    
    # Aggreger per repeat og måned
    chart_agg_df = chart_base_df.groupby(['Year_Month', 'Repeat'], as_index=False).agg({
        'Received_Email': 'sum',
        'Unique_Opens': 'sum',
        'Unique_Clicks': 'sum',
    })
    
    # Formater måneder til visning
    def format_month_label(m):
        parts = m.split('-')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
        return f"{month_names[int(parts[1])-1]} {parts[0][2:]}"
    
    # Hent unikke repeats sorteret efter nummer
    def repeat_sort_key(r):
        match = re.search(r'Repeat\s*(\d+)', str(r))
        return int(match.group(1)) if match else 999
    
    repeats_for_charts = sorted(chart_agg_df['Repeat'].unique(), key=repeat_sort_key)
    
    if len(sorted_chart_months) > 0 and len(repeats_for_charts) > 0:
        num_repeats = len(repeats_for_charts)
        
        height_per_chart = 150
        gap_pixels = 70
        chart_height = num_repeats * height_per_chart + (num_repeats - 1) * gap_pixels
        
        v_spacing = gap_pixels / chart_height if num_repeats > 1 else 0.1
        max_allowed = 1.0 / (num_repeats - 1) if num_repeats > 1 else 0.5
        v_spacing = min(v_spacing, max_allowed * 0.95)
        
        fig = make_subplots(
            rows=num_repeats, cols=1,
            shared_xaxes=False,
            vertical_spacing=v_spacing,
            subplot_titles=[get_short_repeat_name(r) for r in repeats_for_charts],
            specs=[[{"secondary_y": True}] for _ in range(num_repeats)]
        )
        
        for i, repeat in enumerate(repeats_for_charts):
            row = i + 1
            repeat_chart_data = chart_agg_df[chart_agg_df['Repeat'] == repeat]
            
            sent_values = []
            opens_values = []
            clicks_values = []
            for month in sorted_chart_months:
                month_data = repeat_chart_data[repeat_chart_data['Year_Month'] == month]
                if not month_data.empty:
                    sent_values.append(month_data['Received_Email'].sum())
                    opens_values.append(month_data['Unique_Opens'].sum())
                    clicks_values.append(month_data['Unique_Clicks'].sum())
                else:
                    sent_values.append(0)
                    opens_values.append(0)
                    clicks_values.append(0)
            
            x_labels = [format_month_label(m) for m in sorted_chart_months]
            
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
            
            # Smart skalering for clicks y-akse
            current_month = get_current_year_month()
            current_month_label = format_month_short(current_month)
            
            opens_for_calc = [v for v, lbl in zip(opens_values, x_labels) if lbl != current_month_label]
            sent_for_calc = [v for v, lbl in zip(sent_values, x_labels) if lbl != current_month_label]
            clicks_for_calc = [v for v, lbl in zip(clicks_values, x_labels) if lbl != current_month_label]
            
            max_clicks = max(clicks_for_calc) if clicks_for_calc else (max(clicks_values) if clicks_values else 0)
            max_left = max(
                max(sent_for_calc) if sent_for_calc else 0, 
                max(opens_for_calc) if opens_for_calc else 0
            )
            min_opens = min(opens_for_calc) if opens_for_calc else 0
            
            if max_left == 0:
                max_left = max(max(sent_values) if sent_values else 0, max(opens_values) if opens_values else 0)
            if min_opens == 0:
                min_opens = min(opens_values) if opens_values else 0
            
            if max_clicks > 0 and min_opens > 0 and max_left > 0:
                target_visual_height = min_opens * 0.9
                clicks_range = max_clicks * max_left / target_visual_height
                fig.update_yaxes(range=[0, clicks_range], row=row, col=1, secondary_y=True)
            elif max_clicks > 0:
                fig.update_yaxes(range=[0, max_clicks * 2], row=row, col=1, secondary_y=True)
        
        chart_height = max(350, chart_height)
        top_margin = 80
        
        fig.update_layout(
            showlegend=True,
            height=chart_height + top_margin,
            margin=dict(l=60, r=60, t=top_margin, b=40),
            legend=dict(
                orientation="h", 
                yanchor="bottom", y=1.02, 
                xanchor="right", x=1,
                bgcolor='rgba(0,0,0,0)'
            ),
            plot_bgcolor='rgba(250,245,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=13, color='#7B5EA5', family='sans-serif')
            annotation['xanchor'] = 'left'
            annotation['x'] = 0
            annotation['yanchor'] = 'bottom'
            annotation['yshift'] = 10
        
        for i in range(num_repeats):
            fig.update_yaxes(
                title_text="Sendt / Opens",
                title_font=dict(size=10, color='#4A3F55'),
                gridcolor='rgba(212,191,255,0.3)',
                tickformat=',d',
                rangemode='tozero',
                automargin=True,
                ticklabelstandoff=5,
                row=i+1, col=1, secondary_y=False
            )
            fig.update_yaxes(
                title_text="Clicks",
                title_font=dict(size=10, color='#4A3F55'),
                gridcolor='rgba(232,180,203,0.2)',
                showgrid=False,
                tickformat=',d',
                rangemode='tozero',
                automargin=True,
                ticklabelstandoff=5,
                row=i+1, col=1, secondary_y=True
            )
        
        fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', type='category', tickfont=dict(size=10), automargin=True)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_group_overview_content(df, repeat_name, sel_countries, sel_groups, full_df=None, all_months_df=None, ignore_inactive=True, active_groups=None):
    """Render oversigt over groups for en specifik repeat"""
    current_df = df[
        (df['Repeat'] == repeat_name) &
        (df['Country'].isin(sel_countries)) &
        (df['Group'].isin(sel_groups))
    ].copy()

    if current_df.empty:
        st.warning("Ingen data matcher de valgte filtre.")
        return

    display_df = current_df.groupby(['Year_Month', 'Group'], as_index=False).agg({
        'Received_Email': 'sum',
        'Total_Opens': 'sum',
        'Unique_Opens': 'sum',
        'Total_Clicks': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })
    
    display_df['Open_Rate'] = display_df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    display_df['Click_Rate'] = display_df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    display_df['CTR'] = display_df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)

    total_received = display_df['Received_Email'].sum()
    total_opens = display_df['Unique_Opens'].sum()
    total_clicks = display_df['Unique_Clicks'].sum()
    
    open_rate = (total_opens / total_received * 100) if total_received > 0 else 0
    click_rate = (total_clicks / total_received * 100) if total_received > 0 else 0
    ctr = (total_clicks / total_opens * 100) if total_opens > 0 else 0

    prev_received = prev_opens = prev_clicks = prev_or = prev_cr = prev_ctr = None
    
    if full_df is not None:
        selected_months = display_df['Year_Month'].unique().tolist()
        current_month = get_current_year_month()
        
        def month_to_tuple(m):
            parts = m.split('-')
            return (int(parts[0]), int(parts[1]))
        
        sorted_months = sorted(selected_months, key=month_to_tuple)
        oldest_selected = sorted_months[0] if sorted_months else current_month
        num_months = len(selected_months) if selected_months else 1
        
        prev_months = []
        temp_month = oldest_selected
        for _ in range(num_months):
            temp_month = get_previous_month(temp_month)
            prev_months.append(temp_month)
        
        oldest_prev_month = min(prev_months, key=month_to_tuple) if prev_months else oldest_selected
        
        prev_df = full_df[
            (full_df['Year_Month'].isin(prev_months)) &
            (full_df['Repeat'] == repeat_name) &
            (full_df['Country'].isin(sel_countries)) &
            (full_df['Group'].isin(sel_groups))
        ].copy()
        
        if not prev_df.empty:
            prev_agg = prev_df.groupby(['Year_Month'], as_index=False).agg({
                'Received_Email': 'sum',
                'Unique_Opens': 'sum',
                'Unique_Clicks': 'sum',
            })
            
            prev_received = prev_opens = prev_clicks = 0
            current_month_selected = current_month in selected_months
            month_progress = calculate_month_progress() if current_month_selected else 1.0
            
            for _, row in prev_agg.iterrows():
                month = row['Year_Month']
                if month == oldest_prev_month and current_month_selected:
                    prev_received += row['Received_Email'] * month_progress
                    prev_opens += row['Unique_Opens'] * month_progress
                    prev_clicks += row['Unique_Clicks'] * month_progress
                else:
                    prev_received += row['Received_Email']
                    prev_opens += row['Unique_Opens']
                    prev_clicks += row['Unique_Clicks']
            
            if prev_received > 0:
                prev_or = (prev_opens / prev_received * 100)
                prev_cr = (prev_clicks / prev_received * 100)
            if prev_opens > 0:
                prev_ctr = (prev_clicks / prev_opens * 100)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    show_metric(col1, "Emails Sendt", total_received, prev_received)
    show_metric(col2, "Unikke Opens", total_opens, prev_opens)
    show_metric(col3, "Unikke Clicks", total_clicks, prev_clicks)
    show_metric(col4, "Open Rate", open_rate, prev_or, is_percent=True)
    show_metric(col5, "Click Rate", click_rate, prev_cr, is_percent=True)
    show_metric(col6, "Click Through Rate", ctr, prev_ctr, is_percent=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    chart_source_df = all_months_df if all_months_df is not None else df
    chart_source_df = chart_source_df[
        (chart_source_df['Repeat'] == repeat_name) &
        (chart_source_df['Country'].isin(sel_countries)) &
        (chart_source_df['Group'].isin(sel_groups))
    ].copy()
    
    if ignore_inactive and active_groups is not None:
        chart_source_df = chart_source_df[chart_source_df['Group'].isin(active_groups)]
    
    chart_df = chart_source_df.groupby(['Year_Month', 'Group'], as_index=False).agg({
        'Received_Email': 'sum',
    })
    
    chart_df['Month_Label'] = chart_df['Year_Month'].apply(format_month_short)
    
    def month_sort_key(ym):
        parts = ym.split('-')
        return int(parts[0]) * 100 + int(parts[1])
    
    chart_df['Sort_Key'] = chart_df['Year_Month'].apply(month_sort_key)
    chart_df = chart_df.sort_values('Sort_Key')
    
    months_sorted = chart_df.drop_duplicates('Year_Month').sort_values('Sort_Key')['Month_Label'].tolist()
    unique_groups = sorted(chart_df['Group'].unique())

    fig = go.Figure()
    
    for idx, group in enumerate(unique_groups):
        group_data = chart_df[chart_df['Group'] == group].sort_values('Sort_Key')
        color = UNICORN_COLORS[idx % len(UNICORN_COLORS)]
        
        fig.add_trace(
            go.Scatter(
                x=group_data['Month_Label'], 
                y=group_data['Received_Email'],
                name=str(group),
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f'{group}<br>%{{x}}: %{{y:,.0f}} sendt<extra></extra>'
            )
        )
    
    fig.update_layout(
        title="",
        showlegend=True, 
        height=520,
        margin=dict(l=80, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        plot_bgcolor='rgba(250,245,255,0.5)', 
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    def format_y_tick(val):
        if val >= 1000000:
            return f'{val/1000000:.0f}M'
        elif val >= 1000:
            return f'{val/1000:.0f}K'
        return f'{val:.0f}'
    
    max_val = chart_df['Received_Email'].max() if not chart_df.empty else 100000
    tick_step = 50000 if max_val > 100000 else 25000
    tick_vals = list(range(0, int(max_val * 1.2), tick_step))
    tick_text = [format_y_tick(v) for v in tick_vals]
    
    fig.update_yaxes(
        title_text="Antal sendt", title_font=dict(size=12, color='#7B5EA5'),
        gridcolor='rgba(212,191,255,0.3)', tickvals=tick_vals, ticktext=tick_text,
        automargin=True, ticklabelstandoff=10
    )
    fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', categoryorder='array', categoryarray=months_sorted, automargin=True)
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    table_df = display_df.groupby('Group', as_index=False).agg({
        'Received_Email': 'sum',
        'Unique_Opens': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })
    
    if ignore_inactive and active_groups is not None:
        table_df = table_df[table_df['Group'].isin(active_groups)]
    
    table_df['Open_Rate'] = table_df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    table_df['Click_Rate'] = table_df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    table_df['CTR'] = table_df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    table_df = table_df.sort_values('Group', ascending=True)
    table_df = table_df[['Group', 'Received_Email', 'Unique_Opens', 'Unique_Clicks', 'Open_Rate', 'Click_Rate', 'CTR', 'Unsubscribed', 'Bounced']]
    
    table_height = (len(table_df) + 1) * 35 + 3
    
    st.dataframe(
        table_df, use_container_width=True, hide_index=True, height=table_height,
        column_config={
            "Group": st.column_config.TextColumn("Group", width="large"),
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

    # === GRAFER PER GROUP (UNDER TABELLEN) ===
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Hent alle måneder til grafen
    all_months_for_chart = all_months_df['Year_Month'].unique() if all_months_df is not None else df['Year_Month'].unique()
    
    def month_to_sortkey(m):
        parts = m.split('-')
        return int(parts[0]) * 100 + int(parts[1])
    
    sorted_chart_months = sorted(all_months_for_chart, key=month_to_sortkey)
    
    # Hent data til grafen
    chart_base_df = all_months_df if all_months_df is not None else df
    chart_base_df = chart_base_df[
        (chart_base_df['Repeat'] == repeat_name) &
        (chart_base_df['Country'].isin(sel_countries)) &
        (chart_base_df['Group'].isin(sel_groups))
    ].copy()
    
    if ignore_inactive and active_groups is not None:
        chart_base_df = chart_base_df[chart_base_df['Group'].isin(active_groups)]
    
    # Aggreger per group og måned
    chart_agg_df = chart_base_df.groupby(['Year_Month', 'Group'], as_index=False).agg({
        'Received_Email': 'sum',
        'Unique_Opens': 'sum',
        'Unique_Clicks': 'sum',
    })
    
    # Formater måneder til visning
    def format_month_label(m):
        parts = m.split('-')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
        return f"{month_names[int(parts[1])-1]} {parts[0][2:]}"
    
    # Hent unikke groups sorteret
    groups_for_charts = sorted(chart_agg_df['Group'].unique())
    
    if len(sorted_chart_months) > 0 and len(groups_for_charts) > 0:
        num_groups = len(groups_for_charts)
        
        height_per_chart = 150
        gap_pixels = 70
        chart_height = num_groups * height_per_chart + (num_groups - 1) * gap_pixels
        
        v_spacing = gap_pixels / chart_height if num_groups > 1 else 0.1
        max_allowed = 1.0 / (num_groups - 1) if num_groups > 1 else 0.5
        v_spacing = min(v_spacing, max_allowed * 0.95)
        
        fig = make_subplots(
            rows=num_groups, cols=1,
            shared_xaxes=False,
            vertical_spacing=v_spacing,
            subplot_titles=[str(g) for g in groups_for_charts],
            specs=[[{"secondary_y": True}] for _ in range(num_groups)]
        )
        
        for i, group in enumerate(groups_for_charts):
            row = i + 1
            group_chart_data = chart_agg_df[chart_agg_df['Group'] == group]
            
            sent_values = []
            opens_values = []
            clicks_values = []
            for month in sorted_chart_months:
                month_data = group_chart_data[group_chart_data['Year_Month'] == month]
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
            
            # Linje for Opens (venstre y-akse)
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
            
            # Linje for Clicks (højre y-akse)
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
            
            # Smart skalering for clicks y-akse
            current_month = get_current_year_month()
            current_month_label = format_month_short(current_month)
            
            opens_for_calc = [v for v, lbl in zip(opens_values, x_labels) if lbl != current_month_label]
            sent_for_calc = [v for v, lbl in zip(sent_values, x_labels) if lbl != current_month_label]
            clicks_for_calc = [v for v, lbl in zip(clicks_values, x_labels) if lbl != current_month_label]
            
            max_clicks = max(clicks_for_calc) if clicks_for_calc else (max(clicks_values) if clicks_values else 0)
            max_left = max(
                max(sent_for_calc) if sent_for_calc else 0, 
                max(opens_for_calc) if opens_for_calc else 0
            )
            min_opens = min(opens_for_calc) if opens_for_calc else 0
            
            if max_left == 0:
                max_left = max(max(sent_values) if sent_values else 0, max(opens_values) if opens_values else 0)
            if min_opens == 0:
                min_opens = min(opens_values) if opens_values else 0
            
            if max_clicks > 0 and min_opens > 0 and max_left > 0:
                target_visual_height = min_opens * 0.9
                clicks_range = max_clicks * max_left / target_visual_height
                fig.update_yaxes(range=[0, clicks_range], row=row, col=1, secondary_y=True)
            elif max_clicks > 0:
                fig.update_yaxes(range=[0, max_clicks * 2], row=row, col=1, secondary_y=True)
        
        chart_height = max(350, chart_height)
        top_margin = 80
        
        fig.update_layout(
            showlegend=True,
            height=chart_height + top_margin,
            margin=dict(l=60, r=60, t=top_margin, b=40),
            legend=dict(
                orientation="h", 
                yanchor="bottom", y=1.02, 
                xanchor="right", x=1,
                bgcolor='rgba(0,0,0,0)'
            ),
            plot_bgcolor='rgba(250,245,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=13, color='#7B5EA5', family='sans-serif')
            annotation['xanchor'] = 'left'
            annotation['x'] = 0
            annotation['yanchor'] = 'bottom'
            annotation['yshift'] = 10
        
        for i in range(num_groups):
            fig.update_yaxes(
                title_text="Sendt / Opens",
                title_font=dict(size=10, color='#4A3F55'),
                gridcolor='rgba(212,191,255,0.3)',
                tickformat=',d',
                rangemode='tozero',
                automargin=True,
                ticklabelstandoff=5,
                row=i+1, col=1, secondary_y=False
            )
            fig.update_yaxes(
                title_text="Clicks",
                title_font=dict(size=10, color='#4A3F55'),
                gridcolor='rgba(232,180,203,0.2)',
                showgrid=False,
                tickformat=',d',
                rangemode='tozero',
                automargin=True,
                ticklabelstandoff=5,
                row=i+1, col=1, secondary_y=True
            )
        
        fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', type='category', tickfont=dict(size=10), automargin=True)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_group_mail_content(df, repeat_name, group_name, sel_countries, filter_config=None, full_df=None, all_months_df=None):
    """Render mail-niveau detaljer for en specifik group i en repeat"""
    BLANK_LABEL = "Ingen"
    
    group_data = df[
        (df['Repeat'] == repeat_name) &
        (df['Group'] == group_name) &
        (df['Country'].isin(sel_countries))
    ].copy()
    
    if group_data.empty:
        st.warning(f"Ingen data for {group_name} med de valgte filtre.")
        return
    
    # Anvend filtre
    if filter_config is not None:
        if filter_config.get('mails'):
            selected_mails = filter_config['mails']
            include_blank = BLANK_LABEL in selected_mails
            non_blank_mails = [m for m in selected_mails if m != BLANK_LABEL]
            if include_blank:
                group_data = group_data[group_data['Mail'].isin(non_blank_mails) | (group_data['Mail'].str.strip() == '') | group_data['Mail'].isna()]
            else:
                group_data = group_data[group_data['Mail'].isin(non_blank_mails)]
        if filter_config.get('messages'):
            selected_messages = filter_config['messages']
            include_blank = BLANK_LABEL in selected_messages
            non_blank_messages = [m for m in selected_messages if m != BLANK_LABEL]
            if include_blank:
                group_data = group_data[group_data['Message'].isin(non_blank_messages) | (group_data['Message'].str.strip() == '') | group_data['Message'].isna()]
            else:
                group_data = group_data[group_data['Message'].isin(non_blank_messages)]
        if filter_config.get('ab'):
            selected_ab = filter_config['ab']
            include_blank = BLANK_LABEL in selected_ab
            non_blank_ab = [a for a in selected_ab if a != BLANK_LABEL]
            if include_blank:
                group_data = group_data[group_data['AB'].isin(non_blank_ab) | (group_data['AB'].str.strip() == '') | group_data['AB'].isna()]
            else:
                group_data = group_data[group_data['AB'].isin(non_blank_ab)]
    
    if group_data.empty:
        st.warning(f"Ingen data for {group_name} med de valgte filtre.")
        return
    
    scorecard_df = group_data.groupby(['Year_Month'], as_index=False).agg({
        'Received_Email': 'sum',
        'Unique_Opens': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })

    total_received = scorecard_df['Received_Email'].sum()
    total_opens = scorecard_df['Unique_Opens'].sum()
    total_clicks = scorecard_df['Unique_Clicks'].sum()
    
    open_rate = (total_opens / total_received * 100) if total_received > 0 else 0
    click_rate = (total_clicks / total_received * 100) if total_received > 0 else 0
    ctr = (total_clicks / total_opens * 100) if total_opens > 0 else 0

    prev_received = prev_opens = prev_clicks = prev_or = prev_cr = prev_ctr = None
    
    if full_df is not None:
        selected_months = scorecard_df['Year_Month'].unique().tolist()
        current_month = get_current_year_month()
        
        def month_to_tuple(m):
            parts = m.split('-')
            return (int(parts[0]), int(parts[1]))
        
        sorted_months = sorted(selected_months, key=month_to_tuple)
        oldest_selected = sorted_months[0] if sorted_months else current_month
        num_months = len(selected_months) if selected_months else 1
        
        prev_months = []
        temp_month = oldest_selected
        for _ in range(num_months):
            temp_month = get_previous_month(temp_month)
            prev_months.append(temp_month)
        
        oldest_prev_month = min(prev_months, key=month_to_tuple) if prev_months else oldest_selected
        
        prev_filter = (
            (full_df['Year_Month'].isin(prev_months)) &
            (full_df['Repeat'] == repeat_name) &
            (full_df['Group'] == group_name) &
            (full_df['Country'].isin(sel_countries))
        )
        
        prev_data = full_df[prev_filter].copy()
        
        if not prev_data.empty:
            prev_agg = prev_data.groupby(['Year_Month'], as_index=False).agg({
                'Received_Email': 'sum',
                'Unique_Opens': 'sum',
                'Unique_Clicks': 'sum',
            })
            
            prev_received = prev_opens = prev_clicks = 0
            current_month_selected = current_month in selected_months
            month_progress = calculate_month_progress() if current_month_selected else 1.0
            
            for _, row in prev_agg.iterrows():
                month = row['Year_Month']
                if month == oldest_prev_month and current_month_selected:
                    prev_received += row['Received_Email'] * month_progress
                    prev_opens += row['Unique_Opens'] * month_progress
                    prev_clicks += row['Unique_Clicks'] * month_progress
                else:
                    prev_received += row['Received_Email']
                    prev_opens += row['Unique_Opens']
                    prev_clicks += row['Unique_Clicks']
            
            if prev_received > 0:
                prev_or = (prev_opens / prev_received * 100)
                prev_cr = (prev_clicks / prev_received * 100)
            if prev_opens > 0:
                prev_ctr = (prev_clicks / prev_opens * 100)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    show_metric(col1, "Emails Sendt", total_received, prev_received)
    show_metric(col2, "Unikke Opens", total_opens, prev_opens)
    show_metric(col3, "Unikke Clicks", total_clicks, prev_clicks)
    show_metric(col4, "Open Rate", open_rate, prev_or, is_percent=True)
    show_metric(col5, "Click Rate", click_rate, prev_cr, is_percent=True)
    show_metric(col6, "Click Through Rate", ctr, prev_ctr, is_percent=True)

    ignore_mail = filter_config.get('ignore_mail', False) if filter_config else False
    ignore_message = filter_config.get('ignore_message', False) if filter_config else False
    ignore_ab = filter_config.get('ignore_ab', False) if filter_config else False

    # === SKJUL INAKTIVE CHECKBOX (mellem scorecards og tabel) ===
    if filter_config is not None and '_ignore_inactive_key' in filter_config:
        ignore_inactive_key = filter_config['_ignore_inactive_key']
        st.markdown('''
            <style>
            [class*="st-key-rp_ignore_inactive_cb_gm_"] {
                margin-bottom: -14.5px !important;
            }
            </style>
        ''', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.85em; margin-top: 5px;">', unsafe_allow_html=True)
        ignore_inactive_new = st.checkbox(
            "Skjul inaktive mails fra tabel og grafer", 
            value=st.session_state[ignore_inactive_key], 
            key=f"rp_ignore_inactive_cb_gm_{repeat_name}_{group_name}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if ignore_inactive_new != st.session_state[ignore_inactive_key]:
            st.session_state[ignore_inactive_key] = ignore_inactive_new
            st.rerun()

    # === FORBERED DATA TIL TABEL OG GRAFER ===
    table_data = group_data.copy()
    
    # Filtrer inaktive kombinationer hvis ignore_inactive er True
    if filter_config and filter_config.get('ignore_inactive'):
        current_month = get_current_year_month()
        today = datetime.date.today()
        if today.month == 1:
            prev_month = f"{today.year - 1}-12"
        else:
            prev_month = f"{today.year}-{today.month - 1}"
        recent_months = [current_month, prev_month]
        
        # Find kombinationer der har sendt i recent_months
        recent_data = table_data[
            (table_data['Year_Month'].isin(recent_months)) & 
            (table_data['Received_Email'] > 0)
        ]
        
        for col in ['Mail', 'Message', 'AB']:
            if col not in table_data.columns:
                table_data[col] = ''
            if col not in recent_data.columns:
                recent_data[col] = ''
        
        if not recent_data.empty:
            active_combos = recent_data.groupby(['Mail', 'Message', 'AB'], as_index=False).first()[['Mail', 'Message', 'AB']]
            table_data = table_data.merge(active_combos, on=['Mail', 'Message', 'AB'], how='inner')
    
    table_group_cols = []
    if not ignore_mail:
        table_group_cols.append('Mail')
    if not ignore_message:
        table_group_cols.append('Message')
    if not ignore_ab:
        table_group_cols.append('AB')
    
    if not table_group_cols:
        table_data['_dummy'] = 'Total'
        table_group_cols = ['_dummy']
    
    table_df = table_data.groupby(table_group_cols, as_index=False).agg({
        'Received_Email': 'sum',
        'Unique_Opens': 'sum',
        'Unique_Clicks': 'sum',
        'Unsubscribed': 'sum',
        'Bounced': 'sum',
    })
    
    if '_dummy' in table_df.columns:
        table_df = table_df.drop(columns=['_dummy'])
    
    if ignore_mail:
        table_df['Mail'] = ''
    if ignore_message:
        table_df['Message'] = ''
    if ignore_ab:
        table_df['AB'] = ''
    
    table_df['Open_Rate'] = table_df.apply(lambda x: (x['Unique_Opens'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    table_df['Click_Rate'] = table_df.apply(lambda x: (x['Unique_Clicks'] / x['Received_Email'] * 100) if x['Received_Email'] > 0 else 0, axis=1)
    table_df['CTR'] = table_df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    if 'Mail' in table_df.columns:
        table_df['_mail_num'] = table_df['Mail'].apply(lambda m: int(re.search(r'Mail\s*(\d+)', str(m)).group(1)) if re.search(r'Mail\s*(\d+)', str(m)) else 999)
    else:
        table_df['_mail_num'] = 999
    table_df['_msg'] = table_df['Message'].apply(lambda m: str(m) if pd.notna(m) else '') if 'Message' in table_df.columns else ''
    table_df['_ab'] = table_df['AB'].apply(lambda a: str(a) if pd.notna(a) else '') if 'AB' in table_df.columns else ''
    table_df = table_df.sort_values(['_mail_num', '_msg', '_ab'], ascending=True)
    table_df = table_df.drop(columns=['_mail_num', '_msg', '_ab'], errors='ignore')
    
    for col in ['Mail', 'Message', 'AB']:
        if col not in table_df.columns:
            table_df[col] = ''
    
    display_cols = ['Mail', 'Message', 'AB', 'Received_Email', 'Unique_Opens', 'Unique_Clicks', 'Open_Rate', 'Click_Rate', 'CTR', 'Unsubscribed', 'Bounced']
    table_df = table_df[display_cols]
    
    table_height = (len(table_df) + 1) * 35 + 3
    
    st.dataframe(
        table_df, use_container_width=True, hide_index=True, height=table_height,
        column_config={
            "Mail": st.column_config.TextColumn("Mail", width="small"),
            "Message": st.column_config.TextColumn("Message", width="medium"),
            "AB": st.column_config.TextColumn("A/B", width="small"),
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

    # === GRAFER PER MAIL (UNDER TABELLEN) ===
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Hent alle måneder til grafen
    all_chart_months = all_months_df['Year_Month'].unique() if all_months_df is not None else df['Year_Month'].unique()
    
    def month_to_sortkey(m):
        parts = m.split('-')
        return int(parts[0]) * 100 + int(parts[1])
    
    sorted_chart_months = sorted(all_chart_months, key=month_to_sortkey)
    
    # Hent data til grafen
    chart_base_df = all_months_df if all_months_df is not None else df
    chart_base_df = chart_base_df[
        (chart_base_df['Repeat'] == repeat_name) &
        (chart_base_df['Group'] == group_name) &
        (chart_base_df['Country'].isin(sel_countries))
    ].copy()
    
    # Anvend mail/message/ab filtre
    if filter_config is not None:
        if filter_config.get('mails'):
            selected_mails = filter_config['mails']
            include_blank = BLANK_LABEL in selected_mails
            non_blank_mails = [m for m in selected_mails if m != BLANK_LABEL]
            if include_blank:
                chart_base_df = chart_base_df[chart_base_df['Mail'].isin(non_blank_mails) | (chart_base_df['Mail'].str.strip() == '') | chart_base_df['Mail'].isna()]
            else:
                chart_base_df = chart_base_df[chart_base_df['Mail'].isin(non_blank_mails)]
        if filter_config.get('messages'):
            selected_messages = filter_config['messages']
            include_blank = BLANK_LABEL in selected_messages
            non_blank_messages = [m for m in selected_messages if m != BLANK_LABEL]
            if include_blank:
                chart_base_df = chart_base_df[chart_base_df['Message'].isin(non_blank_messages) | (chart_base_df['Message'].str.strip() == '') | chart_base_df['Message'].isna()]
            else:
                chart_base_df = chart_base_df[chart_base_df['Message'].isin(non_blank_messages)]
        if filter_config.get('ab'):
            selected_ab = filter_config['ab']
            include_blank = BLANK_LABEL in selected_ab
            non_blank_ab = [a for a in selected_ab if a != BLANK_LABEL]
            if include_blank:
                chart_base_df = chart_base_df[chart_base_df['AB'].isin(non_blank_ab) | (chart_base_df['AB'].str.strip() == '') | chart_base_df['AB'].isna()]
            else:
                chart_base_df = chart_base_df[chart_base_df['AB'].isin(non_blank_ab)]
        
        # Filtrer inaktive kombinationer hvis ignore_inactive er True
        if filter_config.get('ignore_inactive'):
            current_month = get_current_year_month()
            today = datetime.date.today()
            if today.month == 1:
                prev_month = f"{today.year - 1}-12"
            else:
                prev_month = f"{today.year}-{today.month - 1}"
            recent_months = [current_month, prev_month]
            
            recent_chart_data = chart_base_df[
                (chart_base_df['Year_Month'].isin(recent_months)) & 
                (chart_base_df['Received_Email'] > 0)
            ]
            
            for col in ['Mail', 'Message', 'AB']:
                if col not in chart_base_df.columns:
                    chart_base_df[col] = ''
                if col not in recent_chart_data.columns:
                    recent_chart_data[col] = ''
            
            if not recent_chart_data.empty:
                active_combos = recent_chart_data.groupby(['Mail', 'Message', 'AB'], as_index=False).first()[['Mail', 'Message', 'AB']]
                chart_base_df = chart_base_df.merge(active_combos, on=['Mail', 'Message', 'AB'], how='inner')
    
    # Formatér måneder til visning
    def format_month_label(m):
        parts = m.split('-')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec']
        return f"{month_names[int(parts[1])-1]} {parts[0][2:]}"
    
    # Sørg for at alle kolonner eksisterer
    for col in ['Mail', 'Message', 'AB']:
        if col not in chart_base_df.columns:
            chart_base_df[col] = ''
    
    # Bestem om vi skal ignorere kolonner i grafer
    chart_ignore_mail = filter_config.get('ignore_mail', False) if filter_config else False
    chart_ignore_message = filter_config.get('ignore_message', False) if filter_config else False
    chart_ignore_ab = filter_config.get('ignore_ab', False) if filter_config else False
    
    if chart_ignore_mail or chart_ignore_message or chart_ignore_ab:
        chart_agg_cols = ['Year_Month']
        if not chart_ignore_mail:
            chart_agg_cols.append('Mail')
        if not chart_ignore_message:
            chart_agg_cols.append('Message')
        if not chart_ignore_ab:
            chart_agg_cols.append('AB')
        
        if len(chart_agg_cols) == 1:
            chart_base_df['_dummy'] = 'Total'
            chart_agg_cols.append('_dummy')
        
        chart_base_df = chart_base_df.groupby(chart_agg_cols, as_index=False).agg({
            'Received_Email': 'sum',
            'Unique_Opens': 'sum',
            'Unique_Clicks': 'sum',
            'Unsubscribed': 'sum',
            'Bounced': 'sum',
        })
        
        if '_dummy' in chart_base_df.columns:
            chart_base_df = chart_base_df.drop(columns=['_dummy'])
        
        if chart_ignore_mail:
            chart_base_df['Mail'] = ''
        if chart_ignore_message:
            chart_base_df['Message'] = ''
        if chart_ignore_ab:
            chart_base_df['AB'] = ''
    
    # Hent unikke kombinationer sorteret (matcher tabellen)
    def combo_sort_key(row):
        match = re.search(r'Mail\s*(\d+)', str(row['Mail']))
        mail_num = int(match.group(1)) if match else 999
        msg = str(row['Message']) if pd.notna(row['Message']) else ''
        ab = str(row['AB']) if pd.notna(row['AB']) else ''
        return (mail_num, msg, ab)
    
    unique_combos = chart_base_df.groupby(['Mail', 'Message', 'AB'], as_index=False).first()[['Mail', 'Message', 'AB']]
    unique_combos['_sort_key'] = unique_combos.apply(combo_sort_key, axis=1)
    unique_combos = unique_combos.sort_values('_sort_key').drop(columns=['_sort_key'])
    unique_combos_list = list(unique_combos.itertuples(index=False, name=None))
    
    if len(sorted_chart_months) > 0 and len(unique_combos_list) > 0:
        num_items = len(unique_combos_list)
        
        subplot_titles = []
        for mail, message, ab in unique_combos_list:
            parts = []
            parts.append(str(mail) if pd.notna(mail) and str(mail).strip() else 'Mail')
            if pd.notna(message) and str(message).strip():
                parts.append(str(message))
            if pd.notna(ab) and str(ab).strip():
                parts.append(str(ab))
            subplot_titles.append(' - '.join(parts))
        
        height_per_chart = 150
        gap_pixels = 70
        chart_height = num_items * height_per_chart + (num_items - 1) * gap_pixels
        
        v_spacing = gap_pixels / chart_height if num_items > 1 else 0.1
        max_allowed = 1.0 / (num_items - 1) if num_items > 1 else 0.5
        v_spacing = min(v_spacing, max_allowed * 0.95)
        
        fig = make_subplots(
            rows=num_items, cols=1,
            shared_xaxes=False,
            vertical_spacing=v_spacing,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True}] for _ in range(num_items)]
        )
        
        for i, (mail, message, ab) in enumerate(unique_combos_list):
            row = i + 1
            mask = (chart_base_df['Mail'] == mail) if pd.notna(mail) and str(mail).strip() else (chart_base_df['Mail'].isna() | (chart_base_df['Mail'] == ''))
            
            if pd.notna(message) and str(message).strip():
                mask = mask & (chart_base_df['Message'] == message)
            else:
                mask = mask & (chart_base_df['Message'].isna() | (chart_base_df['Message'] == ''))
            
            if pd.notna(ab) and str(ab).strip():
                mask = mask & (chart_base_df['AB'] == ab)
            else:
                mask = mask & (chart_base_df['AB'].isna() | (chart_base_df['AB'] == ''))
            
            mail_data = chart_base_df[mask].copy()
            
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
            
            current_month_label = format_month_label(get_current_year_month())
            opens_for_calc = [v for v, lbl in zip(opens_values, x_labels) if lbl != current_month_label]
            sent_for_calc = [v for v, lbl in zip(sent_values, x_labels) if lbl != current_month_label]
            clicks_for_calc = [v for v, lbl in zip(clicks_values, x_labels) if lbl != current_month_label]
            
            max_clicks = max(clicks_for_calc) if clicks_for_calc else (max(clicks_values) if clicks_values else 0)
            max_left = max(max(sent_for_calc) if sent_for_calc else 0, max(opens_for_calc) if opens_for_calc else 0)
            min_opens = min(opens_for_calc) if opens_for_calc else 0
            
            if max_left == 0:
                max_left = max(max(sent_values) if sent_values else 0, max(opens_values) if opens_values else 0)
            if min_opens == 0:
                min_opens = min(opens_values) if opens_values else 0
            
            if max_clicks > 0 and min_opens > 0 and max_left > 0:
                target_visual_height = min_opens * 0.9
                clicks_range = max_clicks * max_left / target_visual_height
                fig.update_yaxes(range=[0, clicks_range], row=row, col=1, secondary_y=True)
            elif max_clicks > 0:
                fig.update_yaxes(range=[0, max_clicks * 2], row=row, col=1, secondary_y=True)
        
        chart_height = max(350, chart_height)
        top_margin = 80
        
        fig.update_layout(
            showlegend=True,
            height=chart_height + top_margin,
            margin=dict(l=60, r=60, t=top_margin, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
            plot_bgcolor='rgba(250,245,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=13, color='#7B5EA5', family='sans-serif')
            annotation['xanchor'] = 'left'
            annotation['x'] = 0
            annotation['yanchor'] = 'bottom'
            annotation['yshift'] = 10
        
        for i in range(num_items):
            fig.update_yaxes(
                title_text="Sendt / Opens",
                title_font=dict(size=10, color='#4A3F55'),
                gridcolor='rgba(212,191,255,0.3)',
                tickformat=',d',
                rangemode='tozero',
                automargin=True,
                ticklabelstandoff=5,
                row=i+1, col=1, secondary_y=False
            )
            fig.update_yaxes(
                title_text="Clicks",
                title_font=dict(size=10, color='#4A3F55'),
                gridcolor='rgba(232,180,203,0.2)',
                showgrid=False,
                tickformat=',d',
                rangemode='tozero',
                automargin=True,
                ticklabelstandoff=5,
                row=i+1, col=1, secondary_y=True
            )
        
        fig.update_xaxes(gridcolor='rgba(212,191,255,0.2)', type='category', tickfont=dict(size=10), automargin=True)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_repeat_tab():
    """Render Repeat tab indhold med sub-tabs"""
    
    try:
        with st.spinner('Henter repeat data...'):
            df = load_repeat_data()
        if df.empty:
            st.error("Kunne ikke hente repeat data. Tjek Google Sheets konfiguration.")
            return
    except Exception as e:
        st.error(f"Fejl: {e}")
        return

    available_months = get_available_months(df)
    
    if not available_months:
        st.warning("Ingen måneder tilgængelige i data.")
        return

    if 'rp_selected_countries' not in st.session_state:
        st.session_state.rp_selected_countries = None
    if 'rp_selected_repeats' not in st.session_state:
        st.session_state.rp_selected_repeats = None
    if 'rp_cb_reset_land' not in st.session_state:
        st.session_state.rp_cb_reset_land = 0
    if 'rp_cb_reset_repeat' not in st.session_state:
        st.session_state.rp_cb_reset_repeat = 0

    all_repeats = get_unique_repeats(df)
    
    tab_labels = ["Oversigt"] + [get_short_repeat_name(r) for r in all_repeats]
    sub_tabs = st.tabs(tab_labels)
    
    with sub_tabs[0]:
        render_overview_tab_content(df, available_months)
    
    for i, repeat_name in enumerate(all_repeats):
        with sub_tabs[i + 1]:
            render_repeat_tab_content(df, repeat_name, available_months)


def render_overview_tab_content(df, available_months):
    """Render oversigt tab med filtre"""
    
    def month_to_tuple(m):
        parts = m.split('-')
        return (int(parts[0]), int(parts[1]))
    
    sorted_months = sorted(available_months, key=month_to_tuple)
    
    full_repeat_df = aggregate_to_repeat_level(df)
    all_countries = sorted(full_repeat_df['Country'].unique())
    all_repeats_full = get_unique_repeats(full_repeat_df)
    
    current_month = get_current_year_month()
    today = datetime.date.today()
    if today.month == 1:
        prev_month = f"{today.year - 1}-12"
    else:
        prev_month = f"{today.year}-{today.month - 1}"
    
    recent_months = [current_month, prev_month]
    recent_data = full_repeat_df[
        (full_repeat_df['Year_Month'].isin(recent_months)) & 
        (full_repeat_df['Received_Email'] > 0)
    ]
    active_repeats = set(recent_data['Repeat'].unique())
    
    if 'rp_ignore_inactive_overview' not in st.session_state:
        st.session_state.rp_ignore_inactive_overview = True
    
    all_repeats = [r for r in all_repeats_full if r in active_repeats] if st.session_state.rp_ignore_inactive_overview else all_repeats_full

    if st.session_state.rp_selected_countries is None:
        st.session_state.rp_selected_countries = list(all_countries)
    else:
        st.session_state.rp_selected_countries = [c for c in st.session_state.rp_selected_countries if c in all_countries]

    if st.session_state.rp_selected_repeats is None:
        st.session_state.rp_selected_repeats = list(all_repeats)
    else:
        st.session_state.rp_selected_repeats = [r for r in st.session_state.rp_selected_repeats if r in all_repeats]
        if not st.session_state.rp_selected_repeats:
            st.session_state.rp_selected_repeats = list(all_repeats)

    col_land, col_repeat, col_inaktive, col_slider = st.columns([1, 1, 0.8, 3.2])

    with col_land:
        land_count = len(st.session_state.rp_selected_countries)
        land_label = f"Land ({land_count})" if land_count < len(all_countries) else "Land"
        with st.popover(land_label, use_container_width=True):
            reset_land = st.session_state.rp_cb_reset_land
            all_land_selected = len(st.session_state.rp_selected_countries) == len(all_countries)
            select_all_land = st.checkbox("Vælg alle", value=all_land_selected, key=f"rp_sel_all_land_{reset_land}")
            
            new_selected = []
            only_clicked_land = None
            for country in all_countries:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    checked = country in st.session_state.rp_selected_countries
                    if st.checkbox(country, value=checked, key=f"rp_cb_land_{country}_{reset_land}"):
                        new_selected.append(country)
                with only_col:
                    if st.button("Kun", key=f"rp_only_land_ov_{country}_{reset_land}", type="secondary"):
                        only_clicked_land = country
            
            if only_clicked_land:
                st.session_state.rp_selected_countries = [only_clicked_land]
                st.session_state.rp_cb_reset_land += 1
                st.rerun()
            elif select_all_land and not all_land_selected:
                st.session_state.rp_selected_countries = list(all_countries)
                st.session_state.rp_cb_reset_land += 1
                st.rerun()
            elif not select_all_land and all_land_selected:
                st.session_state.rp_selected_countries = []
                st.session_state.rp_cb_reset_land += 1
                st.rerun()
            elif set(new_selected) != set(st.session_state.rp_selected_countries):
                st.session_state.rp_selected_countries = new_selected
                st.session_state.rp_cb_reset_land += 1
                st.rerun()

    with col_repeat:
        repeat_count = len(st.session_state.rp_selected_repeats)
        repeat_label = f"Repeat ({repeat_count})" if repeat_count < len(all_repeats) else "Repeat"
        with st.popover(repeat_label, use_container_width=True):
            reset_repeat = st.session_state.rp_cb_reset_repeat
            all_repeat_selected = len(st.session_state.rp_selected_repeats) == len(all_repeats)
            select_all_repeat = st.checkbox("Vælg alle", value=all_repeat_selected, key=f"rp_sel_all_repeat_{reset_repeat}")
            
            new_selected_repeats = []
            only_clicked_repeat = None
            for repeat in all_repeats:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    short_name = get_short_repeat_name(repeat)
                    checked = repeat in st.session_state.rp_selected_repeats
                    if st.checkbox(short_name, value=checked, key=f"rp_cb_repeat_{repeat}_{reset_repeat}"):
                        new_selected_repeats.append(repeat)
                with only_col:
                    if st.button("Kun", key=f"rp_only_repeat_{repeat}_{reset_repeat}", type="secondary"):
                        only_clicked_repeat = repeat
            
            if only_clicked_repeat:
                st.session_state.rp_selected_repeats = [only_clicked_repeat]
                st.session_state.rp_cb_reset_repeat += 1
                st.rerun()
            elif select_all_repeat and not all_repeat_selected:
                st.session_state.rp_selected_repeats = list(all_repeats)
                st.session_state.rp_cb_reset_repeat += 1
                st.rerun()
            elif not select_all_repeat and all_repeat_selected:
                st.session_state.rp_selected_repeats = []
                st.session_state.rp_cb_reset_repeat += 1
                st.rerun()
            elif set(new_selected_repeats) != set(st.session_state.rp_selected_repeats):
                st.session_state.rp_selected_repeats = new_selected_repeats
                st.session_state.rp_cb_reset_repeat += 1
                st.rerun()

    with col_inaktive:
        ignore_inactive = st.checkbox(
            "Ignorer Inaktive", 
            value=st.session_state.rp_ignore_inactive_overview, 
            key="rp_ignore_inactive_overview_cb"
        )
        if ignore_inactive != st.session_state.rp_ignore_inactive_overview:
            st.session_state.rp_ignore_inactive_overview = ignore_inactive
            new_all_repeats = [r for r in all_repeats_full if r in active_repeats] if ignore_inactive else all_repeats_full
            st.session_state.rp_selected_repeats = list(new_all_repeats)
            st.session_state.rp_cb_reset_repeat += 1
            st.rerun()

    with col_slider:
        if len(sorted_months) > 1:
            default_end = sorted_months[-1]
            default_start_idx = max(0, len(sorted_months) - 3)
            default_start = sorted_months[default_start_idx]
            
            saved_key = "rp_month_range_overview_saved"
            if saved_key not in st.session_state:
                st.session_state[saved_key] = (default_start, default_end)
            
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
                key="rp_month_range_overview",
                label_visibility="collapsed"
            )
            st.session_state[saved_key] = month_range
            sel_months = get_months_in_range(month_range[0], month_range[1], sorted_months)
        else:
            sel_months = sorted_months
    
    if not sel_months:
        st.warning("Vælg mindst én måned.")
        return

    df_month_filtered = df[df['Year_Month'].isin(sel_months)]
    repeat_df = aggregate_to_repeat_level(df_month_filtered)

    sel_countries = st.session_state.rp_selected_countries
    sel_repeats = st.session_state.rp_selected_repeats

    if not sel_countries:
        st.warning("Vælg mindst ét land.")
        return

    if not sel_repeats:
        st.warning("Vælg mindst én repeat.")
        return

    all_months_repeat_df = aggregate_to_repeat_level(df)
    ignore_inactive = st.session_state.rp_ignore_inactive_overview
    render_overview_content(repeat_df, sel_countries, sel_repeats, full_repeat_df, all_months_repeat_df, ignore_inactive, active_repeats)

    if st.button('Opdater Data', key="rp_refresh_overview"):
        st.rerun()


def render_repeat_tab_content(df, repeat_name, available_months):
    """Render enkelt repeat tab med under-tabs for groups"""
    
    all_groups = get_unique_groups_for_repeat(df, repeat_name)
    
    if not all_groups:
        st.warning(f"Ingen groups fundet for {repeat_name}.")
        return
    
    group_tab_labels = ["Oversigt"] + all_groups
    group_tabs = st.tabs(group_tab_labels)
    
    with group_tabs[0]:
        render_repeat_group_overview_tab(df, repeat_name, available_months, all_groups)
    
    for i, group_name in enumerate(all_groups):
        with group_tabs[i + 1]:
            render_group_mail_tab(df, repeat_name, group_name, available_months)


def render_repeat_group_overview_tab(df, repeat_name, available_months, all_groups):
    """Render group oversigt for en specifik repeat"""
    
    def month_to_tuple(m):
        parts = m.split('-')
        return (int(parts[0]), int(parts[1]))
    
    sorted_months = sorted(available_months, key=month_to_tuple)
    
    repeat_data = df[df['Repeat'] == repeat_name]
    all_countries = sorted(repeat_data['Country'].unique())
    
    safe_repeat = repeat_name.replace(" ", "_")
    land_key = f'rp_countries_go_{safe_repeat}'
    group_key = f'rp_groups_go_{safe_repeat}'
    reset_land_key = f'rp_reset_land_go_{safe_repeat}'
    reset_group_key = f'rp_reset_group_go_{safe_repeat}'
    ignore_key = f'rp_ignore_inactive_go_{safe_repeat}'
    
    if land_key not in st.session_state:
        st.session_state[land_key] = list(all_countries)
    if group_key not in st.session_state:
        st.session_state[group_key] = list(all_groups)
    if reset_land_key not in st.session_state:
        st.session_state[reset_land_key] = 0
    if reset_group_key not in st.session_state:
        st.session_state[reset_group_key] = 0
    if ignore_key not in st.session_state:
        st.session_state[ignore_key] = True
    
    current_month = get_current_year_month()
    today = datetime.date.today()
    if today.month == 1:
        prev_month = f"{today.year - 1}-12"
    else:
        prev_month = f"{today.year}-{today.month - 1}"
    
    recent_months = [current_month, prev_month]
    recent_data = repeat_data[
        (repeat_data['Year_Month'].isin(recent_months)) & 
        (repeat_data['Received_Email'] > 0)
    ]
    active_groups = set(recent_data['Group'].unique())
    
    available_groups = [g for g in all_groups if g in active_groups] if st.session_state[ignore_key] else all_groups
    
    st.session_state[land_key] = [c for c in st.session_state[land_key] if c in all_countries]
    if not st.session_state[land_key]:
        st.session_state[land_key] = list(all_countries)
    
    st.session_state[group_key] = [g for g in st.session_state[group_key] if g in available_groups]
    if not st.session_state[group_key]:
        st.session_state[group_key] = list(available_groups)

    st.markdown(f'<p style="color: #9B7EBD; font-size: 1.1em; font-weight: 500; margin-bottom: 0.5em;">{repeat_name}</p>', unsafe_allow_html=True)

    col_land, col_group, col_inaktive, col_slider = st.columns([1, 1, 0.8, 3.2])

    with col_land:
        land_count = len(st.session_state[land_key])
        land_label = f"Land ({land_count})" if land_count < len(all_countries) else "Land"
        with st.popover(land_label, use_container_width=True):
            reset_land = st.session_state[reset_land_key]
            all_land_selected = len(st.session_state[land_key]) == len(all_countries)
            select_all_land = st.checkbox("Vælg alle", value=all_land_selected, key=f"rp_sel_all_land_go_{safe_repeat}_{reset_land}")
            
            new_selected = []
            only_clicked_land = None
            for country in all_countries:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    checked = country in st.session_state[land_key]
                    if st.checkbox(country, value=checked, key=f"rp_cb_land_go_{safe_repeat}_{country}_{reset_land}"):
                        new_selected.append(country)
                with only_col:
                    if st.button("Kun", key=f"rp_only_land_go_{safe_repeat}_{country}_{reset_land}", type="secondary"):
                        only_clicked_land = country
            
            if only_clicked_land:
                st.session_state[land_key] = [only_clicked_land]
                st.session_state[reset_land_key] += 1
                st.rerun()
            elif select_all_land and not all_land_selected:
                st.session_state[land_key] = list(all_countries)
                st.session_state[reset_land_key] += 1
                st.rerun()
            elif not select_all_land and all_land_selected:
                st.session_state[land_key] = []
                st.session_state[reset_land_key] += 1
                st.rerun()
            elif set(new_selected) != set(st.session_state[land_key]):
                st.session_state[land_key] = new_selected
                st.session_state[reset_land_key] += 1
                st.rerun()

    with col_group:
        group_count = len(st.session_state[group_key])
        group_label = f"Group ({group_count})" if group_count < len(available_groups) else "Group"
        with st.popover(group_label, use_container_width=True):
            reset_group = st.session_state[reset_group_key]
            all_group_selected = len(st.session_state[group_key]) == len(available_groups)
            select_all_group = st.checkbox("Vælg alle", value=all_group_selected, key=f"rp_sel_all_group_go_{safe_repeat}_{reset_group}")
            
            new_selected_groups = []
            only_clicked_group = None
            for group in available_groups:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    checked = group in st.session_state[group_key]
                    if st.checkbox(str(group), value=checked, key=f"rp_cb_group_go_{safe_repeat}_{group}_{reset_group}"):
                        new_selected_groups.append(group)
                with only_col:
                    if st.button("Kun", key=f"rp_only_group_go_{safe_repeat}_{group}_{reset_group}", type="secondary"):
                        only_clicked_group = group
            
            if only_clicked_group:
                st.session_state[group_key] = [only_clicked_group]
                st.session_state[reset_group_key] += 1
                st.rerun()
            elif select_all_group and not all_group_selected:
                st.session_state[group_key] = list(available_groups)
                st.session_state[reset_group_key] += 1
                st.rerun()
            elif not select_all_group and all_group_selected:
                st.session_state[group_key] = []
                st.session_state[reset_group_key] += 1
                st.rerun()
            elif set(new_selected_groups) != set(st.session_state[group_key]):
                st.session_state[group_key] = new_selected_groups
                st.session_state[reset_group_key] += 1
                st.rerun()

    with col_inaktive:
        ignore_inactive = st.checkbox(
            "Ignorer Inaktive", 
            value=st.session_state[ignore_key], 
            key=f"rp_ignore_inactive_cb_go_{safe_repeat}"
        )
        if ignore_inactive != st.session_state[ignore_key]:
            st.session_state[ignore_key] = ignore_inactive
            new_groups = [g for g in all_groups if g in active_groups] if ignore_inactive else all_groups
            st.session_state[group_key] = list(new_groups)
            st.session_state[reset_group_key] += 1
            st.rerun()

    with col_slider:
        if len(sorted_months) > 1:
            default_end = sorted_months[-1]
            default_start_idx = max(0, len(sorted_months) - 3)
            default_start = sorted_months[default_start_idx]
            
            saved_key = f"rp_month_range_go_{safe_repeat}_saved"
            if saved_key not in st.session_state:
                st.session_state[saved_key] = (default_start, default_end)
            
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
                key=f"rp_month_range_go_{safe_repeat}",
                label_visibility="collapsed"
            )
            st.session_state[saved_key] = month_range
            sel_months = get_months_in_range(month_range[0], month_range[1], sorted_months)
        else:
            sel_months = sorted_months
    
    if not sel_months:
        st.warning("Vælg mindst én måned.")
        return

    sel_countries = st.session_state[land_key]
    sel_groups = st.session_state[group_key]

    if not sel_countries:
        st.warning("Vælg mindst ét land.")
        return

    if not sel_groups:
        st.warning("Vælg mindst én group.")
        return

    df_filtered = df[df['Year_Month'].isin(sel_months)]
    
    render_group_overview_content(
        df_filtered, repeat_name, sel_countries, sel_groups, 
        df, df, st.session_state[ignore_key], active_groups
    )

    if st.button('Opdater Data', key=f"rp_refresh_go_{safe_repeat}"):
        st.rerun()


def render_group_mail_tab(df, repeat_name, group_name, available_months):
    """Render mail-niveau detaljer for en specifik group"""
    
    def month_to_tuple(m):
        parts = m.split('-')
        return (int(parts[0]), int(parts[1]))
    
    sorted_months = sorted(available_months, key=month_to_tuple)
    
    group_data = df[(df['Repeat'] == repeat_name) & (df['Group'] == group_name)]
    all_countries = sorted(group_data['Country'].unique())
    
    if not all_countries:
        st.warning(f"Ingen data for {group_name}.")
        return
    
    BLANK_LABEL = "Ingen"
    
    has_blank_mail = any(pd.isna(m) or str(m).strip() == '' for m in group_data['Mail'].unique())
    has_blank_message = any(pd.isna(m) or str(m).strip() == '' for m in group_data['Message'].unique())
    has_blank_ab = any(pd.isna(a) or str(a).strip() == '' for a in group_data['AB'].unique())
    
    all_mails = sorted([m for m in group_data['Mail'].unique() if pd.notna(m) and str(m).strip() != ''])
    all_messages = sorted([m for m in group_data['Message'].unique() if pd.notna(m) and str(m).strip() != ''])
    all_ab = sorted([a for a in group_data['AB'].unique() if pd.notna(a) and str(a).strip() != ''])
    
    if has_blank_mail:
        all_mails = [BLANK_LABEL] + all_mails
    if has_blank_message:
        all_messages = [BLANK_LABEL] + all_messages
    if has_blank_ab:
        all_ab = [BLANK_LABEL] + all_ab
    
    def mail_sort_key(mail):
        if mail == BLANK_LABEL:
            return -1
        match = re.search(r'Mail\s*(\d+)', str(mail))
        return int(match.group(1)) if match else 999
    all_mails = sorted(all_mails, key=mail_sort_key)
    
    safe_key = f"{repeat_name}_{group_name}".replace(" ", "_").replace("-", "_")
    land_key = f'rp_countries_gm_{safe_key}'
    mail_key = f'rp_mails_gm_{safe_key}'
    message_key = f'rp_messages_gm_{safe_key}'
    ab_key = f'rp_ab_gm_{safe_key}'
    reset_land_key = f'rp_reset_land_gm_{safe_key}'
    reset_mail_key = f'rp_reset_mail_gm_{safe_key}'
    reset_message_key = f'rp_reset_message_gm_{safe_key}'
    reset_ab_key = f'rp_reset_ab_gm_{safe_key}'
    ignore_mail_key = f'rp_ignore_mail_gm_{safe_key}'
    ignore_message_key = f'rp_ignore_message_gm_{safe_key}'
    ignore_ab_key = f'rp_ignore_ab_gm_{safe_key}'
    ignore_inactive_key = f'rp_ignore_inactive_gm_{safe_key}'
    
    if land_key not in st.session_state:
        st.session_state[land_key] = list(all_countries)
    if mail_key not in st.session_state:
        st.session_state[mail_key] = list(all_mails)
    if message_key not in st.session_state:
        st.session_state[message_key] = list(all_messages)
    if ab_key not in st.session_state:
        st.session_state[ab_key] = list(all_ab)
    if reset_land_key not in st.session_state:
        st.session_state[reset_land_key] = 0
    if reset_mail_key not in st.session_state:
        st.session_state[reset_mail_key] = 0
    if reset_message_key not in st.session_state:
        st.session_state[reset_message_key] = 0
    if reset_ab_key not in st.session_state:
        st.session_state[reset_ab_key] = 0
    if ignore_mail_key not in st.session_state:
        st.session_state[ignore_mail_key] = False
    if ignore_message_key not in st.session_state:
        st.session_state[ignore_message_key] = False
    if ignore_ab_key not in st.session_state:
        st.session_state[ignore_ab_key] = False
    if ignore_inactive_key not in st.session_state:
        st.session_state[ignore_inactive_key] = True

    st.markdown(f'<p style="color: #9B7EBD; font-size: 1.1em; font-weight: 500; margin-bottom: 0.5em;">{repeat_name} - {group_name}</p>', unsafe_allow_html=True)

    col_land, col_slider = st.columns([1, 5])

    with col_land:
        land_count = len(st.session_state[land_key])
        land_label = f"Land ({land_count})" if land_count < len(all_countries) else "Land"
        with st.popover(land_label, use_container_width=True):
            reset_land = st.session_state[reset_land_key]
            all_land_selected = len(st.session_state[land_key]) == len(all_countries)
            select_all_land = st.checkbox("Vælg alle", value=all_land_selected, key=f"rp_sel_all_land_gm_{safe_key}_{reset_land}")
            
            new_selected = []
            only_clicked_land = None
            for country in all_countries:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    checked = country in st.session_state[land_key]
                    if st.checkbox(country, value=checked, key=f"rp_cb_land_gm_{safe_key}_{country}_{reset_land}"):
                        new_selected.append(country)
                with only_col:
                    if st.button("Kun", key=f"rp_only_land_gm_{safe_key}_{country}_{reset_land}", type="secondary"):
                        only_clicked_land = country
            
            if only_clicked_land:
                st.session_state[land_key] = [only_clicked_land]
                st.session_state[reset_land_key] += 1
                st.rerun()
            elif select_all_land and not all_land_selected:
                st.session_state[land_key] = list(all_countries)
                st.session_state[reset_land_key] += 1
                st.rerun()
            elif not select_all_land and all_land_selected:
                st.session_state[land_key] = []
                st.session_state[reset_land_key] += 1
                st.rerun()
            elif set(new_selected) != set(st.session_state[land_key]):
                st.session_state[land_key] = new_selected
                st.session_state[reset_land_key] += 1
                st.rerun()

    with col_slider:
        if len(sorted_months) > 1:
            default_end = sorted_months[-1]
            default_start_idx = max(0, len(sorted_months) - 3)
            default_start = sorted_months[default_start_idx]
            
            saved_key = f"rp_month_range_gm_{safe_key}_saved"
            if saved_key not in st.session_state:
                st.session_state[saved_key] = (default_start, default_end)
            
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
                key=f"rp_month_range_gm_{safe_key}",
                label_visibility="collapsed"
            )
            st.session_state[saved_key] = month_range
            sel_months = get_months_in_range(month_range[0], month_range[1], sorted_months)
        else:
            sel_months = sorted_months

    col_mail, col_message, col_ab, col_empty = st.columns([1, 1, 1, 3])

    with col_mail:
        mail_count = len([m for m in st.session_state[mail_key] if m in all_mails])
        mail_label = f"Mail ({mail_count})" if mail_count < len(all_mails) else "Mail"
        with st.popover(mail_label, use_container_width=True):
            reset_mail = st.session_state[reset_mail_key]
            current_sel = [m for m in st.session_state[mail_key] if m in all_mails]
            all_mail_selected = len(current_sel) == len(all_mails)
            select_all_mail = st.checkbox("Vælg alle", value=all_mail_selected, key=f"rp_sel_all_mail_gm_{safe_key}_{reset_mail}")
            
            new_selected_mails = []
            only_clicked_mail = None
            for mail in all_mails:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    checked = mail in st.session_state[mail_key]
                    if st.checkbox(str(mail), value=checked, key=f"rp_cb_mail_gm_{safe_key}_{mail}_{reset_mail}"):
                        new_selected_mails.append(mail)
                with only_col:
                    if st.button("Kun", key=f"rp_only_mail_gm_{safe_key}_{mail}_{reset_mail}", type="secondary"):
                        only_clicked_mail = mail
            
            if only_clicked_mail:
                st.session_state[mail_key] = [only_clicked_mail]
                st.session_state[reset_mail_key] += 1
                st.rerun()
            elif select_all_mail and not all_mail_selected:
                st.session_state[mail_key] = list(all_mails)
                st.session_state[reset_mail_key] += 1
                st.rerun()
            elif not select_all_mail and all_mail_selected:
                st.session_state[mail_key] = []
                st.session_state[reset_mail_key] += 1
                st.rerun()
            elif set(new_selected_mails) != set(current_sel):
                st.session_state[mail_key] = new_selected_mails
                st.session_state[reset_mail_key] += 1
                st.rerun()

    with col_message:
        message_count = len([m for m in st.session_state[message_key] if m in all_messages])
        message_label = f"Message ({message_count})" if message_count < len(all_messages) else "Message"
        with st.popover(message_label, use_container_width=True):
            reset_message = st.session_state[reset_message_key]
            current_sel_msg = [m for m in st.session_state[message_key] if m in all_messages]
            all_msg_selected = len(current_sel_msg) == len(all_messages)
            select_all_msg = st.checkbox("Vælg alle", value=all_msg_selected, key=f"rp_sel_all_msg_gm_{safe_key}_{reset_message}")
            
            new_selected_messages = []
            only_clicked_msg = None
            for msg in all_messages:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    checked = msg in st.session_state[message_key]
                    if st.checkbox(str(msg), value=checked, key=f"rp_cb_msg_gm_{safe_key}_{msg}_{reset_message}"):
                        new_selected_messages.append(msg)
                with only_col:
                    if st.button("Kun", key=f"rp_only_msg_gm_{safe_key}_{msg}_{reset_message}", type="secondary"):
                        only_clicked_msg = msg
            
            if only_clicked_msg:
                st.session_state[message_key] = [only_clicked_msg]
                st.session_state[reset_message_key] += 1
                st.rerun()
            elif select_all_msg and not all_msg_selected:
                st.session_state[message_key] = list(all_messages)
                st.session_state[reset_message_key] += 1
                st.rerun()
            elif not select_all_msg and all_msg_selected:
                st.session_state[message_key] = []
                st.session_state[reset_message_key] += 1
                st.rerun()
            elif set(new_selected_messages) != set(current_sel_msg):
                st.session_state[message_key] = new_selected_messages
                st.session_state[reset_message_key] += 1
                st.rerun()

    with col_ab:
        ab_count = len([a for a in st.session_state[ab_key] if a in all_ab])
        ab_label = f"A/B ({ab_count})" if ab_count < len(all_ab) else "A/B"
        with st.popover(ab_label, use_container_width=True):
            reset_ab = st.session_state[reset_ab_key]
            current_sel_ab = [a for a in st.session_state[ab_key] if a in all_ab]
            all_ab_selected = len(current_sel_ab) == len(all_ab)
            select_all_ab_cb = st.checkbox("Vælg alle", value=all_ab_selected, key=f"rp_sel_all_ab_gm_{safe_key}_{reset_ab}")
            
            new_selected_ab = []
            only_clicked_ab = None
            for ab in all_ab:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    checked = ab in st.session_state[ab_key]
                    if st.checkbox(str(ab), value=checked, key=f"rp_cb_ab_gm_{safe_key}_{ab}_{reset_ab}"):
                        new_selected_ab.append(ab)
                with only_col:
                    if st.button("Kun", key=f"rp_only_ab_gm_{safe_key}_{ab}_{reset_ab}", type="secondary"):
                        only_clicked_ab = ab
            
            if only_clicked_ab:
                st.session_state[ab_key] = [only_clicked_ab]
                st.session_state[reset_ab_key] += 1
                st.rerun()
            elif select_all_ab_cb and not all_ab_selected:
                st.session_state[ab_key] = list(all_ab)
                st.session_state[reset_ab_key] += 1
                st.rerun()
            elif not select_all_ab_cb and all_ab_selected:
                st.session_state[ab_key] = []
                st.session_state[reset_ab_key] += 1
                st.rerun()
            elif set(new_selected_ab) != set(current_sel_ab):
                st.session_state[ab_key] = new_selected_ab
                st.session_state[reset_ab_key] += 1
                st.rerun()

    col_ig_mail, col_ig_message, col_ig_ab, col_ig_empty = st.columns([1, 1, 1, 3])
    
    with col_ig_mail:
        ignore_mail = st.checkbox(
            "Ignorer Mail",
            value=st.session_state[ignore_mail_key],
            key=f"rp_ignore_mail_cb_gm_{safe_key}"
        )
        if ignore_mail != st.session_state[ignore_mail_key]:
            st.session_state[ignore_mail_key] = ignore_mail
            st.rerun()
    
    with col_ig_message:
        ignore_message = st.checkbox(
            "Ignorer Message",
            value=st.session_state[ignore_message_key],
            key=f"rp_ignore_message_cb_gm_{safe_key}"
        )
        if ignore_message != st.session_state[ignore_message_key]:
            st.session_state[ignore_message_key] = ignore_message
            st.rerun()
    
    with col_ig_ab:
        ignore_ab = st.checkbox(
            "Ignorer A/B",
            value=st.session_state[ignore_ab_key],
            key=f"rp_ignore_ab_cb_gm_{safe_key}"
        )
        if ignore_ab != st.session_state[ignore_ab_key]:
            st.session_state[ignore_ab_key] = ignore_ab
            st.rerun()

    if not sel_months:
        st.warning("Vælg mindst én måned.")
        return

    sel_countries = st.session_state[land_key]
    if not sel_countries:
        st.warning("Vælg mindst ét land.")
        return

    selected_mails = [m for m in st.session_state[mail_key] if m in all_mails]
    if not selected_mails:
        selected_mails = all_mails
    
    selected_messages = [m for m in st.session_state[message_key] if m in all_messages]
    if not selected_messages:
        selected_messages = all_messages
    
    selected_ab = [a for a in st.session_state[ab_key] if a in all_ab]
    if not selected_ab:
        selected_ab = all_ab
    
    filter_config = {
        'mails': selected_mails,
        'messages': selected_messages,
        'ab': selected_ab,
        'ignore_mail': st.session_state[ignore_mail_key],
        'ignore_message': st.session_state[ignore_message_key],
        'ignore_ab': st.session_state[ignore_ab_key],
        'ignore_inactive': st.session_state[ignore_inactive_key],
        '_ignore_inactive_key': ignore_inactive_key,
    }
    
    df_filtered = df[df['Year_Month'].isin(sel_months)]
    
    render_group_mail_content(df_filtered, repeat_name, group_name, sel_countries, filter_config, df, all_months_df=df)

    if st.button('Opdater Data', key=f"rp_refresh_gm_{safe_key}"):
        st.rerun()
