"""
Newsletters Tab - CRM Dashboard
"""
import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shared import get_gspread_client, show_metric, COUNTRY_ORDER


@st.cache_data(ttl=300, show_spinner=False)  # Cache i 5 minutter
def load_newsletter_data():
    """Henter Newsletter data fra Google Sheet"""
    try:
        gc = get_gspread_client()
        spreadsheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        spreadsheet = gc.open_by_url(spreadsheet_url)
        worksheet = spreadsheet.sheet1
        all_values = worksheet.get_all_values()
        
        if len(all_values) > 2:
            data = all_values[2:]
            raw_df = pd.DataFrame(data)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Fejl ved indlæsning fra Google Sheets: {e}")
        return pd.DataFrame()
    
    # Landekonfiguration
    country_configs = [
        ('DK', 15), ('SE', 21), ('NO', 27), ('FI', 33), ('FR', 39),
        ('UK', 45), ('DE', 51), ('AT', 57), ('NL', 63), ('BE', 69), ('CH', 75),
    ]
    
    all_country_data = []
    
    for country_code, start_col in country_configs:
        try:
            country_df = pd.DataFrame()
            country_df['Send Year'] = raw_df.iloc[:, 0]
            country_df['Send Month'] = raw_df.iloc[:, 1]
            country_df['Send Day'] = raw_df.iloc[:, 2]
            country_df['Send Time'] = raw_df.iloc[:, 3]
            country_df['Number'] = raw_df.iloc[:, 4]
            country_df['Campaign Name'] = raw_df.iloc[:, 5]
            country_df['Email'] = raw_df.iloc[:, 6]
            country_df['Message'] = raw_df.iloc[:, 7]
            country_df['Variant'] = raw_df.iloc[:, 8]
            
            country_df['Total_Received'] = raw_df.iloc[:, start_col + 0]
            country_df['Total_Opens_Raw'] = raw_df.iloc[:, start_col + 1]
            country_df['Unique_Opens'] = raw_df.iloc[:, start_col + 2]
            country_df['Total_Clicks_Raw'] = raw_df.iloc[:, start_col + 3]
            country_df['Unique_Clicks'] = raw_df.iloc[:, start_col + 4]
            country_df['Unsubscribed'] = raw_df.iloc[:, start_col + 5]
            country_df['Country'] = country_code
            
            all_country_data.append(country_df)
        except Exception:
            continue
    
    if not all_country_data:
        return pd.DataFrame()

    df = pd.concat(all_country_data, ignore_index=True)
    
    df['Date'] = pd.to_datetime(
        df['Send Year'].astype(str) + '-' + 
        df['Send Month'].astype(str) + '-' + 
        df['Send Day'].astype(str), 
        errors='coerce'
    )
    df = df.dropna(subset=['Date'])

    numeric_cols = ['Total_Received', 'Unique_Opens', 'Unique_Clicks', 'Unsubscribed']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    df['Open Rate %'] = df.apply(lambda x: (x['Unique_Opens'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1)
    df['Click Rate %'] = df.apply(lambda x: (x['Unique_Clicks'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1)
    df['Click Through Rate %'] = df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    df['ID_Campaign'] = df['Number'].astype(str) + ' - ' + df['Campaign Name'].astype(str)
    df['Email_Message_Base'] = df['Email'].astype(str) + ' - ' + df['Message'].astype(str)
    df['Email_Message_Full'] = df.apply(
        lambda x: f"{x['Email']} - {x['Message']} - {x['Variant']}" 
        if pd.notna(x['Variant']) and str(x['Variant']).strip() not in ['', 'nan', 'None'] 
        else f"{x['Email']} - {x['Message']}", 
        axis=1
    )
    
    return df


def get_quarter_start(date):
    quarter = (date.month - 1) // 3
    return datetime.date(date.year, quarter * 3 + 1, 1)


def calculate_date_range(preset, today, yesterday):
    if preset == "Sidste 7 dage":
        return (today - datetime.timedelta(days=7), yesterday)
    elif preset == "Sidste 30 dage":
        return (today - datetime.timedelta(days=30), yesterday)
    elif preset == "Denne måned":
        return (today.replace(day=1), yesterday)
    elif preset == "Dette kvartal":
        return (get_quarter_start(today), yesterday)
    elif preset == "I år":
        return (today.replace(month=1, day=1), yesterday)
    elif preset == "Sidste måned":
        first_this_month = today.replace(day=1)
        last_day_last_month = first_this_month - datetime.timedelta(days=1)
        return (last_day_last_month.replace(day=1), last_day_last_month)
    elif preset == "Sidste kvartal":
        current_q_start = get_quarter_start(today)
        last_q_end = current_q_start - datetime.timedelta(days=1)
        return (get_quarter_start(last_q_end), last_q_end)
    return None


def filter_data(dataset, start, end, sel_countries, sel_id_campaigns, sel_email_messages, email_col):
    mask = (dataset['Date'] >= pd.to_datetime(start)) & (dataset['Date'] <= pd.to_datetime(end))
    temp_df = dataset.loc[mask].copy()
    
    if len(sel_countries) == 0 or len(sel_id_campaigns) == 0 or len(sel_email_messages) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    temp_df = temp_df[temp_df['Country'].isin(sel_countries)]
    temp_df = temp_df[temp_df['ID_Campaign'].astype(str).isin(sel_id_campaigns)]
    temp_df = temp_df[temp_df[email_col].astype(str).isin(sel_email_messages)]
    
    if not temp_df.empty:
        pivot_df = temp_df.pivot_table(
            index=['Date', 'ID_Campaign', email_col],
            columns='Country',
            values='Total_Received',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        all_countries = COUNTRY_ORDER
        for country in all_countries:
            if country not in pivot_df.columns:
                pivot_df[country] = 0
        
        pivot_df['Total'] = sum(pivot_df[c] for c in all_countries)
        
        agg_df = temp_df.groupby(['Date', 'ID_Campaign', email_col], as_index=False).agg({
            'Total_Received': 'sum',
            'Unique_Opens': 'sum',
            'Unique_Clicks': 'sum',
            'Unsubscribed': 'sum'
        })
        
        agg_df = agg_df.rename(columns={email_col: 'Email_Message'})
        
        agg_df['Open Rate %'] = agg_df.apply(
            lambda x: (x['Unique_Opens'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1
        )
        agg_df['Click Rate %'] = agg_df.apply(
            lambda x: (x['Unique_Clicks'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1
        )
        agg_df['Click Through Rate %'] = agg_df.apply(
            lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1
        )
        
        return agg_df, pivot_df
        
    return temp_df, pd.DataFrame()


def render_newsletters_tab():
    """Render Newsletters tab indhold"""
    
    # Load data
    try:
        with st.spinner('Henter data...'):
            df = load_newsletter_data()
        if df.empty:
            st.error("Kunne ikke hente data. Tjek Secrets.")
            return
    except Exception as e:
        st.error(f"Fejl: {e}")
        return

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    # Session state for dato
    if 'nl_date_preset' not in st.session_state:
        st.session_state.nl_date_preset = "Sidste 30 dage"
    if 'nl_date_range_value' not in st.session_state:
        st.session_state.nl_date_range_value = (today - datetime.timedelta(days=30), yesterday)

    preset_options = [
        "Sidste 7 dage", "Sidste 30 dage", "Denne måned",
        "Dette kvartal", "I år", "Sidste måned", "Sidste kvartal",
    ]

    # Layout - same widths as scorecards below
    col_preset, col_dato, col_land, col_kamp, col_email, col_ab = st.columns(6)

    with col_preset:
        preset_index = preset_options.index(st.session_state.nl_date_preset) if st.session_state.nl_date_preset in preset_options else 1
        selected_preset = st.selectbox(
            "Periode", options=preset_options, index=preset_index,
            label_visibility="collapsed", key="nl_preset"
        )
        if selected_preset != st.session_state.nl_date_preset:
            st.session_state.nl_date_preset = selected_preset
            new_range = calculate_date_range(selected_preset, today, yesterday)
            if new_range:
                st.session_state.nl_date_range_value = new_range
            st.rerun()

    with col_dato:
        current_range = calculate_date_range(st.session_state.nl_date_preset, today, yesterday) or st.session_state.nl_date_range_value
        date_range = st.date_input("Datoer", value=current_range, label_visibility="collapsed", key="nl_dates")
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            if date_range != current_range:
                st.session_state.nl_date_range_value = date_range
        else:
            start_date = date_range[0] if isinstance(date_range, tuple) else date_range
            end_date = start_date

    # Filter data by date first
    date_mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    df_date_filtered = df[date_mask]

    # Track period changes
    current_period_key = f"nl_{start_date}_{end_date}"
    if 'nl_last_period_key' not in st.session_state:
        st.session_state.nl_last_period_key = current_period_key
    period_changed = st.session_state.nl_last_period_key != current_period_key

    # Initialize session states
    if 'nl_selected_campaigns' not in st.session_state:
        st.session_state.nl_selected_campaigns = None
    if 'nl_selected_emails' not in st.session_state:
        st.session_state.nl_selected_emails = None
    if 'nl_selected_countries' not in st.session_state:
        st.session_state.nl_selected_countries = None
    if 'nl_cb_reset_land' not in st.session_state:
        st.session_state.nl_cb_reset_land = 0
    if 'nl_cb_reset_kamp' not in st.session_state:
        st.session_state.nl_cb_reset_kamp = 0
    if 'nl_cb_reset_email' not in st.session_state:
        st.session_state.nl_cb_reset_email = 0
    if 'nl_ignore_ab' not in st.session_state:
        st.session_state.nl_ignore_ab = True

    if period_changed:
        st.session_state.nl_last_period_key = current_period_key
        st.session_state.nl_selected_campaigns = None
        st.session_state.nl_selected_emails = None
        st.session_state.nl_selected_countries = None

    # Filter options
    all_countries = sorted(df_date_filtered['Country'].unique())
    all_id_campaigns = sorted(df_date_filtered['ID_Campaign'].astype(str).unique())
    email_col = 'Email_Message_Base' if st.session_state.nl_ignore_ab else 'Email_Message_Full'
    all_email_messages = sorted(df_date_filtered[email_col].astype(str).unique())

    # Pre-select all
    if st.session_state.nl_selected_countries is None:
        st.session_state.nl_selected_countries = list(all_countries)
    else:
        st.session_state.nl_selected_countries = [c for c in st.session_state.nl_selected_countries if c in all_countries]

    if st.session_state.nl_selected_campaigns is None:
        st.session_state.nl_selected_campaigns = list(all_id_campaigns)
    else:
        st.session_state.nl_selected_campaigns = [c for c in st.session_state.nl_selected_campaigns if c in all_id_campaigns]

    if st.session_state.nl_selected_emails is None:
        st.session_state.nl_selected_emails = list(all_email_messages)
    else:
        st.session_state.nl_selected_emails = [e for e in st.session_state.nl_selected_emails if e in all_email_messages]

    # Land filter
    with col_land:
        with st.popover("Land", use_container_width=True):
            reset_land = st.session_state.nl_cb_reset_land
            all_land_selected = len(st.session_state.nl_selected_countries) == len(all_countries)
            select_all_land = st.checkbox("Vælg alle", value=all_land_selected, key=f"nl_sel_all_land_{reset_land}")
            
            new_selected = []
            for country in all_countries:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    checked = country in st.session_state.nl_selected_countries
                    if st.checkbox(country, value=checked, key=f"nl_cb_land_{country}_{reset_land}"):
                        new_selected.append(country)
                with only_col:
                    if st.button("Kun", key=f"nl_only_land_{country}_{reset_land}"):
                        st.session_state.nl_selected_countries = [country]
                        st.session_state.nl_cb_reset_land += 1
                        st.rerun()
            
            if select_all_land and not all_land_selected:
                st.session_state.nl_selected_countries = list(all_countries)
                st.session_state.nl_cb_reset_land += 1
                st.rerun()
            elif not select_all_land and all_land_selected:
                st.session_state.nl_selected_countries = []
                st.session_state.nl_cb_reset_land += 1
                st.rerun()
            elif set(new_selected) != set(st.session_state.nl_selected_countries):
                st.session_state.nl_selected_countries = new_selected
                st.session_state.nl_cb_reset_land += 1
                st.rerun()

    # Kampagne filter
    with col_kamp:
        kamp_count = len(st.session_state.nl_selected_campaigns)
        kamp_label = f"Kampagne ({kamp_count})" if kamp_count < len(all_id_campaigns) else "Kampagne"
        with st.popover(kamp_label, use_container_width=True):
            reset_kamp = st.session_state.nl_cb_reset_kamp
            all_kamp_selected = len(st.session_state.nl_selected_campaigns) == len(all_id_campaigns)
            select_all_kamp = st.checkbox("Vælg alle", value=all_kamp_selected, key=f"nl_sel_all_kamp_{reset_kamp}")
            
            search_kamp = st.text_input("Søg", key="nl_search_kamp", placeholder="Søg...", label_visibility="collapsed")
            filtered_campaigns = [c for c in all_id_campaigns if search_kamp.lower() in c.lower()] if search_kamp else all_id_campaigns
            
            new_selected = []
            for campaign in filtered_campaigns:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    checked = campaign in st.session_state.nl_selected_campaigns
                    if st.checkbox(campaign, value=checked, key=f"nl_cb_kamp_{campaign}_{reset_kamp}"):
                        new_selected.append(campaign)
                with only_col:
                    if st.button("Kun", key=f"nl_only_kamp_{campaign}_{reset_kamp}"):
                        st.session_state.nl_selected_campaigns = [campaign]
                        st.session_state.nl_cb_reset_kamp += 1
                        st.rerun()
            
            if select_all_kamp and not all_kamp_selected:
                st.session_state.nl_selected_campaigns = list(all_id_campaigns)
                st.session_state.nl_cb_reset_kamp += 1
                st.rerun()
            elif not select_all_kamp and all_kamp_selected:
                st.session_state.nl_selected_campaigns = []
                st.session_state.nl_cb_reset_kamp += 1
                st.rerun()
            elif set(new_selected) != set(st.session_state.nl_selected_campaigns):
                st.session_state.nl_selected_campaigns = new_selected
                st.session_state.nl_cb_reset_kamp += 1
                st.rerun()

    # Email filter
    with col_email:
        email_count = len(st.session_state.nl_selected_emails)
        email_label = f"Email ({email_count})" if email_count < len(all_email_messages) else "Email"
        with st.popover(email_label, use_container_width=True):
            reset_email = st.session_state.nl_cb_reset_email
            all_email_selected = len(st.session_state.nl_selected_emails) == len(all_email_messages)
            select_all_email = st.checkbox("Vælg alle", value=all_email_selected, key=f"nl_sel_all_email_{reset_email}")
            
            search_email = st.text_input("Søg", key="nl_search_email_input", placeholder="Søg...", label_visibility="collapsed")
            filtered_emails = [e for e in all_email_messages if search_email.lower() in e.lower()] if search_email else all_email_messages
            
            new_selected = []
            for email in filtered_emails:
                cb_col, only_col = st.columns([4, 1])
                with cb_col:
                    checked = email in st.session_state.nl_selected_emails
                    if st.checkbox(email, value=checked, key=f"nl_cb_email_{email}_{reset_email}"):
                        new_selected.append(email)
                with only_col:
                    if st.button("Kun", key=f"nl_only_email_{email}_{reset_email}"):
                        st.session_state.nl_selected_emails = [email]
                        st.session_state.nl_cb_reset_email += 1
                        st.rerun()
            
            if select_all_email and not all_email_selected:
                st.session_state.nl_selected_emails = list(all_email_messages)
                st.session_state.nl_cb_reset_email += 1
                st.rerun()
            elif not select_all_email and all_email_selected:
                st.session_state.nl_selected_emails = []
                st.session_state.nl_cb_reset_email += 1
                st.rerun()
            elif set(new_selected) != set(st.session_state.nl_selected_emails):
                st.session_state.nl_selected_emails = new_selected
                st.session_state.nl_cb_reset_email += 1
                st.rerun()

    # Ignorer A/B
    with col_ab:
        ignore_ab = st.checkbox("Ignorer A/B", value=st.session_state.nl_ignore_ab, key="nl_ignore_ab_cb")
        if ignore_ab != st.session_state.nl_ignore_ab:
            st.session_state.nl_ignore_ab = ignore_ab
            st.session_state.nl_selected_emails = None
            st.session_state.nl_cb_reset_email += 1
            st.rerun()

    # Get selections
    sel_id_campaigns = st.session_state.nl_selected_campaigns
    sel_email_messages = st.session_state.nl_selected_emails
    sel_countries = st.session_state.nl_selected_countries

    # Filter and aggregate
    result = filter_data(df, start_date, end_date, sel_countries, sel_id_campaigns, sel_email_messages, email_col)
    if isinstance(result, tuple):
        current_df, display_pivot_df = result
    else:
        current_df = result
        display_pivot_df = pd.DataFrame()

    # Previous period
    period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    prev_end_date = pd.to_datetime(start_date) - pd.Timedelta(days=1)
    prev_start_date = prev_end_date - pd.Timedelta(days=period_days - 1)

    show_delta = (len(sel_id_campaigns) == len(all_id_campaigns)) and (len(sel_email_messages) == len(all_email_messages))

    prev_df = pd.DataFrame()
    if show_delta and len(sel_countries) > 0:
        prev_mask = (df['Date'] >= pd.to_datetime(prev_start_date)) & (df['Date'] <= pd.to_datetime(prev_end_date))
        prev_temp = df.loc[prev_mask].copy()
        prev_temp = prev_temp[prev_temp['Country'].isin(sel_countries)]
        
        if not prev_temp.empty:
            prev_df = prev_temp.groupby(['Date', 'ID_Campaign', email_col], as_index=False).agg({
                'Total_Received': 'sum', 'Unique_Opens': 'sum', 'Unique_Clicks': 'sum', 'Unsubscribed': 'sum'
            })

    # KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    cur_sent = current_df['Total_Received'].sum() if not current_df.empty else 0
    cur_opens = current_df['Unique_Opens'].sum() if not current_df.empty else 0
    cur_clicks = current_df['Unique_Clicks'].sum() if not current_df.empty else 0
    cur_or = (cur_opens / cur_sent * 100) if cur_sent > 0 else 0
    cur_cr = (cur_clicks / cur_sent * 100) if cur_sent > 0 else 0
    cur_ctr = (cur_clicks / cur_opens * 100) if cur_opens > 0 else 0

    prev_sent = prev_df['Total_Received'].sum() if not prev_df.empty and show_delta else None
    prev_opens = prev_df['Unique_Opens'].sum() if not prev_df.empty and show_delta else None
    prev_clicks = prev_df['Unique_Clicks'].sum() if not prev_df.empty and show_delta else None
    prev_or = (prev_opens / prev_sent * 100) if prev_sent and prev_sent > 0 and show_delta else None
    prev_cr = (prev_clicks / prev_sent * 100) if prev_sent and prev_sent > 0 and show_delta else None
    prev_ctr = (prev_clicks / prev_opens * 100) if prev_opens and prev_opens > 0 and show_delta else None

    show_metric(col1, "Emails Sendt", cur_sent, prev_sent)
    show_metric(col2, "Unikke Opens", cur_opens, prev_opens)
    show_metric(col3, "Unikke Clicks", cur_clicks, prev_clicks)
    show_metric(col4, "Open Rate", cur_or, prev_or, is_percent=True)
    show_metric(col5, "Click Rate", cur_cr, prev_cr, is_percent=True)
    show_metric(col6, "Click Through Rate", cur_ctr, prev_ctr, is_percent=True)

    if not current_df.empty:
        display_df = current_df.copy()
        display_df['Date'] = pd.to_datetime(display_df['Date']).dt.date
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        # Chart data
        chart_df = current_df.groupby(['Date', 'Email_Message'], as_index=False).agg({
            'Total_Received': 'sum', 'Unique_Opens': 'sum', 'Unique_Clicks': 'sum'
        })
        chart_df['Open Rate'] = (chart_df['Unique_Opens'] / chart_df['Total_Received'] * 100).round(1)
        chart_df['Click Rate'] = (chart_df['Unique_Clicks'] / chart_df['Total_Received'] * 100).round(2)
        chart_df['CTR'] = (chart_df['Unique_Clicks'] / chart_df['Unique_Opens'] * 100).round(1)
        chart_df['CTR'] = chart_df['CTR'].fillna(0)
        chart_df = chart_df.sort_values('Date')
        chart_df['Email_Short'] = chart_df['Email_Message'].apply(lambda x: x.split(' - ')[-1] if ' - ' in str(x) else str(x))
        
        # Dot Plot
        fig_dot = go.Figure()
        
        # Open Rate dots (lilla)
        fig_dot.add_trace(
            go.Scatter(
                x=chart_df['Open Rate'],
                y=chart_df['Email_Short'],
                mode='markers',
                name='Open Rate',
                marker=dict(color='#9B7EBD', size=14, symbol='circle'),
                hovertemplate='<b>%{y}</b><br>Open Rate: %{x:.1f}%<extra></extra>'
            )
        )
        
        # Click Rate dots (rosa)
        fig_dot.add_trace(
            go.Scatter(
                x=chart_df['Click Rate'],
                y=chart_df['Email_Short'],
                mode='markers',
                name='Click Rate',
                marker=dict(color='#E8B4CB', size=14, symbol='diamond'),
                hovertemplate='<b>%{y}</b><br>Click Rate: %{x:.1f}%<extra></extra>'
            )
        )
        
        # CTR dots (grøn)
        fig_dot.add_trace(
            go.Scatter(
                x=chart_df['CTR'],
                y=chart_df['Email_Short'],
                mode='markers',
                name='CTR',
                marker=dict(color='#A8E6CF', size=14, symbol='square'),
                hovertemplate='<b>%{y}</b><br>CTR: %{x:.1f}%<extra></extra>'
            )
        )
        
        # Tilføj forbindelseslinjer mellem dots for hver email
        for idx, row in chart_df.iterrows():
            min_val = min(row['Click Rate'], row['CTR'])
            max_val = row['Open Rate']
            fig_dot.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[row['Email_Short'], row['Email_Short']],
                    mode='lines',
                    line=dict(color='rgba(155, 126, 189, 0.2)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        
        chart_height_dot = max(400, len(chart_df) * 35)
        
        fig_dot.update_layout(
            title="",
            showlegend=True,
            height=chart_height_dot,
            margin=dict(l=200, r=50, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(250,245,255,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='closest',
            xaxis=dict(
                title="Procent (%)",
                gridcolor='rgba(212,191,255,0.3)',
                ticksuffix='%',
                zeroline=True,
                zerolinecolor='rgba(212,191,255,0.5)'
            ),
            yaxis=dict(
                title="",
                gridcolor='rgba(212,191,255,0.2)',
                categoryorder='array',
                categoryarray=chart_df['Email_Short'].tolist()[::-1],
                automargin=True,
                ticklabelstandoff=10
            )
        )
        
        st.plotly_chart(fig_dot, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Table
        cols_to_show = ['Date', 'ID_Campaign', 'Email_Message', 'Total_Received', 'Unique_Opens', 'Unique_Clicks', 'Open Rate %', 'Click Rate %', 'Click Through Rate %']
        sorted_df = display_df[cols_to_show].sort_values(by='Date', ascending=False)
        table_height = (len(sorted_df) + 1) * 35 + 3
        
        st.dataframe(
            sorted_df, use_container_width=True, hide_index=True, height=table_height,
            column_config={
                "Date": st.column_config.DateColumn("Dato", width="small"),
                "ID_Campaign": st.column_config.TextColumn("Kampagne", width="medium"),
                "Email_Message": st.column_config.TextColumn("Email", width="large"),
                "Total_Received": st.column_config.NumberColumn("Sendt", format="localized", width="small"),
                "Unique_Opens": st.column_config.NumberColumn("Opens", format="localized", width="small"),
                "Unique_Clicks": st.column_config.NumberColumn("Clicks", format="localized", width="small"),
                "Open Rate %": st.column_config.NumberColumn("Open Rate", format="%.1f%%", width="small"),
                "Click Rate %": st.column_config.NumberColumn("Click Rate", format="%.1f%%", width="small"),
                "Click Through Rate %": st.column_config.NumberColumn("CTR", format="%.1f%%", width="small"),
            }
        )
    else:
        st.warning("Ingen data at vise.")

    if st.button('Opdater Data', key="nl_refresh"):
        st.rerun()

