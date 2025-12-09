
import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import extra_streamlit_components as stx
import datetime
import time

# --- SIDE OPSÆTNING ---
st.set_page_config(
    page_title="Sinful KPI Dashboard", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- CSS: DESIGN TILPASNINGER ---
st.markdown("""
    <style>
        /* 1. Fjern luft i toppen af hovedvinduet */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 5rem;
        }
        
        /* 2. Gør sidebaren smallere (og håndter min-width) */
        section[data-testid="stSidebar"] {
            width: 120px !important;
            min-width: 120px !important;
        }

        /* 3. KOMPAKT OG CENTRERET KNAP */
        [data-testid="stSidebar"] .stButton {
            margin-bottom: 0px !important;
            padding-bottom: 0px !important;
        }

        [data-testid="stSidebar"] button {
            /* Størrelse og luft */
            min-height: 0px !important;
            height: auto !important;
            padding-top: 6px !important;    /* Lidt mere luft foroven/forneden ser pænere ud */
            padding-bottom: 6px !important;
            
            /* CENTRERING AF TEKST (Magien sker her) */
            display: flex !important;
            justify-content: center !important; /* Centrer vandret */
            align-items: center !important;     /* Centrer lodret */
            text-align: center !important;
            width: 100% !important;             /* Fyld hele bredden ud, så ser det pænest ud */
            
            /* Tekst styling */
            line-height: 1.2 !important;
            white-space: nowrap !important;
            font-size: 14px !important;
        }

        /* 4. Fjern rød ramme fra ALLE input felter */
        input:focus, input:active,
        [data-baseweb="input"] input:focus,
        [data-baseweb="select"] > div:focus,
        .stTextInput input:focus,
        .stSelectbox > div > div:focus,
        .stMultiSelect > div > div:focus,
        .stDateInput input:focus {
            border-color: #ccc !important;
            box-shadow: none !important;
            outline: none !important;
        }
        
        [data-baseweb="input"],
        [data-baseweb="select"],
        .stTextInput > div > div,
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stDateInput > div > div {
            border-color: #ccc !important;
        }
        
        /* Specifik for multiselect dropdown */
        .stMultiSelect [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div:focus,
        .stMultiSelect [data-baseweb="select"] > div:focus-within,
        .stMultiSelect [data-baseweb="select"] > div:active,
        [data-baseweb="popover"] {
            border-color: #ccc !important;
            box-shadow: none !important;
            outline: none !important;
        }
        
        /* Target selectbox og multiselect wrapper */
        div[data-baseweb="select"] > div {
            border-color: #ccc !important;
        }
        
        div[data-baseweb="select"] > div:focus-within {
            border-color: #0068c9 !important;
            box-shadow: none !important;
        }
        
        /* Popover knap styling: venstrestil tekst, højrestil pil */
        button[data-testid="stPopoverButton"] {
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            width: 100% !important;
        }
        
        button[data-testid="stPopoverButton"] p {
            text-align: left !important;
            margin: 0 !important;
        }

    </style>
""", unsafe_allow_html=True)

# --- LOGIN LOGIK ---
def check_password():
    cookie_manager = stx.CookieManager(key="main_cookie_manager")
    cookie_val = cookie_manager.get("sinful_auth")

    if st.session_state.get("authenticated", False):
        return True

    if cookie_val == "true":
        st.session_state["authenticated"] = True
        return True

    st.title("Newsletter Dashboard 📧")
    st.markdown("🔒 Log ind")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.form("login_form"):
            password_input = st.text_input("Indtast kodeord:", type="password")
            submit_button = st.form_submit_button("Log Ind")

            if submit_button:
                if password_input == st.secrets["PASSWORD"]:
                    st.session_state["authenticated"] = True
                try:
                    expires = datetime.datetime.now() + datetime.timedelta(days=7)
                    cookie_manager.set("sinful_auth", "true", expires_at=expires)
                except Exception:
                    pass
                
                st.success("Login godkendt! Vent venligst...")
                time.sleep(2)
                st.rerun()
            else:
                st.error("😕 Forkert kodeord")
    return False

if not check_password():
    st.stop()

# --- HERUNDER STARTER DASHBOARDET ---

st.title("Newsletter Dashboard 📧")

# Log ud knap i menuen (sidebar er nu kun til log ud)
with st.sidebar:
    if st.button("Log Ud"):
        cookie_manager = stx.CookieManager(key="logout_manager")
        cookie_manager.delete("sinful_auth")
        st.session_state["authenticated"] = False
        st.info("Logger ud...")
        time.sleep(1)
        st.rerun()

# --- DATA INDLÆSNING ---
@st.cache_data(ttl=600)
def load_google_sheet_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        raw_df = conn.read(skiprows=2)  # Skip begge header-rækker
    except Exception:
        return pd.DataFrame()
    
    # Landekonfiguration: (land, startkolonne-index)
    # DK starter i kolonne P (index 15 i 0-indexed)
    country_configs = [
        ('DK', 15),   # Kolonne P (15 i 0-indexed)
        ('SE', 21),   # Kolonne V (21 i 0-indexed) 
        ('NO', 27),   # Kolonne AB (27 i 0-indexed)
        ('FI', 33),   # Kolonne AH (33 i 0-indexed)
        ('FR', 39),   # Kolonne AN (39 i 0-indexed)
        ('UK', 45),   # Kolonne AT (45 i 0-indexed)
        ('DE', 51),   # Kolonne AZ (51 i 0-indexed)
        ('AT', 57),   # Kolonne BF (57 i 0-indexed)
        ('NL', 63),   # Kolonne BL (63 i 0-indexed)
        ('BE', 69),   # Kolonne BR (69 i 0-indexed)
        ('CH', 75),   # Kolonne BX (75 i 0-indexed)
    ]
    
    all_country_data = []
    
    for country_code, start_col in country_configs:
        try:
            # Opret DataFrame for dette land
            country_df = pd.DataFrame()
            
            # Fælles kolonner (0-8)
            country_df['Send Year'] = raw_df.iloc[:, 0]
            country_df['Send Month'] = raw_df.iloc[:, 1]
            country_df['Send Day'] = raw_df.iloc[:, 2]
            country_df['Send Time'] = raw_df.iloc[:, 3]
            country_df['Number'] = raw_df.iloc[:, 4]
            country_df['Campaign Name'] = raw_df.iloc[:, 5]
            country_df['Email'] = raw_df.iloc[:, 6]
            country_df['Message'] = raw_df.iloc[:, 7]
            country_df['Variant'] = raw_df.iloc[:, 8]
            
            # Metrics for dette land
            country_df['Total_Received'] = raw_df.iloc[:, start_col + 0]  # Received Email
            country_df['Total_Opens_Raw'] = raw_df.iloc[:, start_col + 1]  # Total Opens
            country_df['Unique_Opens'] = raw_df.iloc[:, start_col + 2]     # Unique Opens
            country_df['Total_Clicks_Raw'] = raw_df.iloc[:, start_col + 3] # Total Clicks
            country_df['Unique_Clicks'] = raw_df.iloc[:, start_col + 4]    # Unique Clicks
            country_df['Unsubscribed'] = raw_df.iloc[:, start_col + 5]     # Unsubscribed
            
            # Tilføj landekode
            country_df['Country'] = country_code
            
            all_country_data.append(country_df)
        except Exception:
            continue
    
    if not all_country_data:
        return pd.DataFrame()

    # Kombiner alle lande
    df = pd.concat(all_country_data, ignore_index=True)
    
    # Opret dato
    df['Date'] = pd.to_datetime(
        df['Send Year'].astype(str) + '-' + 
        df['Send Month'].astype(str) + '-' + 
        df['Send Day'].astype(str), 
        errors='coerce'
    )
    df = df.dropna(subset=['Date'])

    # Konverter numeriske kolonner
    numeric_cols = ['Total_Received', 'Unique_Opens', 'Unique_Clicks', 'Unsubscribed']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.replace('.', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Beregn rates
    df['Open Rate %'] = df.apply(lambda x: (x['Unique_Opens'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1)
    df['Click Rate %'] = df.apply(lambda x: (x['Unique_Clicks'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1)
    df['Click Through Rate %'] = df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    # Kombinerede kolonner til filtrering
    df['ID_Campaign'] = df['Number'].astype(str) + ' - ' + df['Campaign Name'].astype(str)
    df['Email_Message'] = df['Email'].astype(str) + ' - ' + df['Message'].astype(str)
    
    return df

try:
    with st.spinner('Henter data...'):
        df = load_google_sheet_data()
    if df.empty:
        st.error("Kunne ikke hente data. Tjek Secrets.")
        st.stop()
except Exception as e:
    st.error(f"Fejl: {e}")
    st.stop()


# --- FILTRE & DATO ---

today = datetime.date.today()
default_start = today - datetime.timedelta(days=30)
default_end = today

# Alle filtre på én linje: Periode, Land, Kampagne, Email
col_dato, col_land, col_kamp, col_email = st.columns(4)

with col_dato:
    date_range = st.date_input(
        "Periode",
        value=(default_start, default_end),
        label_visibility="collapsed"
    )
    # Håndter at brugeren kun har valgt én dato
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range[0] if isinstance(date_range, tuple) else date_range
        end_date = start_date

# Filtrer først efter dato - så dropdowns kun viser data fra valgt periode
date_mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
df_date_filtered = df[date_mask]

# Track perioden - nulstil filtre når perioden ændres
current_period_key = f"{start_date}_{end_date}"
if 'last_period_key' not in st.session_state:
    st.session_state.last_period_key = current_period_key

period_changed = st.session_state.last_period_key != current_period_key
if period_changed:
    st.session_state.last_period_key = current_period_key
    # Nulstil alle filtre når perioden ændres
    st.session_state.selected_campaigns = []
    st.session_state.selected_emails = []
    st.session_state.selected_countries = []

# Initialize session states
if 'selected_campaigns' not in st.session_state:
    st.session_state.selected_campaigns = []
if 'selected_emails' not in st.session_state:
    st.session_state.selected_emails = []
if 'selected_countries' not in st.session_state:
    st.session_state.selected_countries = []
if 'search_campaign' not in st.session_state:
    st.session_state.search_campaign = ""
if 'search_email' not in st.session_state:
    st.session_state.search_email = ""
if 'search_country' not in st.session_state:
    st.session_state.search_country = ""

# Land filter først (for at påvirke tilgængelige kampagner)
all_countries = sorted(df_date_filtered['Country'].unique())

# Pre-select alle lande hvis tom (efter periode-ændring eller første gang)
if not st.session_state.selected_countries:
    st.session_state.selected_countries = list(all_countries)

# Synkroniser: fjern lande der ikke findes i perioden
st.session_state.selected_countries = [c for c in st.session_state.selected_countries if c in all_countries]

# Hvis alle lande blev fjernet, vælg alle igen
if not st.session_state.selected_countries:
    st.session_state.selected_countries = list(all_countries)

# Filtrer på land hvis ikke alle er valgt
if len(st.session_state.selected_countries) < len(all_countries):
    df_land_filtered = df_date_filtered[df_date_filtered['Country'].isin(st.session_state.selected_countries)]
else:
    df_land_filtered = df_date_filtered

# Kampagne filter (baseret på dato + land)
all_id_campaigns = sorted(df_land_filtered['ID_Campaign'].astype(str).unique())

# Pre-select alle kampagner hvis tom
if not st.session_state.selected_campaigns:
    st.session_state.selected_campaigns = list(all_id_campaigns)

# Synkroniser: fjern kampagner der ikke findes i filtreret data
st.session_state.selected_campaigns = [c for c in st.session_state.selected_campaigns if c in all_id_campaigns]

# Hvis alle kampagner blev fjernet, vælg alle igen
if not st.session_state.selected_campaigns:
    st.session_state.selected_campaigns = list(all_id_campaigns)

# Land filter
with col_land:
    with st.popover("Land ▾", use_container_width=True):
        all_selected = len(st.session_state.selected_countries) == len(all_countries)
        if st.checkbox("Vælg alle", value=all_selected, key="select_all_countries_cb"):
            if not all_selected:
                st.session_state.selected_countries = list(all_countries)
                st.rerun()
        else:
            if all_selected:
                st.session_state.selected_countries = []
                st.rerun()
        
        search_term = st.text_input("🔍 Søg", key="search_country", label_visibility="collapsed", placeholder="Søg...")
        filtered_countries = [c for c in all_countries if search_term.lower() in c.lower()] if search_term else all_countries
        
        for country in filtered_countries:
            is_selected = country in st.session_state.selected_countries
            if st.checkbox(country, value=is_selected, key=f"country_{country}"):
                if country not in st.session_state.selected_countries:
                    st.session_state.selected_countries.append(country)
            else:
                if country in st.session_state.selected_countries:
                    st.session_state.selected_countries.remove(country)

# Kampagne filter
with col_kamp:
    with st.popover("Kampagne ▾", use_container_width=True):
        all_selected = len(st.session_state.selected_campaigns) == len(all_id_campaigns)
        if st.checkbox("Vælg alle", value=all_selected, key="select_all_campaigns_cb"):
            if not all_selected:
                st.session_state.selected_campaigns = list(all_id_campaigns)
                st.rerun()
        else:
            if all_selected:
                st.session_state.selected_campaigns = []
                st.rerun()
        
        search_term = st.text_input("🔍 Søg", key="search_campaign", label_visibility="collapsed", placeholder="Søg...")
        filtered_campaigns = [c for c in all_id_campaigns if search_term.lower() in c.lower()] if search_term else all_id_campaigns
        
        for campaign in filtered_campaigns:
            is_selected = campaign in st.session_state.selected_campaigns
            if st.checkbox(campaign, value=is_selected, key=f"campaign_{campaign}"):
                if campaign not in st.session_state.selected_campaigns:
                    st.session_state.selected_campaigns.append(campaign)
            else:
                if campaign in st.session_state.selected_campaigns:
                    st.session_state.selected_campaigns.remove(campaign)

sel_id_campaigns = st.session_state.selected_campaigns

# Email filter (afhængig af valgt kampagne, land OG dato)
if len(st.session_state.selected_campaigns) < len(all_id_campaigns):
    filtered_for_email = df_land_filtered[df_land_filtered['ID_Campaign'].astype(str).isin(sel_id_campaigns)]
else:
    filtered_for_email = df_land_filtered

all_email_messages = sorted(filtered_for_email['Email_Message'].astype(str).unique())

if not st.session_state.selected_emails:
    st.session_state.selected_emails = list(all_email_messages)

st.session_state.selected_emails = [e for e in st.session_state.selected_emails if e in all_email_messages]

if not st.session_state.selected_emails:
    st.session_state.selected_emails = list(all_email_messages)

# Email filter
with col_email:
    with st.popover("Email ▾", use_container_width=True):
        all_selected = len(st.session_state.selected_emails) == len(all_email_messages)
        if st.checkbox("Vælg alle", value=all_selected, key="select_all_emails_cb"):
            if not all_selected:
                st.session_state.selected_emails = list(all_email_messages)
                st.rerun()
        else:
            if all_selected:
                st.session_state.selected_emails = []
                st.rerun()
        
        search_term = st.text_input("🔍 Søg", key="search_email", label_visibility="collapsed", placeholder="Søg...")
        filtered_emails = [e for e in all_email_messages if search_term.lower() in e.lower()] if search_term else all_email_messages
        
        for email in filtered_emails:
            is_selected = email in st.session_state.selected_emails
            if st.checkbox(email, value=is_selected, key=f"email_{email}"):
                if email not in st.session_state.selected_emails:
                    st.session_state.selected_emails.append(email)
            else:
                if email in st.session_state.selected_emails:
                    st.session_state.selected_emails.remove(email)

sel_email_messages = st.session_state.selected_emails
sel_countries = st.session_state.selected_countries


# --- DATA FILTRERING OG AGGREGERING ---

def filter_data(dataset, start, end):
    mask = (dataset['Date'] >= pd.to_datetime(start)) & (dataset['Date'] <= pd.to_datetime(end))
    temp_df = dataset.loc[mask].copy()
    
    # Anvend filtre
    if sel_countries:
        temp_df = temp_df[temp_df['Country'].isin(sel_countries)]
    
    if sel_id_campaigns:
        temp_df = temp_df[temp_df['ID_Campaign'].astype(str).isin(sel_id_campaigns)]
    
    if sel_email_messages:
        temp_df = temp_df[temp_df['Email_Message'].astype(str).isin(sel_email_messages)]
    
    # Aggreger data på tværs af lande
    # Gruppér på Date, Campaign, Email og summer metrics
    if not temp_df.empty:
        agg_df = temp_df.groupby(['Date', 'ID_Campaign', 'Email_Message'], as_index=False).agg({
            'Total_Received': 'sum',
            'Unique_Opens': 'sum',
            'Unique_Clicks': 'sum',
            'Unsubscribed': 'sum'
        })
        
        # Genberegn rates baseret på aggregerede tal
        agg_df['Open Rate %'] = agg_df.apply(
            lambda x: (x['Unique_Opens'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1
        )
        agg_df['Click Rate %'] = agg_df.apply(
            lambda x: (x['Unique_Clicks'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1
        )
        agg_df['Click Through Rate %'] = agg_df.apply(
            lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1
        )
        
        return agg_df
        
    return temp_df

current_df = filter_data(df, start_date, end_date)


# --- VISUALISERING ---
col1, col2, col3, col4, col5, col6 = st.columns(6)

def show_metric(col, label, current_val, is_percent=False):
    # Formater hovedværdi
    if is_percent:
        val_fmt = f"{current_val:.1f}%"
    else:
        # Kompakt format: K for tusind, M for million
        if current_val >= 1_000_000:
            val_fmt = f"{current_val / 1_000_000:.1f}M"
        elif current_val >= 1_000:
            val_fmt = f"{current_val / 1_000:.0f}K"
        else:
            val_fmt = f"{current_val:.0f}"
    
    col.metric(label, val_fmt)

cur_sent = current_df['Total_Received'].sum() if not current_df.empty else 0
cur_opens = current_df['Unique_Opens'].sum() if not current_df.empty else 0
cur_clicks = current_df['Unique_Clicks'].sum() if not current_df.empty else 0
cur_or = current_df['Open Rate %'].mean() if not current_df.empty else 0
cur_cr = current_df['Click Rate %'].mean() if not current_df.empty else 0
cur_ctr = (cur_clicks / cur_opens * 100) if cur_opens > 0 else 0

show_metric(col1, "Emails Sendt", cur_sent)
show_metric(col2, "Unikke Opens", cur_opens)
show_metric(col3, "Unikke Clicks", cur_clicks)
show_metric(col4, "Open Rate", cur_or, is_percent=True)
show_metric(col5, "Click Rate", cur_cr, is_percent=True)
show_metric(col6, "Click Through Rate", cur_ctr, is_percent=True)

if not current_df.empty:
    display_df = current_df.copy()
    display_df['Date'] = display_df['Date'].dt.date
    cols_to_show = ['Date', 'ID_Campaign', 'Email_Message', 'Total_Received', 'Unique_Opens', 'Unique_Clicks', 'Open Rate %', 'Click Rate %', 'Click Through Rate %']
    st.dataframe(
        display_df[cols_to_show].sort_values(by='Date', ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", width="small"),
            "ID_Campaign": st.column_config.TextColumn("Kampagne", width="medium"),
            "Email_Message": st.column_config.TextColumn("Email", width="large"),
            "Total_Received": st.column_config.NumberColumn("Emails Sendt", format="%d", width="small"),
            "Unique_Opens": st.column_config.NumberColumn("Unikke Opens", format="%d", width="small"),
            "Unique_Clicks": st.column_config.NumberColumn("Unikke Clicks", format="%d", width="small"),
            "Open Rate %": st.column_config.NumberColumn("Open Rate", format="%.1f%%", width="small"),
            "Click Rate %": st.column_config.NumberColumn("Click Rate", format="%.2f%%", width="small"),
            "Click Through Rate %": st.column_config.NumberColumn("CTR", format="%.1f%%", width="small"),
        }
    )
else:
    st.warning("Ingen data at vise.")

if st.button('🔄 Opdater Data'):
    st.cache_data.clear()
    st.rerun()




