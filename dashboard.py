
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

# --- CSS: CLASSY UNICORN TEMA ---
import os
css_path = os.path.join(os.path.dirname(__file__), 'style.css')
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# JavaScript til dynamisk checkbox fix
st.markdown("""
    <script>
        function fixCheckboxColors() {
            document.querySelectorAll('label[data-baseweb="checkbox"] > span:first-child').forEach(span => {
                if (span.style.borderColor) {
                    span.style.borderColor = '#D4BFFF';
                }
                const input = span.parentElement.querySelector('input[type="checkbox"]');
                if (input && input.checked) {
                    span.style.backgroundColor = '#9B7EBD';
                    span.style.borderColor = '#9B7EBD';
                } else {
                    span.style.backgroundColor = 'white';
                    span.style.borderColor = '#D4BFFF';
                }
            });
        }
        setTimeout(fixCheckboxColors, 100);
        const observer = new MutationObserver(fixCheckboxColors);
        observer.observe(document.body, { childList: true, subtree: true, attributes: true });
    </script>
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

    st.title("Newsletter Dashboard")
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
    return False

if not check_password():
    st.stop()

# --- HERUNDER STARTER DASHBOARDET ---

st.title("Newsletter Dashboard")

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
    
    # Landekonfiguration: (land, startkolonne-index) - Kun Norden
    country_configs = [
        ('DK', 15),   # Kolonne P (15 i 0-indexed)
        ('SE', 21),   # Kolonne V (21 i 0-indexed) 
        ('NO', 27),   # Kolonne AB (27 i 0-indexed)
        ('FI', 33),   # Kolonne AH (33 i 0-indexed)
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

# Initialize session states
if 'selected_campaigns' not in st.session_state:
    st.session_state.selected_campaigns = None  # None = ikke initialiseret
if 'selected_emails' not in st.session_state:
    st.session_state.selected_emails = None
if 'selected_countries' not in st.session_state:
    st.session_state.selected_countries = None
if 'search_campaign' not in st.session_state:
    st.session_state.search_campaign = ""
if 'search_email' not in st.session_state:
    st.session_state.search_email = ""
if 'search_country' not in st.session_state:
    st.session_state.search_country = ""
# Counter til at resette checkbox keys
if 'cb_reset_land' not in st.session_state:
    st.session_state.cb_reset_land = 0
if 'cb_reset_kamp' not in st.session_state:
    st.session_state.cb_reset_kamp = 0
if 'cb_reset_email' not in st.session_state:
    st.session_state.cb_reset_email = 0

if period_changed:
    st.session_state.last_period_key = current_period_key
    # Nulstil alle filtre når perioden ændres (None = vælg alle)
    st.session_state.selected_campaigns = None
    st.session_state.selected_emails = None
    st.session_state.selected_countries = None

# ALLE filter-options baseret på perioden (uafhængige af hinanden)
all_countries = sorted(df_date_filtered['Country'].unique())
all_id_campaigns = sorted(df_date_filtered['ID_Campaign'].astype(str).unique())
all_email_messages = sorted(df_date_filtered['Email_Message'].astype(str).unique())

# Pre-select alle ved første load eller periode-ændring
if st.session_state.selected_countries is None:
    st.session_state.selected_countries = list(all_countries)
else:
    st.session_state.selected_countries = [c for c in st.session_state.selected_countries if c in all_countries]

if st.session_state.selected_campaigns is None:
    st.session_state.selected_campaigns = list(all_id_campaigns)
else:
    st.session_state.selected_campaigns = [c for c in st.session_state.selected_campaigns if c in all_id_campaigns]

if st.session_state.selected_emails is None:
    st.session_state.selected_emails = list(all_email_messages)
else:
    st.session_state.selected_emails = [e for e in st.session_state.selected_emails if e in all_email_messages]

# Land filter med checkboxes
with col_land:
    land_count = len(st.session_state.selected_countries)
    land_label = f"Land ({land_count})" if land_count < len(all_countries) else "Land"
    with st.popover(land_label, use_container_width=True):
        # Vælg alle checkbox
        all_land_selected = len(st.session_state.selected_countries) == len(all_countries)
        select_all_land = st.checkbox("Vælg alle", value=all_land_selected, key=f"sel_all_land_{st.session_state.cb_reset_land}")
        if select_all_land and not all_land_selected:
            st.session_state.selected_countries = list(all_countries)
            st.session_state.cb_reset_land += 1
            st.rerun()
        elif not select_all_land and all_land_selected:
            st.session_state.selected_countries = []
            st.session_state.cb_reset_land += 1
            st.rerun()
        
        st.markdown("<div style='margin: 0;'></div>", unsafe_allow_html=True)
        reset_land = st.session_state.cb_reset_land
        for country in all_countries:
            checked = country in st.session_state.selected_countries
            if st.checkbox(country, value=checked, key=f"cb_land_{country}_{reset_land}"):
                if country not in st.session_state.selected_countries:
                    st.session_state.selected_countries.append(country)
            else:
                if country in st.session_state.selected_countries:
                    st.session_state.selected_countries.remove(country)

# Kampagne filter med checkboxes
with col_kamp:
    kamp_count = len(st.session_state.selected_campaigns)
    kamp_label = f"Kampagne ({kamp_count})" if kamp_count < len(all_id_campaigns) else "Kampagne"
    with st.popover(kamp_label, use_container_width=True):
        # Vælg alle checkbox
        all_kamp_selected = len(st.session_state.selected_campaigns) == len(all_id_campaigns)
        select_all_kamp = st.checkbox("Vælg alle", value=all_kamp_selected, key=f"sel_all_kamp_{st.session_state.cb_reset_kamp}")
        if select_all_kamp and not all_kamp_selected:
            st.session_state.selected_campaigns = list(all_id_campaigns)
            st.session_state.cb_reset_kamp += 1
            st.rerun()
        elif not select_all_kamp and all_kamp_selected:
            st.session_state.selected_campaigns = []
            st.session_state.cb_reset_kamp += 1
            st.rerun()
        
        st.markdown("<div style='margin: 0;'></div>", unsafe_allow_html=True)
        search_kamp = st.text_input("Søg", key="search_kamp", placeholder="Søg...", label_visibility="collapsed")
        filtered_campaigns = [c for c in all_id_campaigns if search_kamp.lower() in c.lower()] if search_kamp else all_id_campaigns
        
        reset_kamp = st.session_state.cb_reset_kamp
        for campaign in filtered_campaigns:
            checked = campaign in st.session_state.selected_campaigns
            if st.checkbox(campaign, value=checked, key=f"cb_kamp_{campaign}_{reset_kamp}"):
                if campaign not in st.session_state.selected_campaigns:
                    st.session_state.selected_campaigns.append(campaign)
            else:
                if campaign in st.session_state.selected_campaigns:
                    st.session_state.selected_campaigns.remove(campaign)

# Email filter med checkboxes
with col_email:
    email_count = len(st.session_state.selected_emails)
    email_label = f"Email ({email_count})" if email_count < len(all_email_messages) else "Email"
    with st.popover(email_label, use_container_width=True):
        # Vælg alle checkbox
        all_email_selected = len(st.session_state.selected_emails) == len(all_email_messages)
        select_all_email = st.checkbox("Vælg alle", value=all_email_selected, key=f"sel_all_email_{st.session_state.cb_reset_email}")
        if select_all_email and not all_email_selected:
            st.session_state.selected_emails = list(all_email_messages)
            st.session_state.cb_reset_email += 1
            st.rerun()
        elif not select_all_email and all_email_selected:
            st.session_state.selected_emails = []
            st.session_state.cb_reset_email += 1
            st.rerun()
        
        st.markdown("<div style='margin: 0;'></div>", unsafe_allow_html=True)
        search_email = st.text_input("Søg", key="search_email_input", placeholder="Søg...", label_visibility="collapsed")
        filtered_emails = [e for e in all_email_messages if search_email.lower() in e.lower()] if search_email else all_email_messages
        
        reset_email = st.session_state.cb_reset_email
        for email in filtered_emails:
            checked = email in st.session_state.selected_emails
            if st.checkbox(email, value=checked, key=f"cb_email_{email}_{reset_email}"):
                if email not in st.session_state.selected_emails:
                    st.session_state.selected_emails.append(email)
            else:
                if email in st.session_state.selected_emails:
                    st.session_state.selected_emails.remove(email)

# Gem valgte værdier til filter_data
sel_id_campaigns = st.session_state.selected_campaigns

sel_email_messages = st.session_state.selected_emails
sel_countries = st.session_state.selected_countries


# --- DATA FILTRERING OG AGGREGERING ---

def filter_data(dataset, start, end):
    mask = (dataset['Date'] >= pd.to_datetime(start)) & (dataset['Date'] <= pd.to_datetime(end))
    temp_df = dataset.loc[mask].copy()
    
    # Hvis nogen filter er tom liste, returner tom data
    if len(sel_countries) == 0 or len(sel_id_campaigns) == 0 or len(sel_email_messages) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Anvend filtre
    temp_df = temp_df[temp_df['Country'].isin(sel_countries)]
    temp_df = temp_df[temp_df['ID_Campaign'].astype(str).isin(sel_id_campaigns)]
    temp_df = temp_df[temp_df['Email_Message'].astype(str).isin(sel_email_messages)]
    
    if not temp_df.empty:
        # Pivot data så vi får en kolonne per land
        pivot_df = temp_df.pivot_table(
            index=['Date', 'ID_Campaign', 'Email_Message'],
            columns='Country',
            values='Total_Received',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        # Sørg for at alle lande-kolonner eksisterer
        for country in ['DK', 'SE', 'NO', 'FI']:
            if country not in pivot_df.columns:
                pivot_df[country] = 0
        
        # Beregn total
        pivot_df['Total'] = pivot_df['DK'] + pivot_df['SE'] + pivot_df['NO'] + pivot_df['FI']
        
        # Aggreger også for metrics (til KPI cards)
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
        
        return agg_df, pivot_df
        
    return temp_df, pd.DataFrame()

result = filter_data(df, start_date, end_date)
if isinstance(result, tuple):
    current_df, display_pivot_df = result
else:
    current_df = result
    display_pivot_df = pd.DataFrame()


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

if not display_pivot_df.empty:
    display_df = display_pivot_df.copy()
    display_df['Date'] = pd.to_datetime(display_df['Date']).dt.date
    cols_to_show = ['Date', 'ID_Campaign', 'Email_Message', 'Total', 'DK', 'SE', 'NO', 'FI']
    sorted_df = display_df[cols_to_show].sort_values(by='Date', ascending=False)
    
    # Beregn højde baseret på antal rækker (35px per række + 38px header)
    table_height = (len(sorted_df) + 1) * 35 + 3
    
    st.dataframe(
        sorted_df,
        use_container_width=True,
        hide_index=True,
        height=table_height,
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            "ID_Campaign": st.column_config.TextColumn("Kampagne"),
            "Email_Message": st.column_config.TextColumn("Email"),
            "Total": st.column_config.NumberColumn("Total", format="localized"),
            "DK": st.column_config.NumberColumn("DK", format="localized"),
            "SE": st.column_config.NumberColumn("SE", format="localized"),
            "NO": st.column_config.NumberColumn("NO", format="localized"),
            "FI": st.column_config.NumberColumn("FI", format="localized"),
        }
    )
else:
    st.warning("Ingen data at vise.")

if st.button('Opdater Data'):
    st.cache_data.clear()
    st.rerun()




