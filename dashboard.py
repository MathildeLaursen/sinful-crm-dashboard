
import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import extra_streamlit_components as stx
import datetime
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            document.querySelectorAll('label[data-baseweb="checkbox"]').forEach(label => {
                const span = label.querySelector('span:first-child');
                const isChecked = label.getAttribute('aria-checked') === 'true';
                const svg = span ? span.querySelector('svg') : null;
                
                if (span) {
                    // Sæt baggrund og border baseret på checked state
                    span.style.setProperty('background-color', isChecked ? '#9B7EBD' : 'white', 'important');
                    span.style.setProperty('border-color', isChecked ? '#9B7EBD' : '#D4BFFF', 'important');
                }
                
                // Fix SVG checkmark visibility
                if (svg) {
                    svg.style.setProperty('visibility', isChecked ? 'visible' : 'hidden', 'important');
                    svg.style.setProperty('opacity', isChecked ? '1' : '0', 'important');
                    
                    // Style polyline (fluebenet)
                    const polyline = svg.querySelector('polyline');
                    if (polyline) {
                        polyline.style.setProperty('stroke', 'white', 'important');
                        polyline.style.setProperty('stroke-width', '2', 'important');
                    }
                }
            });
        }
        
        // Kør flere gange for at fange dynamisk indhold
        setTimeout(fixCheckboxColors, 100);
        setTimeout(fixCheckboxColors, 300);
        setTimeout(fixCheckboxColors, 600);
        setTimeout(fixCheckboxColors, 1000);
        
        // Observer for at fange ændringer
        const observer = new MutationObserver(() => {
            setTimeout(fixCheckboxColors, 50);
        });
        observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['aria-checked'] });
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

# MIDLERTIDIGT DEAKTIVERET - husk at aktivere igen!
# if not check_password():
#     st.stop()

# --- HERUNDER STARTER DASHBOARDET ---

# Titel
st.title("Newsletter Dashboard")

# --- DATA INDLÆSNING ---
@st.cache_data(ttl=600)
def load_google_sheet_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        raw_df = conn.read(skiprows=2)  # Skip begge header-rækker
    except Exception:
        return pd.DataFrame()
    
    # Landekonfiguration: (land, startkolonne-index) - Alle lande
    country_configs = [
        ('DK', 15),   # Kolonne P
        ('SE', 21),   # Kolonne V
        ('NO', 27),   # Kolonne AB
        ('FI', 33),   # Kolonne AH
        ('FR', 39),   # Kolonne AN
        ('UK', 45),   # Kolonne AT
        ('DE', 51),   # Kolonne AZ
        ('AT', 57),   # Kolonne BF
        ('NL', 63),   # Kolonne BL
        ('BE', 69),   # Kolonne BR
        ('CH', 75),   # Kolonne BX
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
            # Fjern tusind-separatorer (komma og anførselstegn), behold decimalpunkt
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Beregn rates
    df['Open Rate %'] = df.apply(lambda x: (x['Unique_Opens'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1)
    df['Click Rate %'] = df.apply(lambda x: (x['Unique_Clicks'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1)
    df['Click Through Rate %'] = df.apply(lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1)
    
    # Kombinerede kolonner til filtrering
    df['ID_Campaign'] = df['Number'].astype(str) + ' - ' + df['Campaign Name'].astype(str)
    # Email_Message UDEN variant (til aggregering)
    df['Email_Message_Base'] = df['Email'].astype(str) + ' - ' + df['Message'].astype(str)
    # Email_Message MED A/B variant hvis den findes
    df['Email_Message_Full'] = df.apply(
        lambda x: f"{x['Email']} - {x['Message']} - {x['Variant']}" 
        if pd.notna(x['Variant']) and str(x['Variant']).strip() not in ['', 'nan', 'None'] 
        else f"{x['Email']} - {x['Message']}", 
        axis=1
    )
    
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
yesterday = today - datetime.timedelta(days=1)

# Helper til at beregne kvartal
def get_quarter_start(date):
    quarter = (date.month - 1) // 3
    return datetime.date(date.year, quarter * 3 + 1, 1)

# Session state for dato
if 'date_preset' not in st.session_state:
    st.session_state.date_preset = "Sidste 30 dage"
if 'date_range_value' not in st.session_state:
    st.session_state.date_range_value = (today - datetime.timedelta(days=30), yesterday)

# Beregn datoer baseret på preset (alle ekskl. dags dato)
def calculate_date_range(preset):
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

# Preset muligheder
preset_options = [
    "Sidste 7 dage",
    "Sidste 30 dage", 
    "Denne måned",
    "Dette kvartal",
    "I år",
    "Sidste måned",
    "Sidste kvartal",
]

# Layout: Preset, Kalender, Land, Kampagne, Email, Ignorer A/B
col_preset, col_dato, col_land, col_kamp, col_email, col_ab = st.columns([1.0, 1.4, 1, 1, 1, 1])

with col_preset:
    preset_index = preset_options.index(st.session_state.date_preset) if st.session_state.date_preset in preset_options else 1
    
    selected_preset = st.selectbox(
        "Periode",
        options=preset_options,
        index=preset_index,
        label_visibility="collapsed"
    )
    
    # Opdater hvis preset ændres
    if selected_preset != st.session_state.date_preset:
        st.session_state.date_preset = selected_preset
        new_range = calculate_date_range(selected_preset)
        if new_range:
            st.session_state.date_range_value = new_range
        st.rerun()

with col_dato:
    # Beregn aktuel range
    current_range = calculate_date_range(st.session_state.date_preset) or st.session_state.date_range_value
    
    date_range = st.date_input(
        "Datoer",
        value=current_range,
        label_visibility="collapsed"
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        # Gem hvis manuelt ændret
        if date_range != current_range:
            st.session_state.date_range_value = date_range
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
if 'ignore_ab' not in st.session_state:
    st.session_state.ignore_ab = True

if period_changed:
    st.session_state.last_period_key = current_period_key
    # Nulstil alle filtre når perioden ændres (None = vælg alle)
    st.session_state.selected_campaigns = None
    st.session_state.selected_emails = None
    st.session_state.selected_countries = None

# ALLE filter-options baseret på perioden (uafhængige af hinanden)
all_countries = sorted(df_date_filtered['Country'].unique())
all_id_campaigns = sorted(df_date_filtered['ID_Campaign'].astype(str).unique())
# Brug Email_Message_Base hvis ignore_ab, ellers Email_Message_Full
email_col = 'Email_Message_Base' if st.session_state.ignore_ab else 'Email_Message_Full'
all_email_messages = sorted(df_date_filtered[email_col].astype(str).unique())

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
    land_label = "Land"
    with st.popover(land_label, use_container_width=True):
        reset_land = st.session_state.cb_reset_land
        all_land_selected = len(st.session_state.selected_countries) == len(all_countries)
        
        # Vælg alle checkbox
        select_all_land = st.checkbox("Vælg alle", value=all_land_selected, key=f"sel_all_land_{reset_land}")
        
        st.markdown("<div style='margin: 0;'></div>", unsafe_allow_html=True)
        
        # Individuelle land checkboxes
        new_selected = []
        for country in all_countries:
            checked = country in st.session_state.selected_countries
            if st.checkbox(country, value=checked, key=f"cb_land_{country}_{reset_land}"):
                new_selected.append(country)
        
        # Håndter "Vælg alle" klik
        if select_all_land and not all_land_selected:
            st.session_state.selected_countries = list(all_countries)
            st.session_state.cb_reset_land += 1
            st.rerun()
        elif not select_all_land and all_land_selected:
            st.session_state.selected_countries = []
            st.session_state.cb_reset_land += 1
            st.rerun()
        # Opdater baseret på individuelle checkboxes
        elif set(new_selected) != set(st.session_state.selected_countries):
            st.session_state.selected_countries = new_selected
            st.session_state.cb_reset_land += 1
            st.rerun()

# Kampagne filter med checkboxes
with col_kamp:
    kamp_count = len(st.session_state.selected_campaigns)
    kamp_label = f"Kampagne ({kamp_count})" if kamp_count < len(all_id_campaigns) else "Kampagne"
    with st.popover(kamp_label, use_container_width=True):
        reset_kamp = st.session_state.cb_reset_kamp
        all_kamp_selected = len(st.session_state.selected_campaigns) == len(all_id_campaigns)
        
        # Vælg alle checkbox
        select_all_kamp = st.checkbox("Vælg alle", value=all_kamp_selected, key=f"sel_all_kamp_{reset_kamp}")
        
        st.markdown("<div style='margin: 0;'></div>", unsafe_allow_html=True)
        search_kamp = st.text_input("Søg", key="search_kamp", placeholder="Søg...", label_visibility="collapsed")
        filtered_campaigns = [c for c in all_id_campaigns if search_kamp.lower() in c.lower()] if search_kamp else all_id_campaigns
        
        # Individuelle kampagne checkboxes
        new_selected = []
        for campaign in filtered_campaigns:
            checked = campaign in st.session_state.selected_campaigns
            if st.checkbox(campaign, value=checked, key=f"cb_kamp_{campaign}_{reset_kamp}"):
                new_selected.append(campaign)
        
        # Håndter "Vælg alle" klik
        if select_all_kamp and not all_kamp_selected:
            st.session_state.selected_campaigns = list(all_id_campaigns)
            st.session_state.cb_reset_kamp += 1
            st.rerun()
        elif not select_all_kamp and all_kamp_selected:
            st.session_state.selected_campaigns = []
            st.session_state.cb_reset_kamp += 1
            st.rerun()
        # Opdater baseret på individuelle checkboxes
        elif set(new_selected) != set(st.session_state.selected_campaigns):
            st.session_state.selected_campaigns = new_selected
            st.session_state.cb_reset_kamp += 1
            st.rerun()

# Email filter med checkboxes
with col_email:
    email_count = len(st.session_state.selected_emails)
    email_label = f"Email ({email_count})" if email_count < len(all_email_messages) else "Email"
    with st.popover(email_label, use_container_width=True):
        reset_email = st.session_state.cb_reset_email
        all_email_selected = len(st.session_state.selected_emails) == len(all_email_messages)
        
        # Vælg alle checkbox
        select_all_email = st.checkbox("Vælg alle", value=all_email_selected, key=f"sel_all_email_{reset_email}")
        
        st.markdown("<div style='margin: 0;'></div>", unsafe_allow_html=True)
        search_email = st.text_input("Søg", key="search_email_input", placeholder="Søg...", label_visibility="collapsed")
        filtered_emails = [e for e in all_email_messages if search_email.lower() in e.lower()] if search_email else all_email_messages
        
        # Individuelle email checkboxes
        new_selected = []
        for email in filtered_emails:
            checked = email in st.session_state.selected_emails
            if st.checkbox(email, value=checked, key=f"cb_email_{email}_{reset_email}"):
                new_selected.append(email)
        
        # Håndter "Vælg alle" klik
        if select_all_email and not all_email_selected:
            st.session_state.selected_emails = list(all_email_messages)
            st.session_state.cb_reset_email += 1
            st.rerun()
        elif not select_all_email and all_email_selected:
            st.session_state.selected_emails = []
            st.session_state.cb_reset_email += 1
            st.rerun()
        # Opdater baseret på individuelle checkboxes
        elif set(new_selected) != set(st.session_state.selected_emails):
            st.session_state.selected_emails = new_selected
            st.session_state.cb_reset_email += 1
            st.rerun()

# Ignorer A/B checkbox
with col_ab:
    ignore_ab = st.checkbox("Ignorer A/B", value=st.session_state.ignore_ab, key="ignore_ab_cb")
    if ignore_ab != st.session_state.ignore_ab:
        st.session_state.ignore_ab = ignore_ab
        st.session_state.selected_emails = None  # Reset email filter
        st.session_state.cb_reset_email += 1
        st.rerun()

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
    temp_df = temp_df[temp_df[email_col].astype(str).isin(sel_email_messages)]
    
    if not temp_df.empty:
        # Pivot data så vi får en kolonne per land
        pivot_df = temp_df.pivot_table(
            index=['Date', 'ID_Campaign', email_col],
            columns='Country',
            values='Total_Received',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        # Sørg for at alle lande-kolonner eksisterer
        all_countries = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK', 'DE', 'AT', 'NL', 'BE', 'CH']
        for country in all_countries:
            if country not in pivot_df.columns:
                pivot_df[country] = 0
        
        # Beregn total
        pivot_df['Total'] = sum(pivot_df[c] for c in all_countries)
        
        # Aggreger også for metrics (til KPI cards)
        agg_df = temp_df.groupby(['Date', 'ID_Campaign', email_col], as_index=False).agg({
            'Total_Received': 'sum',
            'Unique_Opens': 'sum',
            'Unique_Clicks': 'sum',
            'Unsubscribed': 'sum'
        })
        
        # Omdøb email kolonne til standard navn for visning
        agg_df = agg_df.rename(columns={email_col: 'Email_Message'})
        
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

# Beregn foregående periode (samme længde som valgt periode)
period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
prev_end_date = pd.to_datetime(start_date) - pd.Timedelta(days=1)
prev_start_date = prev_end_date - pd.Timedelta(days=period_days - 1)

# Tjek om vi skal vise % ændring (kun hvis ALLE kampagner og emails er valgt)
show_delta = (len(sel_id_campaigns) == len(all_id_campaigns)) and (len(sel_email_messages) == len(all_email_messages))

# Hent data for foregående periode - kun filtreret på lande (ikke kampagner/emails)
prev_df = pd.DataFrame()
if show_delta and len(sel_countries) > 0:
    prev_mask = (df['Date'] >= pd.to_datetime(prev_start_date)) & (df['Date'] <= pd.to_datetime(prev_end_date))
    prev_temp = df.loc[prev_mask].copy()
    prev_temp = prev_temp[prev_temp['Country'].isin(sel_countries)]
    
    if not prev_temp.empty:
        prev_df = prev_temp.groupby(['Date', 'ID_Campaign', email_col], as_index=False).agg({
            'Total_Received': 'sum',
            'Unique_Opens': 'sum',
            'Unique_Clicks': 'sum',
            'Unsubscribed': 'sum'
        })
        prev_df['Open Rate %'] = prev_df.apply(
            lambda x: (x['Unique_Opens'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1
        )
        prev_df['Click Rate %'] = prev_df.apply(
            lambda x: (x['Unique_Clicks'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1
        )
        prev_df['Click Through Rate %'] = prev_df.apply(
            lambda x: (x['Unique_Clicks'] / x['Unique_Opens'] * 100) if x['Unique_Opens'] > 0 else 0, axis=1
        )

# --- VISUALISERING ---
col1, col2, col3, col4, col5, col6 = st.columns(6)

def show_metric(col, label, current_val, prev_val=None, is_percent=False):
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
    
    # Beregn delta hvis prev_val er givet
    if prev_val is not None and prev_val != 0:
        pct_change = ((current_val - prev_val) / prev_val) * 100
        delta_str = f"{pct_change:+.1f}%"
        col.metric(label, val_fmt, delta=delta_str)
    else:
        col.metric(label, val_fmt)

cur_sent = current_df['Total_Received'].sum() if not current_df.empty else 0
cur_opens = current_df['Unique_Opens'].sum() if not current_df.empty else 0
cur_clicks = current_df['Unique_Clicks'].sum() if not current_df.empty else 0
# Beregn rates fra summerede tal (vægtet korrekt)
cur_or = (cur_opens / cur_sent * 100) if cur_sent > 0 else 0
cur_cr = (cur_clicks / cur_sent * 100) if cur_sent > 0 else 0
cur_ctr = (cur_clicks / cur_opens * 100) if cur_opens > 0 else 0

# Foregående periode værdier
prev_sent = prev_df['Total_Received'].sum() if not prev_df.empty and show_delta else None
prev_opens = prev_df['Unique_Opens'].sum() if not prev_df.empty and show_delta else None
prev_clicks = prev_df['Unique_Clicks'].sum() if not prev_df.empty and show_delta else None
# Beregn rates fra summerede tal (vægtet korrekt)
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
    
    # Lidt luft over grafen
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    # --- GRAF: Open Rate & Click Rate per email ---
    # Aggreger per Email_Message for grafen
    chart_df = current_df.groupby(['Date', 'Email_Message'], as_index=False).agg({
        'Total_Received': 'sum',
        'Unique_Opens': 'sum',
        'Unique_Clicks': 'sum'
    })
    chart_df['Open Rate'] = (chart_df['Unique_Opens'] / chart_df['Total_Received'] * 100).round(1)
    chart_df['Click Rate'] = (chart_df['Unique_Clicks'] / chart_df['Total_Received'] * 100).round(2)
    chart_df = chart_df.sort_values('Date')
    
    # Kort email label til grafen (kun Message del)
    chart_df['Email_Short'] = chart_df['Email_Message'].apply(lambda x: x.split(' - ')[-1] if ' - ' in str(x) else str(x))
    
    # Opret graf med to y-akser
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Open Rate søjler (venstre y-akse)
    fig.add_trace(
        go.Bar(
            x=chart_df['Email_Short'], 
            y=chart_df['Open Rate'],
            name='Open Rate',
            marker_color='#9B7EBD',
            text=chart_df['Open Rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            textfont=dict(size=16),
            offsetgroup=0
        ),
        secondary_y=False
    )
    
    # Click Rate søjler (højre y-akse)
    fig.add_trace(
        go.Bar(
            x=chart_df['Email_Short'], 
            y=chart_df['Click Rate'],
            name='Click Rate',
            marker_color='#E8B4CB',
            text=chart_df['Click Rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            textfont=dict(size=12),
            offsetgroup=1
        ),
        secondary_y=True
    )
    
    # Layout styling (unicorn theme)
    fig.update_layout(
        title="",
        showlegend=True,
        height=350,
        margin=dict(l=50, r=50, t=50, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(250,245,255,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        barmode='group',
        bargap=0.3,
        bargroupgap=0.1
    )
    
    # Y-akser styling - tilføj 15% ekstra plads i toppen for labels
    max_open = chart_df['Open Rate'].max() if not chart_df.empty else 50
    max_click = chart_df['Click Rate'].max() if not chart_df.empty else 5
    
    fig.update_yaxes(
        title_text="Open Rate %",
        secondary_y=False,
        gridcolor='rgba(212,191,255,0.3)',
        ticksuffix='%',
        range=[0, max_open * 1.2]
    )
    fig.update_yaxes(
        title_text="Click Rate %",
        secondary_y=True,
        gridcolor='rgba(232,180,203,0.3)',
        ticksuffix='%',
        showgrid=False,
        range=[0, max_click * 1.2]
    )
    fig.update_xaxes(
        gridcolor='rgba(212,191,255,0.2)',
        tickangle=-45,
        type='category',
        tickfont=dict(size=16)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Lidt luft mellem graf og tabel
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # --- TABEL ---
    cols_to_show = ['Date', 'ID_Campaign', 'Email_Message', 'Total_Received', 'Unique_Opens', 'Unique_Clicks', 'Open Rate %', 'Click Rate %', 'Click Through Rate %']
    sorted_df = display_df[cols_to_show].sort_values(by='Date', ascending=False)
    
    # Beregn højde baseret på antal rækker (35px per række + 38px header)
    table_height = (len(sorted_df) + 1) * 35 + 3
    
    st.dataframe(
        sorted_df,
        use_container_width=True,
        hide_index=True,
        height=table_height,
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

if st.button('Opdater Data'):
    st.cache_data.clear()
    st.rerun()




