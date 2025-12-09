
import streamlit as st
import pandas as pd
import plotly.express as px
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

        /* 5. Kompakt spacing i filter boksen (expander) */
        [data-testid="stExpander"] .stVerticalBlock {
            gap: 0.5rem !important;
        }
        
        /* Afstand mellem hovedkolonner (Periode-gruppe → Start-gruppe) */
        [data-testid="stExpander"] > div > div > .stVerticalBlock > div > .stHorizontalBlock {
            gap: 2rem !important;
        }
        
        /* JUSTÉR DENNE: Afstand mellem label-tekst og dropdown INDEN i hver gruppe */
        [data-testid="stExpander"] .stHorizontalBlock .stHorizontalBlock {
            gap: 0.2rem !important;
        }
        
        
        [data-testid="stExpander"] .stSelectbox,
        [data-testid="stExpander"] .stMultiSelect,
        [data-testid="stExpander"] .stDateInput {
            margin-bottom: 0 !important;
        }
        
        /* Reducer luft omkring labels i expander */
        [data-testid="stExpander"] label {
            margin-bottom: 0.25rem !important;
        }
        
        /* Kompakt divider */
        [data-testid="stExpander"] hr {
            margin: 0.5rem 0 !important;
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


# --- TOP-BAR: FILTRE & DATO (COLLAPSIBLE) ---

# Vi bruger st.expander til at lave en boks der kan foldes ud/ind
with st.expander("Filtrér", expanded=True):
    
    # Række 1: Datovælger
    date_options = [
        "Seneste 7 dage",
        "Seneste 30 dage",
        "Uge til dato",
        "Måned til dato",
        "Kvartal til dato",
        "Sidste måned",
        "Sidste kvartal",
        "I år (YTD)"
    ]

    today = datetime.date.today()
    
    # Beregn default datoer baseret på dropdown valg
    def get_date_range(selection):
        if selection == "Måned til dato":
            return today.replace(day=1), today
        elif selection == "Uge til dato":
            return today - datetime.timedelta(days=today.weekday()), today
        elif selection == "Seneste 7 dage":
            return today - datetime.timedelta(days=7), today
        elif selection == "Seneste 30 dage":
            return today - datetime.timedelta(days=30), today
        elif selection == "Sidste måned":
            first_of_this_month = today.replace(day=1)
            end = first_of_this_month - datetime.timedelta(days=1)
            return end.replace(day=1), end
        elif selection == "Kvartal til dato":
            current_q_start_month = 3 * ((today.month - 1) // 3) + 1
            return today.replace(month=current_q_start_month, day=1), today
        elif selection == "Sidste kvartal":
            current_q_start_month = 3 * ((today.month - 1) // 3) + 1
            curr_q_start = today.replace(month=current_q_start_month, day=1)
            end = curr_q_start - datetime.timedelta(days=1)
            prev_q_start_month = 3 * ((end.month - 1) // 3) + 1
            return end.replace(month=prev_q_start_month, day=1), end
        elif selection == "I år (YTD)":
            return today.replace(month=1, day=1), today
        else:
            return today - datetime.timedelta(days=30), today
    
    # Forskellige label-bredder for hver kolonne
    ratio_col1 = [0.18, 0.82]  # Periode/Kampagne (bredere labels)
    ratio_col2 = [0.10, 0.90]  # Start/Email (smallere labels)
    ratio_col3 = [0.08, 0.92]  # Slut/A/B (smallest labels)
    ratio_col4 = [0.10, 0.90]  # Land
    
    # Række 1: Periode, Start, Slut, (tom plads for land på række 2)
    col_periode, col_start_group, col_end_group, col_spacer = st.columns(4)
    
    with col_periode:
        p1, p2 = st.columns(ratio_col1)
        with p1:
            st.markdown("<p style='margin-top: 8px; font-size: 14px; font-weight: bold;'>Periode</p>", unsafe_allow_html=True)
        with p2:
            selected_range = st.selectbox("Periode", date_options, index=1, label_visibility="collapsed")
    
    # Beregn default datoer
    default_start, default_end = get_date_range(selected_range)
    
    with col_start_group:
        s1, s2 = st.columns(ratio_col2)
        with s1:
            st.markdown("<p style='margin-top: 8px; font-size: 14px; font-weight: bold;'>Start</p>", unsafe_allow_html=True)
        with s2:
            start_date = st.date_input("Start dato", default_start, label_visibility="collapsed")
    
    with col_end_group:
        e1, e2 = st.columns(ratio_col3)
        with e1:
            st.markdown("<p style='margin-top: 8px; font-size: 14px; font-weight: bold;'>Slut</p>", unsafe_allow_html=True)
        with e2:
            end_date = st.date_input("Slut dato", default_end, label_visibility="collapsed")

    delta = end_date - start_date
    prev_end_date = start_date - datetime.timedelta(days=1)
    prev_start_date = prev_end_date - delta

    # Række 2: Filtre (Vandret layout) - KASKADERENDE
    
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
        st.session_state.selected_variants = []
        st.session_state.selected_countries = []
    
    # Initialize session states
    if 'selected_campaigns' not in st.session_state:
        st.session_state.selected_campaigns = []
    if 'selected_emails' not in st.session_state:
        st.session_state.selected_emails = []
    if 'selected_variants' not in st.session_state:
        st.session_state.selected_variants = []
    if 'selected_countries' not in st.session_state:
        st.session_state.selected_countries = []
    if 'search_campaign' not in st.session_state:
        st.session_state.search_campaign = ""
    if 'search_email' not in st.session_state:
        st.session_state.search_email = ""
    if 'search_variant' not in st.session_state:
        st.session_state.search_variant = ""
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
    
    # Række 2: Land, Kampagne, Email, A/B (land først for at påvirke kampagner)
    col_land, col_kamp, col_email, col_ab = st.columns(4)
    
    # Land filter først
    with col_land:
        l1, l2 = st.columns(ratio_col4)
        with l1:
            st.markdown("<p style='margin-top: 8px; font-size: 14px; font-weight: bold;'>Land</p>", unsafe_allow_html=True)
        with l2:
            count_text = f"{len(st.session_state.selected_countries)} valgt"
            with st.popover(count_text, use_container_width=True):
                # Select All checkbox
                all_selected = len(st.session_state.selected_countries) == len(all_countries)
                if st.checkbox("Select All", value=all_selected, key="select_all_countries_cb"):
                    if not all_selected:
                        st.session_state.selected_countries = list(all_countries)
                        st.rerun()
                else:
                    if all_selected:
                        st.session_state.selected_countries = []
                        st.rerun()
                
                # Search box
                search_term = st.text_input("🔍 Type to search", key="search_country", label_visibility="collapsed", placeholder="Type to search")
                
                # Reduceret spacing (30% af normal)
                st.markdown("<div style='margin-top: -0.7rem;'></div>", unsafe_allow_html=True)
                
                # Filter countries by search
                filtered_countries = [c for c in all_countries if search_term.lower() in c.lower()] if search_term else all_countries
                
                # Checkboxes for countries
                for country in filtered_countries:
                    is_selected = country in st.session_state.selected_countries
                    if st.checkbox(country, value=is_selected, key=f"country_{country}"):
                        if country not in st.session_state.selected_countries:
                            st.session_state.selected_countries.append(country)
                    else:
                        if country in st.session_state.selected_countries:
                            st.session_state.selected_countries.remove(country)
    
    with col_kamp:
        k1, k2 = st.columns(ratio_col1)
        with k1:
            st.markdown("<p style='margin-top: 8px; font-size: 14px; font-weight: bold;'>Kampagne</p>", unsafe_allow_html=True)
        with k2:
            count_text = f"{len(st.session_state.selected_campaigns)} valgt" if st.session_state.selected_campaigns else "Alle"
            with st.popover(count_text, use_container_width=True):
                # Select All checkbox
                all_selected = len(st.session_state.selected_campaigns) == len(all_id_campaigns)
                if st.checkbox("Select All", value=all_selected, key="select_all_campaigns_cb"):
                    if not all_selected:
                        st.session_state.selected_campaigns = list(all_id_campaigns)
                        st.rerun()
                else:
                    if all_selected:
                        st.session_state.selected_campaigns = []
                        st.rerun()
                
                # Search box
                search_term = st.text_input("🔍 Type to search", key="search_campaign", label_visibility="collapsed", placeholder="Type to search")
                
                # Reduceret spacing (30% af normal)
                st.markdown("<div style='margin-top: -0.7rem;'></div>", unsafe_allow_html=True)
                
                # Filter campaigns by search
                filtered_campaigns = [c for c in all_id_campaigns if search_term.lower() in c.lower()] if search_term else all_id_campaigns
                
                # Checkboxes for campaigns
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
    # Filtrer på kampagner hvis ikke alle er valgt
    if len(st.session_state.selected_campaigns) < len(all_id_campaigns):
        filtered_for_email = df_land_filtered[df_land_filtered['ID_Campaign'].astype(str).isin(sel_id_campaigns)]
    else:
        filtered_for_email = df_land_filtered
    
    all_email_messages = sorted(filtered_for_email['Email_Message'].astype(str).unique())
    
    # Pre-select alle emails hvis tom
    if not st.session_state.selected_emails:
        st.session_state.selected_emails = list(all_email_messages)
    
    # Synkroniser: fjern emails der ikke findes i filtreret data
    st.session_state.selected_emails = [e for e in st.session_state.selected_emails if e in all_email_messages]
    
    # Hvis alle emails blev fjernet, vælg alle igen
    if not st.session_state.selected_emails:
        st.session_state.selected_emails = list(all_email_messages)
    
    with col_email:
        em1, em2 = st.columns(ratio_col2)
        with em1:
            st.markdown("<p style='margin-top: 8px; font-size: 14px; font-weight: bold;'>Email</p>", unsafe_allow_html=True)
        with em2:
            count_text = f"{len(st.session_state.selected_emails)} valgt" if st.session_state.selected_emails else "Alle"
            with st.popover(count_text, use_container_width=True):
                # Select All checkbox
                all_selected = len(st.session_state.selected_emails) == len(all_email_messages)
                if st.checkbox("Select All", value=all_selected, key="select_all_emails_cb"):
                    if not all_selected:
                        st.session_state.selected_emails = list(all_email_messages)
                        st.rerun()
                else:
                    if all_selected:
                        st.session_state.selected_emails = []
                        st.rerun()
                
                # Search box
                search_term = st.text_input("🔍 Type to search", key="search_email", label_visibility="collapsed", placeholder="Type to search")
                
                # Reduceret spacing (30% af normal)
                st.markdown("<div style='margin-top: -0.7rem;'></div>", unsafe_allow_html=True)
                
                # Filter emails by search
                filtered_emails = [e for e in all_email_messages if search_term.lower() in e.lower()] if search_term else all_email_messages
                
                # Checkboxes for emails
                for email in filtered_emails:
                    is_selected = email in st.session_state.selected_emails
                    if st.checkbox(email, value=is_selected, key=f"email_{email}"):
                        if email not in st.session_state.selected_emails:
                            st.session_state.selected_emails.append(email)
                    else:
                        if email in st.session_state.selected_emails:
                            st.session_state.selected_emails.remove(email)
    
    sel_email_messages = st.session_state.selected_emails
    
    # A/B filter (afhængig af valgt email, kampagne, land OG dato)
    # Filtrer på emails hvis ikke alle er valgt
    if len(st.session_state.selected_emails) < len(all_email_messages):
        filtered_for_variant = filtered_for_email[filtered_for_email['Email_Message'].astype(str).isin(sel_email_messages)]
    else:
        filtered_for_variant = filtered_for_email
    
    # Inkluder alle variants, men map "nan" til pæn visning
    all_variants_raw = filtered_for_variant['Variant'].astype(str).unique()
    all_variants_with_nan = sorted(all_variants_raw)
    
    # Map nan til visning
    def display_variant(v):
        if v.lower() == 'nan':
            return "(Ingen)"
        return v
    
    # Pre-select alle variants hvis tom
    if not st.session_state.selected_variants:
        st.session_state.selected_variants = list(all_variants_with_nan)
    
    # Synkroniser: fjern variants der ikke findes i filtreret data
    st.session_state.selected_variants = [v for v in st.session_state.selected_variants if v in all_variants_with_nan]
    
    # Hvis alle variants blev fjernet, vælg alle igen
    if not st.session_state.selected_variants:
        st.session_state.selected_variants = list(all_variants_with_nan)
    
    with col_ab:
        ab1, ab2 = st.columns(ratio_col3)
        with ab1:
            st.markdown("<p style='margin-top: 8px; font-size: 14px; font-weight: bold;'>A/B</p>", unsafe_allow_html=True)
        with ab2:
            count_text = f"{len(st.session_state.selected_variants)} valgt" if st.session_state.selected_variants else "Alle"
            with st.popover(count_text, use_container_width=True):
                # Select All checkbox
                all_selected = len(st.session_state.selected_variants) == len(all_variants_with_nan)
                if st.checkbox("Select All", value=all_selected, key="select_all_variants_cb"):
                    if not all_selected:
                        st.session_state.selected_variants = list(all_variants_with_nan)
                        st.rerun()
                else:
                    if all_selected:
                        st.session_state.selected_variants = []
                        st.rerun()
                
                # Search box
                search_term = st.text_input("🔍 Type to search", key="search_variant", label_visibility="collapsed", placeholder="Type to search")
                
                # Reduceret spacing (30% af normal)
                st.markdown("<div style='margin-top: -0.7rem;'></div>", unsafe_allow_html=True)
                
                # Filter variants by search
                filtered_variants = [v for v in all_variants_with_nan if search_term.lower() in display_variant(v).lower()] if search_term else all_variants_with_nan
                
                # Checkboxes for variants
                for variant in filtered_variants:
                    display_label = display_variant(variant)
                    is_selected = variant in st.session_state.selected_variants
                    if st.checkbox(display_label, value=is_selected, key=f"variant_{variant}"):
                        if variant not in st.session_state.selected_variants:
                            st.session_state.selected_variants.append(variant)
                    else:
                        if variant in st.session_state.selected_variants:
                            st.session_state.selected_variants.remove(variant)
    
    # Hvis ingen variants er valgt manuelt, brug alle (inkl nan)
    if not st.session_state.selected_variants:
        sel_variants = all_variants_with_nan
    else:
        sel_variants = st.session_state.selected_variants
    
    sel_countries = st.session_state.selected_countries
    
    # Tjek om sammenligning er mulig (kun hvis ALLE kampagner, emails og A/B er valgt)
    has_campaign_filter = len(sel_id_campaigns) < len(all_id_campaigns)
    has_email_filter = len(sel_email_messages) < len(all_email_messages)
    has_variant_filter = len(sel_variants) < len(all_variants_with_nan)
    can_compare = not (has_campaign_filter or has_email_filter or has_variant_filter)
    
    # Sammenlignings-tekst inde i filter-boksen
    if can_compare:
        st.caption(f"Sammenlignet med: {prev_start_date} - {prev_end_date}")
    else:
        st.caption("⚠️ Sammenligning ikke mulig når kampagne, email eller A/B filter er aktiv")


# --- DATA FILTRERING OG AGGREGERING ---

def filter_data(dataset, start, end, for_comparison=False):
    mask = (dataset['Date'] >= pd.to_datetime(start)) & (dataset['Date'] <= pd.to_datetime(end))
    temp_df = dataset.loc[mask].copy()
    
    # Land filter bruges altid
    if sel_countries:
        temp_df = temp_df[temp_df['Country'].isin(sel_countries)]
    
    # Kampagne/Email/Variant filtre bruges KUN for visning, IKKE for sammenligning
    if not for_comparison:
        if sel_id_campaigns:
            temp_df = temp_df[temp_df['ID_Campaign'].astype(str).isin(sel_id_campaigns)]
        
        if sel_email_messages:
            temp_df = temp_df[temp_df['Email_Message'].astype(str).isin(sel_email_messages)]
        
        if sel_variants:
            temp_df = temp_df[temp_df['Variant'].astype(str).isin(sel_variants)]
    
    # Aggreger data på tværs af lande
    # Gruppér på Date, Campaign, Email, Variant og summer metrics
    if not temp_df.empty:
        agg_df = temp_df.groupby(['Date', 'ID_Campaign', 'Email_Message', 'Variant'], as_index=False).agg({
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

current_df = filter_data(df, start_date, end_date, for_comparison=False)
# Kun hent prev_df hvis sammenligning er mulig
if can_compare:
    prev_df = filter_data(df, prev_start_date, prev_end_date, for_comparison=True)
else:
    prev_df = pd.DataFrame()  # Tom DataFrame når sammenligning ikke er mulig


# --- VISUALISERING ---
col1, col2, col3, col4, col5, col6 = st.columns(6)

def show_metric(col, label, current_val, prev_val, format_str, is_percent=False, show_delta=True):
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
    
    # Hvis ingen sammenligning, vis kun værdi uden delta
    if not show_delta:
        col.metric(label, val_fmt)
        return
    
    # Beregn absolut og procentuel ændring
    absolute_delta = current_val - prev_val
    if prev_val > 0:
        pct_change = ((current_val - prev_val) / prev_val) * 100
    else:
        pct_change = 0
    
    # Delta: kompakt absolut tal + procent i parentes
    if is_percent:
        # For procent-metrics: kun procent-ændring
        if pct_change >= 0:
            delta_fmt = f"+{pct_change:.1f}%"
        else:
            delta_fmt = f"{pct_change:.1f}%"
    else:
        # For tal-metrics: absolut (kompakt) + procent i parentes
        if absolute_delta >= 1_000_000:
            abs_fmt = f"{absolute_delta / 1_000_000:.1f}M"
        elif absolute_delta >= 1_000:
            abs_fmt = f"{absolute_delta / 1_000:.0f}K"
        elif absolute_delta <= -1_000_000:
            abs_fmt = f"{absolute_delta / 1_000_000:.1f}M"
        elif absolute_delta <= -1_000:
            abs_fmt = f"{absolute_delta / 1_000:.0f}K"
        else:
            abs_fmt = f"{absolute_delta:.0f}"
        
        if absolute_delta >= 0:
            delta_fmt = f"+{abs_fmt} (+{pct_change:.1f}%)"
        else:
            delta_fmt = f"{abs_fmt} ({pct_change:.1f}%)"

    col.metric(label, val_fmt, delta=delta_fmt)

cur_sent = current_df['Total_Received'].sum()
cur_opens = current_df['Unique_Opens'].sum()
cur_clicks = current_df['Unique_Clicks'].sum()
cur_or = current_df['Open Rate %'].mean() if not current_df.empty else 0
cur_cr = current_df['Click Rate %'].mean() if not current_df.empty else 0
cur_ctr = (cur_clicks / cur_opens * 100) if cur_opens > 0 else 0

# Kun beregn prev-værdier hvis sammenligning er mulig
if can_compare and not prev_df.empty:
    prev_sent = prev_df['Total_Received'].sum()
    prev_opens = prev_df['Unique_Opens'].sum()
    prev_clicks = prev_df['Unique_Clicks'].sum()
    prev_or = prev_df['Open Rate %'].mean()
    prev_cr = prev_df['Click Rate %'].mean()
    prev_ctr = (prev_clicks / prev_opens * 100) if prev_opens > 0 else 0
else:
    prev_sent = prev_opens = prev_clicks = prev_or = prev_cr = prev_ctr = 0

show_metric(col1, "Emails Sendt", cur_sent, prev_sent, "{:,.0f}", show_delta=can_compare)
show_metric(col2, "Unikke Opens", cur_opens, prev_opens, "{:,.0f}", show_delta=can_compare)
show_metric(col3, "Unikke Clicks", cur_clicks, prev_clicks, "{:,.0f}", show_delta=can_compare)
show_metric(col4, "Open Rate", cur_or, prev_or, "{:.1f}%", is_percent=True, show_delta=can_compare)
show_metric(col5, "Click Rate", cur_cr, prev_cr, "{:.2f}%", is_percent=True, show_delta=can_compare)
show_metric(col6, "Click Through Rate", cur_ctr, prev_ctr, "{:.1f}%", is_percent=True, show_delta=can_compare)

st.divider()

if not current_df.empty:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    graph_df = current_df.sort_values('Date')
    
    # Dual y-axis plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Open Rate (venstre y-akse) - Blå
    fig.add_trace(
        go.Scatter(
            x=graph_df['Date'], 
            y=graph_df['Open Rate %'],
            name='Open Rate',
            marker=dict(color='#2E86AB', size=8),
            mode='markers'
        ),
        secondary_y=False
    )
    
    # Click Rate (højre y-akse) - Grøn
    fig.add_trace(
        go.Scatter(
            x=graph_df['Date'], 
            y=graph_df['Click Rate %'],
            name='Click Rate',
            marker=dict(color='#28A745', size=8),
            mode='markers'
        ),
        secondary_y=True
    )
    
    # Akse-titler
    fig.update_yaxes(title_text="Open Rate", secondary_y=False)
    fig.update_yaxes(title_text="Click Rate", secondary_y=True)
    fig.update_xaxes(title_text="")
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Ingen data i valgte periode.")

if not current_df.empty:
    display_df = current_df.copy()
    display_df['Date'] = display_df['Date'].dt.date
    cols_to_show = ['Date', 'ID_Campaign', 'Email_Message', 'Variant', 'Total_Received', 'Unique_Opens', 'Unique_Clicks', 'Open Rate %', 'Click Rate %', 'Click Through Rate %']
    st.dataframe(
        display_df[cols_to_show].sort_values(by='Date', ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", width="small"),
            "ID_Campaign": st.column_config.TextColumn("Kampagne", width="medium"),
            "Email_Message": st.column_config.TextColumn("Email", width="large"),
            "Variant": st.column_config.TextColumn("A/B", width="small"),
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




