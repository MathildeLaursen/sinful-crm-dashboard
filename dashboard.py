
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
        df = conn.read(skiprows=1)
    except Exception:
        return pd.DataFrame()
    
    try:
        rename_map = {
            df.columns[0]: 'Send Year',
            df.columns[1]: 'Send Month',
            df.columns[2]: 'Send Day',
            df.columns[3]: 'Send Time',
            df.columns[4]: 'Number',
            df.columns[5]: 'Campaign Name',
            df.columns[6]: 'Email',
            df.columns[7]: 'Message',
            df.columns[8]: 'Variant',
            df.columns[9]: 'Total_Received',
            df.columns[10]: 'Total_Opens_Raw',
            df.columns[11]: 'Unique_Opens',
            df.columns[12]: 'Total_Clicks_Raw',
            df.columns[13]: 'Unique_Clicks',
            df.columns[14]: 'Unsubscribed'
        }
        df = df.rename(columns=rename_map)
    except Exception:
        return pd.DataFrame()

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
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.replace('.', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Open Rate %'] = df.apply(lambda x: (x['Unique_Opens'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1)
    df['Click Rate %'] = df.apply(lambda x: (x['Unique_Clicks'] / x['Total_Received'] * 100) if x['Total_Received'] > 0 else 0, axis=1)
    
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
    
    # Række 1: Periode, Start, Slut
    col_periode, col_start_group, col_end_group = st.columns(3)
    
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
    
    # Kampagne filter (kun kampagner i valgt periode)
    all_id_campaigns = sorted(df_date_filtered['ID_Campaign'].astype(str).unique())
    
    # Række 2: Kampagne, Email, A/B (samme layout som række 1)
    col_kamp, col_email, col_ab = st.columns(3)
    
    with col_kamp:
        k1, k2 = st.columns(ratio_col1)
        with k1:
            st.markdown("<p style='margin-top: 8px; font-size: 14px; font-weight: bold;'>Kampagne</p>", unsafe_allow_html=True)
        with k2:
            sel_id_campaigns = st.multiselect("Kampagne", all_id_campaigns, default=[], placeholder="Vælg...", label_visibility="collapsed")
    
    # Email filter (afhængig af valgt kampagne OG dato)
    if sel_id_campaigns:
        filtered_for_email = df_date_filtered[df_date_filtered['ID_Campaign'].astype(str).isin(sel_id_campaigns)]
    else:
        filtered_for_email = df_date_filtered
    all_email_messages = sorted(filtered_for_email['Email_Message'].astype(str).unique())
    
    with col_email:
        em1, em2 = st.columns(ratio_col2)
        with em1:
            st.markdown("<p style='margin-top: 8px; font-size: 14px; font-weight: bold;'>Email</p>", unsafe_allow_html=True)
        with em2:
            sel_email_messages = st.multiselect("Email", all_email_messages, default=[], placeholder="Vælg...", label_visibility="collapsed")
    
    # A/B filter (afhængig af valgt email OG dato)
    if sel_email_messages:
        filtered_for_variant = filtered_for_email[filtered_for_email['Email_Message'].astype(str).isin(sel_email_messages)]
    else:
        filtered_for_variant = filtered_for_email
    all_variants = sorted(filtered_for_variant['Variant'].astype(str).unique())
    
    with col_ab:
        ab1, ab2 = st.columns(ratio_col3)
        with ab1:
            st.markdown("<p style='margin-top: 8px; font-size: 14px; font-weight: bold;'>A/B</p>", unsafe_allow_html=True)
        with ab2:
            sel_variants = st.multiselect("A/B", all_variants, default=[], placeholder="Vælg...", label_visibility="collapsed")
    
    # Sammenlignings-tekst inde i filter-boksen
    st.caption(f"Sammenlignet med: {prev_start_date} - {prev_end_date}")


# --- DATA FILTRERING ---
def filter_data(dataset, start, end):
    mask = (dataset['Date'] >= pd.to_datetime(start)) & (dataset['Date'] <= pd.to_datetime(end))
    temp_df = dataset.loc[mask]
    
    if sel_id_campaigns:
        temp_df = temp_df[temp_df['ID_Campaign'].astype(str).isin(sel_id_campaigns)]
    if sel_email_messages:
        temp_df = temp_df[temp_df['Email_Message'].astype(str).isin(sel_email_messages)]
    if sel_variants:
        temp_df = temp_df[temp_df['Variant'].astype(str).isin(sel_variants)]
        
    return temp_df

current_df = filter_data(df, start_date, end_date)
prev_df = filter_data(df, prev_start_date, prev_end_date)


# --- VISUALISERING ---
col1, col2, col3, col4, col5, col6 = st.columns(6)

def show_metric(col, label, current_val, prev_val, format_str, is_percent=False):
    delta = 0
    if prev_val > 0:
        delta = current_val - prev_val
    
    if is_percent:
        val_fmt = f"{current_val:.1f}%"
        delta_fmt = f"{delta:.1f}%"
    else:
        val_fmt = f"{current_val:,.0f}"
        delta_fmt = f"{delta:,.0f}"

    col.metric(label, val_fmt, delta=delta_fmt)

cur_sent = current_df['Total_Received'].sum()
prev_sent = prev_df['Total_Received'].sum()
cur_opens = current_df['Unique_Opens'].sum()
prev_opens = prev_df['Unique_Opens'].sum()
cur_clicks = current_df['Unique_Clicks'].sum()
prev_clicks = prev_df['Unique_Clicks'].sum()
cur_or = current_df['Open Rate %'].mean() if not current_df.empty else 0
prev_or = prev_df['Open Rate %'].mean() if not prev_df.empty else 0
cur_cr = current_df['Click Rate %'].mean() if not current_df.empty else 0
prev_cr = prev_df['Click Rate %'].mean() if not prev_df.empty else 0
# Click Through Rate = Clicks / Opens * 100
cur_ctr = (cur_clicks / cur_opens * 100) if cur_opens > 0 else 0
prev_ctr = (prev_clicks / prev_opens * 100) if prev_opens > 0 else 0

show_metric(col1, "Emails Sendt", cur_sent, prev_sent, "{:,.0f}")
show_metric(col2, "Unikke Opens", cur_opens, prev_opens, "{:,.0f}")
show_metric(col3, "Unikke Clicks", cur_clicks, prev_clicks, "{:,.0f}")
show_metric(col4, "Gns. Open Rate", cur_or, prev_or, "{:.1f}%", is_percent=True)
show_metric(col5, "Gns. Click Rate", cur_cr, prev_cr, "{:.2f}%", is_percent=True)
show_metric(col6, "Click Through Rate", cur_ctr, prev_ctr, "{:.1f}%", is_percent=True)

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
            name='Open Rate %',
            line=dict(color='#2E86AB', width=2),
            mode='lines+markers'
        ),
        secondary_y=False
    )
    
    # Click Rate (højre y-akse) - Grøn
    fig.add_trace(
        go.Scatter(
            x=graph_df['Date'], 
            y=graph_df['Click Rate %'],
            name='Click Rate %',
            line=dict(color='#28A745', width=2),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    # Akse-titler
    fig.update_yaxes(title_text="Open Rate %", secondary_y=False)
    fig.update_yaxes(title_text="Click Rate %", secondary_y=True)
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
    cols_to_show = ['Date', 'ID_Campaign', 'Email_Message', 'Variant', 'Total_Received', 'Unique_Opens', 'Unique_Clicks', 'Open Rate %', 'Click Rate %']
    st.dataframe(
        display_df[cols_to_show].sort_values(by='Date', ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID_Campaign": st.column_config.TextColumn("Kampagne (ID - Navn)"),
            "Email_Message": st.column_config.TextColumn("Email - Message"),
            "Open Rate %": st.column_config.NumberColumn(format="%.1f%%"),
            "Click Rate %": st.column_config.NumberColumn(format="%.2f%%"),
            "Total_Received": st.column_config.NumberColumn(format="%d"),
            "Unique_Opens": st.column_config.NumberColumn(format="%d"),
            "Unique_Clicks": st.column_config.NumberColumn(format="%d"),
        }
    )
else:
    st.warning("Ingen data at vise.")

if st.button('🔄 Opdater Data'):
    st.cache_data.clear()
    st.rerun()




