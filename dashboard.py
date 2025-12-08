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

# --- CSS HACK: DESIGN TILPASNINGER ---
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
with st.expander("🔍 Tilpas Dashboard (Dato & Filtre)", expanded=False):
    
    # Række 1: Datovælger
    st.subheader("📅 Vælg Periode")
    date_options = [
        "Seneste 7 dage",
        "Seneste 30 dage",
        "Uge til dato",
        "Måned til dato",
        "Kvartal til dato",
        "Sidste måned",
        "Sidste kvartal",
        "I år (YTD)",
        "Vælg Datoer"
    ]
    
    col_date1, col_date2 = st.columns([1, 3])
    with col_date1:
        selected_range = st.selectbox("Vælg Datoer", date_options)

    today = datetime.date.today()
    start_date = today
    end_date = today

    # Dato Logik
    if selected_range == "Måned til dato":
        start_date = today.replace(day=1)
        end_date = today
    elif selected_range == "Uge til dato":
        start_date = today - datetime.timedelta(days=today.weekday())
        end_date = today
    elif selected_range == "Seneste 7 dage":
        start_date = today - datetime.timedelta(days=7)
        end_date = today
    elif selected_range == "Seneste 30 dage":
        start_date = today - datetime.timedelta(days=30)
        end_date = today
    elif selected_range == "Sidste måned":
        first_of_this_month = today.replace(day=1)
        end_date = first_of_this_month - datetime.timedelta(days=1)
        start_date = end_date.replace(day=1)
    elif selected_range == "Kvartal til dato":
        current_q_start_month = 3 * ((today.month - 1) // 3) + 1
        start_date = today.replace(month=current_q_start_month, day=1)
        end_date = today
    elif selected_range == "Sidste kvartal":
        current_q_start_month = 3 * ((today.month - 1) // 3) + 1
        curr_q_start = today.replace(month=current_q_start_month, day=1)
        end_date = curr_q_start - datetime.timedelta(days=1)
        prev_q_start_month = 3 * ((end_date.month - 1) // 3) + 1
        start_date = end_date.replace(month=prev_q_start_month, day=1)
    elif selected_range == "I år (YTD)":
        start_date = today.replace(month=1, day=1)
        end_date = today
    else: # Brugerdefineret
        with col_date2:
            c1, c2 = st.columns(2)
            start_date = c1.date_input("Start dato", df['Date'].min())
            end_date = c2.date_input("Slut dato", df['Date'].max())

    delta = end_date - start_date
    prev_end_date = start_date - datetime.timedelta(days=1)
    prev_start_date = prev_end_date - delta

    st.divider()

    # Række 2: Filtre (Vandret layout)
    st.subheader("🔍 Detaljerede Filtre")
    
    # Klargør lister (kombinerede kolonner)
    all_id_campaigns = sorted(df['ID_Campaign'].astype(str).unique())
    all_email_messages = sorted(df['Email_Message'].astype(str).unique())
    all_variants = sorted(df['Variant'].astype(str).unique())

    # 3 kolonner til filtrene (kombinerede)
    f1, f2, f3 = st.columns(3)
    
    sel_id_campaigns = f1.multiselect("Kampagne (ID - Navn)", all_id_campaigns, default=[])
    sel_email_messages = f2.multiselect("Email - Message", all_email_messages, default=[])
    sel_variants = f3.multiselect("Variant", all_variants, default=[])


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
st.subheader(f"Overblik: {start_date} - {end_date}")
if selected_range != "Brugerdefineret":
    st.caption(f"Sammenlignet med forrige periode: {prev_start_date} - {prev_end_date}")

col1, col2, col3, col4 = st.columns(4)

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
cur_or = current_df['Open Rate %'].mean() if not current_df.empty else 0
prev_or = prev_df['Open Rate %'].mean() if not prev_df.empty else 0
cur_cr = current_df['Click Rate %'].mean() if not current_df.empty else 0
prev_cr = prev_df['Click Rate %'].mean() if not prev_df.empty else 0

show_metric(col1, "Emails Sendt", cur_sent, prev_sent, "{:,.0f}")
show_metric(col2, "Unikke Opens", cur_opens, prev_opens, "{:,.0f}")
show_metric(col3, "Gns. Open Rate", cur_or, prev_or, "{:.1f}%", is_percent=True)
show_metric(col4, "Gns. Click Rate", cur_cr, prev_cr, "{:.2f}%", is_percent=True)

st.divider()

col_graph1, col_graph2 = st.columns(2)
with col_graph1:
    st.subheader("📈 Udvikling over tid")
    if not current_df.empty:
        graph_df = current_df.sort_values('Date')
        fig_line = px.line(graph_df, x='Date', y='Open Rate %', hover_data=['ID_Campaign', 'Email_Message'], markers=True)
        fig_line.update_traces(line_color='#E74C3C')
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Ingen data i valgte periode.")

with col_graph2:
    st.subheader("🎯 Klik vs. Opens (Matrix)")
    if not current_df.empty:
        fig_scatter = px.scatter(current_df, x='Open Rate %', y='Click Rate %', size='Total_Received', color='ID_Campaign', hover_name='Email_Message')
        st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("📋 Detaljeret Data")
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
