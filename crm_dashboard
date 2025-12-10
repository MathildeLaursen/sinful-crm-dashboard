"""
CRM Dashboard - Hovedscript
"""
import streamlit as st
import extra_streamlit_components as stx
import datetime
import time
import os

# --- SIDE OPSAETNING ---
st.set_page_config(
    page_title="CRM Dashboard", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- CSS TEMA ---
css_path = os.path.join(os.path.dirname(__file__), 'style.css')
with open(css_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# JavaScript til checkbox styling
st.markdown("""
    <script>
        function fixCheckboxColors() {
            document.querySelectorAll('label[data-baseweb="checkbox"]').forEach(label => {
                const span = label.querySelector('span:first-child');
                const isChecked = label.getAttribute('aria-checked') === 'true';
                const svg = span ? span.querySelector('svg') : null;
                
                if (span) {
                    span.style.setProperty('background-color', isChecked ? '#9B7EBD' : 'white', 'important');
                    span.style.setProperty('border-color', isChecked ? '#9B7EBD' : '#D4BFFF', 'important');
                }
                
                if (svg) {
                    svg.style.setProperty('visibility', isChecked ? 'visible' : 'hidden', 'important');
                    svg.style.setProperty('opacity', isChecked ? '1' : '0', 'important');
                    
                    const polyline = svg.querySelector('polyline');
                    if (polyline) {
                        polyline.style.setProperty('stroke', 'white', 'important');
                        polyline.style.setProperty('stroke-width', '2', 'important');
                    }
                }
            });
        }
        
        setTimeout(fixCheckboxColors, 100);
        setTimeout(fixCheckboxColors, 300);
        setTimeout(fixCheckboxColors, 600);
        setTimeout(fixCheckboxColors, 1000);
        
        const observer = new MutationObserver(() => {
            setTimeout(fixCheckboxColors, 50);
        });
        observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['aria-checked'] });
    </script>
""", unsafe_allow_html=True)


# --- LOGIN ---
def check_password():
    cookie_manager = stx.CookieManager(key="main_cookie_manager")
    cookie_val = cookie_manager.get("sinful_auth")

    if st.session_state.get("authenticated", False):
        return True

    if cookie_val == "true":
        st.session_state["authenticated"] = True
        return True

    st.title("CRM Dashboard")
    st.markdown("Log ind")
    
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
                    
                    st.success("Login godkendt!")
                    time.sleep(1)
                    st.rerun()
    return False


if not check_password():
    st.stop()


# --- DASHBOARD ---
st.title("CRM Dashboard")

# Tabs
tab_newsletters, tab_subscribers = st.tabs(["Newsletters", "Subscribers"])

# Import og render tabs
from tab_newsletters import render_newsletters_tab
from tab_subscribers import render_subscribers_tab

with tab_newsletters:
    render_newsletters_tab()

with tab_subscribers:
    render_subscribers_tab()

