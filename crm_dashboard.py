"""
CRM Dashboard - Hovedscript
"""
import streamlit as st

# --- SIDE OPSAETNING (SKAL VAERE FOERST!) ---
st.set_page_config(
    page_title="CRM Dashboard", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Nu kan vi importere resten
import extra_streamlit_components as stx
import datetime
import time
import os
from tab_newsletters import render_newsletters_tab
from tab_subscribers import render_subscribers_tab
from tab_flows import render_flows_tab
from tab_games import render_games_tab

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
        
        // Slider styling - Unicorn tema
        function fixSliderColors() {
            // Find all divs with inline background-color that might be slider parts
            document.querySelectorAll('div').forEach(div => {
                const bg = div.style.backgroundColor;
                if (!bg) return;
                
                // Check if this is inside a slider
                const isInSlider = div.closest('[data-testid="stSlider"]') !== null;
                if (!isInSlider) return;
                
                // Light track (unselected) - various grays/greens
                if (bg.includes('49, 51, 63') || bg.includes('rgb(49') || 
                    bg.includes('14, 17, 23') || bg.includes('rgb(14') ||
                    bg.includes('168, 230') || bg.includes('rgb(168')) {
                    div.style.setProperty('background-color', '#E8D5FF', 'important');
                    div.style.setProperty('height', '10px', 'important');
                }
                
                // Selected track (red/pink accent)
                if (bg.includes('255, 75') || bg.includes('rgb(255, 75') ||
                    bg.includes('255, 43') || bg.includes('rgb(255, 43') ||
                    bg.includes('255, 99') || bg.includes('rgb(255, 99')) {
                    div.style.setProperty('background-color', '#9B7EBD', 'important');
                    div.style.setProperty('height', '10px', 'important');
                }
            });
            
            // Style the thumbs/handles
            document.querySelectorAll('[data-testid="stSlider"] [role="slider"]').forEach(thumb => {
                thumb.style.setProperty('background-color', '#9B7EBD', 'important');
                thumb.style.setProperty('border', '3px solid white', 'important');
                thumb.style.setProperty('box-shadow', '0 2px 8px rgba(155, 126, 189, 0.5)', 'important');
                thumb.style.setProperty('width', '22px', 'important');
                thumb.style.setProperty('height', '22px', 'important');
                thumb.style.setProperty('border-radius', '50%', 'important');
            });
        }
        
        setTimeout(fixSliderColors, 200);
        setTimeout(fixSliderColors, 500);
        setTimeout(fixSliderColors, 1000);
        setTimeout(fixSliderColors, 2000);
        setInterval(fixSliderColors, 3000); // Keep checking periodically
        
        const sliderObserver = new MutationObserver(() => {
            setTimeout(fixSliderColors, 100);
        });
        sliderObserver.observe(document.body, { childList: true, subtree: true });
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
tab_newsletters, tab_flows, tab_games, tab_subscribers = st.tabs(["Newsletters", "Flows", "Games", "Subscribers"])

with tab_newsletters:
    render_newsletters_tab()

with tab_flows:
    render_flows_tab()

with tab_games:
    render_games_tab()

with tab_subscribers:
    render_subscribers_tab()

