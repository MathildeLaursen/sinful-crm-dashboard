"""
Delte funktioner til CRM Dashboard
"""
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# Spreadsheet URLs hentes fra secrets

def get_gspread_client():
    """Returnerer en autoriseret gspread client"""
    gsheets_config = st.secrets["connections"]["gsheets"]
    
    credentials_dict = {
        "type": gsheets_config.get("type", "service_account"),
        "project_id": gsheets_config["project_id"],
        "private_key_id": gsheets_config["private_key_id"],
        "private_key": gsheets_config["private_key"],
        "client_email": gsheets_config["client_email"],
        "client_id": gsheets_config["client_id"],
        "auth_uri": gsheets_config.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri": gsheets_config.get("token_uri", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": gsheets_config.get("auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs"),
        "client_x509_cert_url": gsheets_config.get("client_x509_cert_url", "")
    }
    
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
    return gspread.authorize(credentials)


def format_number(value):
    """Formater tal til kompakt visning (K/M)"""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.0f}K"
    else:
        return f"{value:.0f}"


def show_metric(col, label, current_val, prev_val=None, is_percent=False):
    """Vis en metric med optional delta"""
    if is_percent:
        val_fmt = f"{current_val:.1f}%"
    else:
        val_fmt = format_number(current_val)
    
    if prev_val is not None and prev_val != 0:
        pct_change = ((current_val - prev_val) / prev_val) * 100
        delta_str = f"{pct_change:+.1f}%"
        col.metric(label, val_fmt, delta=delta_str)
    else:
        col.metric(label, val_fmt)


# ===========================================
# PLOTLY GRAF STYLING - Unicorn tema
# ===========================================

# Farvekonstanter til grafer
GRAPH_COLORS = {
    'grid_x': 'rgba(212,191,255,0.2)',
    'grid_y': 'rgba(212,191,255,0.3)',
    'plot_bg': 'rgba(250,245,255,0.5)',
    'paper_bg': 'rgba(0,0,0,0)',
    'purple': '#9B7EBD',
    'pink': '#E8B4CB',
    'lavender': '#D4BFFF',
    'mint': '#A8E6CF',
    'peach': '#FFD3B6',
}


def style_graph(fig, height=400, show_legend=True):
    """
    Anvend standard Unicorn styling på en Plotly figur.
    
    Args:
        fig: Plotly figure objekt
        height: Graf højde i pixels
        show_legend: Vis legend (default True)
    
    Returns:
        fig: Den stylede figur
    """
    fig.update_layout(
        title="",
        showlegend=show_legend,
        height=height,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor=GRAPH_COLORS['plot_bg'],
        paper_bgcolor=GRAPH_COLORS['paper_bg'],
        hovermode='x unified'
    )
    
    fig.update_xaxes(
        gridcolor=GRAPH_COLORS['grid_x'],
        automargin=True,
        ticklabelstandoff=10
    )
    
    fig.update_yaxes(
        gridcolor=GRAPH_COLORS['grid_y'],
        automargin=True,
        ticklabelstandoff=10
    )
    
    return fig
