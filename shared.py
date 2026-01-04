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


def style_graph(fig, height=400, show_legend=True, legend_position='right', 
                 margin_top=30, x_tickformat=None, y_tickformat=None):
    """
    Anvend standard Unicorn styling på en Plotly figur.
    
    Args:
        fig: Plotly figure objekt
        height: Graf højde i pixels
        show_legend: Vis legend (default True)
        legend_position: 'right', 'center' eller dict med custom position
        margin_top: Top margin i pixels (default 30, brug 60 for legend over graf)
        x_tickformat: Format for x-akse labels (f.eks. '%Y-%m')
        y_tickformat: Format for y-akse labels (f.eks. ',')
    
    Returns:
        fig: Den stylede figur
    """
    # Legend position presets
    if legend_position == 'right':
        legend_config = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    elif legend_position == 'center':
        legend_config = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    elif isinstance(legend_position, dict):
        legend_config = legend_position
    else:
        legend_config = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    
    fig.update_layout(
        title="",
        showlegend=show_legend,
        height=height,
        margin=dict(l=50, r=50, t=margin_top, b=50),
        legend=legend_config,
        plot_bgcolor=GRAPH_COLORS['plot_bg'],
        paper_bgcolor=GRAPH_COLORS['paper_bg'],
        hovermode='x unified'
    )
    
    # X-akse styling
    x_kwargs = dict(gridcolor=GRAPH_COLORS['grid_x'], automargin=True, ticklabelstandoff=10)
    if x_tickformat:
        x_kwargs['tickformat'] = x_tickformat
    fig.update_xaxes(**x_kwargs)
    
    # Y-akse styling
    y_kwargs = dict(gridcolor=GRAPH_COLORS['grid_y'], automargin=True, ticklabelstandoff=10)
    if y_tickformat:
        y_kwargs['tickformat'] = y_tickformat
    fig.update_yaxes(**y_kwargs)
    
    return fig


# Standard lande rækkefølge for dropdowns
COUNTRY_ORDER = ['DK', 'SE', 'NO', 'FI', 'FR', 'UK', 'DE', 'AT', 'NL', 'BE', 'CH']