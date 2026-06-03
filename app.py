import os
import logging
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from huggingface_hub import InferenceClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load and prepare HR Dataset
try:
    df = pd.read_csv("employees.csv")
except FileNotFoundError:
    import io
    logger.warning("employees.csv not found, using placeholder layout logic.")
    df = pd.DataFrame(columns=["Employee ID", "Name", "Role", "Department", "Monthly Salary (EGP)", "Source", "Referral Bonus", "Referred By Name"])

# Clean numerical types
df["Monthly Salary (EGP)"] = pd.to_numeric(df["Monthly Salary (EGP)"], errors='coerce').fillna(0)
df["Referral Bonus"] = pd.to_numeric(df["Referral Bonus"], errors='coerce').fillna(0)

HF_TOKEN = os.getenv("HF_TOKEN")
client = None
MODEL = None

if HF_TOKEN:
    try:
        client = InferenceClient(token=HF_TOKEN)
        logger.info("Hugging Face client initialized successfully.")
        MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" 
    except Exception as e:
        logger.error(f"HF init error: {e}")

def generate_insights(dataframe):
    if dataframe.empty or client is None or MODEL is None:
        return "⚡ AI Insights are currently warming up. Adjust filters to update operational analysis."

    try:
        dataframe = dataframe.fillna(0)
        summary_text = f"""
Total Monthly Payroll: {dataframe['Monthly Salary (EGP)'].sum():,.0f} EGP
Total Headcount: {dataframe['Employee ID'].nunique()}
Unique Specialized Roles: {dataframe['Role'].nunique()}
Total Referral Bonus Paid: {dataframe['Referral Bonus'].sum():,.0f} EGP
Top Department Budget: {dataframe.groupby('Department')['Monthly Salary (EGP)'].sum().idxmax()}
Primary Recruitment Source: {dataframe['Source'].value_counts().idxmax()}
"""

        messages = [
            {"role": "system", "content": "You are a senior HR director and talent acquisition strategist. Provide 5 short, actionable, and professional insights based on the organizational data summary provided. your answers is breif short no more than 3 lines and no bolding only insights answer."},
            {"role": "user", "content": summary_text}
        ]

        response = client.chat_completion(
            model=MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI system response: {str(e)[:200]}"

def calculate_kpis(df_):
    if df_.empty:
        return 0, 0, 0, 0
    return (
        df_["Monthly Salary (EGP)"].sum(),
        df_["Employee ID"].nunique(),
        df_["Role"].nunique(),
        df_["Monthly Salary (EGP)"].mean()
    )

# --- BEIGE WARM MINIMALIST STYLING ---
BEIGE_BG_STYLE = {
    "background-color": "#fcfbfa",
    "color": "#2d2a26",
    "font-family": "'Inter', 'Segoe UI', sans-serif",
    "min-height": "100vh",
    "padding": "24px"
}

BEIGE_CARD_STYLE = {
    "background": "#f4f1ea",
    "border": "1px solid #e4dfd5",
    "border-radius": "12px",
    "box-shadow": "0 4px 15px 0 rgba(45, 42, 38, 0.05)",
    "padding": "15px",
    "transition": "transform 0.2s"
}

PLOTLY_LIGHT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#4a4742", "family": "'Inter', sans-serif"},
    "xaxis": {"gridcolor": "#e4dfd5", "zerolinecolor": "#e4dfd5"},
    "yaxis": {"gridcolor": "#e4dfd5", "zerolinecolor": "#e4dfd5"},
    "margin": {"t": 40, "b": 40, "l": 40, "r": 40}
}

# Cohesive soft warm accent colors
COLOR_PALETTE = ["#c5a880", "#a3b19b", "#ce937b", "#8fa4a6"]

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = "Employee Analytics Dashboard"

app.layout = html.Div(style=BEIGE_BG_STYLE, children=[
    dbc.Container([
        
        html.Div([
            html.H1("Employee Analytics Dashboard", 
                    style={"letter-spacing": "1px", "font-weight": "800", "color": "#2d2a26"}),
            html.P("Real-time Employee Intelligence Dashboard", style={"color": "#7a756e", "font-size": "14px", "margin-top": "-5px"})
        ], className="text-center my-4"),

        # Filter Section
        dbc.Row([
            dbc.Col([
                html.Div(children=[
                    html.Label("🏢 Corporate Department", style={"color": "#8c7653", "font-weight": "600", "margin-bottom": "6px"}),
                    dcc.Dropdown(
                        id="department-filter",
                        options=[{"label": d, "value": d} for d in sorted(df["Department"].unique())] if not df.empty else [],
                        value=list(df["Department"].unique()) if not df.empty else [],
                        multi=True
                    )
                ])
            ], md=4, className="mb-3"),

            dbc.Col([
                html.Div(children=[
                    html.Label("🛠️ Operational Role", style={"color": "#8c7653", "font-weight": "600", "margin-bottom": "6px"}),
                    dcc.Dropdown(
                        id="role-filter",
                        options=[{"label": r, "value": r} for r in sorted(df["Role"].unique())] if not df.empty else [],
                        value=list(df["Role"].unique()) if not df.empty else [],
                        multi=True
                    )
                ])
            ], md=4, className="mb-3"),

            dbc.Col([
                html.Div(children=[
                    html.Label("🎯 Sourcing Channel", style={"color": "#8c7653", "font-weight": "600", "margin-bottom": "6px"}),
                    dcc.Dropdown(
                        id="source-filter",
                        options=[{"label": s, "value": s} for s in sorted(df["Source"].unique())] if not df.empty else [],
                        value=list(df["Source"].unique()) if not df.empty else [],
                        multi=True
                    )
                ])
            ], md=4, className="mb-3"),
        ], className="mb-4"),

        # KPI Metrics Cards Section
        dbc.Row([
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[
                html.H6("MONTHLY PAYROLL BUDGET", style={"color": "#7a756e", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="total-payroll", style={"color": "#aa7c57", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[
                html.H6("TOTAL ACTIVE HEADCOUNT", style={"color": "#7a756e", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="total-headcount", style={"color": "#6e8268", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[
                html.H6("UNIQUE SPECIALIZED ROLES", style={"color": "#7a756e", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="unique-roles", style={"color": "#657d80", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[
                html.H6("AVERAGE MONTHLY SALARY", style={"color": "#7a756e", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="avg-salary", style={"color": "#b8785d", "font-weight": "700"})
            ]), md=3, className="mb-3"),
        ], className="mb-4"),

        # AI Insights Section
        html.Div(style={**BEIGE_CARD_STYLE, "background": "#ebdccb", "border-color": "#d5beab"}, children=[
            html.H5("✨ Neural AI Workforce Insights", style={"color": "#5c4d3c", "font-weight": "700", "margin-bottom": "12px"}),
            html.Div(id="ai-insights", style={
                "color": "#3d352b", 
                "whiteSpace": "pre-line", 
                "font-size": "14px", 
                "line-height": "1.7",
                "font-family": "inherit"
            })
        ], className="mb-4"),

        # Row 1 Charts
        dbc.Row([
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[dcc.Graph(id="salary-trend", config={"displayModeBar": False})]), md=6, className="mb-4"),
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[dcc.Graph(id="category-chart", config={"displayModeBar": False})]), md=6, className="mb-4"),
        ]),

        # Row 2 Charts
        dbc.Row([
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[dcc.Graph(id="city-chart", config={"displayModeBar": False})]), md=6, className="mb-4"),
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[dcc.Graph(id="rep-chart", config={"displayModeBar": False})]), md=6, className="mb-4"),
        ]),

        # Data Ledger Section
        html.H4("Active Corporate Personnel Ledger", className="mt-2 mb-3", style={"color": "#4a4742", "font-weight": "600"}),
        html.Div(style={"border-radius": "12px", "overflow": "hidden", "border": "1px solid #e4dfd5"}, children=[
            dash_table.DataTable(
                id="sales-table",
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left", 
                    "backgroundColor": "#fcfbfa", 
                    "color": "#4a4742",
                    "border": "1px solid #e4dfd5",
                    "padding": "12px 15px",
                    "font-family": "'Inter', sans-serif"
                },
                style_header={
                    "backgroundColor": "#f4f1ea",
                    "color": "#2d2a26",
                    "fontWeight": "bold",
                    "border": "1px solid #e4dfd5"
                },
                style_data_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f4f1ea',
                }]
            )
        ], className="mb-5")

    ], fluid=True)
])

@app.callback(
    [
        Output("total-payroll", "children"),
        Output("total-headcount", "children"),
        Output("unique-roles", "children"),
        Output("avg-salary", "children"),
        Output("salary-trend", "figure"),
        Output("category-chart", "figure"),
        Output("city-chart", "figure"),
        Output("rep-chart", "figure"),
        Output("sales-table", "data"),
        Output("sales-table", "columns"),
        Output("ai-insights", "children"),
    ],
    [
        Input("department-filter", "value"),
        Input("role-filter", "value"),
        Input("source-filter", "value"),
    ]
)
def update_dashboard(departments, roles, sources):
    departments = departments or list(df["Department"].unique())
    roles = roles or list(df["Role"].unique())
    sources = sources or list(df["Source"].unique())

    filtered = df[
        df["Department"].isin(departments) &
        df["Role"].isin(roles) &
        df["Source"].isin(sources)
    ]

    payroll, headcount, unique_roles, avg_sal = calculate_kpis(filtered)

    if not filtered.empty:
        # Chart 1: Average Salary Scaling per Role Group
        trend_df = filtered.groupby("Role")["Monthly Salary (EGP)"].mean().reset_index().sort_values(by="Monthly Salary (EGP)", ascending=False)
        trend = px.bar(trend_df, x="Role", y="Monthly Salary (EGP)", title="Role Compensation Scaling (Avg EGP)")
        trend.update_traces(marker_color="#c5a880", marker_line_color="#b0956f", marker_line_width=1)
        
        # Chart 2: Payroll Expenditures Allocated Across Corporate Departments
        cat_df = filtered.groupby("Department")["Monthly Salary (EGP)"].sum().reset_index()
        category = px.bar(cat_df, x="Department", y="Monthly Salary (EGP)", title="Payroll Budget Allocation Across Verticals")
        category.update_traces(marker_color="#a3b19b", marker_line_color="#8d9c85", marker_line_width=1)
        
        # Chart 3: Recruitment Channels Share
        city_df = filtered.groupby("Source")["Employee ID"].count().reset_index().rename(columns={"Employee ID": "Count"})
        city = px.pie(city_df, names="Source", values="Count", title="Recruitment Sourcing Share Channels", hole=0.4)
        city.update_traces(textinfo='percent+label', marker=dict(colors=COLOR_PALETTE))
        
        # Chart 4: Top Employee Referral Networks
        referral_df = filtered[filtered["Referred By Name"].notna() & (filtered["Referred By Name"] != "")]
        rep_df = referral_df.groupby("Referred By Name")["Referral Bonus"].sum().reset_index().sort_values(by="Referral Bonus", ascending=True)
        
        if not rep_df.empty:
            rep = px.bar(rep_df, x="Referral Bonus", y="Referred By Name", orientation='h', title="Top Sourcing Referral Pipelines (Bonus EGP)")
            rep.update_traces(marker_color="#ce937b", marker_line_color="#b87f68", marker_line_width=1)
        else:
            rep = px.bar(title="Top Sourcing Referral Pipelines (No Data)")
    else:
        trend, category, city, rep = px.bar(), px.bar(), px.pie(), px.bar()

    for fig in [trend, category, city, rep]:
        fig.update_layout(**PLOTLY_LIGHT_LAYOUT)
        fig.update_layout(title={"font": {"size": 14, "color": "#2d2a26"}})

    city.update_layout(showlegend=False)

    return (
        f"{payroll:,.0f} EGP",
        f"{headcount:,}",
        f"{unique_roles:,}",
        f"{avg_sal:,.0f} EGP",
        trend,
        category,
        city,
        rep,
        filtered.to_dict("records"),
        [{"name": i, "id": i} for i in filtered.columns],
        generate_insights(filtered)
    )

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))