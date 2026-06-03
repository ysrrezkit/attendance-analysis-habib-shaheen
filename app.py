import os
import logging
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from huggingface_hub import InferenceClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

df = pd.read_csv("attendance_data.csv")

df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df["Day_of_Week"] = df["Date"].dt.day_name()
df["Is_Present"] = df["Status"].str.contains("Present", case=False, na=False).astype(int)

HF_TOKEN = os.getenv("HF_TOKEN")
client = None
MODEL = None

if HF_TOKEN:
    try:
        client = InferenceClient(token=HF_TOKEN)
        MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    except Exception as e:
        logger.error(f"HF init error: {e}")

def generate_insights(dataframe):
    if dataframe.empty or client is None or MODEL is None:
        return "AI Insights are currently warming up. Adjust filters to update operational analysis."

    try:
        attendance_counts = dataframe[dataframe["Is_Present"] == 1]["Name"].value_counts()
        day_counts = dataframe[dataframe["Is_Present"] == 1]["Day_of_Week"].value_counts()
        
        most_present = attendance_counts.idxmax() if not attendance_counts.empty else "N/A"
        least_present = attendance_counts.idxmin() if not attendance_counts.empty else "N/A"
        most_present_day = day_counts.idxmax() if not day_counts.empty else "N/A"
        least_present_day = day_counts.idxmin() if not day_counts.empty else "N/A"

        summary_text = f"""
Total Records: {len(dataframe)}
Total Present Cases: {dataframe['Is_Present'].sum()}
Most Present Employee: {most_present}
Least Present Employee: {least_present}
Highest Attendance Day: {most_present_day}
Lowest Attendance Day: {least_present_day}
"""
        messages = [
            {"role": "system", "content": "You are a senior HR director. Provide 10 short, actionable, and professional insights based on the attendance data summary provided. your answers is breif short no more than 3 lines and no bolding only insights answer."},
            {"role": "user", "content": summary_text}
        ]
        response = client.chat_completion(model=MODEL, messages=messages, max_tokens=300, temperature=0.5)
        return response.choices[0].message.content
    except Exception as e:
        return f"AI system response: {str(e)[:200]}"

def calculate_kpis(df_):
    if df_.empty:
        return 0, 0, 0, 0
    total_records = len(df_)
    total_present = df_["Is_Present"].sum()
    attendance_rate = (total_present / total_records) * 100 if total_records > 0 else 0
    unique_employees = df_["Employee ID"].nunique()
    return total_records, total_present, attendance_rate, unique_employees

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

COLOR_PALETTE = ["#c5a880", "#a3b19b", "#ce937b", "#8fa4a6"]

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = "Employee Attendance Dashboard"

app.layout = html.Div(style=BEIGE_BG_STYLE, children=[
    dbc.Container([
        
        html.Div([
            html.H1("Employee Attendance Dashboard", style={"letter-spacing": "1px", "font-weight": "800", "color": "#2d2a26"}),
            html.P("Real-time Attendance Intelligence Dashboard", style={"color": "#7a756e", "font-size": "14px", "margin-top": "-5px"})
        ], className="text-center my-4"),

        dbc.Row([
            dbc.Col([
                html.Div(children=[
                    html.Label("Attendance Status", style={"color": "#8c7653", "font-weight": "600", "margin-bottom": "6px"}),
                    dcc.Dropdown(
                        id="status-filter",
                        options=[{"label": s, "value": s} for s in sorted(df["Status"].dropna().unique())] if not df.empty else [],
                        value=list(df["Status"].dropna().unique()) if not df.empty else [],
                        multi=True
                    )
                ])
            ], md=6, className="mb-3"),

            dbc.Col([
                html.Div(children=[
                    html.Label("Day of Week", style={"color": "#8c7653", "font-weight": "600", "margin-bottom": "6px"}),
                    dcc.Dropdown(
                        id="day-filter",
                        options=[{"label": d, "value": d} for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]],
                        value=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                        multi=True
                    )
                ])
            ], md=6, className="mb-3"),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[
                html.H6("TOTAL LOGGED DAYS", style={"color": "#7a756e", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="total-records", style={"color": "#aa7c57", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[
                html.H6("TOTAL PRESENT DAYS", style={"color": "#7a756e", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="total-present", style={"color": "#6e8268", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[
                html.H6("ATTENDANCE RATE", style={"color": "#7a756e", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="attendance-rate", style={"color": "#657d80", "font-weight": "700"})
            ]), md=3, className="mb-3"),
            
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[
                html.H6("TOTAL EMPLOYEES", style={"color": "#7a756e", "font-weight": "700", "letter-spacing": "1px"}),
                html.H2(id="total-employees", style={"color": "#b8785d", "font-weight": "700"})
            ]), md=3, className="mb-3"),
        ], className="mb-4"),

        html.Div(style={**BEIGE_CARD_STYLE, "background": "#ebdccb", "border-color": "#d5beab"}, children=[
            html.H5("Workforce Attendance Insights", style={"color": "#5c4d3c", "font-weight": "700", "margin-bottom": "12px"}),
            html.Div(id="ai-insights", style={
                "color": "#3d352b", 
                "whiteSpace": "pre-line", 
                "font-size": "14px", 
                "line-height": "1.7",
                "font-family": "inherit"
            })
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[dcc.Graph(id="employee-attendance-chart", config={"displayModeBar": False})]), md=6, className="mb-4"),
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[dcc.Graph(id="day-attendance-chart", config={"displayModeBar": False})]), md=6, className="mb-4"),
        ]),

        dbc.Row([
            dbc.Col(html.Div(style=BEIGE_CARD_STYLE, children=[dcc.Graph(id="status-distribution-chart", config={"displayModeBar": False})]), md=12, className="mb-4"),
        ]),

        html.H4("Attendance Ledger", className="mt-2 mb-3", style={"color": "#4a4742", "font-weight": "600"}),
        html.Div(style={"border-radius": "12px", "overflow": "hidden", "border": "1px solid #e4dfd5"}, children=[
            dash_table.DataTable(
                id="attendance-table",
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
                    'backgroundColor': '#f4f1ea', #type:ignore
                }]
            )
        ], className="mb-5")

    ], fluid=True)
])

@app.callback(
    [
        Output("total-records", "children"),
        Output("total-present", "children"),
        Output("attendance-rate", "children"),
        Output("total-employees", "children"),
        Output("employee-attendance-chart", "figure"),
        Output("day-attendance-chart", "figure"),
        Output("status-distribution-chart", "figure"),
        Output("attendance-table", "data"),
        Output("attendance-table", "columns"),
        Output("ai-insights", "children"),
    ],
    [
        Input("status-filter", "value"),
        Input("day-filter", "value"),
    ]
)
def update_dashboard(statuses, days):
    statuses = statuses or list(df["Status"].dropna().unique())
    days = days or list(df["Day_of_Week"].unique())

    filtered = df[
        df["Status"].isin(statuses) &
        df["Day_of_Week"].isin(days)
    ]

    records, present, rate, employees = calculate_kpis(filtered)

    if not filtered.empty:
        emp_df = filtered.groupby("Name")["Is_Present"].sum().reset_index().sort_values(by="Is_Present", ascending=False)
        top_bottom_emp = pd.concat([emp_df.head(5), emp_df.tail(5)]).drop_duplicates()
        employee_chart = px.bar(top_bottom_emp, x="Name", y="Is_Present", title="Top and Bottom Employee Attendance (Total Days Present)")
        employee_chart.update_traces(marker_color="#c5a880", marker_line_color="#b0956f", marker_line_width=1)
        
        day_df = filtered.groupby("Day_of_Week")["Is_Present"].sum().reset_index()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_df["Day_of_Week"] = pd.Categorical(day_df["Day_of_Week"], categories=day_order, ordered=True)
        day_df = day_df.sort_values("Day_of_Week")
        day_chart = px.bar(day_df, x="Day_of_Week", y="Is_Present", title="Attendance Analysis by Day of Week")
        day_chart.update_traces(marker_color="#a3b19b", marker_line_color="#8d9c85", marker_line_width=1)
        
        status_df = filtered.groupby("Status")["Employee ID"].count().reset_index().rename(columns={"Employee ID": "Count"})
        status_chart = px.pie(status_df, names="Status", values="Count", title="Attendance Status Distribution Channel Share", hole=0.4)
        status_chart.update_traces(textinfo='percent+label', marker=dict(colors=COLOR_PALETTE))
    else:
        employee_chart, day_chart, status_chart = px.bar(), px.bar(), px.pie()

    for fig in [employee_chart, day_chart, status_chart]:
        fig.update_layout(**PLOTLY_LIGHT_LAYOUT)
        fig.update_layout(title={"font": {"size": 14, "color": "#2d2a26"}})

    status_chart.update_layout(showlegend=False)

    table_columns = [{"name": i, "id": i} for i in filtered.columns if i not in ["Is_Present", "Day_of_Week"]]

    return (
        f"{records:,}",
        f"{present:,}",
        f"{rate:.1f}%",
        f"{employees:,}",
        employee_chart,
        day_chart,
        status_chart,
        filtered.to_dict("records"),
        table_columns,
        generate_insights(filtered)
    )

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))