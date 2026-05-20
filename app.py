import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="IRR Predictor — Real Estate ML",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main .block-container { max-width: 1300px; padding: 1.5rem 2rem; }
section[data-testid="stSidebar"] { display: none; }
.header-box {
    background: linear-gradient(135deg, #0A2463 0%, #1E5FA8 100%);
    padding: 26px 36px; border-radius: 12px; margin-bottom: 24px;
}
.header-title { color: white; font-size: 26px; font-weight: 900; letter-spacing: 1px; }
.header-sub   { color: #D6E4F7; font-size: 13px; margin-top: 4px; }
.kpi-card {
    background: white; border-radius: 10px; padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
}
.kpi-val  { font-size: 22px; font-weight: 900; color: #0A2463; }
.kpi-lbl  { font-size: 11px; color: #888; text-transform: uppercase; font-weight: 600; }
.pred-box {
    background: linear-gradient(135deg, #0A2463, #1E5FA8);
    border-radius: 14px; padding: 28px; text-align: center; color: white;
}
.pred-val { font-size: 52px; font-weight: 900; }
.pred-lbl { font-size: 14px; color: #D6E4F7; margin-top: 4px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #E0E0E0; }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 14px; padding: 10px 20px; border-radius: 8px 8px 0 0; color: #555; }
.stTabs [aria-selected="true"] { background:#0A2463 !important; color:white !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <div class="header-title">🏗️ Real Estate IRR Predictor</div>
  <div class="header-sub">Machine Learning · Random Forest · Scikit-learn · Streamlit</div>
</div>
""", unsafe_allow_html=True)

# ── DATOS Y MODELO ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

@st.cache_resource
def train_model(df):
    le_type = LabelEncoder()
    le_loc  = LabelEncoder()
    df = df.copy()
    df["project_type_enc"] = le_type.fit_transform(df["project_type"])
    df["location_enc"]     = le_loc.fit_transform(df["location"])

    features = ["land_cost","total_cost","capex_ratio","units","sqm",
                "cost_per_sqm","exit_value","holding_years",
                "project_type_enc","location_enc"]
    X = df[features]
    y = df["irr"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    return model, le_type, le_loc, features, X_test, y_test, y_pred, mae, r2

df = load_data()
model, le_type, le_loc, features, X_test, y_test, y_pred, mae, r2 = train_model(df)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Análisis Exploratorio", "🤖 Predictor IRR", "📈 Rendimiento del Modelo"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Exploración del Dataset")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(df)}</div><div class="kpi-lbl">Proyectos</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{df["irr"].mean()*100:.1f}%</div><div class="kpi-lbl">IRR Promedio</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">${df["total_cost"].mean()/1e6:.1f}M</div><div class="kpi-lbl">CAPEX Promedio</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{df["holding_years"].mean():.1f} años</div><div class="kpi-lbl">Tenencia Promedio</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="irr", nbins=30, color_discrete_sequence=["#0A2463"],
                           title="Distribución de IRR", labels={"irr": "IRR"})
        fig.update_traces(xbins=dict(size=0.01))
        fig.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        irr_by_type = df.groupby("project_type")["irr"].mean().reset_index().sort_values("irr", ascending=False)
        fig2 = px.bar(irr_by_type, x="project_type", y="irr",
                      color="irr", color_continuous_scale=["#D6E4F7","#0A2463"],
                      title="IRR Promedio por Tipo de Proyecto",
                      labels={"project_type":"Tipo","irr":"IRR"})
        fig2.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20), coloraxis_showscale=False)
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(df, x="total_cost", y="irr", color="project_type",
                          title="CAPEX Total vs IRR",
                          labels={"total_cost":"CAPEX Total ($)","irr":"IRR","project_type":"Tipo"},
                          color_discrete_sequence=px.colors.qualitative.Bold)
        fig3.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
        fig3.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        irr_by_loc = df.groupby("location")["irr"].mean().reset_index().sort_values("irr", ascending=False)
        fig4 = px.bar(irr_by_loc, x="irr", y="location", orientation="h",
                      color="irr", color_continuous_scale=["#D6E4F7","#0A2463"],
                      title="IRR Promedio por Ubicación",
                      labels={"location":"Ubicación","irr":"IRR"})
        fig4.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20), coloraxis_showscale=False)
        fig4.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Predice el IRR de tu proyecto")
    st.markdown("Ingresa las características del proyecto y el modelo estimará el IRR esperado.")

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        c1, c2 = st.columns(2)
        with c1:
            land_cost_in   = st.number_input("💰 Costo del Terreno ($)", 200_000, 5_000_000, 1_500_000, step=50_000)
            units_in       = st.number_input("🏠 Unidades", 5, 60, 20)
            exit_year_in   = st.selectbox("📅 Año de Salida", [2025,2026,2027,2028,2029], index=1)
            proj_type_in   = st.selectbox("🏗️ Tipo de Proyecto", ["Residencial","Comercial","Mixto","Hotel/Boutique"])
        with c2:
            total_cost_in  = st.number_input("🏗️ CAPEX Total ($)", 500_000, 20_000_000, 4_500_000, step=100_000)
            sqm_in         = st.number_input("📐 Área Construida (m²)", 200, 6_000, 1_800, step=50)
            exit_value_in  = st.number_input("🏷️ Valor de Salida ($)", 500_000, 30_000_000, 8_000_000, step=100_000)
            location_in    = st.selectbox("📍 Ubicación", ["Casco Antiguo","Santa Ana","Miraflores","San Francisco","Bella Vista"])

        predict_btn = st.button("🔮 Predecir IRR", use_container_width=True, type="primary")

    with col_result:
        if predict_btn:
            capex_ratio_in  = total_cost_in / land_cost_in if land_cost_in > 0 else 2.0
            cost_per_sqm_in = total_cost_in / sqm_in if sqm_in > 0 else 2000
            holding_in      = exit_year_in - 2022

            type_enc = le_type.transform([proj_type_in])[0]
            loc_enc  = le_loc.transform([location_in])[0]

            X_pred = pd.DataFrame([{
                "land_cost":       land_cost_in,
                "total_cost":      total_cost_in,
                "capex_ratio":     capex_ratio_in,
                "units":           units_in,
                "sqm":             sqm_in,
                "cost_per_sqm":    cost_per_sqm_in,
                "exit_value":      exit_value_in,
                "holding_years":   holding_in,
                "project_type_enc":type_enc,
                "location_enc":    loc_enc,
            }])

            irr_pred = model.predict(X_pred)[0]
            profit   = exit_value_in - total_cost_in
            roi_pred = profit / total_cost_in if total_cost_in > 0 else 0

            # Nivel de riesgo
            if irr_pred >= 0.20:
                nivel, color = "Alto Retorno 🟢", "#2ECC71"
            elif irr_pred >= 0.12:
                nivel, color = "Retorno Moderado 🟡", "#F39C12"
            else:
                nivel, color = "Retorno Bajo 🔴", "#E74C3C"

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-lbl">IRR Estimado</div>
                <div class="pred-val">{irr_pred*100:.1f}%</div>
                <div style="margin-top:8px;font-size:14px;color:{color};font-weight:700;">{nivel}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("ROI", f"{roi_pred*100:.1f}%")
            with m2:
                st.metric("Utilidad", f"${profit:,.0f}")
            with m3:
                st.metric("CAPEX/Terreno", f"{capex_ratio_in:.2f}x")

            # Comparar con el mercado
            market_avg = df["irr"].mean()
            diff = irr_pred - market_avg
            st.markdown(f"""
            <div style="background:#EFF3FA;border-radius:8px;padding:12px;margin-top:12px;font-size:13px;">
                📊 IRR promedio del mercado: <b>{market_avg*100:.1f}%</b><br>
                Tu proyecto está <b style="color:{'#2ECC71' if diff>=0 else '#E74C3C'}">
                {'▲' if diff>=0 else '▼'} {abs(diff)*100:.1f}pp {'por encima' if diff>=0 else 'por debajo'}</b> del promedio.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#EFF3FA;border-radius:12px;padding:40px;text-align:center;color:#888;">
                <div style="font-size:40px;">🔮</div>
                <div style="font-size:14px;margin-top:12px;">Completa los datos del proyecto<br>y presiona <b>Predecir IRR</b></div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODELO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Rendimiento del Modelo")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{r2*100:.1f}%</div><div class="kpi-lbl">R² Score</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{mae*100:.2f}pp</div><div class="kpi-lbl">Error Absoluto Medio</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">200</div><div class="kpi-lbl">Árboles (n_estimators)</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=y_test*100, y=y_pred*100, mode="markers",
                                       marker=dict(color="#0A2463", opacity=0.5, size=5),
                                       name="Predicciones"))
        lim = [min(y_test.min(), y_pred.min())*100, max(y_test.max(), y_pred.max())*100]
        fig_pred.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                       line=dict(color="#E74C3C", dash="dash"), name="Ideal"))
        fig_pred.update_layout(title="Real vs Predicho", height=340,
                               xaxis_title="IRR Real (%)", yaxis_title="IRR Predicho (%)",
                               margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_pred, use_container_width=True)

    with col_b:
        importances = pd.DataFrame({
            "Feature": ["Valor Salida","CAPEX Total","Terreno","Costo/m²","Área m²",
                        "CAPEX Ratio","Holding","Unidades","Ubicación","Tipo Proyecto"],
            "Importancia": model.feature_importances_
        }).sort_values("Importancia", ascending=True)

        fig_imp = px.bar(importances, x="Importancia", y="Feature", orientation="h",
                         color="Importancia", color_continuous_scale=["#D6E4F7","#0A2463"],
                         title="Importancia de Variables")
        fig_imp.update_layout(height=340, margin=dict(t=40,b=20,l=20,r=20), coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.markdown("**Algoritmo:** Random Forest Regressor · **Train/Test split:** 80/20 · **Dataset:** 600 proyectos inmobiliarios sintéticos con distribuciones basadas en datos reales del mercado panameño.")
