import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="RegressLab", page_icon="🚀", layout="wide")

# ---------- CSS ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(124,92,255,.18), transparent 40%),
            radial-gradient(circle at 85% 20%, rgba(0,224,255,.14), transparent 40%),
            #0b0f1a;
    }
    #MainMenu, footer {visibility: hidden;}

    @keyframes fadeUp { from {opacity:0; transform: translateY(18px);} to {opacity:1; transform: none;} }
    @keyframes shine { 0% {background-position: 0% 50%;} 100% {background-position: 200% 50%;} }

    .hero {
        padding: 2.4rem 2.6rem; border-radius: 24px; margin-bottom: 1.6rem;
        background: linear-gradient(120deg, rgba(124,92,255,.35), rgba(0,224,255,.20));
        border: 1px solid rgba(255,255,255,.12);
        backdrop-filter: blur(14px); animation: fadeUp .7s ease both;
    }
    .hero h1 {
        margin: 0; font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(90deg,#fff,#a8b9ff,#6ef3ff,#fff);
        background-size: 200% auto; -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; animation: shine 6s linear infinite;
    }
    .hero p { margin: .5rem 0 0; color: #c9d2f0; font-size: 1.05rem; }
    .badge {
        display:inline-block; padding: .25rem .8rem; margin: .8rem .4rem 0 0;
        border-radius: 999px; font-size: .78rem; font-weight: 600;
        background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18);
    }

    .kpi {
        padding: 1.2rem 1.3rem; border-radius: 20px; animation: fadeUp .8s ease both;
        background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1);
        backdrop-filter: blur(10px); transition: all .25s ease;
    }
    .kpi:hover { transform: translateY(-5px); border-color: rgba(124,92,255,.8);
                 box-shadow: 0 10px 30px rgba(124,92,255,.35); }
    .kpi .lbl { font-size: .8rem; color: #9fb0e8; letter-spacing: .08em; text-transform: uppercase; }
    .kpi .val { font-size: 1.9rem; font-weight: 800; margin-top: .2rem;
                background: linear-gradient(90deg,#8f7bff,#4fe3ff);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .kpi .sub { font-size: .78rem; color: #8895c4; }

    .insight {
        padding: 1.1rem 1.4rem; border-radius: 16px; margin-top: 1rem;
        background: rgba(0,224,255,.07); border-left: 4px solid #4fe3ff; color: #dbe6ff;
    }
    .stTabs [data-baseweb="tab-list"] { gap: .5rem; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px; padding: .6rem 1.2rem; background: rgba(255,255,255,.05);
    }
    .stTabs [aria-selected="true"] { background: linear-gradient(90deg,#7c5cff,#3ab8ff) !important; color: #fff !important; }
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.08); }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# ---------- Sidebar ----------
st.sidebar.markdown("## 🚀 RegressLab")
sumber = st.sidebar.radio("Sumber data", ["Data contoh", "Upload CSV"])


@st.cache_data
def data_contoh():
    rng = np.random.default_rng(42)
    jam = rng.uniform(1, 10, 100)
    nilai = 38 + 6 * jam - 0.18 * jam**2 + rng.normal(0, 5, 100)
    return pd.DataFrame({"Jam_Belajar": jam.round(2), "Nilai_Ujian": nilai.round(2)})


if sumber == "Upload CSV":
    file = st.sidebar.file_uploader("Pilih file CSV", type="csv")
    if file is None:
        st.markdown(
            '<div class="hero"><h1>RegressLab</h1>'
            "<p>Upload CSV di sidebar, atau pilih <b>Data contoh</b>.</p></div>",
            unsafe_allow_html=True,
        )
        st.stop()
    df = pd.read_csv(file)
else:
    df = data_contoh()

kolom = df.select_dtypes(include="number").columns.tolist()
if len(kolom) < 2:
    st.error("Data harus punya minimal 2 kolom numerik.")
    st.stop()

x_col = st.sidebar.selectbox("Variabel X", kolom, index=0)
y_col = st.sidebar.selectbox("Variabel Y", kolom, index=1)
derajat = st.sidebar.slider("Derajat polinomial", 1, 5, 1,
                            help="1 = regresi linear biasa")
tampil_ci = st.sidebar.toggle("Tampilkan interval kepercayaan 95%", value=True)

if x_col == y_col:
    st.warning("Pilih variabel X dan Y yang berbeda.")
    st.stop()

data = df[[x_col, y_col]].dropna().sort_values(x_col)
X = data[[x_col]].values
y = data[y_col].values
n = len(data)

# ---------- Model ----------
poly = PolynomialFeatures(degree=derajat, include_bias=False)
Xp = poly.fit_transform(X)
model = LinearRegression().fit(Xp, y)
y_pred = model.predict(Xp)

r2 = r2_score(y, y_pred)
k = Xp.shape[1]
r2_adj = 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
resid = y - y_pred
se = np.sqrt(np.sum(resid**2) / max(n - k - 1, 1))

# ---------- Hero ----------
st.markdown(
    f"""
    <div class="hero">
        <h1>RegressLab 🚀</h1>
        <p>Analisis regresi interaktif: dari data mentah sampai prediksi dalam hitungan detik.</p>
        <span class="badge">X: {x_col}</span>
        <span class="badge">Y: {y_col}</span>
        <span class="badge">{n} observasi</span>
        <span class="badge">Derajat {derajat}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "R²", f"{r2:.3f}", f"{r2*100:.1f}% variasi terjelaskan")
kpi(c2, "Adjusted R²", f"{r2_adj:.3f}", "dikoreksi jumlah parameter")
kpi(c3, "RMSE", f"{rmse:.3f}", "rata-rata galat kuadrat")
kpi(c4, "MAE", f"{mae:.3f}", "rata-rata galat absolut")

if r2 >= 0.7:
    kekuatan = "kuat"
elif r2 >= 0.4:
    kekuatan = "sedang"
else:
    kekuatan = "lemah"

if derajat == 1:
    arah = "positif" if model.coef_[0] > 0 else "negatif"
    teks = (
        f"Setiap kenaikan 1 satuan <b>{x_col}</b> diikuti perubahan "
        f"<b>{model.coef_[0]:+.3f}</b> pada <b>{y_col}</b> (hubungan {arah}). "
        f"Kecocokan model tergolong <b>{kekuatan}</b> (R² = {r2:.3f})."
    )
else:
    teks = (
        f"Model polinomial derajat {derajat} menjelaskan <b>{r2*100:.1f}%</b> variasi "
        f"<b>{y_col}</b> (kecocokan {kekuatan}). Bandingkan Adjusted R² dengan derajat 1 "
        f"untuk mengecek apakah kenaikan derajat benar-benar membantu."
    )
st.markdown(f'<div class="insight">💡 {teks}</div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Regresi", "🔍 Diagnostik", "📊 Eksplorasi", "🔮 Prediksi", "📄 Data"]
)

TEMPLATE = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)")

# ---------- Tab 1: Regresi ----------
with tab1:
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_line = model.predict(poly.transform(x_line))

    fig = go.Figure()
    if tampil_ci:
        t = stats.t.ppf(0.975, max(n - k - 1, 1))
        fig.add_trace(go.Scatter(
            x=np.r_[x_line.ravel(), x_line.ravel()[::-1]],
            y=np.r_[y_line + t * se, (y_line - t * se)[::-1]],
            fill="toself", fillcolor="rgba(124,92,255,0.15)",
            line=dict(width=0), name="Interval prediksi 95%", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=data[x_col], y=data[y_col], mode="markers", name="Data",
        marker=dict(size=10, color=data[y_col], colorscale="Turbo", opacity=0.85,
                    line=dict(width=1, color="white"), showscale=False),
    ))
    fig.add_trace(go.Scatter(
        x=x_line.ravel(), y=y_line, mode="lines", name="Model",
        line=dict(color="#4fe3ff", width=4),
    ))
    fig.update_layout(**TEMPLATE, height=520, xaxis_title=x_col, yaxis_title=y_col,
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if derajat == 1:
        tanda = "+" if model.intercept_ >= 0 else "-"
        st.latex(rf"\hat{{y}} = {model.coef_[0]:.4f}\,x {tanda} {abs(model.intercept_):.4f}")

# ---------- Tab 2: Diagnostik ----------
with tab2:
    d1, d2 = st.columns(2)
    with d1:
        f1 = px.scatter(x=y_pred, y=resid, labels={"x": "Nilai prediksi", "y": "Residual"},
                        title="Residual vs Prediksi", color=np.abs(resid),
                        color_continuous_scale="Sunsetdark")
        f1.add_hline(y=0, line_dash="dash", line_color="#4fe3ff")
        f1.update_layout(**TEMPLATE, height=400, coloraxis_showscale=False)
        st.plotly_chart(f1, use_container_width=True)
    with d2:
        f2 = px.histogram(x=resid, nbins=20, title="Distribusi Residual",
                          color_discrete_sequence=["#7c5cff"])
        f2.update_layout(**TEMPLATE, height=400, xaxis_title="Residual")
        st.plotly_chart(f2, use_container_width=True)

    teori = stats.probplot(resid, dist="norm")
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=teori[0][0], y=teori[0][1], mode="markers",
                            marker=dict(color="#4fe3ff"), name="Residual"))
    f3.add_trace(go.Scatter(x=teori[0][0], y=teori[1][1] + teori[1][0] * teori[0][0],
                            mode="lines", line=dict(color="#ff5c8a", dash="dash"), name="Normal"))
    f3.update_layout(**TEMPLATE, height=380, title="Q-Q Plot (uji kenormalan residual)",
                     xaxis_title="Kuantil teoretis", yaxis_title="Kuantil sampel")
    st.plotly_chart(f3, use_container_width=True)

    if derajat == 1 and n > 2:
        r, p = stats.pearsonr(X.ravel(), y)
        st.markdown(
            f'<div class="insight">Korelasi Pearson r = <b>{r:.4f}</b>, '
            f'p-value = <b>{p:.2e}</b> → '
            f'{"signifikan" if p < 0.05 else "tidak signifikan"} pada α = 0,05.</div>',
            unsafe_allow_html=True,
        )

# ---------- Tab 3: Eksplorasi ----------
with tab3:
    e1, e2 = st.columns(2)
    with e1:
        fx = px.histogram(data, x=x_col, nbins=20, marginal="box",
                          title=f"Distribusi {x_col}", color_discrete_sequence=["#7c5cff"])
        fx.update_layout(**TEMPLATE, height=420)
        st.plotly_chart(fx, use_container_width=True)
    with e2:
        fy = px.histogram(data, x=y_col, nbins=20, marginal="box",
                          title=f"Distribusi {y_col}", color_discrete_sequence=["#3ab8ff"])
        fy.update_layout(**TEMPLATE, height=420)
        st.plotly_chart(fy, use_container_width=True)

    if len(kolom) > 2:
        korelasi = df[kolom].corr()
        fh = px.imshow(korelasi, text_auto=".2f", color_continuous_scale="Tealrose",
                       title="Matriks Korelasi", aspect="auto")
        fh.update_layout(**TEMPLATE, height=480)
        st.plotly_chart(fh, use_container_width=True)
    st.dataframe(data.describe().T, use_container_width=True)

# ---------- Tab 4: Prediksi ----------
with tab4:
    p1, p2 = st.columns([1, 1])
    with p1:
        nilai_x = st.slider(
            f"Nilai {x_col}",
            float(X.min()), float(X.max()), float(X.mean()),
        )
    hasil = float(model.predict(poly.transform([[nilai_x]]))[0])
    with p2:
        kpi(st, f"Prediksi {y_col}", f"{hasil:.3f}", f"untuk {x_col} = {nilai_x:.2f}")
    t = stats.t.ppf(0.975, max(n - k - 1, 1))
    st.markdown(
        f'<div class="insight">Rentang prediksi 95%: '
        f'<b>{hasil - t*se:.3f}</b> sampai <b>{hasil + t*se:.3f}</b></div>',
        unsafe_allow_html=True,
    )
    out = data.copy()
    out["Prediksi"] = y_pred
    out["Residual"] = resid
    st.download_button("⬇️ Unduh hasil (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="hasil_regresi.csv", mime="text/csv")

# ---------- Tab 5: Data ----------
with tab5:
    st.dataframe(data, use_container_width=True, height=450)
    st.caption(f"{n} baris data valid dipakai untuk model.")
