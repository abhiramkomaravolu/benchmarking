import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD DATA
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_excel("buildings_synthetic_output.xlsx")

df = load_data()

# Define columns
numeric_cols = ["GFA (m²)", "Height", "CapEx ($M)"]
cat_cols_all = ["Project ID", "Type"]
pie_cols     = [c for c in cat_cols_all if c != "Project ID"]  # → ["Type"]

# ────────────────────────────────────────────────────────────────────────────────
# 2) UPLOAD CURRENT & FILTER HISTORICAL
# ────────────────────────────────────────────────────────────────────────────────

cur = None
hist = df.copy()

uploaded = st.sidebar.file_uploader("Upload current building (1-row CSV)", type="csv")
if uploaded:
    cur = pd.read_csv(uploaded)
    # filter hist to same Type
    if "Type" in cur.columns:
        hist = df[df["Type"] == cur["Type"].iloc[0]].copy()

# ────────────────────────────────────────────────────────────────────────────────
# 3) SIDEBAR CONTROLS
# ────────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Buildings Benchmarking")
show_dash     = st.sidebar.checkbox("0. Dashboard", True)
show_overview = st.sidebar.checkbox("1. Data Overview", False)
show_dist     = st.sidebar.checkbox("2. Distribution & Outliers", False)
show_corr     = st.sidebar.checkbox("3. Correlation & Significance", False)
show_scatter  = st.sidebar.checkbox("4. Pairwise Scatter Plots", False)
show_norm     = st.sidebar.checkbox("5. Normalization & Clustering", False)
show_top10    = st.sidebar.checkbox("6. Top 10", False)
show_dl       = st.sidebar.checkbox("7. Download Reports", False)
show_insights = st.sidebar.checkbox("8. Actionable Insights", False)

st.sidebar.markdown("---")
st.sidebar.write("Numeric fields:")
for c in numeric_cols: st.sidebar.write(f"• {c}")
st.sidebar.write("Categorical fields:")
for c in cat_cols_all: st.sidebar.write(f"• {c}")

# ────────────────────────────────────────────────────────────────────────────────
# 4) PANEL 0: DASHBOARD
# ────────────────────────────────────────────────────────────────────────────────

if show_dash:
    st.header("0. Dashboard Overview")
    st.metric("Historical Projects", hist.shape[0])
    st.metric("Avg GFA (m²)", f"{hist['GFA (m²)'].mean():.0f}")
    st.metric("Avg Height", f"{hist['Height'].mean():.1f}")
    st.metric("Avg CapEx ($M)", f"{hist['CapEx ($M)'].mean():.1f}")
    if cur is not None:
        st.markdown("**Current Project**")
        st.table(cur)

# ────────────────────────────────────────────────────────────────────────────────
# 5) PANEL 1: DATA OVERVIEW
# ────────────────────────────────────────────────────────────────────────────────

elif show_overview:
    st.header("1. Data Overview")
    st.subheader("Historical Sample")
    st.dataframe(hist.head())
    if cur is not None:
        st.subheader("Current Project")
        st.table(cur)

# ────────────────────────────────────────────────────────────────────────────────
# 6) PANEL 2: DISTRIBUTION & OUTLIERS
# ────────────────────────────────────────────────────────────────────────────────

elif show_dist:
    st.header("2. Distribution & Outlier Analysis")
    col = st.selectbox("Select a numeric column:", numeric_cols)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(hist[col].dropna(), bins=20, edgecolor="black", alpha=0.7)
    if cur is not None:
        v = cur[col].iloc[0]
        ax.axvline(v, color="red", linewidth=2, label="Current")
        ax.legend()
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    Q1, Q3 = hist[col].quantile([.25, .75])
    IQR = Q3 - Q1
    st.write(f"Bounds: [{Q1-1.5*IQR:.2f}, {Q3+1.5*IQR:.2f}] → Outliers: {((hist[col]<Q1-1.5*IQR)|(hist[col]>Q3+1.5*IQR)).sum()}")
    if st.checkbox("Show outlier rows"):
        mask = (hist[col]<Q1-1.5*IQR)|(hist[col]>Q3+1.5*IQR)
        st.dataframe(hist.loc[mask].head(10))

    if st.checkbox("Show boxplot for all numeric"):
        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.boxplot([hist[c].dropna() for c in numeric_cols], labels=numeric_cols, vert=False)
        ax2.set_title("Boxplot of All Numeric Columns")
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Categorical Distribution")
    cat = st.selectbox("Select a categorical column:", pie_cols)
    counts = hist[cat].value_counts()
    fig3, ax3 = plt.subplots(figsize=(5,5))
    ax3.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    ax3.set_title(f"{cat} Distribution")
    st.pyplot(fig3)

# ────────────────────────────────────────────────────────────────────────────────
# 7) PANEL 3: CORRELATION & SIGNIFICANCE
# ────────────────────────────────────────────────────────────────────────────────

elif show_corr:
    st.header("3. Correlation & Significance")
    if cur is not None:
        st.subheader("Current Project Metrics")
        st.table(cur[numeric_cols])

    corr = hist[numeric_cols].corr()
    st.write("### Pearson Correlation Matrix")
    st.dataframe(corr)

    if st.checkbox("Show heatmap"):
        fig4, ax4 = plt.subplots(figsize=(6,6))
        im = ax4.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax4.set_xticks(range(len(numeric_cols))); ax4.set_xticklabels(numeric_cols, rotation=45)
        ax4.set_yticks(range(len(numeric_cols))); ax4.set_yticklabels(numeric_cols)
        fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        st.pyplot(fig4)

    if st.checkbox("Show p-value matrix"):
        pmat = pd.DataFrame(np.ones((len(numeric_cols),)*2), index=numeric_cols, columns=numeric_cols)
        for i,a in enumerate(numeric_cols):
            for j,b in enumerate(numeric_cols[i+1:], i+1):
                _,p = pearsonr(hist[a].dropna(), hist[b].dropna())
                pmat.loc[a,b] = pmat.loc[b,a] = p
        st.write("### P-Value Matrix")
        st.dataframe(pmat.style.format("{:.2e}"))

    if st.checkbox("Show R² (Correlation²) Matrix"):
        st.write("### R² Matrix")
        st.dataframe(corr**2)

# ────────────────────────────────────────────────────────────────────────────────
# 8) PANEL 4: PAIRWISE SCATTER PLOTS
# ────────────────────────────────────────────────────────────────────────────────

elif show_scatter:
    st.header("4. Pairwise Scatter Plots")
    x_col = st.selectbox("X-axis column:", numeric_cols, index=0, key="sx")
    y_col = st.selectbox("Y-axis column:", numeric_cols, index=1, key="sy")
    fig5, ax5 = plt.subplots(figsize=(6,4))
    ax5.scatter(hist[x_col], hist[y_col], alpha=0.3, color="gray")
    if cur is not None:
        ax5.scatter(cur[x_col], cur[y_col], color="red", s=100, label="Current")
        ax5.legend()
    ax5.set_xlabel(x_col); ax5.set_ylabel(y_col)
    st.pyplot(fig5)
    if len(hist[x_col].dropna())>1 and len(hist[y_col].dropna())>1:
        r,p = pearsonr(hist[x_col].dropna(), hist[y_col].dropna())
        st.write(f"r = {r:.2f}, p = {p:.2e}, R² = {r*r:.2f}")
    if st.checkbox("Show scatter matrix"):
        sel = st.multiselect("Pick ≥2 columns:", numeric_cols, default=numeric_cols[:3], key="sm")
        if len(sel)>=2:
            fig6 = scatter_matrix(hist[sel], alpha=0.6, diagonal="hist", figsize=(8,8))
            st.pyplot(plt.gcf())

# ────────────────────────────────────────────────────────────────────────────────
# 9) PANEL 5: NORMALIZATION & CLUSTERING
# ────────────────────────────────────────────────────────────────────────────────

elif show_norm:
    st.header("5. Normalization & Clustering")
    if st.checkbox("Show Z-score normalized"):
        norm = hist[numeric_cols].apply(lambda s:(s-s.mean())/s.std())
        st.dataframe(norm.head())
        if cur is not None:
            cn = (cur[numeric_cols].iloc[0]-norm.mean())/norm.std()
            st.write("Current normalized:", cn.to_frame().T)
    if st.checkbox("Run K-Means"):
        norm = hist[numeric_cols].apply(lambda s:(s-s.mean())/s.std())
        k = st.slider("Clusters", 2, 10, 3)
        km = KMeans(n_clusters=k, random_state=42).fit(norm)
        pcs = PCA(2).fit_transform(norm)
        fig7, ax7 = plt.subplots(figsize=(6,4))
        scatter = ax7.scatter(pcs[:,0], pcs[:,1], c=km.labels_, cmap="tab10", alpha=0.7)
        ax7.legend(*scatter.legend_elements(), title="Cluster")
        if cur is not None:
            pc_cur = PCA(2).fit(norm).transform((cur[numeric_cols]-norm.mean())/norm.std())
            ax7.scatter(pc_cur[0,0], pc_cur[0,1], color="red", s=100, label="Current")
            ax7.legend()
        st.pyplot(fig7)

# ────────────────────────────────────────────────────────────────────────────────
# 10) PANEL 6: TOP 10 PROJECTS
# ────────────────────────────────────────────────────────────────────────────────

elif show_top10:
    st.header("6. Top 10 Projects")
    metric = st.selectbox("Metric:", numeric_cols, key="t10")
    order  = st.radio("Sort order:", ["Descending","Ascending"])
    asc    = (order=="Ascending")
    df10   = hist.sort_values(metric, ascending=asc).head(10)
    st.dataframe(df10.reset_index(drop=True))
    fig8, ax8 = plt.subplots(figsize=(6,4))
    ax8.barh(df10["Project ID"].astype(str), df10[metric], color="skyblue")
    ax8.invert_yaxis()
    if cur is not None:
        ax8.axvline(cur[metric].iloc[0], color="red", linewidth=2, label="Current")
        ax8.legend()
    st.pyplot(fig8)

# ────────────────────────────────────────────────────────────────────────────────
# 11) PANEL 7: DOWNLOAD REPORTS
# ────────────────────────────────────────────────────────────────────────────────

elif show_dl:
    st.header("7. Download Reports")
    st.download_button("Descriptive Stats", hist[numeric_cols].describe().to_csv(), "desc.csv")
    st.download_button("Correlation Matrix", hist[numeric_cols].corr().to_csv(), "corr.csv")

# ────────────────────────────────────────────────────────────────────────────────
# 12) PANEL 8: ACTIONABLE INSIGHTS
# ────────────────────────────────────────────────────────────────────────────────

elif show_insights:
    st.header("8. Actionable Insights")
    corr = hist[numeric_cols].corr()
    pmat = pd.DataFrame(np.ones((len(numeric_cols),)*2), index=numeric_cols, columns=numeric_cols)
    for i,a in enumerate(numeric_cols):
        for j,b in enumerate(numeric_cols[i+1:], i+1):
            _,p = pearsonr(hist[a].dropna(), hist[b].dropna())
            pmat.loc[a,b] = pmat.loc[b,a] = p

    strong = [
        f"{a}↔{b}: r={corr.loc[a,b]:.2f}, p={pmat.loc[a,b]:.2e}"
        for a in numeric_cols for b in numeric_cols
        if a!=b and abs(corr.loc[a,b])>0.5 and pmat.loc[a,b]<0.05
    ]
    if strong:
        st.write("**Strong correlations:**")
        for s in strong: st.write("-", s)
    else:
        st.write("No strong correlations found.")
