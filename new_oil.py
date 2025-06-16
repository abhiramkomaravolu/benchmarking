import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ────────────────────────────────────────────────────────────────────────────────
# Helper for both historical and current parsing
def parse_scope(val):
    if pd.isna(val):
        return (np.nan, None)
    parts = str(val).split()
    if len(parts) == 2:
        try:
            num = float(parts[0].replace(",", ""))
        except:
            num = np.nan
        return (num, parts[1])
    return (np.nan, None)

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD & CLEAN DATA
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_and_clean():
    df = pd.read_csv("synthetic_oil_projects.csv")
    # apply parse_scope to historical
    sc = df["Scope (km or m3)"].apply(parse_scope)
    df["Scope_Value"] = sc.map(lambda x: x[0])
    df["Scope_Unit"]  = sc.map(lambda x: x[1])
    # convert percentages
    for col in ["Delay %", "Rework Cost %"]:
        if col in df:
            df[col + "_Num"] = (
                df[col].astype(str)
                      .str.rstrip("%")
                      .replace("", np.nan)
                      .astype(float)
            )
    return df

df = load_and_clean()

# ────────────────────────────────────────────────────────────────────────────────
# 1.3) DEFINE COLUMNS
# ────────────────────────────────────────────────────────────────────────────────

numeric_cols = [c for c in ["CapEx ($M)", "Duration (months)", "Safety Score",
                            "Scope_Value", "Delay %_Num", "Rework Cost %_Num"]
                if c in df.columns]
cat_cols_all = [c for c in ["Project ID", "Location", "Type", "Scope_Unit", "Delivery Model"]
                if c in df.columns]
pie_cols     = [c for c in cat_cols_all if c != "Project ID"]

# ────────────────────────────────────────────────────────────────────────────────
# 2) SIDEBAR & TAB SELECTION
# ────────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Oil Projects Benchmarking")
tabs = ["Dashboard","Overview","Distribution","Correlation","Scatter",
        "Normalize","Top10","Download","Insights","Compare"]
selection = st.sidebar.radio("Select Tab", tabs)

# Always show uploader when not in Dashboard
uploaded = None
cur = None
hist = df.copy()
if selection != "Dashboard":
    uploaded = st.sidebar.file_uploader("Upload new project (1-row CSV)", type="csv")
    if uploaded:
        cur = pd.read_csv(uploaded)
        # clean current
        sc2 = parse_scope(cur["Scope (km or m3)"].iloc[0]) if "Scope (km or m3)" in cur else (np.nan,None)
        cur["Scope_Value"], cur["Scope_Unit"] = sc2
        for col in ["Delay %", "Rework Cost %"]:
            if col in cur:
                cur[col + "_Num"] = float(str(cur[col].iloc[0]).rstrip("%") or np.nan)
        # filter historical peers
        mask = pd.Series(True, index=df.index)
        if "Location" in cur:
            mask &= df["Location"] == cur["Location"].iloc[0]
        if "Type" in cur:
            mask &= df["Type"] == cur["Type"].iloc[0]
        if "Delivery Model" in cur and "Delivery Model" in df:
            mask &= df["Delivery Model"] == cur["Delivery Model"].iloc[0]
        hist = df[mask].copy()

# ────────────────────────────────────────────────────────────────────────────────
# 3) PANELS
# ────────────────────────────────────────────────────────────────────────────────

# Dashboard
if selection == "Dashboard":
    st.header("Dashboard Overview")
    st.metric("Historical Projects", hist.shape[0])
    for c in numeric_cols:
        st.metric(f"Avg {c}", f"{hist[c].mean():.2f}")
    if cur is not None:
        st.markdown("**Current Project**")
        st.table(cur)

# Data Overview
elif selection == "Overview":
    st.header("Data Overview")
    st.subheader("Historical Sample")
    st.dataframe(hist.head())
    if cur is not None:
        st.subheader("Current Project")
        st.table(cur)

# Distribution & Outliers
elif selection == "Distribution":
    st.header("Distribution & Outliers")
    col = st.selectbox("Numeric column:", numeric_cols)
    fig, ax = plt.subplots()
    ax.hist(hist[col].dropna(), bins=20, edgecolor="black", alpha=0.7)
    if cur is not None:
        val = cur[col].iloc[0]
        ax.axvline(val, color="red", linewidth=2, label="Current")
        ax.legend()
    st.pyplot(fig)
    Q1, Q3 = hist[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    st.write(f"Outlier bounds: [{Q1-1.5*IQR:.2f}, {Q3+1.5*IQR:.2f}]")
    if pie_cols:
        st.markdown("---")
        cat = st.selectbox("Categorical:", pie_cols)
        fig2, ax2 = plt.subplots()
        ax2.pie(hist[cat].value_counts(), labels=hist[cat].value_counts().index, autopct="%1.1f%%")
        st.pyplot(fig2)

# Correlation & Significance
elif selection == "Correlation":
    st.header("Correlation & Significance")
    if cur is not None:
        st.subheader("Current Project Metrics")
        st.table(cur[numeric_cols])
    # apply additional filters if desired
    corr_df = hist
    corr = corr_df[numeric_cols].corr()
    # p-value matrix
    pmat = pd.DataFrame(np.ones((len(numeric_cols),)*2), index=numeric_cols, columns=numeric_cols)
    for i,a in enumerate(numeric_cols):
        for j,b in enumerate(numeric_cols[i+1:], i+1):
            _, p = pearsonr(corr_df[a].dropna(), corr_df[b].dropna())
            pmat.loc[a,b] = pmat.loc[b,a] = p
    st.subheader("P-Value Matrix")
    st.dataframe(pmat.style.format("{:.2e}"))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Historical Correlation Heatmap")
        fig, ax = plt.subplots()
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols))); ax.set_xticklabels(numeric_cols, rotation=45)
        ax.set_yticks(range(len(numeric_cols))); ax.set_yticklabels(numeric_cols)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
    with col2:
        if cur is not None:
            st.subheader("Current Normalized Heatmap")
            cur_vals = cur[numeric_cols].iloc[0]
            mn, mx = hist[numeric_cols].min(), hist[numeric_cols].max()
            normed = (cur_vals - mn) / (mx - mn)
            fig2, ax2 = plt.subplots()
            im2 = ax2.imshow([normed.values], cmap="Reds", vmin=0, vmax=1)
            ax2.set_xticks(range(len(numeric_cols))); ax2.set_xticklabels(numeric_cols, rotation=45)
            ax2.set_yticks([0]); ax2.set_yticklabels(["Current"])
            fig2.colorbar(im2, ax=ax2)
            st.pyplot(fig2)

# Pairwise Scatter
elif selection == "Scatter":
    st.header("Pairwise Scatter Plots")
    x_col = st.selectbox("X-axis:", numeric_cols)
    y_col = st.selectbox("Y-axis:", numeric_cols, index=1)
    fig, ax = plt.subplots()
    ax.scatter(hist[x_col], hist[y_col], alpha=0.3, color="gray")
    if cur is not None:
        ax.scatter(cur[x_col], cur[y_col], color="red", s=100, label="Current")
        ax.legend()
    st.pyplot(fig)
    if len(hist[x_col].dropna()) > 1 and len(hist[y_col].dropna()) > 1:
        r, p = pearsonr(hist[x_col].dropna(), hist[y_col].dropna())
        st.write(f"r = {r:.2f}, p = {p:.2e}, R² = {r*r:.2f}")

# Normalization & Clustering
elif selection == "Normalize":
    st.header("Normalization & Clustering")
    if st.checkbox("Show Z–score Normalized"):
        norm = hist[numeric_cols].apply(lambda s: (s - s.mean()) / s.std())
        st.dataframe(norm.head())
        if cur is not None:
            cn = (cur[numeric_cols].iloc[0] - norm.mean()) / norm.std()
            st.write("Current Normalized:", cn.to_frame().T)
    if st.checkbox("Run K–Means"):
        norm = hist[numeric_cols].apply(lambda s: (s - s.mean()) / s.std())
        k = st.slider("Clusters", 2, 10, 3)
        km = KMeans(n_clusters=k, random_state=1).fit(norm)
        pcs = PCA(n_components=2).fit_transform(norm)
        fig, ax = plt.subplots()
        ax.scatter(pcs[:,0], pcs[:,1], c=km.labels_, alpha=0.7)
        if cur is not None:
            pc_cur = PCA(n_components=2).fit(norm).transform((cur[numeric_cols] - norm.mean()) / norm.std())
            ax.scatter(pc_cur[0,0], pc_cur[0,1], color="red", s=100, label="Current")
            ax.legend()
        st.pyplot(fig)

# Top 10 Projects
elif selection == "Top10":
    st.header("Top 10 Projects")
    metric = st.selectbox("Metric:", numeric_cols)
    df10 = hist.sort_values(metric, ascending=False).head(10)
    st.dataframe(df10)
    fig, ax = plt.subplots()
    ax.barh(df10["Project ID"], df10[metric])
    ax.invert_yaxis()
    if cur is not None:
        ax.axvline(cur[metric].iloc[0], color="red", linewidth=2, label="Current")
        ax.legend()
    st.pyplot(fig)

# Download Reports
elif selection == "Download":
    st.header("Download Reports")
    st.download_button("Descriptive Stats", hist[numeric_cols].describe().to_csv(), "desc.csv")
    st.download_button("Correlation Matrix", hist[numeric_cols].corr().to_csv(), "corr.csv")

# Actionable Insights
elif selection == "Insights":
    st.header("Actionable Insights")
    corr = hist[numeric_cols].corr()
    pmat = pd.DataFrame(np.ones((len(numeric_cols),)*2), index=numeric_cols, columns=numeric_cols)
    for i,a in enumerate(numeric_cols):
        for j,b in enumerate(numeric_cols[i+1:], i+1):
            _, p = pearsonr(hist[a].dropna(), hist[b].dropna())
            pmat.loc[a,b] = pmat.loc[b,a] = p
    strong = [
        f"{a}↔{b}: r={corr.loc[a,b]:.2f}, p={pmat.loc[a,b]:.2e}"
        for a in numeric_cols for b in numeric_cols
        if a!=b and abs(corr.loc[a,b])>0.5 and pmat.loc[a,b]<0.05
    ]
    if strong:
        st.write("Strong correlations:", strong)
    else:
        st.write("No strong correlations found.")

# Current vs Historical (dedicated tab)
elif selection == "Compare":
    st.header("Current vs Historical Comparison")
    if cur is not None:
        st.subheader("Current Project")
        st.table(cur)
        st.subheader(f"{len(hist)} Historical Peers")
        st.dataframe(hist)
        col = st.selectbox("Compare Metric:", numeric_cols)
        hvals = hist[col].dropna()
        fig, ax = plt.subplots()
        ax.hist(hvals, bins=20, edgecolor="black", alpha=0.7)
        cv = cur[col].iloc[0]
        ax.axvline(cv, color="red", linewidth=2, label="Current")
        ax.legend()
        st.pyplot(fig)
        st.write(f"Percentile: {(hvals < cv).mean()*100:.1f}th")
    else:
        st.info("Upload a current project to compare.")
