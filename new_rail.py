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
    return pd.read_csv("rail_synthetic_formatted.csv")

df = load_data()

numeric_cols  = df.select_dtypes("number").columns.tolist()
cat_cols_all  = df.select_dtypes("object").columns.tolist()
location_col  = "Region" if "Region" in df.columns else None
industry_col  = "Delivery Model" if "Delivery Model" in df.columns else None
project_col   = "Project ID"
pie_cols      = [c for c in cat_cols_all if c != project_col]
carbon_col    = next((c for c in numeric_cols if "Carbon" in c), None)

# ────────────────────────────────────────────────────────────────────────────────
# 2) UPLOAD CURRENT & FILTER HISTORICAL
# ────────────────────────────────────────────────────────────────────────────────

cur = None
hist = df.copy()

uploaded = st.sidebar.file_uploader("Upload current project (1-row CSV)", type="csv")
if uploaded:
    cur = pd.read_csv(uploaded)

    # filter hist to same Region & Delivery Model
    mask = pd.Series(True, index=df.index)
    if location_col and location_col in cur.columns:
        mask &= df[location_col] == cur[location_col].iloc[0]
    if industry_col and industry_col in cur.columns:
        mask &= df[industry_col] == cur[industry_col].iloc[0]
    hist = df[mask].copy()

    # 2.1) Compute Cost_Eff = Scope_Value / CapEx if present, else fallback to 1/CapEx
    if "Scope_Value" in hist.columns and "CapEx ($M)" in hist.columns:
        hist["Cost_Eff"] = hist["Scope_Value"] / hist["CapEx ($M)"]
        cur["Cost_Eff"]  = cur["Scope_Value"].iloc[0] / cur["CapEx ($M)"].iloc[0]
    elif "CapEx ($M)" in hist.columns:
        hist["Cost_Eff"] = 1.0 / hist["CapEx ($M)"]
        cur["Cost_Eff"]  = 1.0 / cur["CapEx ($M)"].iloc[0]

    # 2.2) Compute Quality_Index = 1 - normalized safety incidents
    if "Safety Incidents" in hist.columns:
        mn, mx = hist["Safety Incidents"].min(), hist["Safety Incidents"].max()
        hist["Quality_Index"] = 1 - (hist["Safety Incidents"] - mn) / (mx - mn)
        val = cur["Safety Incidents"].iloc[0]
        cur["Quality_Index"] = 1 - (val - mn) / (mx - mn)

# ────────────────────────────────────────────────────────────────────────────────
# 3) SIDEBAR NAVIGATION
# ────────────────────────────────────────────────────────────────────────────────

tabs = [
    "Dashboard","Overview","Distribution","Correlation","Scatter",
    "Normalize","Top10","Download","Insights","Visualizations"
]
selection = st.sidebar.radio("Choose panel", tabs)

# ────────────────────────────────────────────────────────────────────────────────
# 4) PANEL 0: DASHBOARD
# ────────────────────────────────────────────────────────────────────────────────

if selection == "Dashboard":
    st.header("0. Dashboard Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Projects", hist.shape[0])
    if carbon_col:
        c2.metric("Avg Carbon Intensity", f"{hist[carbon_col].mean():.1f}")
    else:
        c2.metric("Avg Carbon Intensity", "N/A")
    c3.metric("Avg CapEx ($M)", f"{hist['CapEx ($M)'].mean():.1f}")
    c4.metric("Median Duration (months)", f"{hist['Duration (months)'].median():.1f}")
    c5.metric("Avg Safety Incidents", f"{hist['Safety Incidents'].mean():.1f}")
    if cur is not None:
        st.markdown("**Current Project**")
        st.table(cur)

# ────────────────────────────────────────────────────────────────────────────────
# 5) PANEL 1: OVERVIEW
# ────────────────────────────────────────────────────────────────────────────────

elif selection == "Overview":
    st.header("1. Data Overview")
    st.subheader("Historical Sample")
    st.dataframe(hist.head())
    if cur is not None:
        st.subheader("Current Project")
        st.table(cur)

# ────────────────────────────────────────────────────────────────────────────────
# 6) PANEL 2: DISTRIBUTION & OUTLIERS
# ────────────────────────────────────────────────────────────────────────────────

elif selection == "Distribution":
    st.header("2. Distribution & Outlier Analysis")
    col = st.selectbox("Numeric column:", numeric_cols)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(hist[col].dropna(), bins=20, edgecolor="black", alpha=0.7)
    if cur is not None:
        v = cur[col].iloc[0]
        ax.axvline(v, color="crimson", linewidth=2, label="Current")
        ax.legend()
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    Q1, Q3 = hist[col].quantile([.25, .75])
    IQR = Q3 - Q1
    st.write(f"Outlier bounds: [{Q1-1.5*IQR:.2f}, {Q3+1.5*IQR:.2f}]")

    if pie_cols:
        st.markdown("### Categorical Distribution")
        cat = st.selectbox("Category for pie:", pie_cols)
        counts = hist[cat].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5,5))
        ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
        ax2.axis("equal")
        st.pyplot(fig2)

# ────────────────────────────────────────────────────────────────────────────────
# 7) PANEL 3: CORRELATION & SIGNIFICANCE
# ────────────────────────────────────────────────────────────────────────────────

elif selection == "Correlation":
    st.header("3. Correlation & Significance")
    if cur is not None:
        st.subheader("Current Project Metrics")
        st.table(cur[numeric_cols])

    corr = hist[numeric_cols].corr()
    pmat = pd.DataFrame(np.ones((len(numeric_cols),)*2),
                        index=numeric_cols, columns=numeric_cols)
    for i,a in enumerate(numeric_cols):
        for j,b in enumerate(numeric_cols[i+1:], i+1):
            _, p = pearsonr(hist[a].dropna(), hist[b].dropna())
            pmat.loc[a,b] = pmat.loc[b,a] = p

    st.subheader("P-Value Matrix")
    st.dataframe(pmat.style.format("{:.2e}"))

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5,5))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_yticklabels(numeric_cols)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
    with c2:
        if cur is not None:
            cur_vals = cur[numeric_cols].iloc[0]
            mn, mx = hist[numeric_cols].min(), hist[numeric_cols].max()
            normed = (cur_vals - mn) / (mx - mn)
            fig2, ax2 = plt.subplots(figsize=(5,2))
            im2 = ax2.imshow([normed.values], cmap="Reds", vmin=0, vmax=1)
            ax2.set_xticks(range(len(numeric_cols)))
            ax2.set_xticklabels(numeric_cols, rotation=45, ha="right")
            ax2.set_yticks([0])
            ax2.set_yticklabels(["Current"])
            fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            st.pyplot(fig2)

# ────────────────────────────────────────────────────────────────────────────────
# 8) PANEL 4: PAIRWISE SCATTER
# ────────────────────────────────────────────────────────────────────────────────

elif selection == "Scatter":
    st.header("4. Pairwise Scatter Plots")
    x = st.selectbox("X-axis:", numeric_cols, key="sx")
    y = st.selectbox("Y-axis:", numeric_cols, index=1, key="sy")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(hist[x], hist[y], alpha=0.4, color="slategray")
    if cur is not None:
        ax.scatter(cur[x], cur[y], color="crimson", s=100, label="Current")
        ax.legend()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)

    if len(hist[x].dropna())>1 and len(hist[y].dropna())>1:
        r, p = pearsonr(hist[x].dropna(), hist[y].dropna())
        st.write(f"r = {r:.2f}, p = {p:.2e}, R² = {r*r:.2f}")

# ────────────────────────────────────────────────────────────────────────────────
# 9) PANEL 5: NORMALIZATION & CLUSTERING
# ────────────────────────────────────────────────────────────────────────────────

elif selection == "Normalize":
    st.header("5. Normalization & Clustering")
    if st.checkbox("Show Z-score normalized"):
        norm = hist[numeric_cols].apply(lambda s: (s - s.mean()) / s.std())
        st.dataframe(norm.head())
        if cur is not None:
            cn = (cur[numeric_cols].iloc[0] - norm.mean()) / norm.std()
            st.write("Current normalized:", cn.to_frame().T)
    if st.checkbox("Run K-Means clustering"):
        norm = hist[numeric_cols].apply(lambda s: (s - s.mean()) / s.std())
        k = st.slider("Clusters:", 2, 10, 3)
        km = KMeans(n_clusters=k, random_state=42).fit(norm)
        pcs = PCA(n_components=2).fit_transform(norm)
        fig, ax = plt.subplots(figsize=(6,4))
        sc = ax.scatter(pcs[:,0], pcs[:,1], c=km.labels_, cmap="tab10", alpha=0.7)
        legend1 = ax.legend(*sc.legend_elements(), title="Cluster")
        ax.add_artist(legend1)
        if cur is not None:
            pc_cur = PCA(n_components=2).fit(norm).transform((cur[numeric_cols] - norm.mean()) / norm.std())
            ax.scatter(pc_cur[0,0], pc_cur[0,1], color="crimson", s=120, label="Current")
            ax.legend()
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────────
# 10) PANEL 6: TOP 10 PROJECTS
# ────────────────────────────────────────────────────────────────────────────────

elif selection == "Top10":
    st.header("6. Top 10 Projects by Metric")
    metric = st.selectbox("Metric:", numeric_cols, key="t10")
    df10 = hist.sort_values(metric, ascending=False).head(10)
    st.table(df10[[project_col, metric]].assign(Rank=range(1,11)).set_index("Rank"))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(df10[project_col].astype(str), df10[metric], color="teal")
    ax.invert_yaxis()
    if cur is not None:
        ax.axvline(cur[metric].iloc[0], color="crimson", linewidth=2, label="Current")
        ax.legend()
    ax.set_xlabel(metric)
    st.pyplot(fig)

# ────────────────────────────────────────────────────────────────────────────────
# 11) PANEL 7: DOWNLOAD REPORTS
# ────────────────────────────────────────────────────────────────────────────────

elif selection == "Download":
    st.header("7. Download Reports")
    st.download_button("Descriptive Statistics", hist[numeric_cols].describe().to_csv(), "desc.csv")
    st.download_button("Correlation Matrix", hist[numeric_cols].corr().to_csv(), "corr.csv")

# ────────────────────────────────────────────────────────────────────────────────
# 12) PANEL 8: ACTIONABLE INSIGHTS
# ────────────────────────────────────────────────────────────────────────────────

elif selection == "Insights":
    st.header("8. Actionable Insights")
    corr = hist[numeric_cols].corr()
    pmat = pd.DataFrame(np.ones((len(numeric_cols),)*2), index=numeric_cols, columns=numeric_cols)
    for i,a in enumerate(numeric_cols):
        for j,b in enumerate(numeric_cols[i+1:], i+1):
            _, p = pearsonr(hist[a].dropna(), hist[b].dropna())
            pmat.loc[a,b] = pmat.loc[b,a] = p
    strong = [
        f"{a}↔{b}: r={corr.loc[a,b]:.2f}, p={pmat.loc[a,b]:.2e}"
        for a in numeric_cols for b in numeric_cols if a!=b
        if abs(corr.loc[a,b])>0.5 and pmat.loc[a,b]<0.05
    ]
    if strong:
        st.write("**Strong correlations:**")
        for s in strong:
            st.write("•", s)
    else:
        st.write("No strong correlations found.")

# ────────────────────────────────────────────────────────────────────────────────
# 13) PANEL 9: VISUALIZATIONS
# ────────────────────────────────────────────────────────────────────────────────

elif selection == "Visualizations":
    st.header("9. Custom Visualizations")

    # 1) Projects by Region
    st.subheader("Projects by Region")
    counts = hist[location_col].value_counts().reindex(
        ["USA","Africa","Asia","South America","Australia","Europe","Canada"],
        fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(counts.index, counts.values, color="#A8D568")
    ax.set_xlabel("Region"); ax.set_ylabel("Number of Projects")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    st.pyplot(fig)

    st.markdown("---")

    # 2) Delivery Model Breakdown
    st.subheader("Delivery Model Breakdown")
    dm_counts = hist[industry_col].value_counts()
    colors = ["#5DA5DA","#FAA43A","#60BD68","#F17CB0","#B2912F"]
    fig2, ax2 = plt.subplots(figsize=(5,5))
    wedges, texts, autotexts = ax2.pie(
        dm_counts.values,
        labels=dm_counts.index,
        colors=colors[:len(dm_counts)],
        autopct="%1.0f%%",
        textprops=dict(color="white", weight="bold"),
        startangle=140
    )
    ax2.axis("equal")
    ax2.legend(wedges, dm_counts.index, loc="lower center", bbox_to_anchor=(0.5,-0.1), ncol=3)
    st.pyplot(fig2)

    st.markdown("---")

    # 3) Avg CapEx by Region
    st.subheader("Avg CapEx by Region")
    avg_by_region = hist.groupby(location_col)["CapEx ($M)"].mean().reindex(counts.index)
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.bar(avg_by_region.index, avg_by_region.values, color="#6DD3B0")
    ax3.set_xlabel("Region"); ax3.set_ylabel("Avg CapEx ($M)")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    st.pyplot(fig3)

    st.markdown("---")

    # 4) Schedule Performance (Radar)
    st.subheader("Schedule Performance")
    categories   = ["Design","Construction","Testing","Commissioning","Handover"]
    baseline     = [0.6,0.5,0.55,0.4,0.5]
    current_vals = [0.7,0.6,0.65,0.5,0.6] if cur is not None else [0]*5
    angles       = np.linspace(0,2*np.pi,len(categories),endpoint=False).tolist()
    vals_b       = baseline + baseline[:1]
    vals_c       = current_vals + current_vals[:1]
    angles       = angles + angles[:1]

    fig4, ax4 = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax4.set_theta_offset(np.pi/2)
    ax4.set_theta_direction(-1)
    ax4.set_ylim(0,1)
    ax4.yaxis.grid(True, color="gray", linestyle="--", alpha=0.5)
    ax4.xaxis.grid(False)
    ax4.plot(angles, vals_b, color="gray", linewidth=1, linestyle="--")
    ax4.plot(angles, vals_c, color="#A8D568", linewidth=2)
    ax4.fill(angles, vals_c, color="#A8D568", alpha=0.25)
    ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(categories)
    ax4.set_yticklabels([])
    ax4.legend(["Baseline","Current"], loc="upper right", bbox_to_anchor=(1.1,1.1))
    st.pyplot(fig4)

    st.markdown("---")

    # 5) Cost Effectiveness
    st.subheader("Cost Effectiveness")
    bench_ce = hist["Cost_Eff"].mean() if "Cost_Eff" in hist.columns else 0
    ours_ce  = cur["Cost_Eff"].iloc[0] if (cur is not None and "Cost_Eff" in cur.columns) else 0
    fig5, ax5 = plt.subplots(figsize=(7,2))
    ax5.barh(["Industry Benchmark"], [bench_ce], color="#D3D3D3")
    ax5.barh(["Our Project"],       [ours_ce],  color="#FFD23F")
    ax5.set_xlabel("Scope per $M")
    ax5.set_xlim(0, max(bench_ce, ours_ce)*1.2)
    ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)
    st.pyplot(fig5)

    st.markdown("---")

    # 6) Quality Index (Donut)
    st.subheader("Quality Index")
    bench_q = hist["Quality_Index"].mean() if "Quality_Index" in hist.columns else 0
    ours_q  = cur["Quality_Index"].iloc[0] if (cur is not None and "Quality_Index" in cur.columns) else 0
    labels  = ["Our Project","Industry Benchmark"]
    sizes   = [ours_q, bench_q]
    if sum(sizes) == 0:
        sizes = [1, 0]
    fig6, ax6 = plt.subplots(figsize=(5,5))
    wedges, texts, autotexts = ax6.pie(
        sizes,
        labels=labels,
        colors=["#FF8A9B","#D3D3D3"],
        startangle=90,
        wedgeprops=dict(width=0.3),
        pctdistance=0.8,
        autopct=lambda pct: f"{pct:.0f}%",
        textprops=dict(color="black", weight="bold")
    )
    ax6.axis("equal")
    ax6.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5,-0.1), ncol=2)
    st.pyplot(fig6)
