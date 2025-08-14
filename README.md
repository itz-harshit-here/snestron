    # snestron
    # smart_data_analyzer.py
    # Run: streamlit run smart_data_analyzer.py
    
    import io
    import json
    import textwrap
    from datetime import datetime
    
    import numpy as np
    import pandas as pd
    import streamlit as st
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        r2_score,
        mean_absolute_error,
        mean_squared_error,
    )
    
    st.set_page_config(
        page_title="Smart Data Analyzer",
        page_icon="📊",
        layout="wide",
    )
    
    # ---------------------------
    # Utility functions
    # ---------------------------
    
    def load_data(uploaded_file) -> pd.DataFrame:
        if uploaded_file is None:
            return None
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file)
        if name.endswith(".json"):
            # Try JSON lines first, then normal JSON
            try:
                return pd.read_json(uploaded_file, lines=True)
            except ValueError:
                uploaded_file.seek(0)
                data = json.load(uploaded_file)
                return pd.json_normalize(data)
        # Fallback: try CSV parsing
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    
    
    def memory_friendly_info(df: pd.DataFrame) -> pd.DataFrame:
        d = []
        for col in df.columns:
            d.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_count": df[col].notna().sum(),
                "null_count": df[col].isna().sum(),
                "unique": df[col].nunique(dropna=True),
                "sample": str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] else ""
            })
        return pd.DataFrame(d)
    
    
    def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            return pd.DataFrame()
        desc = df[num_cols].describe().T
        desc["missing"] = df[num_cols].isna().sum()
        return desc
    
    
    def describe_categorical(df: pd.DataFrame, top_k=5) -> pd.DataFrame:
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        rows = []
        for col in cat_cols:
            vc = df[col].value_counts(dropna=True)
            common = ", ".join([f"{idx} ({cnt})" for idx, cnt in vc.head(top_k).items()])
            rows.append({
                "column": col,
                "unique": df[col].nunique(dropna=True),
                "top_values": common
            })
        return pd.DataFrame(rows)
    
    
    def handle_missing_values(df: pd.DataFrame, strategy: str, columns=None, fill_value=None) -> pd.DataFrame:
        df = df.copy()
        if columns is None or len(columns) == 0:
            columns = df.columns.tolist()
        if strategy == "drop-rows":
            return df.dropna(subset=columns)
        if strategy in {"mean", "median", "mode"}:
            for col in columns:
                if strategy == "mode":
                    val = df[col].mode(dropna=True)
                    val = val.iloc[0] if not val.empty else np.nan
                elif strategy == "mean":
                    val = df[col].mean()
                else:
                    val = df[col].median()
                df[col] = df[col].fillna(val)
            return df
        if strategy == "constant":
            return df.fillna({c: fill_value for c in columns})
        return df
    
    
    def detect_outliers_iqr(df: pd.DataFrame, k=1.5) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            return pd.DataFrame(index=df.index, data={"is_outlier": False})
        bounds = {}
        for col in num_cols:
            q1, q3 = np.percentile(df[col].dropna(), [25, 75]) if df[col].notna().any() else (np.nan, np.nan)
            iqr = q3 - q1 if pd.notna(q3) and pd.notna(q1) else np.nan
            if pd.isna(iqr) or iqr == 0:
                bounds[col] = (np.nan, np.nan)
            else:
                lb = q1 - k * iqr
                ub = q3 + k * iqr
                bounds[col] = (lb, ub)

    mask = pd.Series(False, index=df.index)
    for col, (lb, ub) in bounds.items():
        if pd.notna(lb) and pd.notna(ub):
            mask |= (df[col] < lb) | (df[col] > ub)

    out = pd.DataFrame({"is_outlier": mask})
    return out


    def detect_outliers_zscore(df: pd.DataFrame, z_thresh=3.0) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            return pd.DataFrame(index=df.index, data={"is_outlier": False})
        zmask = pd.Series(False, index=df.index)
        for col in num_cols:
            series = df[col]
            mu = series.mean()
            sigma = series.std(ddof=0)
            if pd.isna(sigma) or sigma == 0:
                continue
            z = (series - mu) / sigma
            zmask |= z.abs() > z_thresh
        return pd.DataFrame({"is_outlier": zmask})
    
    
    def auto_task_type(series: pd.Series) -> str:
        if series.dtype.kind in "biufc":
            # Numeric: decide by number of distinct values
            uniq = series.dropna().nunique()
            return "regression" if uniq > 15 else "classification"
        return "classification"

    
    def build_pipeline(df: pd.DataFrame, target: str, task: str):
        X = df.drop(columns=[target])
        y = df[target]

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipe, X, y


    def top_correlations(df: pd.DataFrame, top_n=10):
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] < 2:
            return pd.DataFrame(columns=["feature_1", "feature_2", "corr"])
        corr = num_df.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().reset_index()
        pairs.columns = ["feature_1", "feature_2", "corr"]
        pairs = pairs.sort_values("corr", ascending=False).head(top_n)
        return pairs
    
    
    def generate_report_md(
        df_original: pd.DataFrame,
        df_clean: pd.DataFrame,
        info_df: pd.DataFrame,
        num_desc: pd.DataFrame,
        cat_desc: pd.DataFrame,
        outlier_summary: dict,
        corr_pairs: pd.DataFrame,
        model_summary: dict | None,
    ) -> str:
        lines = []
        lines.append(f"# Smart Data Analyzer Report")
        lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    lines.append("## Dataset overview")
    lines.append(f"- Original shape: {df_original.shape[0]} rows × {df_original.shape[1]} columns")
    lines.append(f"- Cleaned shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns\n")

    lines.append("### Schema snapshot")
    head = info_df[["column", "dtype", "non_null_count", "null_count", "unique"]].copy()
    lines.append(head.to_markdown(index=False))
    lines.append("")

    lines.append("## Descriptive statistics")
    if not num_desc.empty:
        lines.append("### Numeric columns")
        lines.append(num_desc.round(4).to_markdown())
        lines.append("")
    if not cat_desc.empty:
        lines.append("### Categorical columns (top values)")
        lines.append(cat_desc.to_markdown(index=False))
        lines.append("")

    lines.append("## Outliers")
    lines.append(f"- Method: {outlier_summary.get('method')}")
    lines.append(f"- Outlier rows flagged: {outlier_summary.get('count', 0)}")
    lines.append("")

    lines.append("## Correlations (top pairs by absolute value)")
    if corr_pairs.empty:
        lines.append("- Not enough numeric columns for correlation.")
    else:
        lines.append(corr_pairs.to_markdown(index=False))
    lines.append("")

    if model_summary:
        lines.append("## Quick modeling")
        lines.append(f"- Task: {model_summary['task']}")
        lines.append(f"- Target: {model_summary['target']}")
        lines.append(f"- Train/Test split: {model_summary['split']}")
        lines.append(f"- Metrics:")
        for k, v in model_summary["metrics"].items():
            lines.append(f"  - {k}: {v}")
        if model_summary.get("top_features"):
            lines.append("- Top features:")
            for name, imp in model_summary["top_features"]:
                lines.append(f"  - {name}: {imp:.4f}")
        lines.append("")

    lines.append("---")
    lines.append("Report generated by Smart Data Analyzer.")
    return "\n".join(lines)


    # ---------------------------
    # Sidebar — Controls
    # ---------------------------
    
    with st.sidebar:
        st.title("📊 Smart Data Analyzer")
        st.caption("Upload your dataset and explore it intelligently.")

    uploaded = st.file_uploader("Upload a CSV, Excel (.xlsx), or JSON file", type=["csv", "xlsx", "xls", "json"])

    st.markdown("---")
    st.subheader("Data cleaning")
    drop_dupes = st.checkbox("Drop duplicate rows", value=True)

    miss_strategy = st.selectbox(
        "Missing value strategy",
        ["none", "drop-rows", "mean", "median", "mode", "constant"],
        index=1
    )
    fill_value = None
    if miss_strategy == "constant":
        fill_value = st.text_input("Fill constant value (applied as string; numeric cols will attempt cast)", value="0")

    st.markdown("---")
    st.subheader("Outlier detection")
    outlier_method = st.selectbox("Method", ["IQR", "Z-score"], index=0)
    iqr_k = st.slider("IQR k (1.0–3.0 typical)", 0.5, 5.0, 1.5, 0.1)
    z_thresh = st.slider("Z-score threshold", 1.0, 6.0, 3.0, 0.1)

    st.markdown("---")
    st.subheader("Quick modeling")
    enable_model = st.checkbox("Enable quick ML modeling", value=False)
    
    # ---------------------------
    # Main — App Body
    # ---------------------------
    
    st.title("Smart Data Analyzer")
    st.write("A one-stop app to explore, clean, visualize, and model your data — fast.")
    
    if uploaded is None:
        st.info("Upload a dataset to begin.")
        st.stop()
    
    df_original = load_data(uploaded)
    if df_original is None or df_original.empty:
        st.error("Failed to load the file or the dataset is empty.")
        st.stop()
    
    # Clone for cleaning operations
    df = df_original.copy()

    # Drop duplicates if selected
    if drop_dupes:
        before = len(df)
        df = df.drop_duplicates()
        st.caption(f"Removed {before - len(df)} duplicate rows.")
    
    # Missing value handling
    if miss_strategy != "none":
        cols_for_missing = st.multiselect(
            "Columns to apply missing-value strategy to (leave blank for all)",
            options=df.columns.tolist(),
            default=[]
        )
        if miss_strategy == "constant" and fill_value is not None:
            # Attempt numeric cast where possible
            try:
                fv_cast = float(fill_value)
            except ValueError:
                fv_cast = fill_value
            df = handle_missing_values(df, "constant", columns=cols_for_missing, fill_value=fv_cast)
        else:
            df = handle_missing_values(df, miss_strategy, columns=cols_for_missing or None)

    # ---------------------------
    # Tabs
    # ---------------------------
    
    tab_overview, tab_visuals, tab_outliers, tab_model, tab_report = st.tabs(
        ["Overview", "Visualize", "Outliers", "Model", "Report"]
    )
    
    with tab_overview:
        st.subheader("Dataset preview")
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Schema and quality")
    info_df = memory_friendly_info(df)
    st.dataframe(info_df, use_container_width=True, height=300)

    st.subheader("Descriptive statistics")
    num_desc = describe_numeric(df)
    if not num_desc.empty:
        st.markdown("Numeric columns")
        st.dataframe(num_desc, use_container_width=True)
    else:
        st.info("No numeric columns found.")

    cat_desc = describe_categorical(df)
    if not cat_desc.empty:
        st.markdown("Categorical columns (top values)")
        st.dataframe(cat_desc, use_container_width=True)

    st.subheader("Correlation heatmap (numeric)")
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(num_df.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns to compute correlations.")

    with tab_visuals:
        st.subheader("Univariate")
        cols_num = df.select_dtypes(include=np.number).columns.tolist()
        cols_cat = df.select_dtypes(exclude=np.number).columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        if cols_num:
            col_num = st.selectbox("Numeric column", cols_num, key="uni_num_col")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[col_num].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col_num}")
            st.pyplot(fig)
        else:
            st.info("No numeric columns found.")

    with c2:
        if cols_cat:
            col_cat = st.selectbox("Categorical column", cols_cat, key="uni_cat_col")
            vc = df[col_cat].value_counts(dropna=False).head(30)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=vc.values, y=vc.index, ax=ax)
            ax.set_title(f"Top categories of {col_cat}")
            st.pyplot(fig)
        else:
            st.info("No categorical columns found.")

    st.subheader("Bivariate")
    if len(cols_num) >= 2:
        xcol = st.selectbox("X (numeric)", cols_num, key="bi_x")
        ycol = st.selectbox("Y (numeric)", cols_num, key="bi_y")
        hue = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist(), key="bi_hue")
        fig, ax = plt.subplots(figsize=(7, 5))
        if hue != "None":
            sns.scatterplot(data=df, x=xcol, y=ycol, hue=hue, ax=ax)
        else:
            sns.scatterplot(data=df, x=xcol, y=ycol, ax=ax)
        ax.set_title(f"{ycol} vs {xcol}")
        st.pyplot(fig)
    else:
        st.info("Need at least two numeric columns for scatter plot.")
    
    with tab_outliers:
        st.subheader("Outlier detection")
        if outlier_method == "IQR":
            out_mask_df = detect_outliers_iqr(df, k=iqr_k)
            out_method_used = f"IQR (k={iqr_k})"
        else:
            out_mask_df = detect_outliers_zscore(df, z_thresh=z_thresh)
            out_method_used = f"Z-score (threshold={z_thresh})"

    out_count = int(out_mask_df["is_outlier"].sum())
    st.markdown(f"- Method: **{out_method_used}**")
    st.markdown(f"- Outlier rows flagged: **{out_count}**")

    show_outliers = st.checkbox("Show outlier rows", value=(out_count <= 200))
    if show_outliers and out_count > 0:
        st.dataframe(df[out_mask_df["is_outlier"]], use_container_width=True, height=300)

    # Optionally remove outliers
    remove_outliers = st.checkbox("Remove flagged outliers from the working dataset")
    if remove_outliers:
        df = df.loc[~out_mask_df["is_outlier"]].copy()
        st.success(f"Removed {out_count} rows. New shape: {df.shape[0]} × {df.shape[1]}")
    
    with tab_model:
        st.subheader("Quick ML modeling")
        if not enable_model:
            st.info("Enable 'Quick modeling' in the sidebar to use this feature.")
        else:
            target = st.selectbox("Select target column", df.columns.tolist())
            # Infer task if user doesn't override
            inferred = auto_task_type(df[target])
            task = st.selectbox("Task type", ["classification", "regression"], index=0 if inferred == "classification" else 1,
                                help="Automatically inferred from the target; override if needed.")
            test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)

        try:
            pipe, X, y = build_pipeline(df, target, task)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if task == "classification" and y.nunique() > 1 else None
            )
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            metrics = {}
            if task == "classification":
                metrics["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
                # Compute F1 macro if valid
                try:
                    metrics["f1_macro"] = round(float(f1_score(y_test, preds, average="macro")), 4)
                except Exception:
                    pass
                st.write("Metrics:", metrics)
                st.text("Classification report:")
                try:
                    st.code(classification_report(y_test, preds), language="text")
                except Exception:
                    st.info("Classification report unavailable for the current target format.")
            else:
                metrics["R2"] = round(float(r2_score(y_test, preds)), 4)
                metrics["MAE"] = round(float(mean_absolute_error(y_test, preds)), 4)
                metrics["RMSE"] = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
                st.write("Metrics:", metrics)

            # Feature importances
            try:
                model = pipe.named_steps["model"]
                pre = pipe.named_steps["preprocess"]
                feature_names = []
                if hasattr(pre, "get_feature_names_out"):
                    feature_names = pre.get_feature_names_out()
                importances = getattr(model, "feature_importances_", None)
                if importances is not None and len(feature_names) == len(importances):
                    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
                    imp_df = imp_df.sort_values("importance", ascending=False).head(20)
                    st.markdown("Top features by importance")
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Feature importances not available for this configuration.")
            except Exception as e:
                st.info("Could not compute feature importances.")

            # Keep summary in session state for report
            st.session_state["model_summary"] = {
                "task": task,
                "target": target,
                "split": f"{int((1-test_size)*100)}% train / {int(test_size*100)}% test",
                "metrics": metrics,
                "top_features": list(zip(imp_df["feature"], imp_df["importance"])) if "imp_df" in locals() else None
            }
        except Exception as e:
            st.error(f"Modeling failed: {e}")

    with tab_report:
        st.subheader("Generate report")

    # Recompute items for report context (based on current df)
    info_now = memory_friendly_info(df)
    num_desc_now = describe_numeric(df)
    cat_desc_now = describe_categorical(df)
    if outlier_method == "IQR":
        out_mask_now = detect_outliers_iqr(df, k=iqr_k)
        out_method_used = f"IQR (k={iqr_k})"
    else:
        out_mask_now = detect_outliers_zscore(df, z_thresh=z_thresh)
        out_method_used = f"Z-score (threshold={z_thresh})"
    outlier_summary = {
        "method": out_method_used,
        "count": int(out_mask_now["is_outlier"].sum())
    }
    corr_pairs_now = top_correlations(df, top_n=10)

    model_summary = st.session_state.get("model_summary")

    report_md = generate_report_md(
        df_original=df_original,
        df_clean=df,
        info_df=info_now,
        num_desc=num_desc_now,
        cat_desc=cat_desc_now,
        outlier_summary=outlier_summary,
        corr_pairs=corr_pairs_now,
        model_summary=model_summary
    )

    st.markdown("Preview")
    st.code(report_md, language="markdown")

    filename = f"Smart_Data_Analyzer_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    st.download_button(
        label="Download Markdown report",
        data=report_md.encode("utf-8"),
        file_name=filename,
        mime="text/markdown"
    )

    st.caption("Tip: Use the tabs to explore, clean, visualize, model, and then export a report.")
