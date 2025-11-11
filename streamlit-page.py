import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt


st.set_page_config(page_title="Terms Analysis Results - Map & Charts", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str)
    def parse_counts(text: str):
        text = (text or "").strip().strip('"')
        if not text:
            return 0, {}
        total = 0
        details = {}
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                chap, val = token.split(":", 1)
                chap = chap.strip()
                try:
                    n = int(val.strip())
                except Exception:
                    n = 0
                total += n
                details[chap] = details.get(chap, 0) + n
            else:
                details[token] = details.get(token, 0) + 1
                total += 1
        return total, details
    totals, details = zip(*df["章節出現次數"].map(parse_counts))
    df["總次數"] = totals
    df["章節明細"] = details
    def parse_people(s: str):
        s = (s or "").strip().strip('"')
        if not s:
            return []
        return [x.strip() for x in s.split(",") if x.strip()]
    df["同行人物"] = df["同行出現人物匯總"].map(parse_people)
    city_coords = {
        "北京": (39.9042, 116.4074),
        "南京": (32.0603, 118.7969),
        "揚州": (32.3936, 119.4127),
        "蘇州": (31.2989, 120.5853),
        "杭州": (30.2741, 120.1551),
    }
    df["緯度"] = df["城市"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[0])
    df["經度"] = df["城市"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[1])
    return df


def explode_chapter_details(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        d = r["章節明細"] or {}
        for chap, n in d.items():
            rows.append({"稱呼": r["稱呼"], "城市": r["城市"], "章節": chap, "次數": n})
    if not rows:
        return pd.DataFrame(columns=["稱呼", "城市", "章節", "次數"])
    return pd.DataFrame(rows)


def make_city_agg(df: pd.DataFrame) -> pd.DataFrame:
    city_agg = (
        df.groupby(["城市"], as_index=False)["總次數"]
        .sum()
        .sort_values("總次數", ascending=False)
    )
    city_agg = city_agg.merge(
        df[["城市", "緯度", "經度"]].drop_duplicates(), on="城市", how="left"
    )
    city_agg = city_agg.dropna(subset=["緯度", "經度"])
    top_by_city = (
        df.sort_values(["城市", "總次數"], ascending=[True, False])
        .groupby("城市")
        .apply(
            lambda g: ", ".join(
                (g["稱呼"] + "(" + g["總次數"].astype(str) + ")").head(5).tolist()
            )
        )
        .rename("Top稱呼")
        .reset_index()
    )
    city_agg = city_agg.merge(top_by_city, on="城市", how="left")
    if not city_agg.empty:
        min_r, max_r = 4000, 20000
        v = city_agg["總次數"].to_numpy(dtype=float)
        v_min, v_max = float(v.min()), float(v.max())
        city_agg["半徑"] = (
            (min_r + max_r) / 2.0 if v_max == v_min else min_r + (v - v_min) / (v_max - v_min) * (max_r - min_r)
        )
    else:
        city_agg["半徑"] = 8000
    return city_agg


def aggregate_people(df: pd.DataFrame) -> pd.DataFrame:
    # 人物出现次数（出现即+1），亦可加權：與稱呼總次數相乘
    counts = {}
    weighted = {}
    for _, r in df.iterrows():
        people = r["同行人物"]
        for p in people:
            counts[p] = counts.get(p, 0) + 1
            weighted[p] = weighted.get(p, 0) + r["總次數"]
    rows = [{"人物": p, "出現行數": counts[p], "加權次數": weighted[p]} for p in counts]
    return pd.DataFrame(rows).sort_values("加權次數", ascending=False)


def main():
    base_dir = Path(__file__).parent
    csv_path = base_dir / "稱呼分析結果.csv"
    if not csv_path.exists():
        st.error(f"Data file not found: {csv_path}")
        st.stop()

    df = load_data(csv_path)

    st.title("Terms Analysis Results - Map & Charts")
    st.caption("Data source: 稱呼分析結果.csv")

    # 收集所有人物
    all_people = sorted({p for lst in df["同行人物"] for p in lst})

    with st.sidebar:
        st.header("Filters")
        all_cities = [c for c in df["城市"].dropna().unique().tolist() if c]
        selected_cities = st.multiselect("Select City", options=all_cities, default=all_cities)
        selected_people = st.multiselect("Select Character (co-occurrence)", options=all_people)
        kw = st.text_input("Terms keyword (fuzzy match)", value="")
        min_total = st.slider("Minimum total count", 0, int(df["總次數"].max() or 0), 0, step=1)
        top_k = st.slider("Top N (Terms bar chart)", 5, 30, 15, step=1)
        top_p = st.slider("Top N (Character co-occurrence)", 5, 30, 15, step=1)
        show_table = st.checkbox("Show Terms details table", value=False)
        show_people_table = st.checkbox("Show Character co-occurrence table", value=False)

    f = df.copy()
    if selected_cities:
        f = f[f["城市"].isin(selected_cities)]
    if kw.strip():
        key = kw.strip()
        f = f[f["稱呼"].str.contains(key, case=False, na=False)]
    if min_total > 0:
        f = f[f["總次數"] >= min_total]
    if selected_people:
        # 要求同行人物与所选有交集
        f = f[f["同行人物"].apply(lambda lst: any(p in lst for p in selected_people))]

    col_map, col_chart = st.columns([1.2, 1])

    city_agg = make_city_agg(f)
    with col_map:
        st.subheader("Map: Aggregated by City")
        if city_agg.empty:
            st.info("No matching city data")
        else:
            lat = np.average(city_agg["緯度"], weights=city_agg["總次數"])
            lon = np.average(city_agg["經度"], weights=city_agg["總次數"])
            tooltip = {
                "html": "<b>City:</b> {城市}<br/><b>Total Count:</b> {總次數}<br/><b>Top Termss:</b> {Top稱呼}",
                "style": {"backgroundColor": "white", "color": "black"},
            }
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=city_agg,
                get_position="[經度, 緯度]",
                get_radius="半徑",
                get_fill_color="[255, 99, 71, 140]",
                pickable=True,
            )
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=4.5),
                tooltip=tooltip,
            )
            st.pydeck_chart(deck)

    with col_chart:
        st.subheader("Chart: Top N Termss (by total count)")
        if f.empty:
            st.info("No matching Terms data")
        else:
            top_df = f.sort_values("總次數", ascending=False).head(top_k).copy()
            chart = (
                alt.Chart(top_df)
                .mark_bar()
                .encode(
                    x=alt.X("總次數:Q", title="Total Count"),
                    y=alt.Y("稱呼:N", sort="-x", title="Terms"),
                    color=alt.Color("城市:N", title="City"),
                    tooltip=[
                        alt.Tooltip("稱呼:N", title="Terms"),
                        alt.Tooltip("城市:N", title="City"),
                        alt.Tooltip("總次數:Q", title="Total Count"),
                    ],
                )
                .properties(height=420)
            )
            st.altair_chart(chart)

    st.subheader("Chapter Distribution (filtered results aggregated)")
    ch_long = explode_chapter_details(f)
    if ch_long.empty:
        st.info("No chapter distribution data")
    else:
        chap_agg = (
            ch_long.groupby("章節", as_index=False)["次數"]
            .sum()
            .sort_values("次數", ascending=False)
        )
        chap_chart = (
            alt.Chart(chap_agg)
            .mark_bar()
            .encode(
                x=alt.X("章節:N", sort=None, title="Chapter"),
                y=alt.Y("次數:Q", title="Count"),
                tooltip=[alt.Tooltip("章節:N", title="Chapter"), alt.Tooltip("次數:Q", title="Count")],
            )
            .properties(height=280)
        )
        st.altair_chart(chap_chart)

    st.subheader("Character Co-occurrence Analysis")
    people_agg = aggregate_people(f)
    if people_agg.empty:
        st.info("No character co-occurrence data")
    else:
        top_people = people_agg.head(top_p)
        p_chart = (
            alt.Chart(top_people)
            .mark_bar()
            .encode(
                x=alt.X("加權次數:Q", title="Weighted Count (sum of Terms total count)"),
                y=alt.Y("人物:N", sort="-x", title="Character"),
                tooltip=[
                    alt.Tooltip("人物:N", title="Character"),
                    alt.Tooltip("出現行數:Q", title="Occurrence Rows"),
                    alt.Tooltip("加權次數:Q", title="Weighted Count"),
                ],
                color=alt.Color("出現行數:Q", title="Occurrence Rows", scale=alt.Scale(scheme="reds")),
            )
            .properties(height=360)
        )
        st.altair_chart(p_chart)

    if show_table:
        st.subheader("Terms Details (current filters)")
        display_cols = [
            "稱呼",
            "城市",
            "總次數",
            "章節出現次數",
            "同行出現人物匯總",
        ]
        st.dataframe(
            f[display_cols].sort_values(["城市", "總次數"], ascending=[True, False])
        )
        csv1 = f[display_cols].to_csv(index=False, encoding="utf-8-sig")
        st.download_button("Download Terms filtered results CSV", data=csv1, file_name="稱呼分析_篩選結果.csv", mime="text/csv")

    if show_people_table and not people_agg.empty:
        st.subheader("Character Co-occurrence Table (current filters)")
        st.dataframe(people_agg.reset_index(drop=True))
        csv2 = people_agg.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("Download Character co-occurrence results CSV", data=csv2, file_name="人物共現_篩選結果.csv", mime="text/csv")


if __name__ == "__main__":
    main()