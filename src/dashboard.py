import json
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Liquidity Stress Index", layout="wide")


@st.cache_data
def load_lsi():
    df = pd.read_csv("data/final/lsi_ml.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date")


@st.cache_data
def load_lsi_formula():
    try:
        df = pd.read_csv("data/final/lsi.csv")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.sort_values("date")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_importance():
    try:
        return pd.read_csv("data/final/feature_importance.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["feature", "importance"])


@st.cache_data
def load_backtest():
    try:
        return pd.read_csv("data/final/backtest_report.csv")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_sensitivity():
    try:
        return pd.read_csv("data/final/sensitivity_report.csv")
    except FileNotFoundError:
        return pd.DataFrame()


def status_from_lsi(value: float) -> str:
    if value < 40:
        return "Норма"
    if value < 70:
        return "Напряжение"
    return "Стресс"


def show_status_alert(status: str, lsi: float, active_flags_count: int) -> None:
    if status == "Стресс":
        st.error(f"Критический уровень стресса ликвидности. LSI = {lsi:.2f}")
    elif status == "Напряжение":
        st.warning(f"Повышенное напряжение ликвидности. LSI = {lsi:.2f}")
    elif active_flags_count > 0:
        st.warning(f"LSI в норме ({lsi:.2f}), но обнаружены локальные стресс-флаги: {active_flags_count}.")
    else:
        st.success(f"Ликвидность в норме. LSI = {lsi:.2f}")


def get_active_flags(row: pd.Series) -> pd.DataFrame:
    flag_cols = {
        "m1_flag_end_of_period": "М1: конец периода усреднения",
        "m2_flag_demand": "М2: повышенный спрос на РЕПО",
        "m3_flag_nedospros": "М3: недоспрос ОФЗ",
        "m3_flag_perespros": "М3: переспрос ОФЗ",
        "m4_tax_week_flag": "М4: налоговая неделя",
        "m4_main_tax_day_flag": "М4: основной налоговый день",
        "m4_vat_period_flag": "М4: период НДС",
        "m4_profit_tax_flag": "М4: налог на прибыль",
        "m4_end_of_month_flag": "М4: конец месяца",
        "m4_end_of_quarter_flag": "М4: конец квартала",
        "m5_flag_budget_drain": "М5: отток бюджетной ликвидности",
        "double_count_correction_flag": "Коррекция двойного счёта активна",
    }
    rows = []
    for col, desc in flag_cols.items():
        if col in row.index and pd.notna(row[col]) and float(row[col]) > 0:
            rows.append({"flag": col, "description": desc, "value": row[col]})
    return pd.DataFrame(rows)


def filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    mapping = {"3M": 90, "6M": 180, "1Y": 365, "3Y": 365 * 3}
    days = mapping.get(period)
    if days is None:
        return df.copy()
    max_date = df["date"].max()
    return df[df["date"] >= max_date - pd.Timedelta(days=days)].copy()


def plot_line(df: pd.DataFrame, y_cols: list[str], title: str):
    cols = [c for c in y_cols if c in df.columns]
    if not cols:
        st.info("Нет данных для отображения.")
        return
    st.plotly_chart(px.line(df, x="date", y=cols, title=title))


def plot_module_block(df, score_cols, metric_blocks):
    plot_line(df, score_cols, "Индексные показатели модуля")
    for title, cols in metric_blocks:
        plot_line(df, cols, title)


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5-coder:7b"


def _call_ollama(messages: list[dict], timeout: int = 30) -> Optional[str]:
    body = json.dumps({"model": OLLAMA_MODEL, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data.get("message", {}).get("content", "")
    except Exception as e:
        st.warning(f"Ollama недоступен: {e}")
        return None


def build_rag_context(df: pd.DataFrame, df_formula: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    recent = df.tail(30)
    lsi_min = float(df["lsi"].min())
    lsi_min_date = df.loc[df["lsi"].idxmin(), "date"]
    lsi_max = float(df["lsi"].max())
    lsi_max_date = df.loc[df["lsi"].idxmax(), "date"]
    contribs = {f"m{m}": float(latest.get(f"m{m}_contribution", 0)) for m in range(1, 6)}
    top_module = max(contribs, key=contribs.get)
    flags = [col for col in df.columns if "flag" in col and col in latest.index and float(latest[col]) > 0]

    context = f"""
Текущая дата данных: {latest['date'].strftime('%Y-%m-%d')}
Формульный LSI: {latest['lsi']:.1f} (шкала 0-100, где 0-39=Норма, 40-69=Напряжение, 70-100=Стресс)
Статус: {latest['status']}
ML LSI: {latest.get('ml_lsi', 'N/A')}
Вероятность стресса (ML): {latest.get('stress_probability', 'N/A')}

Максимальный LSI за всю историю: {lsi_max:.1f} ({lsi_max_date.strftime('%Y-%m-%d')})
Минимальный LSI за всю историю: {lsi_min:.1f} ({lsi_min_date.strftime('%Y-%m-%d')})

Вклад модулей:
- М1 (резервы): {contribs.get('m1', 0):.1f}
- М2 (РЕПО ЦБ): {contribs.get('m2', 0):.1f}
- М3 (ОФЗ): {contribs.get('m3', 0):.1f}
- М4 (налоги): {contribs.get('m4', 0):.1f}
- М5 (казначейство): {contribs.get('m5', 0):.1f}
Наибольший вклад: {top_module}

Активные флаги: {', '.join(flags) if flags else 'нет'}

Динамика LSI за последние 30 дней:
- Было: {recent['lsi'].iloc[0]:.1f}
- Стало: {recent['lsi'].iloc[-1]:.1f}
- Изменение: {recent['lsi'].iloc[-1] - recent['lsi'].iloc[0]:.1f}
"""
    return context


def answer_question(query: str, df: pd.DataFrame, df_formula: pd.DataFrame) -> str:
    context = build_rag_context(df, df_formula)
    system_prompt = (
        "Ты — аналитик системы раннего предупреждения стресса ликвидности (Liquidity Stress Index, LSI). "
        "Отвечай на русском языке, коротко и по делу. Используй предоставленный контекст данных. "
        "Если данных недостаточно для ответа, скажи об этом."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Контекст данных системы:\n{context}\n\nВопрос пользователя: {query}"},
    ]
    answer = _call_ollama(messages)
    if answer:
        return answer
    return _fallback_answer(query, df)


def _fallback_answer(query: str, df: pd.DataFrame) -> str:
    query_lower = query.lower()
    if "максимальн" in query_lower or "пик" in query_lower or "самый высок" in query_lower:
        max_row = df.loc[df["lsi"].idxmax()]
        return f"Максимальный LSI ({max_row['lsi']:.1f}) зафиксирован {max_row['date'].strftime('%d.%m.%Y')}. Статус: «{max_row['status']}»."
    if "минимальн" in query_lower or "самый низк" in query_lower:
        min_row = df.loc[df["lsi"].idxmin()]
        return f"Минимальный LSI ({min_row['lsi']:.1f}) зафиксирован {min_row['date'].strftime('%d.%m.%Y')}."
    if "2022" in query:
        period = df[df["date"].between("2022-02-01", "2022-03-31")]
        if not period.empty:
            return f"В феврале-марте 2022 средний LSI был {period['lsi'].mean():.1f}, максимум {period['lsi'].max():.1f}."
    if "2023" in query:
        period = df[df["date"].between("2023-08-01", "2023-09-15")]
        if not period.empty:
            return f"В августе-сентябре 2023 средний LSI был {period['lsi'].mean():.1f}, максимум {period['lsi'].max():.1f}."
    if "2014" in query:
        period = df[df["date"].between("2014-12-01", "2014-12-31")]
        if not period.empty:
            return f"В декабре 2014 средний LSI был {period['lsi'].mean():.1f}, максимум {period['lsi'].max():.1f}."
    if "модул" in query or "вклад" in query:
        latest = df.iloc[-1]
        contribs = {f"М{m}": latest.get(f"m{m}_contribution", 0) for m in range(1, 6)}
        contrib_str = ", ".join(f"{k}: {v:.1f}" for k, v in sorted(contribs.items(), key=lambda x: -x[1]))
        return f"Вклад модулей на {latest['date'].strftime('%d.%m.%Y')}: {contrib_str}."
    if "тренд" in query or "динамик" in query:
        recent = df.tail(30)
        if len(recent) > 1:
            trend = "растёт" if recent["lsi"].iloc[-1] > recent["lsi"].iloc[0] else "снижается"
            return f"За последние 30 дней LSI {trend}. Текущее значение: {recent['lsi'].iloc[-1]:.1f}, было {recent['lsi'].iloc[0]:.1f}."
    return ("Я могу ответить на вопросы о LSI, модулях, флагах и исторических периодах. "
            "Например: «Что было в марте 2022?», «Какой вклад модулей?», «Максимальный LSI?».")


def generate_auto_comment(df: pd.DataFrame, active_flags: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    top_module = "m2"
    max_contrib = 0
    for m in ["m1", "m2", "m3", "m4", "m5"]:
        val = float(latest.get(f"{m}_contribution", 0))
        if val > max_contrib:
            max_contrib = val
            top_module = m
    module_names = {"m1": "резервов (М1)", "m2": "РЕПО ЦБ (М2)", "m3": "ОФЗ (М3)", "m4": "налогов (М4)", "m5": "казначейства (М5)"}
    parts = []
    parts.append(f"Текущий LSI составляет {latest['lsi']:.1f} — это уровень «{latest['status']}».")
    parts.append(f"ML-модель оценивает вероятность стресса в {latest.get('ml_lsi', 0):.1f} баллов.")
    parts.append(f"Наибольший вклад вносит модуль {module_names.get(top_module, top_module)}.")
    parts.append(f"Вклад модулей: М1={latest.get('m1_contribution', 0):.1f}, М2={latest.get('m2_contribution', 0):.1f}, М3={latest.get('m3_contribution', 0):.1f}, М4={latest.get('m4_contribution', 0):.1f}, М5={latest.get('m5_contribution', 0):.1f}.")
    if not active_flags.empty:
        flags_desc = "; ".join(active_flags["description"].tolist())
        parts.append(f"Активные флаги: {flags_desc}.")
    else:
        parts.append("Активных стресс-флагов нет.")
    df_prev = df[df["date"] < latest["date"] - pd.Timedelta(days=7)]
    if not df_prev.empty:
        prev_lsi = df_prev["lsi"].iloc[-1]
        delta = latest["lsi"] - prev_lsi
        if delta > 10:
            parts.append(f"За неделю LSI вырос на {delta:.1f} пункта — тенденция к усилению напряжения.")
        elif delta < -10:
            parts.append(f"За неделю LSI снизился на {abs(delta):.1f} пункта — напряжение ослабевает.")
        else:
            parts.append("За неделю LSI существенно не изменился.")
    base_comment = " ".join(parts)
    system_prompt = (
        "Ты — аналитик LSI. Перепиши следующий комментарий более естественным языком, "
        "сохранив все факты и цифры. Ответь одним абзацем на русском."
    )
    llm_answer = _call_ollama([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_comment},
    ])
    return llm_answer if llm_answer else base_comment


st.sidebar.title("Liquidity Stress Index")
page = st.sidebar.radio("Страница", ["Дашборд", "Аналитик (LLM-чат)"])

df = load_lsi()
df_formula = load_lsi_formula()
importance = load_importance()
backtest = load_backtest()
sensitivity = load_sensitivity()

if "status" not in df.columns:
    df["status"] = df["lsi"].apply(status_from_lsi)

df = df.dropna(subset=["date"]).sort_values("date")
latest = df.iloc[-1]
active_flags = get_active_flags(latest)

if page == "Дашборд":
    st.title("Liquidity Stress Index")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Формульный LSI", round(float(latest["lsi"]), 2))
    col2.metric("Статус", str(latest["status"]))
    col3.metric("ML LSI", round(float(latest["ml_lsi"]), 2))
    col4.metric("Дата", latest["date"].strftime("%Y-%m-%d"))
    show_status_alert(str(latest["status"]), float(latest["lsi"]), len(active_flags))
    st.divider()
    period = st.selectbox("Период отображения графиков", ["3M", "6M", "1Y", "3Y", "ALL"], index=2)
    plot_df = filter_period(df, period)
    st.subheader("Динамика индексов")
    fig_lsi = px.line(plot_df, x="date", y=["lsi", "ml_lsi"], title="Формульный LSI и ML LSI")
    fig_lsi.add_hrect(y0=0, y1=40, opacity=0.1)
    fig_lsi.add_hrect(y0=40, y1=70, opacity=0.1)
    fig_lsi.add_hrect(y0=70, y1=100, opacity=0.1)
    st.plotly_chart(fig_lsi)
    st.subheader("Активные флаги")
    if active_flags.empty:
        st.success("Активных стресс-флагов на последнюю дату нет.")
    else:
        st.warning("Обнаружены активные флаги.")
        st.dataframe(active_flags)
    st.subheader("Вклад модулей в формульный LSI")
    contrib_cols = ["m1_contribution", "m2_contribution", "m3_contribution", "m4_contribution", "m5_contribution"]
    latest_contrib = latest[contrib_cols].reset_index()
    latest_contrib.columns = ["module", "contribution"]
    total_contribution = latest_contrib["contribution"].sum()
    if total_contribution > 0:
        latest_contrib["share_pct"] = latest_contrib["contribution"] / total_contribution * 100
    else:
        latest_contrib["share_pct"] = 0.0
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(px.bar(latest_contrib, x="module", y="contribution", title="Абсолютный вклад модулей"))
    with col_b:
        st.plotly_chart(px.bar(latest_contrib, x="module", y="share_pct", title="Доля модулей в текущем LSI, %"))
    st.dataframe(latest_contrib)
    st.subheader("Графики модулей М1–М5")
    module_tabs = st.tabs(["М1 Резервы", "М2 РЕПО", "М3 ОФЗ", "М4 Налоги", "М5 Казначейство"])
    with module_tabs[0]:
        st.markdown("### М1: резервы и RUONIA")
        plot_module_block(plot_df, score_cols=["m1_score"], metric_blocks=[
            ("Спред резервов", ["spread"]),
            ("RUONIA", ["ruonia"]),
            ("MAD-сигналы М1", ["m1_mad_score_spread", "m1_mad_score_ruonia"]),
        ])
    with module_tabs[1]:
        st.markdown("### М2: РЕПО ЦБ")
        plot_module_block(plot_df, score_cols=["m2_score"], metric_blocks=[
            ("Объём требований по РЕПО", ["repo_outstanding"]),
            ("Cover ratio proxy", ["cover_ratio_proxy"]),
            ("Rate spread proxy", ["rate_spread_proxy"]),
            ("MAD-сигналы М2", [c for c in plot_df.columns if "m2_mad_score" in c]),
        ])
    with module_tabs[2]:
        st.markdown("### М3: ОФЗ")
        plot_module_block(plot_df, score_cols=["m3_score"], metric_blocks=[
            ("Cover ratio ОФЗ", ["cover_ratio"]),
            ("Yield spread ОФЗ", ["yield_spread"]),
            ("MAD-сигналы М3", [c for c in plot_df.columns if "m3_mad_score" in c]),
        ])
    with module_tabs[3]:
        st.markdown("### М4: налоговый календарь")
        plot_module_block(plot_df, score_cols=["m4_score"], metric_blocks=[
            ("Налоговые флаги", [c for c in plot_df.columns if "m4_" in c and "flag" in c]),
            ("Seasonal factor", ["m4_seasonal_factor"]),
        ])
    with module_tabs[4]:
        st.markdown("### М5: казначейство")
        plot_module_block(plot_df, score_cols=["m5_score"], metric_blocks=[
            ("Дефицит / профицит ликвидности", ["liquidity_deficit"]),
            ("Недельная и месячная дельта", ["weekly_delta", "monthly_delta"]),
            ("MAD-сигналы М5", [c for c in plot_df.columns if "m5_mad_score" in c]),
        ])
    st.subheader("Важность признаков ML-модели")
    if not importance.empty:
        st.plotly_chart(px.bar(importance.head(25), x="feature", y="importance", title="Top-25 feature importance"))
        st.dataframe(importance)
    else:
        st.warning("Файл feature_importance.csv не найден.")
    st.subheader("Sensitivity analysis ±20%")
    if not sensitivity.empty:
        st.dataframe(sensitivity)
        st.plotly_chart(px.bar(sensitivity, x="scenario", y="latest_diff", title="Изменение LSI при изменении весов ±20%"))
    else:
        st.warning("Файл sensitivity_report.csv не найден. Запусти: python3 src/sensitivity.py")
    st.subheader("Backtest на стрессовых периодах")
    if not backtest.empty:
        st.dataframe(backtest)
        cols = [c for c in ["mean_lsi", "max_lsi", "mean_ml_lsi", "max_ml_lsi"] if c in backtest.columns]
        if cols:
            st.plotly_chart(px.bar(backtest, x="period", y=cols, barmode="group", title="LSI на исторических стрессовых периодах"))
    else:
        st.warning("Файл backtest_report.csv не найден. Запусти: python3 src/backtest.py")
    st.subheader("Автоматический комментарий")
    st.info(generate_auto_comment(df, active_flags))
    st.subheader("Последние значения")
    cols = [c for c in ["date", "lsi", "status", "ml_lsi", "stress_probability",
                        "m1_score", "m2_score", "m3_score", "m4_score", "m5_score",
                        "double_count_correction_flag"] if c in df.columns]
    st.dataframe(df[cols].tail(30))

elif page == "Аналитик (LLM-чат)":
    st.title("Аналитик — LLM-чат по данным системы")
    st.markdown("Задайте вопрос о ликвидности, LSI, модулях или исторических периодах.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    query = st.chat_input("Введите вопрос...")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner("Анализирую данные системы..."):
            answer = answer_question(query, df, df_formula)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
