import os
import re
import time
import math
import difflib
import html as _html
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_TITLE = "Dynamic public transport demand forecasting and resource allocation using machine learning"

st.set_page_config(page_title="Bus Demand Planner", page_icon="🚌", layout="wide",
                   initial_sidebar_state="collapsed")

# ============================================================
# Load the files saved by the notebook
# ============================================================
REQUIRED_FILES = ["model_calendar.pkl", "model_with_lag.pkl", "app_info.pkl"]
BASE_DIR = os.path.dirname(os.path.abspath(globals().get('__file__', 'app.py')))


def find_file(name):
    """Finds a data file next to the app. Also accepts small naming slips such as 'model_with_lag (1).pkl',
    'Model With Lag.pkl' or a file placed in a sub-folder."""
    roots = [BASE_DIR] + ([os.getcwd()] if os.getcwd() != BASE_DIR else [])
    for root in roots:
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    squash = lambda t: re.sub(r'[^a-z0-9]', '', re.sub(r'\(\d+\)', '', t.lower()))
    key = squash(name[:-4])
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames if not x.startswith('.') and x != '__pycache__']
            for f in filenames:
                if f.lower().endswith('.pkl') and squash(f[:-4]) == key:
                    return os.path.join(dirpath, f)
    return None


PATHS = {n: find_file(n) for n in REQUIRED_FILES + ["recent_riders.pkl"]}
missing = [f for f in REQUIRED_FILES if PATHS[f] is None]
if missing:
    st.error("These files are missing from the project folder: " + ", ".join(missing))
    found = sorted(f for f in os.listdir(BASE_DIR) if not f.startswith('.'))
    st.info("Upload them to the same place as app.py (the top level of the GitHub repository), with exactly these "
            "names. Files the app can see right now: " + (", ".join(found) if found else "none"))
    st.stop()


@st.cache_resource
def load_artifacts():
    info = joblib.load(PATHS["app_info.pkl"])
    problems, models = [], {}
    for key, fname in (("cal", "model_calendar.pkl"), ("lag", "model_with_lag.pkl")):
        try:
            models[key] = joblib.load(PATHS[fname])
        except Exception as e:          # usually a scikit-learn version difference between Colab and this PC
            models[key] = None
            problems.append(f"{fname}: {type(e).__name__}: {e}")
    return models["cal"], models["lag"], info, problems


model_cal, model_lag, info, LOAD_PROBLEMS = load_artifacts()
HAS_LAG = model_lag is not None
ROUTE_CODES = info['route_codes']
ROUTE_AVG = info['route_avg']
FEAT_CAL = info['features_calendar']
FEAT_LAG = info['features_lag']
HOLIDAYS = set(info['holidays'])
RESULTS = info['results']
PROFILE = info['profile']                       # usual riders per route, weekday and hour (training year)
LAG_MODEL_NAME = info.get('model_with_lag_name', 'Random Forest (with last week)')


@st.cache_resource
def load_recent():
    """Optional file: real hourly riders per route (late Dec 2024 to Mar 2025). Gives the real 'last week' number."""
    if PATHS.get("recent_riders.pkl") is None:
        return None
    try:
        return joblib.load(PATHS["recent_riders.pkl"])
    except Exception:
        return None


RECENT = load_recent()


def _route_sort_key(r):
    m = re.match(r'^([A-Za-z]*)(\d*)(.*)$', str(r))
    return (m.group(1).upper(), int(m.group(2)) if m.group(2) else 0, m.group(3))


ROUTES = sorted(ROUTE_CODES, key=_route_sort_key)
ROUTE_POS = {r: i for i, r in enumerate(ROUTES)}


def borough(r):
    """Routes start with a borough code: M, Bx, B, Q, S. Express routes use QM, BM, BxM, SIM or X."""
    m = re.match(r'^([A-Za-z]+)', str(r))
    p = m.group(1).upper() if m else ''
    if p in ('QM', 'BM', 'BXM', 'SIM', 'X'):
        return 'Express'
    return {'M': 'Manhattan', 'BX': 'Bronx', 'B': 'Brooklyn', 'Q': 'Queens', 'S': 'Staten Island'}.get(p, 'Other')


BOROUGHS = ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island', 'Express', 'Other']


@st.cache_resource
def build_lookups():
    usual = {k: g.set_index('bus_route')['riders'] for k, g in PROFILE.groupby(['dow', 'hour'])}
    peak = PROFILE.groupby(['bus_route', 'dow'])['riders'].max()
    # the same profile as one array: [weekday 0-6, hour 0-23, route position]
    arr = np.zeros((7, 24, len(ROUTES)))
    piv = (PROFILE.pivot_table(index=['dow', 'hour'], columns='bus_route', values='riders')
           .reindex(columns=ROUTES).fillna(0))
    for (dw, hr), row in piv.iterrows():
        arr[int(dw), int(hr)] = row.values
    return usual, peak, arr


USUAL_TABLE, PEAK, USUAL_ARR = build_lookups()


def _nth_weekday(year, month, weekday, n):
    """n-th given weekday of a month (n = -1 for the last one)."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


@st.cache_data(show_spinner=False)
def federal_holidays(year):
    """US federal holidays for any year. The saved list only covers the years used in the notebook."""
    days = {date(year, 1, 1), _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3), _nth_weekday(year, 5, 0, -1),
            date(year, 6, 19), date(year, 7, 4), _nth_weekday(year, 9, 0, 1), _nth_weekday(year, 10, 0, 2),
            date(year, 11, 11), _nth_weekday(year, 11, 3, 4), date(year, 12, 25)}
    return {x.strftime('%Y-%m-%d') for x in days}


def is_holiday(d):
    key = d.strftime('%Y-%m-%d')
    return int(key in HOLIDAYS or key in federal_holidays(d.year))


def eff_dow(d):
    """Weekday used for the 'usual' pattern. A US federal holiday on a weekday runs like a Sunday."""
    return 6 if (is_holiday(d) and d.weekday() < 5) else d.weekday()


def usual_at(dow, hour):
    return USUAL_TABLE.get((dow, hour), pd.Series(dtype=float))


@st.cache_resource
def build_trend():
    """Learns from the saved real 2025 riders (a) how each route's level compares with the training-year usual,
    (b) the typical percentage miss per route. Also runs two honest back-tests that are shown in the About page."""
    ones = np.ones(len(ROUTES))
    out = {'trend': ones, 'err': np.full(len(ROUTES), 0.2), 'bt': None}
    if RECENT is None:
        return out
    cols = [r for r in ROUTES if r in RECENT.columns]
    if len(cols) < len(ROUTES) // 2:
        return out
    pos = [ROUTE_POS[r] for r in cols]
    idx = RECENT.index
    act = RECENT[cols].values.astype(float)
    holiday = np.array([t.strftime('%Y-%m-%d') in HOLIDAYS for t in idx])
    edow = np.where(holiday & (idx.dayofweek < 5), 6, idx.dayofweek)
    usual = USUAL_ARR[edow, idx.hour][:, pos]

    def factors(mask):
        num, den = act[mask].sum(0), usual[mask].sum(0)
        raw = np.where(den > 1000, num / np.maximum(den, 1e-9), 1.0)
        return 1 + (np.clip(raw, 0.5, 2.0) - 1) * 0.75            # shrunk 25% toward "no change"

    trend = np.ones(len(ROUTES))
    trend[pos] = factors(np.ones(len(idx), dtype=bool))
    err = np.full(len(ROUTES), 0.2)
    pct = np.abs(act - usual * trend[pos]).sum(0) / np.maximum(act.sum(0), 1.0)
    err[pos] = np.clip(np.where(act.sum(0) > 1000, pct, 0.25), 0.08, 0.6)
    bt = None
    try:
        tr, te = idx < pd.Timestamp('2025-03-01'), idx >= pd.Timestamp('2025-03-01')
        f = factors(tr)
        plain_dow = idx.dayofweek
        usual_plain = USUAL_ARR[plain_dow, idx.hour][:, pos]
        hol = holiday & (idx.dayofweek < 5)
        bt = {'base': float(np.abs(usual[te] - act[te]).mean()),
              'adj': float(np.abs(usual[te] * f - act[te]).mean()),
              'hol_plain': float(np.abs(usual_plain[hol] - act[hol]).mean()) if hol.any() else None,
              'hol_sunday': float(np.abs(usual[hol] - act[hol]).mean()) if hol.any() else None}
    except Exception:
        bt = None
    out.update({'trend': trend, 'err': err, 'bt': bt})
    return out


_TR = build_trend()
TREND, ERR_PCT, BACKTEST = _TR['trend'], _TR['err'], _TR['bt']


def trend_for(routes):
    return np.array([TREND[ROUTE_POS[r]] for r in routes], dtype=float)


def err_for(route, real_lag=False):
    e = float(ERR_PCT[ROUTE_POS[route]]) if route in ROUTE_POS else 0.2
    return e * (0.85 if real_lag else 1.0)


# Which method forecasts better when last week's number is NOT known? Decided from the test results.
def _mae(name):
    return RESULTS.get(name, {}).get('MAE')


_base_mae, _cal_mae = _mae("Baseline: usual route-day-hour average"), _mae("Gradient Boosting (calendar only)")
USE_PROFILE = (model_cal is None) or ((_base_mae is not None) and (_cal_mae is None or _base_mae <= _cal_mae))

DEFAULT_RES = (RESULTS.get("Baseline: usual route-day-hour average") if USE_PROFILE
               else RESULTS.get("Gradient Boosting (calendar only)")) or {}
_cal_would_be_better = (_cal_mae is not None) and (_base_mae is None or _cal_mae < _base_mae)

# If last week's number is not known, the Random Forest is given the route's usual last-week level. The notebook tests
# this on the 2025 data (PROXY_NAME). If that test is WORSE than the simple usual-average rule, the app switches back
# to the usual riders (adjusted to the recent level of each route, and holiday-aware). If no test was saved, ML is used.
PROXY_NAME = "Random Forest (usual level, no last week)"
_proxy_mae = _mae(PROXY_NAME)
USE_ML_DEFAULT = HAS_LAG and not (_proxy_mae is not None and _base_mae is not None and _proxy_mae >= _base_mae)
if USE_ML_DEFAULT and PROXY_NAME in RESULTS:
    DEFAULT_RES = RESULTS[PROXY_NAME]
ML_ON = (HAS_LAG and USE_ML_DEFAULT) or (not USE_PROFILE)

LOW_RATIO, HIGH_RATIO = 0.35, 0.75                # demand level = expected riders / usual busiest hour of the route
TOP_ROUTE = max(info['top_routes'], key=info['top_routes'].get) if info.get('top_routes') else ROUTES[0]


# ============================================================
# Forecasting helpers
# ============================================================
def fmt_hour(h):
    h = int(h)
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def pax_text(n):
    return f"{n} rider" if n == 1 else f"{n} riders"


def usual_for(routes, d, hour):
    """Usual riders (training-year average) for these routes on this type of day and hour."""
    return np.array([USUAL_ARR[eff_dow(d), hour, ROUTE_POS[r]] for r in routes], dtype=float)


def calendar_frame(routes, d, hour):
    df = pd.DataFrame({'bus_route': list(routes)})
    df['route_code'] = df['bus_route'].map(ROUTE_CODES)
    df['route_avg'] = df['bus_route'].map(ROUTE_AVG)
    df['hour'] = hour
    df['dow'] = d.weekday()
    df['month'] = d.month
    df['is_weekend'] = int(d.weekday() >= 5)
    df['is_holiday'] = is_holiday(d)
    return df


def has_real_lag(d, hour):
    """True if the real riders from exactly one week earlier are saved for this date and hour."""
    if RECENT is None:
        return False
    return (pd.Timestamp(d.year, d.month, d.day, hour) - pd.Timedelta(hours=168)) in RECENT.index


def recent_lag(routes, d, hour):
    """Real riders one week earlier for each route (array), or None if not saved."""
    if not has_real_lag(d, hour):
        return None
    ts = pd.Timestamp(d.year, d.month, d.day, hour) - pd.Timedelta(hours=168)
    return RECENT.loc[ts].reindex(list(routes)).fillna(0).values.astype(float)


def actual_day(route, d):
    """Real riders for each hour of this date (only for dates inside the saved 2025 data), else None."""
    if RECENT is None or route not in RECENT.columns:
        return None
    idx = pd.date_range(pd.Timestamp(d.year, d.month, d.day), periods=24, freq="h")
    if not idx.isin(RECENT.index).all():
        return None
    return RECENT.loc[idx, route].values.astype(float)


def uses_real_lag_day(d):
    return HAS_LAG and has_real_lag(d, 0) and has_real_lag(d, 23)


def method_caption(d, hour):
    if HAS_LAG and has_real_lag(d, hour):
        return ("✅ The real riders from one week earlier are saved for this date, so the Random Forest model "
                "is used with them (the most accurate way).")
    if USE_ML_DEFAULT:
        return ("ℹ️ No saved data for the week before this date, so the Random Forest model is given each "
                "route's usual level.")
    return ("ℹ️ No saved data for the week before this date, so the forecast is each route's usual riders for "
            "this weekday and hour, adjusted to the route's recent level"
            + " and treated like a Sunday on US federal holidays.")


def recent_range_text():
    if RECENT is None:
        return None
    a = RECENT.index.min() + pd.Timedelta(hours=168)
    b = RECENT.index.max() + pd.Timedelta(hours=168)
    return f"{a.strftime('%d %b %Y')} to {b.strftime('%d %b %Y')}"


def predict_routes(routes, d, hour, lag=None):
    """Expected riders per route for one date and hour."""
    if lag is None and HAS_LAG:
        real = recent_lag(routes, d, hour)                # real riders one week earlier (saved 2025 data)
        if real is not None:
            df = calendar_frame(routes, d, hour)
            df['lag_168'] = real
            return np.clip(model_lag.predict(df[FEAT_LAG]), 0, None)
    if HAS_LAG and (lag is not None or USE_ML_DEFAULT):   # Random Forest model
        df = calendar_frame(routes, d, hour)
        if lag is None:                                   # last week unknown: use the route's usual level
            df['lag_168'] = usual_at(d.weekday(), hour).reindex(list(routes)).fillna(0).values.astype(float)
        else:
            df['lag_168'] = lag
        return np.clip(model_lag.predict(df[FEAT_LAG]), 0, None)
    if USE_PROFILE:                                       # usual riders, adjusted to the route's recent level
        return usual_for(routes, d, hour) * trend_for(routes)
    df = calendar_frame(routes, d, hour)
    return np.clip(model_cal.predict(df[FEAT_CAL]), 0, None)


@st.cache_data(show_spinner=False)
def day_curve(route, day_iso):
    """Expected riders for each hour 0-23 of the day."""
    d = date.fromisoformat(day_iso)
    if uses_real_lag_day(d):
        frames = []
        for h in range(24):
            f = calendar_frame([route], d, h)
            f['lag_168'] = recent_lag([route], d, h)[0]
            frames.append(f)
        return np.clip(model_lag.predict(pd.concat(frames, ignore_index=True)[FEAT_LAG]), 0, None)
    if HAS_LAG and USE_ML_DEFAULT:
        frames = []
        for h in range(24):
            f = calendar_frame([route], d, h)
            f['lag_168'] = float(usual_at(d.weekday(), h).get(route, 0.0))
            frames.append(f)
        return np.clip(model_lag.predict(pd.concat(frames, ignore_index=True)[FEAT_LAG]), 0, None)
    if USE_PROFILE:
        return USUAL_ARR[eff_dow(d), :, ROUTE_POS[route]] * TREND[ROUTE_POS[route]]
    df = pd.concat([calendar_frame([route], d, h) for h in range(24)], ignore_index=True)
    return np.clip(model_cal.predict(df[FEAT_CAL]), 0, None)


def usual_curve(route, d):
    return USUAL_ARR[eff_dow(d), :, ROUTE_POS[route]].astype(float)


def level_of(ratio):
    if ratio >= HIGH_RATIO:
        return "High"
    if ratio >= LOW_RATIO:
        return "Moderate"
    return "Low"


def level_for(route, d, riders):
    pk = float(PEAK.get((route, eff_dow(d)), 0.0))
    ratio = riders / pk if pk > 0 else 0.0
    return level_of(ratio), ratio


def buses_for(riders, cap):
    return int(math.ceil(round(float(riders)) / max(int(cap), 1)))


@st.cache_data(show_spinner="Calculating demand for all routes...")
def route_table(day_iso, hour, cap):
    """One row per route: forecast, usual riders, buses needed now and buses normally run at this hour."""
    d = date.fromisoformat(day_iso)
    pred = predict_routes(ROUTES, d, hour)
    usual = usual_for(ROUTES, d, hour)
    peak = np.array([float(PEAK.get((r, eff_dow(d)), 0.0)) for r in ROUTES])
    ratio = np.where(peak > 0, pred / np.where(peak > 0, peak, 1), 0.0)
    levels = [level_of(r) for r in ratio]
    need = np.ceil(np.round(pred) / max(cap, 1)).astype(int)
    normal = np.ceil(np.round(usual) / max(cap, 1)).astype(int)
    return pd.DataFrame({
        'Route': ROUTES, 'Borough': [borough(r) for r in ROUTES],
        'riders': pred, 'Expected riders': np.round(pred).astype(int),
        'Usual riders': np.round(usual).astype(int), 'ratio': ratio, 'level': levels,
        'Level': [f"{LEVEL_EMOJI[l]} {l}" for l in levels],
        'Normal buses': normal, 'Buses needed': need, 'Change': need - normal,
        'avg': [ROUTE_AVG.get(r, 0.0) for r in ROUTES],
    }).sort_values('riders', ascending=False).reset_index(drop=True)


def build_moves(tbl, limit=12):
    """Pairs routes that can spare buses with routes that need more. Same borough first."""
    give = [[r, b, int(-c)] for r, b, c in zip(tbl['Route'], tbl['Borough'], tbl['Change']) if c < 0]
    need = [[r, b, int(c)] for r, b, c in zip(tbl['Route'], tbl['Borough'], tbl['Change']) if c > 0]
    give.sort(key=lambda x: -x[2])
    need.sort(key=lambda x: -x[2])
    moves = []
    for same_only in (True, False):
        for t in need:
            for g in give:
                if t[2] == 0:
                    break
                if g[2] == 0:
                    continue
                same = (g[1] == t[1]) and g[1] != 'Other'
                if same_only and not same:
                    continue
                n = min(g[2], t[2])
                moves.append({'From route': g[0], 'To route': t[0], 'Buses to move': n,
                              'Same borough': "Yes" if same else "No (check depots)"})
                g[2] -= n
                t[2] -= n
    moves.sort(key=lambda m: (-m['Buses to move'], m['Same borough'] != "Yes"))
    return pd.DataFrame(moves[:limit], columns=['From route', 'To route', 'Buses to move', 'Same borough'])


LEVEL_EMOJI = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
LEVEL_PLAIN = {
    "High": "A busy time for this route.",
    "Moderate": "Fairly busy, but manageable.",
    "Low": "A quiet time for this route.",
}
LEVEL_NOTE = (f"Demand level compares the expected riders with the route's usual busiest hour on that type of day. "
              f"Low: under {int(LOW_RATIO * 100)}%  |  Moderate: {int(LOW_RATIO * 100)}% to {int(HIGH_RATIO * 100)}%  |  "
              f"High: {int(HIGH_RATIO * 100)}% or more.")


def get_cap():
    return int(st.session_state.get('cap', 100))


# ============================================================
# LOOK AND FEEL: colours, CSS and small HTML components
# ============================================================
INDIGO, NAVY, TEAL, AMBER, CORAL = "#4F5FE0", "#16224F", "#2FA891", "#E0A83A", "#D96461"
LEVEL_COLOR = {"Low": TEAL, "Moderate": AMBER, "High": CORAL}

APP_CSS = """<style>
[data-testid="stMainBlockContainer"], .block-container {max-width: 1120px; padding-top: 1.8rem;}
[data-testid="stChatMessage"] {border-radius: 18px; padding: 14px 16px; border: 1px solid rgba(127,127,127,.2); background: rgba(79,95,224,.05);}
div.stButton > button, div[data-testid="stButton"] > button {border-radius: 12px; border: 1px solid rgba(79,95,224,.45); font-weight: 600; transition: all .15s ease;}
div.stButton > button:hover, div[data-testid="stButton"] > button:hover {border-color: #4F5FE0; color: #4F5FE0; transform: translateY(-1px); box-shadow: 0 3px 10px rgba(79,95,224,.18);}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {background: #4F5FE0; border-color: #4F5FE0; color: #fff;}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {background: #3E4CC9; border-color: #3E4CC9; color: #fff;}
button[data-baseweb="tab"] p {font-weight: 700;}
.tc-hero {position: relative; overflow: hidden; border-radius: 22px; padding: 30px 32px; margin-bottom: 20px; color: #fff; background: linear-gradient(135deg, #16224F 0%, #3446C4 55%, #4F5FE0 100%);}
.tc-hero:after {content: ""; position: absolute; right: -70px; top: -70px; width: 260px; height: 260px; border-radius: 50%; background: rgba(255,255,255,.08);}
.tc-hero:before {content: ""; position: absolute; right: 120px; bottom: -90px; width: 200px; height: 200px; border-radius: 50%; background: rgba(255,255,255,.05);}
.tc-hero-title {font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin-bottom: 8px; position: relative; z-index: 1;}
.tc-hero-sub {font-size: 1.05rem; opacity: .92; max-width: 640px; position: relative; z-index: 1;}
.tc-pills {display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; position: relative; z-index: 1;}
.tc-pill {background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.28); border-radius: 999px; padding: 6px 14px; font-size: .86rem; color: #fff;}
.tc-pagehead {margin: 0 0 18px 0; padding: 18px 22px; border-radius: 18px; color: #fff; background: linear-gradient(135deg, #16224F 0%, #3446C4 100%);}
.tc-pagehead-t {font-size: 1.55rem; font-weight: 800;}
.tc-pagehead-s {opacity: .88; font-size: .98rem; margin-top: 2px;}
.tc-compact {padding: 10px 14px; margin-bottom: 12px; border-radius: 14px; background: rgba(79,95,224,.08); border: 1px solid rgba(79,95,224,.25); font-size: .92rem;}
.tc-brand {display: flex; align-items: center; gap: 10px; margin-bottom: 4px;}
.tc-brand-logo {width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; background: linear-gradient(135deg, #3446C4, #4F5FE0);}
.tc-brand-name {font-weight: 800; font-size: 1.25rem; line-height: 1.1;}
.tc-brand-tag {font-size: .78rem; opacity: .65;}
.tc-origin {display: inline-block; margin-bottom: 6px; padding: 2px 10px; border-radius: 999px; font-size: .74rem; font-weight: 700; letter-spacing: .02em; background: rgba(79,95,224,.14); color: #4F5FE0; border: 1px solid rgba(79,95,224,.35);}
.tc-card {border: 1px solid rgba(127,127,127,.25); border-left: 6px solid #4F5FE0; border-radius: 16px; padding: 16px 18px; margin: 8px 0 4px 0; background: rgba(127,127,127,.06);}
.tc-top {display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;}
.tc-eyebrow {font-weight: 700; font-size: 1.05rem;}
.tc-sub {opacity: .7; font-size: .86rem; margin-top: 2px;}
.tc-badge {display: inline-block; color: #fff; border-radius: 999px; padding: 4px 12px; font-weight: 700; font-size: .84rem; white-space: nowrap;}
.tc-big {font-size: 2.7rem; font-weight: 800; line-height: 1.1; margin: 10px 0 2px;}
.tc-big span {font-size: 1rem; font-weight: 500; opacity: .75; margin-left: 8px; color: inherit;}
.tc-gauge {position: relative; height: 10px; border-radius: 999px; margin: 16px 2px 6px;}
.tc-dot {position: absolute; top: 50%; width: 18px; height: 18px; border-radius: 50%; background: #fff; border: 4px solid #4F5FE0; transform: translate(-50%, -50%); box-shadow: 0 1px 4px rgba(0,0,0,.35);}
.tc-scale {display: flex; justify-content: space-between; font-size: .72rem; opacity: .65;}
.tc-note {margin-top: 10px; font-size: .95rem;}
.tc-tip {margin-top: 12px; padding: 10px 12px; border-radius: 12px; font-size: .9rem; background: rgba(79,95,224,.12); border: 1px dashed rgba(79,95,224,.5);}
.tc-foot {margin-top: 8px; font-size: .78rem; opacity: .6;}
.tc-hours {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 10px 0 6px;}
.tc-hour {border: 2px solid #2FA891; border-radius: 16px; padding: 12px; text-align: center; background: rgba(47,168,145,.08);}
.tc-hour-t {font-size: 1.6rem; font-weight: 800;}
.tc-hour-n {font-size: .85rem; opacity: .75; margin: 2px 0 8px;}
.tc-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 6px 0 18px;}
.tc-kpi {border: 1px solid rgba(127,127,127,.22); border-top: 5px solid #4F5FE0; border-radius: 16px; padding: 14px 16px; background: rgba(127,127,127,.06);}
.tc-kpi-label {font-size: .82rem; opacity: .7; font-weight: 600;}
.tc-kpi-value {font-size: 1.9rem; font-weight: 800; margin: 2px 0; line-height: 1.15;}
.tc-kpi-sub {font-size: .78rem; opacity: .6;}
.tc-steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 8px 0 18px;}
.tc-step {display: flex; gap: 12px; align-items: flex-start; border: 1px solid rgba(127,127,127,.22); border-radius: 16px; padding: 14px 16px; background: rgba(127,127,127,.06);}
.tc-step-n {flex: 0 0 34px; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; color: #fff; background: #4F5FE0;}
.tc-step-t {font-weight: 700; margin-bottom: 2px;}
.tc-step-d {font-size: .88rem; opacity: .78;}
.tc-srcs {margin: 8px 0 2px; font-size: .82rem;}
.tc-src {display: inline-block; margin: 2px 6px 2px 0; padding: 3px 10px; border-radius: 999px; background: rgba(127,127,127,.12); border: 1px solid rgba(127,127,127,.3);}
.tc-glance {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 8px 0 18px;}
.tc-gcard {border: 1px solid rgba(127,127,127,.22); border-top: 5px solid #4F5FE0; border-radius: 18px; padding: 16px 18px; background: rgba(127,127,127,.06);}
.tc-gcard-h {font-weight: 800; font-size: 1.05rem; margin-bottom: 6px;}
.tc-gcard-big {font-size: 2.1rem; font-weight: 800; line-height: 1.1; margin: 2px 0 4px;}
.tc-gcard ul {margin: 6px 0 0 0; padding-left: 18px; font-size: .88rem; line-height: 1.55;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {display: none !important;}
header[data-testid="stHeader"] {display: none;}
#MainMenu, footer {visibility: hidden;}
div[data-testid="stRadio"] > div[role="radiogroup"] {gap: 8px; flex-wrap: wrap; margin-bottom: 6px;}
div[data-testid="stRadio"] label {background: rgba(79,95,224,.08); border: 1px solid rgba(79,95,224,.28); border-radius: 999px; padding: 6px 18px; cursor: pointer; transition: all .15s ease;}
div[data-testid="stRadio"] label:hover {border-color: #4F5FE0;}
div[data-testid="stRadio"] label > div:first-child {display: none;}
div[data-testid="stRadio"] label:has(input:checked) {background: #4F5FE0; border-color: #4F5FE0;}
div[data-testid="stRadio"] label:has(input:checked) p {color: #fff !important; font-weight: 700;}
.w-result {border: 1px solid rgba(127,127,127,.25); border-left: 8px solid #4F5FE0; border-radius: 20px; padding: 22px 26px; background: rgba(127,127,127,.06); margin: 12px 0;}
.w-res-top {display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap;}
.w-route {font-size: 1.5rem; font-weight: 800;}
.w-when {opacity: .7;}
.w-big {font-size: 3.6rem; font-weight: 800; line-height: 1.05; margin: 12px 0 4px;}
.w-big span {font-size: 1.15rem; font-weight: 500; opacity: .75; margin-left: 10px; color: inherit;}
.w-range {font-size: .98rem; opacity: .8; margin-bottom: 6px;}
.w-advice {font-size: 1.1rem; font-weight: 600; margin: 8px 0;}
.w-line {font-size: .98rem; opacity: .85; margin-top: 4px;}
.w-trust {margin-top: 14px; padding: 10px 14px; border-radius: 12px; background: rgba(47,168,145,.12); border: 1px solid rgba(47,168,145,.35); font-size: .9rem;}
.w-move {display: grid; grid-template-columns: 1fr auto 1fr; gap: 14px; align-items: center; margin: 8px 0;}
.w-move-col {border: 1px solid rgba(127,127,127,.25); border-top: 6px solid #4F5FE0; border-radius: 18px; padding: 16px 18px; background: rgba(127,127,127,.06);}
.w-move-label {font-size: .8rem; font-weight: 700; opacity: .8; letter-spacing: .03em;}
.w-move-route {font-size: 1.8rem; font-weight: 800; margin: 4px 0;}
.w-move-n {font-size: 1rem;}
.w-move-s {font-size: .85rem; opacity: .7; margin-top: 2px;}
.w-arrow {font-size: 2.2rem; font-weight: 800; color: #4F5FE0; text-align: center;}
.w-arrow small {display: block; font-size: .8rem; font-weight: 700;}
@media (max-width: 700px) {.w-move {grid-template-columns: 1fr;} .w-arrow {transform: rotate(90deg);}}
</style>"""


def esc(x):
    return _html.escape(str(x))


def inject_css():
    st.markdown(APP_CSS, unsafe_allow_html=True)


def hero_html(title, subtitle, pills=()):
    pills_html = "".join(f'<span class="tc-pill">{esc(p)}</span>' for p in pills)
    return (f'<div class="tc-hero"><div class="tc-hero-title">{esc(title)}</div>'
            f'<div class="tc-hero-sub">{esc(subtitle)}</div><div class="tc-pills">{pills_html}</div></div>')


def pagehead_html(title, subtitle):
    return (f'<div class="tc-pagehead"><div class="tc-pagehead-t">{esc(title)}</div>'
            f'<div class="tc-pagehead-s">{esc(subtitle)}</div></div>')


def kpi_html(label, value, sub="", color=INDIGO):
    return (f'<div class="tc-kpi" style="border-top-color:{color}"><div class="tc-kpi-label">{esc(label)}</div>'
            f'<div class="tc-kpi-value">{esc(value)}</div><div class="tc-kpi-sub">{esc(sub)}</div></div>')


def kpi_grid_html(items):
    return '<div class="tc-grid">' + "".join(kpi_html(*i) for i in items) + '</div>'


def steps_html(items):
    out = "".join(
        f'<div class="tc-step"><div class="tc-step-n">{i}</div><div><div class="tc-step-t">{esc(t)}</div>'
        f'<div class="tc-step-d">{esc(d)}</div></div></div>' for i, (t, d) in enumerate(items, 1))
    return f'<div class="tc-steps">{out}</div>'


def gauge_html(ratio, level, labels=("quiet", "usual busiest hour of this route = 100%", "busy")):
    top = 1.3
    a, b = LOW_RATIO / top * 100, HIGH_RATIO / top * 100
    pos = min(max(ratio / top * 100, 2), 98)
    bg = (f"linear-gradient(to right,{TEAL} 0%,{TEAL} {a:.1f}%,{AMBER} {a:.1f}%,{AMBER} {b:.1f}%,"
          f"{CORAL} {b:.1f}%,{CORAL} 100%)")
    return (f'<div class="tc-gauge" style="background:{bg}"><div class="tc-dot" '
            f'style="left:{pos:.1f}%;border-color:{LEVEL_COLOR[level]}"></div></div>'
            f'<div class="tc-scale"><span>{esc(labels[0])}</span><span>{esc(labels[1])}</span>'
            f'<span>{esc(labels[2])}</span></div>')


def badge_html(level):
    return f'<span class="tc-badge" style="background:{LEVEL_COLOR[level]}">{LEVEL_EMOJI[level]} {level}</span>'


def pred_card_html(c):
    col = LEVEL_COLOR[c['level']]
    tip = f'<div class="tc-tip">{esc(c["tip"])}</div>' if c.get('tip') else ''
    foot = f'<div class="tc-foot">{esc(c["foot"])}</div>' if c.get('foot') else ''
    return (f'<div class="tc-card" style="border-left-color:{col}">'
            f'<div class="tc-top"><div><div class="tc-eyebrow">{esc(c["title"])}</div>'
            f'<div class="tc-sub">{esc(c["subtitle"])}</div></div>{badge_html(c["level"])}</div>'
            f'<div class="tc-big" style="color:{col}">{c["n"]:,}<span>{"rider" if c["n"] == 1 else "riders"}'
            f' expected</span></div>{gauge_html(c["ratio"], c["level"])}'
            f'<div class="tc-note">{esc(c["note"])}</div>{tip}{foot}</div>')


def bus_word(n):
    return f"{n} bus" if n == 1 else f"{n} buses"


def compare_html(c):
    """Shows the best suggested bus move: spare buses on one route go to a route that needs more."""
    if c.get('none'):
        return f'<div class="tc-tip">💡 {esc(c["advice"])}</div>'
    f, t = c['from'], c['to']
    return ('<div class="w-move">'
            f'<div class="w-move-col" style="border-top-color:{TEAL}">'
            f'<div class="w-move-label">🟢 CAN SPARE {esc(bus_word(c["n"]).upper())}</div>'
            f'<div class="w-move-route">Route {esc(f["route"])}</div>'
            f'<div class="w-move-n">{f["riders"]:,} riders expected</div>'
            f'<div class="w-move-s">Normally {esc(bus_word(f["normal"]))}, needs {f["need"]}</div></div>'
            f'<div class="w-arrow">➜<small>{c["n"]}</small></div>'
            f'<div class="w-move-col" style="border-top-color:{CORAL}">'
            f'<div class="w-move-label">🔴 NEEDS MORE BUSES</div>'
            f'<div class="w-move-route">Route {esc(t["route"])}</div>'
            f'<div class="w-move-n">{t["riders"]:,} riders expected</div>'
            f'<div class="w-move-s">Normally {esc(bus_word(t["normal"]))}, needs {t["need"]}</div></div></div>'
            f'<div class="tc-tip">💡 {esc(c["advice"])}</div>')


def hours_html(c):
    chips = "".join(
        f'<div class="tc-hour" style="border-color:{LEVEL_COLOR[lvl]}"><div class="tc-hour-t">{esc(h)}</div>'
        f'<div class="tc-hour-n">about {n:,} riders</div>{badge_html(lvl)}</div>'
        for h, n, lvl in c['hours'])
    return (f'<div class="tc-card"><div class="tc-eyebrow">{esc(c["title"])}</div>'
            f'<div class="tc-sub">{esc(c["subtitle"])}</div><div class="tc-hours">{chips}</div>'
            f'<div class="tc-foot">{esc(c["foot"])}</div></div>')


def card_html(c):
    return {'prediction': pred_card_html, 'compare': compare_html, 'hours': hours_html}[c['kind']](c)


def glance_html():
    """Four cards: dataset, technology, models, accuracy."""
    best = RESULTS.get(LAG_MODEL_NAME, {})
    base = RESULTS.get("Baseline: usual route-day-hour average", {})
    acc = []
    if best:
        acc.append(f"With last week's riders: ±{best['MAE']:.1f} riders (WAPE {best['WAPE %']:.1f}%)")
    if base:
        acc.append(f"Without it (usual riders): ±{base['MAE']:.1f} riders")
        acc.append("R² is high because routes differ a lot in size")
    cards = [
        ("📦 Dataset", INDIGO, "", [f"MTA New York bus ridership", f"{info['n_routes']} routes, hourly (data.ny.gov)",
                                    f"Train on {info['train_period']}", f"Test on {info['test_period']} (separate data)"]),
        ("🛠️ Technology", TEAL, "", ["Python and Google Colab", "pandas and scikit-learn", "Streamlit and Plotly"]),
        ("🧠 Models", AMBER, "", ["3 simple rules (baselines)", "Linear Regression", "Gradient Boosting",
                                 "Random Forest (best)"]),
        ("🎯 Accuracy", CORAL, f"±{best['MAE']:.1f} riders" if best else "", acc or ["See the test results"]),
    ]
    out = ""
    for title, color, big, lines in cards:
        bigh = f'<div class="tc-gcard-big" style="color:{color}">{esc(big)}</div>' if big else ''
        li = "".join(f"<li>{esc(x)}</li>" for x in lines)
        out += (f'<div class="tc-gcard" style="border-top-color:{color}"><div class="tc-gcard-h">{esc(title)}</div>'
                f'{bigh}<ul>{li}</ul></div>')
    return f'<div class="tc-glance">{out}</div>'


ORIGIN_LABEL = {'model': '🤖 FORECAST', 'docs': '📚 FROM PROJECT DOCUMENTS', 'info': 'ℹ️ PROJECT INFO'}


def origin_html(kind):
    return f'<div class="tc-origin">{ORIGIN_LABEL[kind]}</div>' if kind in ORIGIN_LABEL else ''


def source_chips_html(sources):
    if not sources:
        return ''
    chips = "".join(
        f'<span class="tc-src">📎 {esc(s["title"])}'
        + (f' · {s["score"] * 100:.0f}% match' if s.get("score") is not None else '') + '</span>'
        for s in sources)
    return f'<div class="tc-srcs">Sources: {chips}</div>'


def _clean_fig(fig, height=None):
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(size=13))
    if height:
        fig.update_layout(height=height)
    return fig


def style_fig(fig, kind="line"):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(size=13))
    if kind == "line":
        fig.update_traces(line=dict(color=INDIGO, width=3), marker=dict(size=7, color=INDIGO))
    else:
        fig.update_traces(marker_color=INDIGO)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(127,127,127,.2)")
    return fig


def curve_figure(route, d, hour=None, show_real=True):
    """Expected riders for the day with the likely range, the usual pattern and (if saved) the real riders."""
    labels = [fmt_hour(h) for h in range(24)]
    y = np.asarray(day_curve(route, d.isoformat()), dtype=float)
    e = err_for(route, real_lag=uses_real_lag_day(d))
    fig = go.Figure()
    fig.add_scatter(x=labels, y=y * (1 + e), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=labels, y=y * (1 - e), mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(79,95,224,.15)", name="Likely range", hoverinfo="skip")
    fig.add_scatter(x=labels, y=y, mode="lines+markers", name="Our forecast",
                    line=dict(color=INDIGO, width=3), marker=dict(size=6))
    fig.add_scatter(x=labels, y=usual_curve(route, d), mode="lines", name="Usual riders",
                    line=dict(color=AMBER, width=2, dash="dash"))
    act = actual_day(route, d) if show_real else None
    if act is not None:
        fig.add_scatter(x=labels, y=act, mode="lines+markers", name="Real riders",
                        line=dict(color=TEAL, width=3), marker=dict(size=6))
    if hour is not None:
        fig.add_shape(type="line", x0=labels[hour], x1=labels[hour], y0=0, y1=1, yref="paper",
                      line=dict(color=CORAL, width=2, dash="dot"))
    fig.update_layout(title=f"Route {route}: expected riders through the day", margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.2), yaxis_title="Riders")
    fig.update_yaxes(gridcolor="rgba(127,127,127,.2)", rangemode="tozero")
    return fig


def buses_by_hour_fig(route, d, cap):
    """Buses needed in each hour (bars) against the buses normally run (line)."""
    labels = [fmt_hour(h) for h in range(24)]
    pred = np.asarray(day_curve(route, d.isoformat()), dtype=float)
    need = [buses_for(v, cap) for v in pred]
    normal = [buses_for(v, cap) for v in usual_curve(route, d)]
    colors = [CORAL if n > m else (TEAL if n < m else INDIGO) for n, m in zip(need, normal)]
    fig = go.Figure()
    fig.add_bar(x=labels, y=need, marker_color=colors, name="Buses needed")
    fig.add_scatter(x=labels, y=normal, mode="lines+markers", name="Buses normally run",
                    line=dict(color=AMBER, width=2, dash="dash"), marker=dict(size=5))
    fig.update_layout(title=f"Route {route}: buses needed each hour", yaxis_title="Buses",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(gridcolor="rgba(127,127,127,.2)", rangemode="tozero", dtick=1 if max(need + normal) <= 12 else None)
    return _clean_fig(fig, 360), need, normal


@st.cache_data(show_spinner=False)
def week_outlook(route, start_iso):
    """Total riders and busiest hour for each of the next 7 days."""
    start = date.fromisoformat(start_iso)
    rows = []
    for i in range(7):
        d = start + timedelta(days=i)
        c = np.asarray(day_curve(route, d.isoformat()), dtype=float)
        tag = "Holiday" if (is_holiday(d) and d.weekday() < 5) else ("Weekend" if d.weekday() >= 5 else "Weekday")
        rows.append({'Day': d.strftime('%a %d %b'), 'Total riders': int(round(c.sum())),
                     'Busiest hour': fmt_hour(int(np.argmax(c))), 'Type': tag})
    return pd.DataFrame(rows)


def compare_day_fig(route, d):
    """Forecast and real riders for each hour of one day."""
    labels = [fmt_hour(h) for h in range(24)]
    fig = go.Figure()
    fig.add_scatter(x=labels, y=day_curve(route, d.isoformat()), mode="lines+markers", name="Our forecast",
                    line=dict(color=INDIGO, width=3), marker=dict(size=6))
    act = actual_day(route, d)
    if act is not None:
        fig.add_scatter(x=labels, y=act, mode="lines+markers", name="Real riders",
                        line=dict(color=TEAL, width=3), marker=dict(size=6))
    fig.update_layout(legend=dict(orientation="h", y=-0.2), yaxis_title="Riders")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(127,127,127,.2)", rangemode="tozero")
    return _clean_fig(fig, 340)


def level_donut(tbl):
    order = ["High", "Moderate", "Low"]
    counts = tbl['level'].value_counts()
    fig = go.Figure(go.Pie(labels=["Very busy", "Fairly busy", "Quiet"], values=[int(counts.get(o, 0)) for o in order],
                           hole=0.62, sort=False, marker=dict(colors=[CORAL, AMBER, TEAL]), textinfo="label+value"))
    fig.update_layout(showlegend=False,
                      annotations=[dict(text=f"{len(tbl)}<br>routes", x=0.5, y=0.5, showarrow=False, font=dict(size=16))])
    return _clean_fig(fig, 300)


def change_bar_fig(tbl, n=8):
    """Routes that need the most extra buses (red) and routes that can spare the most (green)."""
    more = tbl[tbl['Change'] > 0].sort_values('Change', ascending=False).head(n)
    spare = tbl[tbl['Change'] < 0].sort_values('Change').head(n)
    t = pd.concat([more, spare])
    t = t.iloc[::-1]
    fig = go.Figure(go.Bar(x=t['Change'].astype(int), y=t['Route'].astype(str), orientation="h",
                           marker_color=[CORAL if c > 0 else TEAL for c in t['Change']],
                           text=[f"{int(c):+d}" for c in t['Change']], textposition="outside"))
    lim = max(float(t['Change'].abs().max()), 1) * 1.3
    fig.update_xaxes(range=[-lim, lim], title="Buses to add (+) or spare (-)", gridcolor="rgba(127,127,127,.2)",
                     zeroline=True, zerolinecolor="rgba(127,127,127,.6)")
    return _clean_fig(fig, max(260, 30 * len(t) + 80))


# ============================================================
# CHAT ASSISTANT: understanding the question
# ============================================================
ROUTE_LOOKUP = {re.sub(r'[\s\-]', '', str(r).lower()): r for r in ROUTES}
ROUTE_RE = re.compile(r'(?<![a-z0-9])([a-z]{1,3}\d{1,3}[a-z]{0,3}\+?)(?![a-z0-9])')
ROUTE_WORD_RE = re.compile(r'\b(?:route|bus|line)\s+([a-z]{0,3}\d{1,3}[a-z]{0,3}\+?)(?![a-z0-9])')
DAY_WORDS = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
             'friday': 4, 'saturday': 5, 'sunday': 6}


def find_route(t):
    """Returns (route_id or None, unknown_text or None)."""
    sbs = bool(re.search(r'sbs|select bus', t))
    for m in ROUTE_RE.finditer(t):
        raw = m.group(1)
        base = raw.rstrip('+')
        order = [base + 'sbs', base + '+', raw, base] if sbs else [raw, base + '+', base]   # 'SBS' means the + route
        for cand in order:
            if cand in ROUTE_LOOKUP:
                return ROUTE_LOOKUP[cand], None
    m = ROUTE_WORD_RE.search(t)
    if m:
        return None, m.group(1).upper()
    return None, None


def route_suggestions(text, n=3):
    keys = difflib.get_close_matches(re.sub(r'[\s\-]', '', text.lower()), list(ROUTE_LOOKUP), n=n, cutoff=0.5)
    return [ROUTE_LOOKUP[k] for k in keys]


def parse_hour(t):
    """Finds a time like '8 AM', '5:30 pm', '17:00', 'at 15', 'morning'. Returns 0-23 or None."""
    m = re.search(r'\b(\d{1,2})(?::[0-5]\d)?\s*([ap])\.?m\b', t)
    if m and 1 <= int(m.group(1)) <= 12:
        return (int(m.group(1)) % 12) + (12 if m.group(2) == 'p' else 0)
    m = re.search(r'\b([01]?\d|2[0-3]):[0-5]\d\b', t)
    if m:
        return int(m.group(1))
    if re.search(r'\bmidnight\b', t):
        return 0
    if re.search(r'\bnoon\b', t):
        return 12
    for word, h in (('morning', 8), ('afternoon', 15), ('evening', 18), ('tonight', 21), ('night', 21)):
        if re.search(r'\b' + word + r'\b', t):
            return h
    m = re.search(r'\b(?:at|around|by|@)\s*(\d{1,2})\b(?![/\-:.]\d)', t)
    if m and 0 <= int(m.group(1)) <= 23:
        return int(m.group(1))
    return None


def _phrase_for_date(d):
    today = date.today()
    name = d.strftime('%A')
    if d == today:
        return f"today ({name})"
    if d == today + timedelta(days=1):
        return f"tomorrow ({name})"
    if d == today - timedelta(days=1):
        return f"yesterday ({name})"
    return "on " + d.strftime('%A, %d %b %Y')


def parse_day(t):
    """Returns (date, phrase) or None if the message does not mention a day."""
    today = date.today()
    d = None
    m = re.search(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', t)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            d = None
    if d is None:
        m = re.search(r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b', t)   # day/month/year
        if m:
            try:
                d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                d = None
    if d is None:
        if re.search(r'\bday after tomorrow\b', t):
            d = today + timedelta(days=2)
        elif re.search(r'\btomorrow\b', t):
            d = today + timedelta(days=1)
        elif re.search(r'\b(today|tonight)\b', t):
            d = today
        elif re.search(r'\byesterday\b', t):
            d = today - timedelta(days=1)
    if d is None:
        for name, idx in DAY_WORDS.items():
            if re.search(r'\b' + name + r's?\b', t):
                d = today + timedelta(days=(idx - today.weekday()) % 7)
                break
    if d is None and re.search(r'\bweekends?\b', t):
        d = today + timedelta(days=(5 - today.weekday()) % 7)
    if d is None and re.search(r'\bweekdays?\b', t):
        d = today if today.weekday() < 5 else today + timedelta(days=7 - today.weekday())
    return (d, _phrase_for_date(d)) if d is not None else None


def parse_last_week(t):
    m = (re.search(r'(?:last week|previous week|week before|a week ago)[^0-9]{0,40}(\d{1,6})', t)
         or re.search(r'(\d{1,6})\s*(?:riders|passengers|people)?\s*(?:last week|a week ago|the week before)', t))
    return int(m.group(1)) if m else None


# ============================================================
# CHAT ASSISTANT: remembering the conversation
# ============================================================
def new_ctx():
    return {'route': None, 'hour': None, 'date': None, 'phrase': None}


def get_ctx():
    if 'chat_ctx' not in st.session_state:
        st.session_state['chat_ctx'] = new_ctx()
    return st.session_state['chat_ctx']


def reply(text, table=None, chart=None, card=None, suggestions=None, origin=None, sources=None):
    return {'text': text, 'table': table, 'chart': chart, 'card': card,
            'suggestions': suggestions or [], 'origin': origin, 'sources': sources or []}


STARTERS = [f"How busy is {TOP_ROUTE} at 8 AM tomorrow?",
            f"Best time to travel on {TOP_ROUTE}",
            "Which routes need more buses at 8 AM?",
            "Why was this model chosen?"]


def _date_or_default(day, ctx):
    if day:
        return day
    if ctx['date'] is not None:
        return ctx['date'], ctx['phrase']
    today = date.today()
    return today, f"today ({today.strftime('%A')})"


def quiet_tip(route, d, h, level):
    """If a nearby hour is clearly quieter, suggest it."""
    if level == "Low":
        return None
    curve = day_curve(route, d.isoformat())
    lo, hi = max(5, h - 2), min(23, h + 2)
    window = curve[lo:hi + 1]
    if len(window) == 0:
        return None
    best = lo + int(np.argmin(window))
    now = float(curve[h])
    if best != h and now - float(window.min()) >= max(5.0, 0.1 * now) and float(window.min()) <= 0.85 * now:
        return (f"Tip: around {fmt_hour(best)} it should be quieter "
                f"(about {round(float(window.min())):,} riders instead of {round(now):,}).")
    return None


def build_prediction_card(route, d, phrase, h, riders, lag_used):
    level, ratio = level_for(route, d, riders)
    usual = float(usual_for([route], d, h)[0])
    n = int(round(riders))
    real = (not lag_used) and has_real_lag(d, h)
    if (lag_used or real) and usual > 0:
        pct = (riders / usual - 1) * 100
        note = f"{LEVEL_PLAIN[level]} About {abs(pct):.0f}% {'above' if pct >= 0 else 'below'} the usual {usual:,.0f} riders."
        method = ("Used the real riders from one week earlier (saved 2025 data) with the Random Forest model."
                  if real else "Used your last-week number with the more accurate model.")
    elif HAS_LAG and USE_ML_DEFAULT:
        note = f"{LEVEL_PLAIN[level]} Forecast by the machine-learning model for this route, date and hour."
        method = "Method: Random Forest model, using this route's usual last-week level."
    elif USE_PROFILE:
        note = (f"{LEVEL_PLAIN[level]} Based on the usual riders for this route, weekday and hour "
                f"({usual:,.0f}), adjusted to the route's recent level.")
        method = "Method: usual riders, adjusted to the recent level of the route (it beat the calendar model in testing)."
    else:
        note = LEVEL_PLAIN[level]
        method = "Method: calendar model (route, date, hour, holidays)."
    need, normal = buses_for(riders, get_cap()), buses_for(usual, get_cap())
    foot = (f"Needs about {bus_word(need)}; the usual service at this hour is about {bus_word(normal)} "
            f"(if one bus carries {get_cap()} riders an hour, an assumption you can change on the Bus Planner page). "
            f"{method}")
    return {'kind': 'prediction', 'title': f"Route {route}", 'subtitle': f"{fmt_hour(h)} · {phrase}", 'n': n,
            'level': level, 'ratio': ratio, 'note': note, 'tip': quiet_tip(route, d, h, level), 'foot': foot}


def unknown_route_reply(unknown):
    close = route_suggestions(unknown)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    return reply(f"I could not find route **{unknown}** in the data.{hint} Route names include letters, "
                 f"for example **{TOP_ROUTE}**.", suggestions=STARTERS[:2])


def compare_data(tbl):
    """The best suggested bus move (a card) from a route table, or a card saying nothing needs to move."""
    moves = build_moves(tbl, limit=1)
    n_more = int((tbl['Change'] > 0).sum())
    n_spare = int((tbl['Change'] < 0).sum())
    if len(moves) == 0:
        return {'kind': 'compare', 'none': True, 'n_more': n_more, 'n_spare': n_spare,
                'advice': "Normal service is enough on every route at this time, so there is no need to move buses."}
    mv = moves.iloc[0]
    rows = tbl.set_index('Route')

    def side(r):
        x = rows.loc[r]
        return {'route': r, 'riders': int(x['Expected riders']), 'normal': int(x['Normal buses']),
                'need': int(x['Buses needed'])}
    same = mv['Same borough'] == "Yes"
    n = int(mv['Buses to move'])
    advice = (f"Move {bus_word(n)} from Route {mv['From route']} to Route {mv['To route']}. "
              + ("Both are in the same borough, so sharing buses may be practical. " if same else
                 "They are in different boroughs, so moving buses may not be practical. ")
              + "This is a suggestion only. A planner must check depots, vehicles, drivers and schedules.")
    return {'kind': 'compare', 'none': False, 'n': n, 'from': side(mv['From route']), 'to': side(mv['To route']),
            'n_more': n_more, 'n_spare': n_spare, 'advice': advice}


def answer_help():
    return reply(
        "I can forecast **how many riders a bus route will have** and answer questions about the project.\n\n"
        "**Forecasts**\n"
        f"- *How busy is {TOP_ROUTE} at 8 AM tomorrow?*\n"
        f"- *Best time to travel on {TOP_ROUTE}*\n"
        f"- *When is {TOP_ROUTE} busiest on Saturday?*\n"
        "- *Which routes need more buses at 8 AM?*\n"
        f"- *{TOP_ROUTE} at 5 PM tomorrow, last week it had 420 riders* (uses the more accurate model)\n\n"
        "**The project (from the documents)**\n"
        "- *Why was this model chosen?*\n"
        "- *How accurate is it?*\n"
        "- *What are the limitations?*\n\n"
        "Routes have letters, like M15, Bx12 or Q58. I understand times (8 AM, 17:00), days (today, tomorrow, "
        "Saturday, 25/12/2026) and I remember your last route, time and day for follow-ups like *what about 5 PM?*",
        origin='info', suggestions=STARTERS)


def answer_accuracy():
    best = RESULTS.get(LAG_MODEL_NAME, {})
    base = RESULTS.get("Baseline: usual route-day-hour average", {})
    cal = RESULTS.get("Gradient Boosting (calendar only)", {})
    lines = [f"The forecasts were tested on **{info['test_rows']:,} route-hours** from a separate dataset "
             f"({info['test_period']}), which the model never saw during training."]
    if best:
        lines.append(f"- **With last week's riders ({LAG_MODEL_NAME}):** average error **{best['MAE']:.1f} riders** "
                     f"per route-hour, R² {best['R2']:.2f}, WAPE {best['WAPE %']:.1f}%")
    if base:
        lines.append(f"- **Without last week's riders (usual riders for that day and hour):** error {base['MAE']:.1f} riders")
    if cal:
        lines.append(f"- **Calendar-only model:** error {cal['MAE']:.1f} riders")
    proxy_res = RESULTS.get(PROXY_NAME)
    if proxy_res:
        lines.append(f"- **Random Forest without last week (given the usual level):** error {proxy_res['MAE']:.1f} riders")
    lines.append("\nThese are estimates, not exact counts, and they do not know about weather or special events."
                 + ("" if (USE_ML_DEFAULT and proxy_res) else " The model needs last week's number to beat the simple rule."))
    return reply("\n".join(lines), origin='info',
                 suggestions=["Is the model better than a simple rule?", "What are the limitations?",
                              "How was the model tested?"])


def answer_levels():
    return reply(
        "**Demand levels** compare the expected riders with the **usual busiest hour** of that route on that "
        "type of day (weekday, Saturday or Sunday):\n\n"
        f"- 🟢 **Low (quiet):** under {int(LOW_RATIO * 100)}% of the usual peak\n"
        f"- 🟡 **Moderate (fairly busy):** {int(LOW_RATIO * 100)}% to {int(HIGH_RATIO * 100)}%\n"
        f"- 🔴 **High (very busy):** {int(HIGH_RATIO * 100)}% or more\n\n"
        "Using each route's own peak keeps small and big routes comparable. A route that is *very busy* at its "
        "usual rush hour does not automatically need extra buses, because its normal service already covers that.",
        origin='info', suggestions=[STARTERS[0], STARTERS[2]])


def answer_peak(route, day, overall):
    ctx = get_ctx()
    if route is None and not overall:
        route = ctx['route']
    if route is None:
        hourly = pd.Series(info['hourly_avg']).sort_index()
        hi, lo = int(hourly.idxmax()), int(hourly.idxmin())
        return reply(
            f"Across all routes, buses are busiest around **{fmt_hour(hi)}** (about {hourly[hi]:.0f} riders per "
            f"route-hour) and quietest around **{fmt_hour(lo)}** (about {hourly[lo]:.0f}).\n\n"
            f"Ask about one route, for example *When is {TOP_ROUTE} busiest?*",
            chart={'x': [fmt_hour(h) for h in hourly.index], 'y': [round(float(v), 1) for v in hourly.values],
                   'title': 'Average riders per route-hour, by time of day'},
            origin='info', suggestions=[f"When is {TOP_ROUTE} busiest?", f"Best time to travel on {TOP_ROUTE}"])
    d, phrase = _date_or_default(day, ctx)
    ctx['route'], ctx['date'], ctx['phrase'] = route, d, phrase
    curve = day_curve(route, d.isoformat())
    hi, lo = int(np.argmax(curve)), int(np.argmin(curve))
    hint = ("\n\n_I used the route from your last question. Say **overall** to see all routes together._"
            if st.session_state.get('_route_from_ctx', False) else "")
    return reply(
        f"For **Route {route}** ({phrase}), the busiest time is around **{fmt_hour(hi)}** "
        f"(about {curve[hi]:,.0f} riders) and the quietest around **{fmt_hour(lo)}** (about {curve[lo]:,.0f}).{hint}",
        chart={'x': [fmt_hour(h) for h in range(24)], 'y': [round(float(v), 1) for v in curve],
               'title': f'Expected riders by time of day: Route {route}'},
        origin='model', suggestions=["Best time to travel", f"How busy is Route {route} at {fmt_hour(hi)}?",
                                     "Which routes need more buses?"])


def _window_from_text(t):
    if re.search(r'\bmorning\b', t):
        return 6, 11, "in the morning"
    if re.search(r'\bafternoon\b', t):
        return 12, 16, "in the afternoon"
    if re.search(r'\bevening\b', t):
        return 17, 21, "in the evening"
    return 6, 21, "between 6 AM and 9 PM"


def answer_besttime(t, route, day):
    ctx = get_ctx()
    lo_h, hi_h, label = _window_from_text(t)
    overall = bool(re.search(r'overall|in general|all routes|every route|whole|network', t))
    if route is None and not overall:
        route = ctx['route']
    if route is None:
        hourly = pd.Series(info['hourly_avg']).sort_index().loc[lo_h:hi_h]
        best = hourly.nsmallest(3).sort_index()
        hours = [(fmt_hour(h), round(float(v)), "Low") for h, v in best.items()]
        peak_h = int(hourly.idxmax())
        card = {'kind': 'hours', 'title': 'Least crowded times (all routes)', 'subtitle': label, 'hours': hours,
                'foot': f"Busiest in this window: {fmt_hour(peak_h)} (about {hourly[peak_h]:.0f} riders per route-hour). "
                        "Average of all routes."}
        return reply(f"Across all routes, the quietest times {label} are:", card=card, origin='info',
                     suggestions=[f"Best time to travel on {TOP_ROUTE}", f"When is {TOP_ROUTE} busiest?"])
    d, phrase = _date_or_default(day, ctx)
    ctx['route'], ctx['date'], ctx['phrase'] = route, d, phrase
    curve = day_curve(route, d.isoformat())
    idx = np.arange(lo_h, hi_h + 1)
    order = idx[np.argsort(curve[idx])][:3]
    best_hours = sorted(int(h) for h in order)
    hours = [(fmt_hour(h), int(round(curve[h])), level_for(route, d, float(curve[h]))[0]) for h in best_hours]
    peak_h = int(idx[np.argmax(curve[idx])])
    card = {'kind': 'hours', 'title': f'Least crowded times: Route {route}', 'subtitle': f"{label}, {phrase}",
            'hours': hours,
            'foot': f"Busiest in this window: {fmt_hour(peak_h)} (about {curve[peak_h]:,.0f} riders). "
                    "Estimates for the whole route."}
    return reply(f"If you want a seat, these are the best times to travel on **Route {route}**:", card=card,
                 chart={'x': [fmt_hour(h) for h in idx], 'y': [round(float(curve[h]), 1) for h in idx],
                        'title': f'Expected riders {label}: Route {route}'},
                 origin='model', suggestions=[f"How busy is Route {route} at {fmt_hour(best_hours[0])}?",
                                              "When is it busiest?", "Which routes need more buses?"])


def answer_compare(hour, day):
    ctx = get_ctx()
    assumed = False
    if hour is None:
        hour = ctx['hour']
    if hour is None:
        hour, assumed = 8, True
    d, phrase = _date_or_default(day, ctx)
    ctx['hour'], ctx['date'], ctx['phrase'] = hour, d, phrase

    tbl = route_table(d.isoformat(), hour, get_cap())
    card = compare_data(tbl)
    text = (f"At **{fmt_hour(hour)}**, {phrase}: **{card['n_more']}** routes need more buses than usual and "
            f"**{card['n_spare']}** can spare some. The 5 routes with the biggest change:")
    if assumed:
        text = "_You did not mention a time, so I used 8 AM._\n\n" + text
    big = tbl.reindex(tbl['Change'].abs().sort_values(ascending=False).index).head(5)
    table = big[['Route', 'Expected riders', 'Normal buses', 'Buses needed', 'Change']].reset_index(drop=True)
    return reply(text, table=table, card=card, origin='model',
                 suggestions=[f"When is Route {tbl.iloc[0]['Route']} busiest?", "What about 5 PM?",
                              "How does resource allocation work?"])


def answer_predict(route, unknown, hour, day, lag):
    ctx = get_ctx()
    if unknown:
        return unknown_route_reply(unknown)
    if hour is not None:
        ctx['hour'] = hour
    if day:
        ctx['date'], ctx['phrase'] = day
    if route is None:
        route = ctx['route']
    if route is None:
        return reply("Which route should I check? For example: " + f"*{TOP_ROUTE} at 8 AM tomorrow*.",
                     suggestions=STARTERS[:2])
    ctx['route'] = route
    h = hour if hour is not None else ctx['hour']
    if h is None:
        return reply(f"What time should I check for **Route {route}**? For example: *8 AM* or *5:30 PM*.",
                     suggestions=["8 AM", "12 PM", "5 PM"])
    d, phrase = _date_or_default(None, ctx)
    ctx['date'], ctx['phrase'] = d, phrase

    note_no_lag = ""
    if lag is not None and not HAS_LAG:
        lag, note_no_lag = None, "\n\n_The last-week model could not be loaded on this computer, so I used the usual riders._"
    riders = float(predict_routes([route], d, h, lag)[0])
    card = build_prediction_card(route, d, phrase, h, riders, lag_used=lag is not None)
    alt = 17 if h < 12 else 8
    sugg = [f"What about {fmt_hour(alt)}?", "Best time to travel", "Which routes need more buses?"]
    if card['level'] != "Low":
        sugg.append("What do the levels mean?")
    return reply(f"Here is what I expect for **Route {route}** at **{fmt_hour(h)}**, {phrase}:" + note_no_lag,
                 card=card, origin='model', suggestions=sugg)


def answer(prompt):
    """Understands one chat message and returns a reply dict."""
    result = _answer(prompt)
    h = parse_hour(ROUTE_RE.sub(' ', prompt.lower()))
    if h is not None and re.search(r'\b\d{1,2}:(?!00\b)[0-5]\d\b', prompt.lower()):
        result['text'] += f"\n\n_Note: I work in whole hours, so I used {fmt_hour(h)}._"
    return result


def _answer(prompt):
    t = prompt.lower().strip()
    route, unknown = find_route(t)
    t2 = ROUTE_RE.sub(' ', t)                           # text without route names, so "m15" is not read as a time
    hour, day, lag = parse_hour(t2), parse_day(t2), parse_last_week(t2)
    has_slot = bool(route or unknown or hour is not None or day)
    st.session_state['_route_from_ctx'] = False

    if not (route or unknown) and len(t.split()) <= 4:
        if re.match(r'^(hi+|hello+|hey+|hai|vanakkam|good (morning|afternoon|evening)|start)\b', t):
            return reply("Hi! 👋 I'm **Bus Demand Planner**. Ask me how busy a New York bus route will be, or ask about the "
                         "project. Tap a suggestion below, or type **help** to see what I can do.",
                         suggestions=STARTERS)
        if re.match(r'^(thanks|thank you|thx|nandri|ok|okay|great|nice)\b', t):
            return reply("You're welcome! Ask me anything else about bus demand. 🚌", suggestions=STARTERS[:3])
    if re.search(r'^\s*(help|what can you do|how do you work|how to use)', t):
        return answer_help()

    if re.search(r'accura|\bmae\b|\br2\b|r²|\bwape\b|\berror\b|reliab|\btrust\b|how good|how well', t) \
            and not re.search(r'simple rule|baseline', t):
        return answer_accuracy()
    if re.search(r'demand levels?|(what|meaning|explain).{0,25}\b(high|low|moderate)\b|\blevels?\b', t):
        return answer_levels()

    if re.search(r'\b(where|from)\b.{0,30}\b(data|dataset)\b|which dataset|what dataset|data ?source|kaggle|'
                 r'dataset (name|link|source)|\bmta\b', t):
        doc = answer_doc("Data sources")
        if doc:
            return doc

    if unknown and not route:
        return unknown_route_reply(unknown)

    besttime_kw = re.search(r'best time|avoid (the )?(crowd|rush)|least crowded|less crowded|not (too )?crowded|'
                            r'when should i|when to (travel|go|leave)|quietest (time|hour)|quiet time', t)
    if besttime_kw:
        st.session_state['_route_from_ctx'] = route is None
        return answer_besttime(t, route, day)

    peak_kw = re.search(r'\b(peak|rush)\b|busiest (time|hour)|'
                        r'\b(when|what time|which hour)\b.{0,40}\b(busy|busiest|crowded|quiet|quietest|full|peak)', t)
    if peak_kw:
        overall = bool(re.search(r'overall|in general|all routes|every route|whole|network', t))
        st.session_state['_route_from_ctx'] = route is None and not overall
        return answer_peak(route, day, overall)

    if re.search(r'how (does|do|is|are)\b.{0,40}(allocation|buses needed|bus plan)|what is (the )?resource allocation', t):
        doc = answer_doc("Buses needed and buses normally run" if re.search(r'buses needed', t) else "Resource allocation advice")
        if doc:
            return doc

    compare_kw = re.search(r'busiest|quietest|least busy|most busy|most crowded|least crowded|which routes?\b|'
                           r'compare (the |all )?routes|reallocat|allocat|move (some )?buses|send (more )?buses|'
                           r'need more buses|spare buses|where should|top routes|all routes', t)
    if compare_kw:
        return answer_compare(hour, day)

    if has_slot:
        return answer_predict(route, unknown, hour, day, lag)

    from_docs = rag_for_chat(prompt)
    if from_docs:
        return from_docs
    return reply("Sorry, I did not quite understand that. 🙈 I can forecast bus riders and answer questions "
                 "about the project.\n\nTry one of the suggestions below, or type **help**.", suggestions=STARTERS)


# ============================================================
# PROJECT KNOWLEDGE (RAG): passages + retrieval
# ============================================================
def build_knowledge():
    """Short plain-English passages about the project. Numbers come from the saved test results."""
    best = RESULTS.get(LAG_MODEL_NAME, {})
    base = RESULTS.get("Baseline: usual route-day-hour average", {})
    cal = RESULTS.get("Gradient Boosting (calendar only)", {})
    res_tbl = (pd.DataFrame(RESULTS).T.reset_index().rename(columns={'index': 'Model'})
               .sort_values('MAE').round(2)) if RESULTS else None
    n_routes, test_rows = info['n_routes'], info['test_rows']
    gain = (100 * (base['MAE'] - best['MAE']) / base['MAE']) if best and base and base['MAE'] else None
    bt = BACKTEST
    cards = []

    def card(title, text, keywords="", table=None, cat="Project"):
        cards.append({'title': title, 'text': text, 'keywords': keywords, 'table': table,
                      'source': 'Project documents', 'cat': cat})

    card("Project overview",
         "Bus Demand Planner forecasts how many riders a New York City bus route will have in a given hour, for a given "
         "date. Planners can use it to see when routes are busy or quiet and where buses might be moved. The "
         "forecasting model is trained on one MTA dataset and tested on a second, separate MTA dataset.",
         "about purpose aim objective goal summary introduction what is transitai project")
    card("Data sources",
         "Both datasets come from the MTA (New York) through the Open Data NY portal (data.ny.gov). "
         "Training: 'MTA Bus Hourly "
         "Ridership: 2020-2024' (dataset id kv7t-n8in). Testing: 'MTA Bus Hourly "
         "Ridership: Beginning 2025' (dataset id gxb3-akrn). Both give hourly ridership per bus route.",
         "dataset where data come from source kaggle mta new york data.ny.gov open data portal download "
         "which dataset what dataset where did you get the data origin", cat="Data")
    card("What ridership means",
         "Ridership is the estimated number of riders who entered a bus with OMNY or MetroCard in a given hour on "
         "a given route. It is a fare-payment count, not a passenger-counter measurement, so riders who did not pay "
         "with these methods are not included.",
         "ridership meaning definition omny metrocard fare payment riders count target", cat="Data")
    card("Training and testing design",
         f"The model learns only from {info['train_period']} (training dataset). It is tested on {info['test_period']} "
         f"from a separate dataset that was never used for training. {n_routes} routes appear in both datasets, and "
         f"{test_rows:,} route-hours were used for testing.",
         "train test split design separate dataset validation how was the model tested evaluation period",
         cat="Data")
    card("How the data was downloaded",
         "The full files are very large, so the portal was asked for one total per route and hour for the dates "
         "needed, with all payment types added up. The results were downloaded page by page and saved.",
         "download api size large files route hour totals aggregate", cat="Data")
    card("Data cleaning",
         "Cleaning steps: timestamps were converted to real date-times, ridership was turned into numbers and "
         "negative values were set to 0, rows were reduced to one per route and hour, and only routes present in both "
         "datasets were kept. Missing hours in the full hourly table count as no riders.",
         "preprocessing clean cleaning missing remove steps prepared", cat="Data")
    card("Routes used",
         f"Only routes that appear in both the training and the testing dataset are used: {n_routes} routes. A route "
         f"the model has never seen cannot be forecast.",
         "routes how many routes covered missing route not found unknown route", cat="Data")
    card("Features used",
         "The model uses: the route (its code and its usual average riders in the training year), the hour of the "
         "day, the day of the week, the month, a weekend flag, a US federal holiday flag, and optionally the riders "
         "on the same route and hour one week earlier (lag_168).",
         "features inputs variables columns which features used predictors", cat="Model")
    card("Why last week's riders matter",
         "The riders at the same hour one week earlier are the strongest clue about the next value. "
         + (f"With it the error drops from {cal['MAE']:.1f} to {best['MAE']:.1f} riders per route-hour."
            if best and cal else "Models with this number are more accurate."),
         "lag last week same hour previous week lag_168 previous passenger count", cat="Model")
    card("Route average and fairness",
         "A route's usual average riders is calculated from the training year only, never from the 2025 test data. "
         "The test period contributes to no feature, so the test is fair.",
         "route_avg leakage fair test no leakage data leakage training only", cat="Model")
    card("Baselines",
         "Three simple rules were used as a bar to beat: always guess the average, repeat the same hour of last week, "
         "and use the usual average for that route, weekday and hour. "
         + (f"The usual route-day-hour average is the strongest of the three here (error {base['MAE']:.1f} riders)."
            if base else ""),
         "baseline simple rule naive benchmark compare usual average", cat="Model")
    card("Models compared",
         "Four machine-learning models were compared on the separate 2025 dataset: Linear Regression, Gradient "
         "Boosting (with last week), Gradient Boosting (calendar only) and Random Forest. The table shows every "
         "result, lower MAE is better.",
         "compare comparison models which models linear regression gradient boosting random forest results table "
         "why this model chosen selected best", table=res_tbl, cat="Model")
    if best:
        card("Best model and scores",
             f"The best model is {LAG_MODEL_NAME}: average error (MAE) {best['MAE']:.1f} riders per route-hour, "
             f"RMSE {best['RMSE']:.1f}, R² {best['R2']:.2f} and WAPE {best['WAPE %']:.1f}% on {test_rows:,} route-hours "
             f"from the separate 2025 dataset.",
             "best model scores results accuracy final model performance mae rmse r2 wape", cat="Model")
    card("Accuracy: MAE",
         "MAE (mean absolute error) is the average number of riders by which a forecast is off for one route and "
         "one hour. A smaller MAE is better.",
         "mae error average mistake how far wrong accurate", cat="Model")
    card("Accuracy: R² and WAPE",
         "R² is very high partly because routes differ hugely in size, so even simple guesses look good on R². "
         + (f"WAPE is more honest: total error as a percentage of total riders, here {best['WAPE %']:.1f}%."
            if best else "WAPE is total error as a percentage of total riders, which is easier to compare."),
         "r2 r squared wape percentage error accuracy score why r2 is high", cat="Model")
    card("Is the model better than a simple rule?",
         ("Yes, but modestly. " if gain and gain > 0 else "")
         + (f"The best model has an error of {best['MAE']:.1f} riders against {base['MAE']:.1f} for the usual-average "
            f"rule, which is {gain:.0f}% lower. " if gain is not None else "")
         + "Because the simple rule is already strong, this project reports both honestly.",
         "better than simple rule baseline improvement honest worth it", cat="Model")
    card("Calendar-only forecasts",
         "Without last week's number the machine-learning calendar model "
         + (f"has an error of {cal['MAE']:.1f} riders" if cal else "was tested")
         + (", which is worse than the usual-average rule, so the app uses the usual riders in that case."
            if USE_PROFILE else ", which is better than the usual-average rule, so the app uses it in that case."),
         "calendar only without last week no lag fallback which method app uses", cat="Model")
    if USE_PROFILE:
        card("Recent-level adjustment and holidays",
             "When last week's number is not known, the forecast is the usual riders for the route, weekday and hour, "
             "with two improvements. First, it is scaled by a per-route factor learned from the saved real 2025 riders, "
             "so a route that is now busier or quieter than in the training year is followed. Second, a US federal "
             "holiday on a weekday is treated like a Sunday."
             + (f" In a back-test (factors learned on data before March 2025, tested on March 2025) the average error "
                f"fell from {bt['base']:.1f} to {bt['adj']:.1f} riders." if bt else "")
             + (f" On weekday holidays in the saved data, using the Sunday pattern cut the error from "
                f"{bt['hol_plain']:.0f} to {bt['hol_sunday']:.0f} riders." if bt and bt.get('hol_plain') else ""),
             "adjustment trend recent level holiday sunday scaled factor dynamic forecast how forecast changes",
             cat="Model")
    card("Demand levels",
         f"Demand levels compare the expected riders with the usual busiest hour of that route on that type of day. "
         f"Low (quiet) is under {int(LOW_RATIO * 100)}% of that peak, Moderate (fairly busy) is {int(LOW_RATIO * 100)}% to "
         f"{int(HIGH_RATIO * 100)}%, and High (very busy) is {int(HIGH_RATIO * 100)}% or more. Using each route's own "
         f"peak keeps small and big routes comparable.",
         "high low moderate threshold cutoff level meaning colour color label very busy quiet", cat="App")
    card("How a prediction is made",
         "You choose a route, a date and an hour. If you also know how many riders the route had at that hour last "
         + ("week, the Random Forest model uses that number. Otherwise the model is given the route's usual last-week "
            "level (the training-year average for that route, weekday and hour)."
            if (HAS_LAG and USE_ML_DEFAULT) else
            "week, the more accurate Random Forest model is used. Otherwise the app uses the usual riders for that "
            "route, weekday and hour, adjusted to the route's recent level, because that beat the calendar model "
            "in testing."),
         "how prediction works inputs steps process method used", cat="App")
    card("Buses needed and buses normally run",
         "Buses needed = expected riders divided by how many riders one bus can carry per hour (an assumption set on "
         "the Bus Planner page, because the data has no bus capacity or timetable). Buses normally run = the same "
         "calculation for the usual riders at that hour. The difference is how many buses to add or spare.",
         "buses needed capacity assumption how many buses fleet size normal service change spare add", cat="App")
    card("Resource allocation advice",
         "The Bus Planner compares, for every route, the buses needed for the forecast with the buses the usual "
         "service would run at that hour. Routes that need more are paired with routes that can spare buses, "
         "same borough first. It is only a suggestion: planners must check depots, vehicles, drivers and schedules.",
         "reallocation buses move recommendation fleet allocate suggestion planner resource resource allocation "
         "how does resource allocation work", cat="App")
    card("Holidays",
         "The holiday flag marks US federal holidays only. A weekday holiday is forecast with the Sunday pattern. "
         "Local events, school holidays, weather and service changes are not in the data.",
         "holiday holidays federal festival event weather", cat="App")
    card("Limitations",
         "Main limitations: the model learned from one year (2024) and was tested on January to March 2025 only; "
         "ridership counts fare payments, not passenger counters; only US federal holidays are used; weather, events "
         "and service changes are not in the data; forecasts are per route, not per stop or vehicle; the bus "
         "capacity is an assumption; and ridership may have changed since 2025.",
         "limitations weakness problems drawbacks cannot risk issues shortcomings")
    card("Future improvements",
         "Future work: train on several years, retrain with 2025 data, add weather and event data, add more "
         "holidays, use the MTA stop-level dataset for stop forecasts, use real bus counts for allocation, and put the "
         "app online.",
         "future work improve next steps enhance deploy online more data")
    card("Technology used",
         "Data was downloaded from data.ny.gov and analysed in Google Colab with Python (pandas and scikit-learn). "
         "The app is built with Streamlit and Plotly. The trained models and lookup tables are saved with joblib as "
         ".pkl files.",
         "tools libraries python streamlit colab stack software technology built made")
    card("Chat Assistant",
         "The Assistant understands route names (like M15), times, days and dates, remembers your last route, time "
         "and day, and can compare routes or suggest the best time to travel. It is rule-based: it looks for "
         "keywords, not a general-purpose AI.",
         "chatbot chat assistant how works rule based follow-up best time", cat="App")
    card("How the assistant uses documents (RAG)",
         "For project questions the assistant works like RAG (retrieval-augmented generation). Your question is "
         "matched against short passages from the project documents using TF-IDF similarity, and the best passages "
         "are shown with their sources. No new text is generated, so answers cannot invent facts, but it can only "
         "answer what is written in the documents.",
         "rag retrieval augmented generation tf-idf how this page works search knowledge base documents",
         cat="App")
    return cards


def upload_chunks(files):
    """Turns uploaded .txt / .md notes into extra passages (one per paragraph)."""
    out, budget = [], 200_000
    for f in files or []:
        text = f.getvalue().decode('utf-8', errors='ignore')[:budget]
        budget -= len(text)
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) >= 40]
        for i, p in enumerate(paras, 1):
            out.append({'title': f"{f.name} (part {i})", 'text': p, 'keywords': "", 'table': None,
                        'source': f.name, 'cat': "Your notes"})
        if budget <= 0:
            break
    return out


def all_chunks():
    return build_knowledge() + st.session_state.get('extra_chunks', [])


def _stem(w):
    for suf in ('ing', 'ed', 's'):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[:-len(suf)]
            break
    if len(w) > 3 and w.endswith('e'):
        w = w[:-1]
    return w


IGNORED_WORDS = set(ENGLISH_STOP_WORDS) | {'did', 'does', 'do', 'doing', 'done', 'tell', 'please', 'give', 'show'}


def _tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower().replace('²', '2'))
    return [_stem(w) for w in words if w not in IGNORED_WORDS and len(w) > 1]


@st.cache_resource(show_spinner=False)
def build_index(chunk_keys):
    texts = [f"{t}. {t}. {x} {k}" for t, x, k in chunk_keys]
    vec = TfidfVectorizer(tokenizer=_tokens, token_pattern=None, lowercase=False,
                          ngram_range=(1, 2), sublinear_tf=True)
    return vec, vec.fit_transform(texts)


CHAT_RAG_MIN = 0.15      # chat: stricter than the search page, so random text is not matched


def rag_search(query, chunks, top_k=3):
    vec, mat = build_index(tuple((c['title'], c['text'], c['keywords']) for c in chunks))
    sims = cosine_similarity(vec.transform([query]), mat).ravel()
    order = sims.argsort()[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in order if sims[i] > 0]


CARD_QUESTIONS = {
    "Project overview": "What is this project about?",
    "Data sources": "Where does the data come from?",
    "What ridership means": "What does ridership mean?",
    "Training and testing design": "How was the model tested?",
    "Features used": "Which features were used?",
    "Models compared": "Which models did you compare?",
    "Best model and scores": "How accurate is the model?",
    "Is the model better than a simple rule?": "Is the model better than a simple rule?",
    "Demand levels": "What do Low, Moderate and High mean?",
    "How a prediction is made": "How is a prediction made?",
    "Resource allocation advice": "How does resource allocation work?",
    "Buses needed and buses normally run": "How are the buses needed calculated?",
    "Limitations": "What are the limitations?",
    "Future improvements": "What are the future improvements?",
}
DOC_DEFAULT_QUESTIONS = ["What are the limitations?", "Which models did you compare?", "How accurate is the model?"]


def answer_doc(title):
    for c in all_chunks():
        if c['title'] == title:
            sugg = [CARD_QUESTIONS[x] for x in ("Training and testing design", "What ridership means",
                                                "Features used") if x in CARD_QUESTIONS]
            sugg = [q for q in sugg if q != CARD_QUESTIONS.get(title)][:3] or DOC_DEFAULT_QUESTIONS
            return reply(f"**{c['title']}**\n\n{c['text']}", table=c.get('table'), origin='docs',
                         sources=[{'title': c['title'], 'source': c['source'], 'score': None, 'text': c['text']}],
                         suggestions=sugg)
    return None


def rag_for_chat(prompt):
    hits = rag_search(prompt, all_chunks())
    if not hits or hits[0][1] < CHAT_RAG_MIN:
        return None
    best, score = hits[0]
    used = [(c, s) for c, s in hits if s >= max(CHAT_RAG_MIN * 0.6, 0.5 * score)]
    sugg = [CARD_QUESTIONS[c['title']] for c, _ in hits[1:] if c['title'] in CARD_QUESTIONS]
    for q in DOC_DEFAULT_QUESTIONS:
        if len(sugg) >= 3:
            break
        if q not in sugg:
            sugg.append(q)
    sugg = [q for q in sugg if q != CARD_QUESTIONS.get(best['title'])][:3]
    return reply(f"**{best['title']}**\n\n{best['text']}", table=best.get('table'), origin='docs',
                 sources=[{'title': c['title'], 'source': c['source'], 'score': s, 'text': c['text']}
                          for c, s in used], suggestions=sugg)


# ============================================================
# CHAT SCREEN: drawing messages
# ============================================================
def stream_words(text):
    for chunk in re.split(r'(\s+)', text):
        yield chunk
        time.sleep(0.01)


def draw_extras(m):
    if m.get('card'):
        st.markdown(card_html(m['card']), unsafe_allow_html=True)
    if m.get('table') is not None:
        st.dataframe(m['table'], width="stretch", hide_index=True)
    if m.get('chart'):
        c = m['chart']
        fig = style_fig(px.line(x=c['x'], y=c['y'], markers=True,
                                labels={'x': 'Time of day', 'y': 'Expected riders'}, title=c['title']))
        st.plotly_chart(fig, width="stretch")
    if m.get('sources'):
        st.markdown(source_chips_html(m['sources']), unsafe_allow_html=True)
        with st.expander("See the passages this answer was built from"):
            for s in m['sources']:
                st.markdown(f"**{s['title']}** · {s['source']}")
                st.caption(s['text'])


def draw_message(m):
    with st.chat_message(m['role'], avatar="🚌" if m['role'] == 'assistant' else "🧑"):
        if m.get('origin'):
            st.markdown(origin_html(m['origin']), unsafe_allow_html=True)
        st.markdown(m['content'])
        draw_extras(m)


def draw_chips(suggestions, key_prefix):
    suggestions = [s for s in suggestions if s][:4]
    if not suggestions:
        return
    st.caption("Ask next:")
    cols = st.columns(len(suggestions))
    for i, (col, q) in enumerate(zip(cols, suggestions)):
        if col.button(q, key=f"{key_prefix}_{i}", width="stretch"):
            st.session_state['pending'] = q
            st.rerun()


# ============================================================
# WEBSITE-STYLE LAYOUT: top menu, plain-language pages
# ============================================================
HOME, FORECAST, PLANNER = "🏠 Home", "🔮 Bus Forecast", "🚌 Bus Planner"
INSIGHTS, ASK, ABOUT = "📊 Insights", "💬 Ask a Question", "ℹ️ About"
PAGES = [HOME, FORECAST, PLANNER, INSIGHTS, ASK, ABOUT]

LEVEL_WORD = {"High": "Very busy", "Moderate": "Fairly busy", "Low": "Quiet"}
PLAIN_METHOD = {
    "Random Forest (with last week)": "Random Forest + last week's riders",
    "Gradient Boosting (with last week)": "Gradient Boosting + last week's riders",
    "Baseline: usual route-day-hour average": "Usual riders (simple rule)",
    "Random Forest (usual level, no last week)": "Random Forest, no last week",
    "Baseline: same hour last week": "Same as last week (simple rule)",
    "Linear Regression (with last week)": "Linear Regression + last week's riders",
    "Gradient Boosting (calendar only)": "Gradient Boosting, calendar only",
    "Baseline: always the average": "Same number every time (simple rule)",
}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_best = RESULTS.get(LAG_MODEL_NAME, {})
_hourly = pd.Series(info['hourly_avg']).sort_index()
if 'cap' not in st.session_state:
    st.session_state['cap'] = 100


def goto(p):
    st.session_state['page'] = p


def plain_status(level):
    return f"{LEVEL_EMOJI[level]} {LEVEL_WORD[level]}"


def when_bar(prefix, cols=None):
    """One date and one time, remembered between pages (the widgets themselves are re-created on every page)."""
    if cols is None:
        cols = st.columns(2)
    kd, kh = f"{prefix}_date", f"{prefix}_hour"
    if kd not in st.session_state:
        st.session_state[kd] = st.session_state.get('_pref_date', date.today() + timedelta(days=1))
    if kh not in st.session_state:
        st.session_state[kh] = st.session_state.get('_pref_hour', 8)
    d = cols[0].date_input("📅 Date", key=kd)
    h = cols[1].selectbox("🕗 Time", list(range(24)), format_func=fmt_hour, key=kh)
    st.session_state['_pref_date'], st.session_state['_pref_hour'] = d, h
    return d, h


def cap_setting():
    """Riders one bus can carry per hour. An assumption, so the person can change it."""
    if 'w_cap' not in st.session_state:
        st.session_state['w_cap'] = int(st.session_state.get('cap', 100))
    cap = st.number_input("How many riders can one bus carry in an hour?", min_value=20, max_value=1000,
                          step=10, key="w_cap",
                          help="We do not have this number in the data, so it is an assumption.")
    st.session_state['cap'] = int(cap)
    return int(cap)


def reliability_line(real):
    res = _best if (real and _best) else (DEFAULT_RES or _best)
    if not res:
        return ""
    return (f"How reliable is this? In our tests, forecasts like this missed by about {round(res['WAPE %'])} "
            f"riders in every 100. It is a good estimate, not an exact count.")


def simple_gauge_html(ratio, level):
    return gauge_html(ratio, level, labels=("Quiet", "Fairly busy", "Very busy"))


def badge_plain(level):
    return f'<span class="tc-badge" style="background:{LEVEL_COLOR[level]}">{plain_status(level)}</span>'


def result_html(route, d, hour, n, lo, hi, level, ratio, need, normal, usual, tip, trust):
    col = LEVEL_COLOR[level]
    if n == 0 and usual <= 0:
        advice = "👉 No riders are expected. This route may not run at this hour."
    elif need > normal:
        advice = f"👉 Add {bus_word(need - normal)} to the usual service (about {need} needed in total)."
    elif need < normal:
        advice = f"👉 {bus_word(normal - need).capitalize()} can be spared (about {need} needed, usually {normal})."
    else:
        advice = f"👉 Normal service is enough (about {bus_word(need)} needed)."
    usual_line = (f'<div class="w-line">Normally this route has about {round(usual):,} riders at this time.</div>'
                  if usual > 0 else '')
    tip_html = f'<div class="tc-tip">💡 {esc(tip)}</div>' if tip else ''
    trust_html = f'<div class="w-trust">{esc(trust)}</div>' if trust else ''
    rng = (f'<div class="w-range">Likely range: about {lo:,} to {hi:,} riders.</div>' if n > 0 else '')
    return (f'<div class="w-result" style="border-left-color:{col}">'
            f'<div class="w-res-top"><div><div class="w-route">Route {esc(route)}</div>'
            f'<div class="w-when">{esc(d.strftime("%A, %d %b %Y"))} · {esc(fmt_hour(hour))}</div></div>{badge_plain(level)}</div>'
            f'<div class="w-big" style="color:{col}">{n:,}<span>{"rider" if n == 1 else "riders"} expected</span></div>'
            f'{rng}<div class="w-advice">{esc(advice)}</div>{usual_line}{simple_gauge_html(ratio, level)}'
            f'{tip_html}{trust_html}</div>')


# ---------------- Home ----------------
def page_home():
    st.markdown(hero_html(
        "Know how busy your bus will be",
        "Pick a bus route and a time. We tell you how many people will ride, and how many buses are needed.",
        pills=[f"{info['n_routes']} bus routes", "Hour-by-hour forecasts", "Built on official MTA data"]),
        unsafe_allow_html=True)
    st.markdown(f'<div class="tc-compact">🎓 <b>Project:</b> {esc(PROJECT_TITLE)}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.button("🔮 Check a bus route", type="primary", width="stretch", on_click=goto, args=(FORECAST,))
    c2.button("🚌 See which routes need buses", width="stretch", on_click=goto, args=(PLANNER,))

    st.subheader("How it works")
    st.markdown(steps_html([
        ("Choose", "Pick a bus route, a date and a time."),
        ("See the forecast", "We show how many riders to expect and whether the bus will be quiet or very busy."),
        ("Plan the buses", "We show which routes need more buses and which can spare some."),
    ]), unsafe_allow_html=True)

    st.subheader("Example")
    tomorrow = date.today() + timedelta(days=1)
    st.caption(f"Tomorrow, {tomorrow.strftime('%A %d %b')}, 8 AM")
    tbl = route_table(tomorrow.isoformat(), 8, get_cap())
    st.markdown(compare_html(compare_data(tbl)), unsafe_allow_html=True)
    st.caption("Forecasts are estimates based on past patterns. The final decision is always made by the transport planner.")


# ---------------- Bus Forecast ----------------
def page_forecast():
    st.markdown(pagehead_html("🔮 How busy will the bus be?",
                              "Choose a route, a date and a time. We show what to expect."), unsafe_allow_html=True)
    with st.container(border=True):
        route = st.selectbox("🚌 Bus route", ROUTES, key="w_route",
                             index=ROUTES.index(TOP_ROUTE) if TOP_ROUTE in ROUTES else 0)
        d, hour = when_bar("fc")
        lag = None
        if HAS_LAG:
            with st.expander("Optional: I know how many riders this route had last week"):
                if st.checkbox("Use last week's number", key="w_know"):
                    lag = float(st.number_input("Riders last week (same route, same hour)",
                                                min_value=0, value=0, step=10, key="w_lag"))
    cap = get_cap()
    riders = float(predict_routes([route], d, hour, lag)[0])
    n = int(round(riders))
    real = lag is not None or has_real_lag(d, hour)
    e = err_for(route, real_lag=real)
    lo, hi = int(round(riders * (1 - e))), int(round(riders * (1 + e)))
    level, ratio = level_for(route, d, riders)
    usual = float(usual_for([route], d, hour)[0])
    need, normal = buses_for(riders, cap), buses_for(usual, cap)
    st.markdown(result_html(route, d, hour, n, lo, hi, level, ratio, need, normal, usual,
                            quiet_tip(route, d, hour, level), reliability_line(real)),
                unsafe_allow_html=True)
    act = actual_day(route, d)
    if act is not None:
        st.success(f"✅ Real riders on this date at {fmt_hour(hour)}: **{int(round(act[hour])):,}** "
                   f"(our forecast: {n:,})")

    t1, t2, t3 = st.tabs(["📈 Through the day", "🚌 Buses needed each hour", "📅 Next 7 days"])
    with t1:
        st.plotly_chart(curve_figure(route, d, hour), width="stretch")
        st.caption("The dotted line marks the time you chose. The blue band is the likely range. "
                   "The dashed line is what this route usually has."
                   + (" The green line shows the real riders on this date." if act is not None else ""))
    with t2:
        fig, need_h, normal_h = buses_by_hour_fig(route, d, cap)
        st.plotly_chart(fig, width="stretch")
        extra = sum(max(a - b, 0) for a, b in zip(need_h, normal_h))
        spare = sum(max(b - a, 0) for a, b in zip(need_h, normal_h))
        st.caption(f"Red bars need more buses than usual, green bars can spare some. Over the whole day: "
                   f"{extra} extra bus-hours needed and {spare} bus-hours spare "
                   f"(if one bus carries {cap} riders an hour, which you can change on the Bus Planner page).")
    with t3:
        wk = week_outlook(route, d.isoformat())
        colors = {'Weekday': INDIGO, 'Weekend': AMBER, 'Holiday': CORAL}
        fig = go.Figure(go.Bar(x=wk['Day'], y=wk['Total riders'], marker_color=[colors[t] for t in wk['Type']],
                               text=[f"{v:,}" for v in wk['Total riders']], textposition="outside"))
        fig.update_layout(title=f"Route {route}: total riders per day", yaxis_title="Riders in the day")
        fig.update_yaxes(gridcolor="rgba(127,127,127,.2)", rangemode="tozero")
        st.plotly_chart(_clean_fig(fig, 340), width="stretch")
        st.dataframe(wk, width="stretch", hide_index=True)
        st.caption("Blue = weekday, amber = weekend, red = US federal holiday (forecast like a Sunday).")


# ---------------- Bus Planner ----------------
def page_planner():
    st.markdown(pagehead_html("🚌 Which routes need more buses?",
                              "We compare the buses each route needs with the buses it normally runs at that time."),
                unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        d, hour = when_bar("pl", (c1, c2))
        area = c3.selectbox("🗺️ Borough", ["All boroughs"] + BOROUGHS, key="pl_area")
        with st.expander("Settings"):
            cap = cap_setting()
    full = route_table(d.isoformat(), hour, cap)
    tbl = full if area == "All boroughs" else full[full['Borough'] == area].reset_index(drop=True)
    if len(tbl) == 0:
        st.info("No routes in this borough.")
        return
    when_txt = f"{fmt_hour(hour)} on {d.strftime('%A %d %b')}"
    n_more, n_spare = int((tbl['Change'] > 0).sum()), int((tbl['Change'] < 0).sum())
    extra, spare = int(tbl.loc[tbl['Change'] > 0, 'Change'].sum()), int(-tbl.loc[tbl['Change'] < 0, 'Change'].sum())

    st.markdown(f'<div class="tc-compact">📌 <b>{esc(when_txt)}:</b> {n_more} routes need more buses (+{extra} in total) '
                f'and {n_spare} routes can spare buses (−{spare} in total).</div>', unsafe_allow_html=True)
    st.subheader("Suggestion")
    st.markdown(compare_html(compare_data(tbl)), unsafe_allow_html=True)
    st.caption("This is only a suggestion. A planner must check depots, buses, drivers and schedules.")

    st.markdown(kpi_grid_html([
        ("🔴 Routes needing more buses", f"{n_more}", f"+{extra} buses in total", CORAL),
        ("🟢 Routes that can spare buses", f"{n_spare}", f"−{spare} buses in total", TEAL),
        ("🚌 Buses needed now", f"{int(tbl['Buses needed'].sum()):,}",
         f"normally {int(tbl['Normal buses'].sum()):,} · {cap} riders per bus per hour", INDIGO),
    ]), unsafe_allow_html=True)

    moves = build_moves(tbl, limit=30)
    st.subheader("🔁 Suggested bus moves")
    if moves.empty:
        st.info("No bus moves are needed with these settings.")
    else:
        st.dataframe(moves, width="stretch", hide_index=True)
        if extra > spare:
            st.caption(f"After these moves, about {extra - spare} more buses would still be needed in total.")
        elif spare > extra:
            st.caption(f"After these moves, about {spare - extra} buses would still be spare.")

    g1, g2 = st.columns([3, 2])
    with g1:
        st.subheader("📊 Biggest changes")
        if n_more + n_spare:
            st.plotly_chart(change_bar_fig(tbl), width="stretch")
        else:
            st.info("Every route can keep its normal service.")
    with g2:
        st.subheader("🚦 How busy are the routes?")
        st.plotly_chart(level_donut(tbl), width="stretch")
        st.caption("Compared with each route's own busiest hour. Very busy does not always mean extra buses: "
                   "the normal service already covers the usual rush hour.")

    def view(df):
        out = df.assign(Status=df['level'].map(plain_status))[
            ['Route', 'Borough', 'Expected riders', 'Usual riders', 'Normal buses', 'Buses needed', 'Change', 'Status']]
        return out.rename(columns={'Expected riders': 'Riders expected', 'Usual riders': 'Normally at this hour'})

    t1, t2, t3 = st.tabs(["🔴 Need more buses", "🟢 Can spare buses", "📋 All routes"])
    with t1:
        more = tbl[tbl['Change'] > 0].sort_values('Change', ascending=False)
        if len(more):
            st.dataframe(view(more.head(15)), width="stretch", hide_index=True)
        else:
            st.info("No route needs extra buses.")
    with t2:
        sp = tbl[tbl['Change'] < 0].sort_values('Change')
        if len(sp):
            st.dataframe(view(sp.head(15)), width="stretch", hide_index=True)
        else:
            st.info("No route can spare buses.")
    with t3:
        st.dataframe(view(tbl), width="stretch", hide_index=True)
    plan = view(tbl)
    st.download_button("⬇️ Download the bus plan (CSV)", plan.to_csv(index=False).encode("utf-8"),
                       file_name=f"bus_plan_{d.isoformat()}_{hour:02d}h.csv", mime="text/csv")
    st.caption("Normal buses = buses needed for the usual riders at this hour. Buses needed = buses for the forecast. "
               "Both use the riders-per-bus setting above.")


# ---------------- Insights ----------------
def page_insights():
    st.markdown(pagehead_html("📊 Insights", "When people ride, which routes are busiest, and how good our forecasts are"),
                unsafe_allow_html=True)
    top = pd.Series(info['top_routes']).sort_values(ascending=False)
    dow = pd.Series(info['dow_avg']).sort_index()
    hi, lo = int(_hourly.idxmax()), int(_hourly.idxmin())
    st.markdown(kpi_grid_html([
        ("🕗 Busiest time of day", fmt_hour(hi), "on a typical day", CORAL),
        ("😴 Quietest time", fmt_hour(lo), "fewest riders", TEAL),
        ("🚍 Busiest route", str(top.index[0]), "most riders over the year", INDIGO),
        ("📅 Busiest day", DAY_NAMES[int(dow.idxmax())], "on average", AMBER),
    ]), unsafe_allow_html=True)

    st.subheader("⏰ When do people ride?")
    fig1 = style_fig(px.line(x=[fmt_hour(h) for h in _hourly.index], y=_hourly.values, markers=True,
                             labels={'x': 'Time of day', 'y': 'Average riders per route'}))
    st.plotly_chart(fig1, width="stretch")
    st.caption(f"Buses are fullest around {fmt_hour(hi)} and emptiest around {fmt_hour(lo)}.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📅 Which days are busiest?")
        fig2 = px.bar(x=[DAY_NAMES[int(i)][:3] for i in dow.index], y=dow.values,
                      labels={'x': '', 'y': 'Average riders per route'})
        st.plotly_chart(style_fig(fig2, "bar"), width="stretch")
    with c2:
        st.subheader("🚍 Which routes are busiest?")
        fig3 = px.bar(x=top.head(10).values, y=[str(r) for r in top.head(10).index], orientation='h',
                      labels={'x': 'Total riders in the year', 'y': ''})
        fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(style_fig(fig3, "bar"), width="stretch")
        st.caption("A + after a route (like M15+) means Select Bus Service, the faster limited-stop version.")

    if RESULTS:
        st.subheader("🎯 How good are our forecasts?")
        res = pd.DataFrame(RESULTS).T.sort_values("MAE")
        show = res.iloc[:-1] if res["MAE"].iloc[-1] > 3 * res["MAE"].iloc[0] else res
        vals = [float(v) for v in show["MAE"].values]
        names = [PLAIN_METHOD.get(n, n) for n in show.index]
        colors = [TEAL] + [("#9AA3B5" if str(n).startswith("Baseline") else INDIGO) for n in list(show.index)[1:]]
        fig4 = go.Figure(go.Bar(x=vals[::-1], y=names[::-1], orientation="h", marker_color=colors[::-1],
                                text=[f"off by {v:.1f}" for v in vals[::-1]], textposition="outside"))
        fig4.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Average mistake, in riders (shorter bar is better)",
                           height=60 + 46 * len(vals))
        fig4.update_xaxes(range=[0, max(vals) * 1.3], gridcolor="rgba(127,127,127,.2)")
        st.plotly_chart(fig4, width="stretch")
        st.caption(f"The best method misses by about {vals[0]:.0f} riders on average for one route in one hour. "
                   f"Methods that use last week's riders do best, because last week is the strongest clue. "
                   f"Grey bars are simple rules."
                   + (f" Guessing the same number every time misses by {res['MAE'].iloc[-1]:.0f} (left out of the chart)."
                      if len(show) < len(res) else ""))

    if RECENT is not None:
        st.subheader("🔍 Check our forecast against a real day")
        start = (RECENT.index.min() + pd.Timedelta(days=8)).date()
        end = RECENT.index.max().date()
        if 'chk_date' not in st.session_state:
            st.session_state['chk_date'] = min(max(date(2025, 2, 12), start), end)
        k1, k2 = st.columns(2)
        r = k1.selectbox("Route", ROUTES, key="chk_route", index=ROUTES.index(TOP_ROUTE) if TOP_ROUTE in ROUTES else 0)
        dd = k2.date_input("Date (saved real data)", min_value=start, max_value=end, key="chk_date")
        act = actual_day(r, dd)
        if act is None:
            st.info("Real riders are not saved for this date. Pick another date.")
        else:
            fc = np.asarray(day_curve(r, dd.isoformat()), dtype=float)
            miss = float(np.abs(fc - act).mean())
            tot_f, tot_a = float(fc.sum()), float(act.sum())
            st.markdown(kpi_grid_html([
                ("Real riders in the day", f"{int(round(tot_a)):,}", "from the saved 2025 data", TEAL),
                ("Our forecast for the day", f"{int(round(tot_f)):,}", f"{abs(tot_f - tot_a) / max(tot_a, 1) * 100:.0f}% away", INDIGO),
                ("Average miss per hour", f"{miss:.0f} riders", "forecast vs real", AMBER),
            ]), unsafe_allow_html=True)
            st.plotly_chart(compare_day_fig(r, dd), width="stretch")
            st.caption("The forecast uses the real riders from the week before, as it would in real use.")


# ---------------- page switch ----------------
inject_css()
st.markdown(
    '<div class="tc-brand" style="margin-bottom:8px"><div class="tc-brand-logo">🚌</div><div>'
    '<div class="tc-brand-name">Bus Demand Planner</div>'
    '<div class="tc-brand-tag">Know how busy your bus will be</div></div></div>', unsafe_allow_html=True)
if 'page' not in st.session_state:
    st.session_state['page'] = HOME
page = st.radio("Menu", PAGES, horizontal=True, key="page", label_visibility="collapsed")

if page == HOME:
    page_home()
elif page == FORECAST:
    page_forecast()
elif page == PLANNER:
    page_planner()
elif page == INSIGHTS:
    page_insights()
elif page == ASK:
    if 'chat' not in st.session_state:
        st.session_state['chat'] = []
    history = st.session_state['chat']
    if history and st.button("🗑️ New conversation", key="new_chat"):
        st.session_state['chat'] = []
        st.session_state['chat_ctx'] = new_ctx()
        st.rerun()

    typed = st.chat_input(f"Ask about bus demand or the project, e.g. How busy is {TOP_ROUTE} at 8 AM tomorrow?")
    prompt = typed or st.session_state.pop('pending', None)

    if not history and not prompt:
        st.markdown(hero_html(
            "How busy will your bus be?",
            "Ask in plain English. I forecast hourly riders for New York City bus routes from MTA data, and I can "
            "also explain the project from its documents.",
            pills=[f"{info['n_routes']} routes", f"Tested on {info['test_rows']:,} route-hours",
                   (f"Typical error {(DEFAULT_RES or _best)['MAE']:.0f} riders per route-hour"
                    if (DEFAULT_RES or _best) else "Tested on a separate dataset")]),
            unsafe_allow_html=True)
        st.markdown("##### What would you like to know?")
        for row in (STARTERS[:2], STARTERS[2:]):
            cols = st.columns(2)
            for col, q in zip(cols, row):
                if col.button(q, key=f"starter_{q}", width="stretch"):
                    st.session_state['pending'] = q
                    st.rerun()
        st.caption("Or type your own question below. Type **help** to see everything I can do.")
    else:
        st.markdown('<div class="tc-compact">💬 <b>Assistant</b> · forecasts come from the trained model '
                    'and the project documents</div>', unsafe_allow_html=True)

    for i, m in enumerate(history):
        draw_message(m)
        if i == len(history) - 1 and not prompt and m['role'] == 'assistant':
            draw_chips(m.get('suggestions'), f"chip_{i}")

    if prompt:
        history.append({'role': 'user', 'content': prompt})
        draw_message({'role': 'user', 'content': prompt})

        with st.chat_message("assistant", avatar="🚌"):
            with st.spinner("Thinking..."):
                try:
                    result = answer(prompt)
                except Exception as e:           # never show a crash to the user
                    result = reply("Sorry, something went wrong while I was working that out. "
                                   "Please try asking in a different way.", suggestions=STARTERS)
                    print("Chat error:", repr(e))
            if result.get('origin'):
                st.markdown(origin_html(result['origin']), unsafe_allow_html=True)
            st.write_stream(stream_words(result['text']))
            draw_extras(result)

        history.append({'role': 'assistant', 'content': result['text'], 'table': result['table'],
                        'chart': result['chart'], 'card': result['card'], 'origin': result['origin'],
                        'sources': result['sources'], 'suggestions': result['suggestions']})
        draw_chips(result['suggestions'], f"chip_{len(history) - 1}")


elif page == ABOUT:
    st.markdown(pagehead_html("ℹ️ About this project", PROJECT_TITLE), unsafe_allow_html=True)
    st.markdown(glance_html(), unsafe_allow_html=True)

    best = RESULTS.get(LAG_MODEL_NAME, {})
    base = RESULTS.get("Baseline: usual route-day-hour average", {})
    cal = RESULTS.get("Gradient Boosting (calendar only)", {})
    dow_s = pd.Series(info['dow_avg']).sort_index()
    hi_h, lo_h = int(_hourly.idxmax()), int(_hourly.idxmin())
    busiest_day = DAY_NAMES[int(dow_s.idxmax())]
    gain = (100 * (base['MAE'] - best['MAE']) / base['MAE']) if best and base and base.get('MAE') else None
    bt = BACKTEST

    table_md = ""
    if RESULTS:
        res = pd.DataFrame(RESULTS).T.sort_values("MAE").round(2)
        cols = list(res.columns)
        rows = ["| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for name, row in res.iterrows():
            rows.append(f"| {name} | " + " | ".join(str(row[c]) for c in cols) + " |")
        table_md = "\n".join(rows)

    results_text = ""
    if best:
        results_text += (f"The best model is **{LAG_MODEL_NAME}**. On {info['test_rows']:,} route-hours of the separate "
                         f"2025 dataset its average error (MAE) is **{best['MAE']:.1f} riders** per route-hour, "
                         f"RMSE {best['RMSE']:.1f}, R² {best['R2']:.2f} and WAPE {best['WAPE %']:.1f}%. ")
    if gain is not None and gain > 0:
        results_text += (f"This is {gain:.0f}% lower than the usual-average baseline "
                         f"(MAE {base['MAE']:.1f}). ")
    if cal:
        results_text += (f"Without last week's number the calendar-only model has an MAE of {cal['MAE']:.1f}. ")
    _px = RESULTS.get(PROXY_NAME)
    if _px:
        _cmp = ("slightly worse than" if (base and _px['MAE'] > base['MAE']) else "better than") if base else "compared with"
        results_text += (f"When last week's number is not known and the Random Forest is given the route's usual "
                         f"level instead, its MAE is {_px['MAE']:.1f} (WAPE {_px['WAPE %']:.1f}%), which is {_cmp} the "
                         f"usual-average rule" + (f" ({base['MAE']:.1f})" if base else "") + ". "
                         + ("" if USE_ML_DEFAULT else "In that case the app uses the usual riders. "))

    adjust_text = ""
    if USE_PROFILE and not USE_ML_DEFAULT:
        adjust_text = ("When last week's number is not known, the app uses the usual riders for the route, weekday and "
                       "hour, with two improvements: a per-route factor learned from the saved real 2025 riders, and "
                       "the Sunday pattern for US federal holidays on weekdays.")
        if bt:
            adjust_text += (f" In a back-test (factors learned before March 2025, tested on March 2025) the error fell "
                            f"from {bt['base']:.1f} to {bt['adj']:.1f} riders.")
            if bt.get('hol_plain'):
                adjust_text += (f" On weekday holidays in the saved data the Sunday pattern cut the error from "
                                f"{bt['hol_plain']:.0f} to {bt['hol_sunday']:.0f} riders.")
        adjust_text += "\n\n"

    sections = [
        ("1. Introduction",
         "Public transport demand changes with the hour of the day, the day of the week and the season. Planners need "
         "to know in advance when a route will be busy, so that buses can be placed where they are needed.\n\n"
         "This project builds a machine-learning system that forecasts hourly bus ridership for New York City bus "
         "routes, and uses the forecasts to suggest how buses could be shared between busy and quiet routes.\n\n"
         "**Objectives**\n"
         "- Forecast the number of riders for any route, date and hour.\n"
         "- Compare machine-learning models with simple baseline rules.\n"
         "- Estimate the buses needed and suggest where buses can be moved.\n"
         "- Present everything in an easy web application."),
        ("2. Data",
         "The data comes from the MTA (New York) through the Open Data NY portal (data.ny.gov). It gives hourly "
         "ridership for every bus route.\n\n"
         f"- **Training data:** MTA Bus Hourly Ridership 2020-2024 (dataset id kv7t-n8in), using {info['train_period']}.\n"
         f"- **Testing data:** MTA Bus Hourly Ridership, beginning 2025 (dataset id gxb3-akrn), using {info['test_period']}.\n"
         f"- **Routes:** {info['n_routes']} routes appear in both datasets and are used.\n"
         f"- **Test size:** {info['test_rows']:,} route-hours.\n\n"
         "Ridership is the number of riders who entered a bus with OMNY or MetroCard in that hour. It is a fare "
         "payment count, not a passenger-counter measurement.\n\n"
         "**Cleaning:** timestamps were converted to date-times, ridership was converted to numbers and negative "
         "values were set to 0, the data was reduced to one row per route and hour, and only routes present in both "
         "datasets were kept."),
        ("3. Methodology",
         "**Features:** route code, the route's usual average riders (from the training year only), hour, day of the "
         "week, month, weekend flag, US federal holiday flag, and optionally the riders on the same route and hour "
         "one week earlier (lag_168). "
         + ("When last week's number is not known, the model is given the route's usual last-week level.\n\n"
            if USE_ML_DEFAULT else "\n\n")
         + adjust_text
         + (f"For dates from {recent_range_text()}, the real riders from one week earlier are saved in the app, so "
            "the Random Forest model uses them.\n\n" if recent_range_text() else "") +
         "**Baselines:** always guess the average, repeat last week's same hour, and use the usual average for the "
         "route, weekday and hour.\n\n"
         "**Models:** Linear Regression, Random Forest, and Gradient Boosting (with and without last week's number).\n\n"
         "**Fair testing:** the models learn only from the training year. They are tested on a separate 2025 dataset "
         "that was never used for training, and no feature uses 2025 data. The recent-level factors above use the "
         "saved 2025 riders only inside the app, after the test."),
        ("4. Results",
         results_text + "\n\n" + table_md +
         "\n\nLower MAE, RMSE and WAPE are better. R² is high partly because routes differ a lot in size, so WAPE "
         "is the fairer score."),
        ("5. Demand patterns",
         f"- Busiest hour of the day: **{fmt_hour(hi_h)}** (about {_hourly.max():.0f} riders per route-hour).\n"
         f"- Quietest hour: **{fmt_hour(lo_h)}** (about {_hourly.min():.0f} riders per route-hour).\n"
         f"- Busiest weekday: **{busiest_day}**.\n"
         f"- Route with the most riders in the training year: **{TOP_ROUTE}**."),
        ("6. Resource allocation",
         "For a chosen date and hour, the system forecasts every route and works out two numbers:\n\n"
         "- **Buses needed** = forecast riders ÷ riders one bus can carry per hour.\n"
         "- **Buses normally run** = the same calculation for the route's usual riders at that hour.\n\n"
         "The difference is how many buses to add (+) or spare (−). Routes that need more buses are paired with "
         "routes that can spare some, same borough first (borough is read from the route name: M, Bx, B, Q, S, or "
         "express routes). The system also labels each route Quiet, Fairly busy or Very busy compared with its own "
         f"busiest hour ({int(LOW_RATIO * 100)}% and {int(HIGH_RATIO * 100)}% limits). A route can be very busy at its usual "
         "rush hour without needing extra buses, because its normal service already covers that.\n\n"
         "The data has no bus capacity or timetable, so riders per bus per hour is an assumption that can be changed "
         "on the Bus Planner page. Moves are only suggestions. A planner must check depots, vehicles, drivers and "
         "schedules."),
        ("7. Limitations",
         "- The model learned from one year (2024) and was tested on January to March 2025 only.\n"
         "- Ridership counts fare payments, not passenger counters.\n"
         "- Only US federal holidays are used. Weather, events and service changes are not in the data.\n"
         "- Forecasts are per route, not per stop or vehicle.\n"
         "- The bus capacity and the normal bus numbers are assumptions, not real timetable data.\n"
         "- Ridership may have changed since 2025."),
        ("8. Future work",
         "- Train on several years and retrain with 2025 data.\n"
         "- Add weather, events and more holidays.\n"
         "- Add real bus counts and an optimization method for bus allocation.\n"
         "- Use stop-level data and put the app online."),
        ("9. Conclusion",
         "The project shows that hourly bus demand can be forecast from past patterns and the calendar, and that "
         "the forecasts can support simple, practical decisions about where buses are needed. "
         + ("Using last week's riders makes the forecast more accurate than the simple rules. " if best else "")
         + "The results are estimates, so planners should use them as guidance, not as exact counts."),
    ]

    for title, body in sections:
        st.markdown(f"### {title}")
        st.markdown(body)

    report_md = (f"# {PROJECT_TITLE}\n\n"
                 + "\n\n".join(f"## {t}\n\n{b}" for t, b in sections))
    st.divider()
    st.download_button("⬇️ Download report (Markdown)", report_md, file_name="project_report.md",
                       mime="text/markdown")

    if LOAD_PROBLEMS:
        with st.expander("Technical notes"):
            st.caption("Some model files could not be loaded (probably a scikit-learn version difference). "
                       "The app still works.")
            for msg in LOAD_PROBLEMS:
                st.caption(msg)
