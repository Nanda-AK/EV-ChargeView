# Predefine Function working, but user query fails with list issue. 
# EV Review Streamlit App with GPT-4o and 10 Predefined Analysis Functions

import streamlit as st
import pandas as pd
import json
import openai
from datetime import datetime
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# --- CONFIG ---
openai.api_key = st.secrets["OpenAI_API_KEY"]  # Store key in Streamlit secrets
llm = OpenAI(api_token=openai.api_key, model="gpt-3.5-turbo")

# --- LOAD JSON DATA ---
@st.cache_data
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.json_normalize(data)

raw_df = load_data("cleaned_ev_data.json")

# --- FILTER INCOMPLETE RECORDS (no drops, just filter) ---
raw_df = raw_df[
    raw_df['EV Vendor'].notna() &
    raw_df['address'].notna() &
    raw_df['reviewsCount'].notna() &
    raw_df['totalScore'].notna()
]

# --- CREATE SYNTHETIC STATION ID ---
raw_df['station_id'] = raw_df['EV Vendor'] + " - " + raw_df['address']

# --- INIT LLM-AWARE DATAFRAME ---
df = SmartDataframe(raw_df, config={"llm": llm})

# --- APP LAYOUT ---
st.set_page_config(page_title="EV Review Insights", layout="wide")
st.title("🔌 EV Charging Station Review Explorer")

# --- SIDEBAR INPUT ---
st.sidebar.header("Ask a Question or Choose Analysis")
user_query = st.sidebar.text_input("Ad hoc Query (e.g., stations with long wait time)")

# --- PREDEFINED FUNCTIONS ---
def get_top_stations():
    return raw_df.groupby('station_id')['reviewsCount'].sum().sort_values(ascending=False).head(10)

def get_worst_stations():
    return raw_df.groupby('station_id')['totalScore'].mean().sort_values().head(10)

def average_rating_by_vendor():
    return raw_df.groupby('EV Vendor')['totalScore'].mean().sort_values(ascending=False)

def review_distribution():
    col_name = 'reviewsDistribution.1'
    if col_name in raw_df.columns:
        return raw_df[col_name].value_counts().sort_index()
    else:
        return pd.Series(dtype=int)

def wait_time_mentions():
    return raw_df['reviews'].apply(lambda x: 'wait' in str(x).lower()).value_counts()

def stations_with_long_wait():
    return raw_df[raw_df['reviews'].apply(lambda x: 'long wait' in str(x).lower())]

def top_complaints():
    return df.chat("What are the top complaints across all reviews?")

def summarize_by_city():
    return df.chat("Summarize the best and worst EV stations by city.")

def vendor_sentiment():
    return df.chat("Compare sentiment for each vendor across all reviews.")

def peak_occupancy_analysis():
    return df.chat("What are the busiest times of day across EV stations?")

# --- RENDER PREDEFINED INSIGHTS ---
st.subheader("📊 Key Insights")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top 10 Stations by Review Volume**")
    st.dataframe(get_top_stations())

    st.markdown("**Worst 10 Stations by Avg Rating**")
    st.dataframe(get_worst_stations())

    st.markdown("**Avg Rating by Vendor**")
    st.dataframe(average_rating_by_vendor())

with col2:
    st.markdown("**Review Distribution (1-star only)**")
    dist_series = review_distribution()
    if not dist_series.empty:
        st.bar_chart(dist_series)
    else:
        st.info("No review distribution data available.")

    st.markdown("**Mentions of Wait Time**")
    st.dataframe(wait_time_mentions())

    st.markdown("**Stations with 'Long Wait' in Reviews**")
    st.dataframe(stations_with_long_wait()[['station_id', 'address']].drop_duplicates())

# --- USER QUERY TO GPT-4o ---
if user_query:
    st.subheader("🤖 LLM Answer")
    try:
        answer = df.chat(user_query)
        st.write(answer)
    except Exception as e:
        st.error(f"Error: {e}")

# --- BONUS INSIGHTS ---
st.subheader("✨ Bonus LLM Insights")
st.write("Top complaints across stations:")
st.write(top_complaints())

st.write("City-based Summary:")
st.write(summarize_by_city())

st.write("Vendor Sentiment Summary:")
st.write(vendor_sentiment())

st.write("Occupancy Patterns:")
st.write(peak_occupancy_analysis())
