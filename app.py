import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Walmart VoC Insight Engine", layout="wide")

# Download NLTK resources silently
nltk.download('vader_lexicon', quiet=True)

# --- 2. THE AUTOMATION LOGIC (The "Backend") ---
def analyze_data(df):
    """
    This function automates the entire analyst workflow:
    Cleaning -> Sentiment Scoring -> Logic Tagging
    """
    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()
    
    # Clean Data
    if 'Review Text' in df.columns:
        df = df.rename(columns={'Review Text': 'review_text', 'Department Name': 'department_name'})
    
    df = df.dropna(subset=['review_text'])
    
    # 1. AI Scoring
    df['sentiment_score'] = df['review_text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    
    # 2. Categorization Logic
    def get_sentiment_label(score):
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        else: return 'Neutral'
    
    df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)
    
    # 3. AUTOMATED ISSUE TAGGING (The "Walmart Value Add")
    # This simulates an AI detecting specific business problems
    def tag_issue(text):
        text = str(text).lower()
        if 'size' in text or 'fit' in text or 'small' in text or 'large' in text:
            return 'Sizing Issue'
        elif 'fabric' in text or 'quality' in text or 'tear' in text:
            return 'Quality Issue'
        elif 'shipping' in text or 'delivery' in text or 'late' in text:
            return 'Supply Chain'
        elif 'price' in text or 'expensive' in text:
            return 'Pricing'
        else:
            return 'General'
            
    df['issue_tag'] = df['review_text'].apply(tag_issue)
    
    return df

# --- 3. THE USER INTERFACE (The "Frontend") ---
st.title("🛒 Walmart Voice of Customer (VoC) Engine")
st.markdown("""
**Automated Analyst Tool:** Upload raw review data to instantly identify 
merchandising risks and supply chain pain points.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload Monthly Review CSV", type=['csv'])

if uploaded_file is not None:
    # A. Load and Process
    with st.spinner('Running AI Analysis...'):
        raw_df = pd.read_csv(uploaded_file)
        processed_df = analyze_data(raw_df)
    
    st.success("Analysis Complete!")
    
    # B. KPI Row
    col1, col2, col3 = st.columns(3)
    
    total_reviews = len(processed_df)
    avg_sentiment = processed_df['sentiment_score'].mean()
    negative_count = len(processed_df[processed_df['sentiment_label'] == 'Negative'])
    
    col1.metric("Total Feedback Volume", f"{total_reviews:,}")
    col2.metric("Avg Customer Sentiment", f"{avg_sentiment:.2f}")
    col3.metric("Urgent Negative Reviews", f"{negative_count}", delta_color="inverse")
    
    st.divider()

    # C. Interactive Charts (Plotly)
    
    # Chart 1: Problems by Department (The "Where?")
    st.subheader("⚠️ Risk Analysis by Department")
    dept_risk = processed_df[processed_df['sentiment_label'] == 'Negative'].groupby('department_name').size().reset_index(name='count')
    
    fig_bar = px.bar(dept_risk, x='department_name', y='count', 
                     title="Negative Feedback Volume by Department",
                     color='count', color_continuous_scale='Reds')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Chart 2: The "Why?" (Automated Tags)
    st.subheader("🔍 Root Cause Identification")
    # Filter for negative reviews only for this chart
    neg_reviews = processed_df[processed_df['sentiment_label'] == 'Negative']
    fig_pie = px.pie(neg_reviews, names='issue_tag', title="Drivers of Negative Sentiment")
    st.plotly_chart(fig_pie, use_container_width=True)

    # D. The "Action List" (Drill Down)
    st.subheader("📋 Urgent Action List")
    st.markdown("Reviews flagged as **Negative** with **Quality** or **Sizing** issues.")
    
    # Filter for the table
    urgent_table = processed_df[
        (processed_df['sentiment_label'] == 'Negative')
    ][['department_name', 'issue_tag', 'review_text', 'sentiment_score']]
    
    st.dataframe(urgent_table, use_container_width=True)
    
    # E. Download Button (The Automation)
    st.download_button(
        label="📥 Download Processed Report (CSV)",
        data=processed_df.to_csv(index=False).encode('utf-8'),
        file_name='walmart_voc_analyzed.csv',
        mime='text/csv',
    )