import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar

# Set page configuration
st.set_page_config(
    page_title="Financial Transaction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_data():
    # Load JSON data
    with open('categorized_transactions.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Clean data
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d %b %y')
    
    # Extract transaction month and day of week
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()
    df['day'] = df['date'].dt.day
    
    # Convert debit and credit to numeric values
    # Use raw strings for regex to avoid escape sequence warnings
    df['debit'] = df['debit'].replace(r'[\$,]', '', regex=True).replace('', '0').astype(float)
    df['credit'] = df['credit'].replace(r'[\$,]', '', regex=True).replace('', '0').astype(float)
    
    # Create a unified amount column (positive for credits, negative for debits)
    df['amount'] = df['credit'] - df['debit']
    df['abs_amount'] = abs(df['amount'])
    
    # Extract transaction type
    df['transaction_type'] = df.apply(
        lambda row: 'Credit' if row['credit'] > 0 else 'Debit', axis=1
    )
    
    # Clean balance column
    df['balance'] = df['balance'].replace(r'[\$, CR]', '', regex=True).astype(float)
    
    return df

# Load data
df = load_data()

# Title and overview
st.markdown("<h1 class='main-header'>💰 Financial Transaction Dashboard</h1>", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("📊 Filters")

# Date range filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Category filter
all_categories = sorted(df['category'].unique())
selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=all_categories,
    default=all_categories
)

# Transaction type filter
transaction_types = st.sidebar.multiselect(
    "Transaction Type",
    options=['Debit', 'Credit'],
    default=['Debit', 'Credit']
)

# Confidence level filter
confidence_levels = st.sidebar.multiselect(
    "Category Confidence",
    options=sorted(df['category_confidence'].unique()),
    default=sorted(df['category_confidence'].unique())
)

# Apply filters
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[
        (df['date'].dt.date >= start_date) & 
        (df['date'].dt.date <= end_date)
    ]
else:
    filtered_df = df.copy()

filtered_df = filtered_df[
    filtered_df['category'].isin(selected_categories) &
    filtered_df['transaction_type'].isin(transaction_types) &
    filtered_df['category_confidence'].isin(confidence_levels)
]

# Display key metrics
st.markdown("<h2 class='sub-header'>📈 Key Financial Metrics</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_debits = filtered_df[filtered_df['debit'] > 0]['debit'].sum()
    st.metric("Total Debits", f"${total_debits:,.2f}")

with col2:
    total_credits = filtered_df[filtered_df['credit'] > 0]['credit'].sum()
    st.metric("Total Credits", f"${total_credits:,.2f}")

with col3:
    net_flow = total_credits - total_debits
    st.metric("Net Cash Flow", f"${net_flow:,.2f}")

with col4:
    num_transactions = len(filtered_df)
    st.metric("Total Transactions", f"{num_transactions:,}")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "💰 Spending Analysis", 
    "📅 Time Analysis", 
    "🏷️ Category Breakdown",
    "📋 Transaction Details"
])

with tab1:
    st.markdown("<h3 class='sub-header'>Financial Overview</h3>", unsafe_allow_html=True)
    
    # Balance over time
    fig = px.line(
        filtered_df.sort_values('date'),
        x='date',
        y='balance',
        title='Account Balance Over Time',
        markers=True
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Balance ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True, width='stretch')
    
    # Daily transaction volume
    daily_counts = filtered_df.groupby(filtered_df['date'].dt.date).size().reset_index()
    daily_counts.columns = ['date', 'count']
    
    fig2 = px.bar(
        daily_counts,
        x='date',
        y='count',
        title='Daily Transaction Count',
        color='count',
        color_continuous_scale='blues'
    )
    fig2.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Transactions"
    )
    st.plotly_chart(fig2, use_container_width=True, width='stretch')

with tab2:
    st.markdown("<h3 class='sub-header'>Spending Analysis</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top spending categories
        category_spending = filtered_df[filtered_df['debit'] > 0].groupby('category', observed=True)['debit'].sum().reset_index()
        category_spending = category_spending.sort_values('debit', ascending=False)
        
        fig = px.pie(
            category_spending,
            values='debit',
            names='category',
            title='Spending by Category',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True, width='stretch')
    
    with col2:
        # Top transactions by amount
        top_debits = filtered_df[filtered_df['debit'] > 0].nlargest(10, 'debit')[['date', 'description/particulars', 'debit', 'category']]
        
        fig = px.bar(
            top_debits,
            x='debit',
            y='description/particulars',
            orientation='h',
            title='Top 10 Largest Debits',
            color='category',
            text='debit'
        )
        fig.update_layout(
            xaxis_title="Amount ($)",
            yaxis_title="Description",
            yaxis={'categoryorder': 'total ascending'}
        )
        fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True, width='stretch')
    
    # Monthly spending trend
    monthly_spending = filtered_df[filtered_df['debit'] > 0].copy()
    monthly_spending['month_year'] = monthly_spending['date'].dt.strftime('%b %Y')
    monthly_totals = monthly_spending.groupby('month_year', observed=True)['debit'].sum().reset_index()
    
    # Sort by date properly
    monthly_totals['sort_date'] = pd.to_datetime(monthly_totals['month_year'], format='%b %Y')
    monthly_totals = monthly_totals.sort_values('sort_date')
    
    fig = px.line(
        monthly_totals,
        x='month_year',
        y='debit',
        title='Monthly Spending Trend',
        markers=True
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Total Spending ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True, width='stretch')

with tab3:
    st.markdown("<h3 class='sub-header'>Time-Based Analysis</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Spending by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_spending = filtered_df[filtered_df['debit'] > 0].copy()
        day_spending['day_of_week'] = pd.Categorical(day_spending['day_of_week'], categories=day_order, ordered=True)
        day_totals = day_spending.groupby('day_of_week', observed=True)['debit'].sum().reset_index()
        
        fig = px.bar(
            day_totals,
            x='day_of_week',
            y='debit',
            title='Spending by Day of Week',
            color='debit',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Total Spending ($)"
        )
        st.plotly_chart(fig, use_container_width=True, width='stretch')
    
    with col2:
        # Spending by day of month
        day_of_month_spending = filtered_df[filtered_df['debit'] > 0].copy()
        day_of_month_totals = day_of_month_spending.groupby('day', observed=True)['debit'].sum().reset_index()
        
        fig = px.line(
            day_of_month_totals,
            x='day',
            y='debit',
            title='Spending by Day of Month',
            markers=True
        )
        fig.update_layout(
            xaxis_title="Day of Month",
            yaxis_title="Total Spending ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True, width='stretch')
    
    # Heatmap of spending by category and day of week
    heatmap_data = filtered_df[filtered_df['debit'] > 0].copy()
    heatmap_data['day_of_week'] = pd.Categorical(heatmap_data['day_of_week'], categories=day_order, ordered=True)
    
    pivot_table = pd.pivot_table(
        heatmap_data,
        values='debit',
        index='category',
        columns='day_of_week',
        aggfunc='sum',
        fill_value=0
    )
    
    fig = px.imshow(
        pivot_table,
        title='Spending Heatmap: Category vs Day of Week',
        color_continuous_scale='YlOrRd',
        labels=dict(x="Day of Week", y="Category", color="Spending ($)")
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, width='stretch')

with tab4:
    st.markdown("<h3 class='sub-header'>Category Analysis</h3>", unsafe_allow_html=True)
    
    # Select a category to analyze
    selected_category = st.selectbox(
        "Select a category to analyze in detail:",
        options=sorted(filtered_df['category'].unique())
    )
    
    category_data = filtered_df[filtered_df['category'] == selected_category]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category transaction count
        fig = px.histogram(
            category_data,
            x='date',
            title=f'Transaction Count for {selected_category}',
            nbins=20,
            color_discrete_sequence=['#3B82F6']
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Transactions"
        )
        st.plotly_chart(fig, use_container_width=True, width='stretch')
    
    with col2:
        # Category amount distribution
        if len(category_data) > 0:
            fig = px.box(
                category_data,
                y='abs_amount',
                title=f'Amount Distribution for {selected_category}',
                points='all'
            )
            fig.update_layout(
                yaxis_title="Amount ($)"
            )
            st.plotly_chart(fig, use_container_width=True, width='stretch')
    
    # Confidence level distribution for selected category
    if len(category_data) > 0:
        confidence_counts = category_data['category_confidence'].value_counts().reset_index()
        confidence_counts.columns = ['confidence', 'count']
        
        fig = px.pie(
            confidence_counts,
            values='count',
            names='confidence',
            title=f'Confidence Level Distribution for {selected_category}',
            hole=0.3,
            color='confidence',
            color_discrete_map={'high': '#10B981', 'medium': '#F59E0B', 'low': '#EF4444'}
        )
        st.plotly_chart(fig, use_container_width=True, width='stretch')

with tab5:
    st.markdown("<h3 class='sub-header'>Transaction Details</h3>", unsafe_allow_html=True)
    
    # Search and filter transactions
    search_term = st.text_input("Search in transaction descriptions:", "")
    
    if search_term:
        display_df = filtered_df[filtered_df['description/particulars'].str.contains(search_term, case=False, na=False)]
    else:
        display_df = filtered_df
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by:",
        options=['date', 'debit', 'credit', 'category', 'balance'],
        index=0
    )
    
    sort_order = st.radio("Sort order:", ["Ascending", "Descending"], horizontal=True)
    
    if sort_order == "Descending":
        display_df = display_df.sort_values(sort_by, ascending=False)
    else:
        display_df = display_df.sort_values(sort_by, ascending=True)
    
    # Format columns for display
    display_df_display = display_df.copy()
    display_df_display['date'] = display_df_display['date'].dt.strftime('%d %b %Y')
    display_df_display['debit'] = display_df_display['debit'].apply(lambda x: f"${x:,.2f}" if x > 0 else "")
    display_df_display['credit'] = display_df_display['credit'].apply(lambda x: f"${x:,.2f}" if x > 0 else "")
    display_df_display['balance'] = display_df_display['balance'].apply(lambda x: f"${x:,.2f}")
    
    # Select columns to display
    columns_to_display = ['date', 'description/particulars', 'debit', 'credit', 'balance', 'category', 'category_confidence']
    display_df_display = display_df_display[columns_to_display]
    
    # Display the dataframe
    st.dataframe(
        display_df_display,
        use_container_width=True,
        height=400
    )
    
    # Download option
    csv = display_df[columns_to_display].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_transactions.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280;'>
        <p>Financial Transaction Dashboard • Data from categorized_transactions.json</p>
        <p>Visualizations created with Plotly and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
