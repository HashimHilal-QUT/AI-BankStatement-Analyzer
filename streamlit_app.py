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
import io

# Set page configuration
st.set_page_config(
    page_title="AI Analyzer - Bank Statement",
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
    .upload-section {
        background-color: #F0F9FF;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border: 2px dashed #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Function to process data
def process_data(df):
    """Process and clean the transaction dataframe"""
    if df.empty:
        return df
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        try:
            df['date'] = pd.to_datetime(df['date'], format='%d %b %y')
        except:
            # Try other common date formats
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract transaction month and day of week
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()
    df['day'] = df['date'].dt.day
    
    # Convert debit and credit to numeric values
    # Use raw strings for regex to avoid escape sequence warnings
    if 'debit' in df.columns:
        df['debit'] = df['debit'].replace(r'[\$,]', '', regex=True).replace('', '0').astype(float)
    else:
        df['debit'] = 0.0
    
    if 'credit' in df.columns:
        df['credit'] = df['credit'].replace(r'[\$,]', '', regex=True).replace('', '0').astype(float)
    else:
        df['credit'] = 0.0
    
    # Create a unified amount column (positive for credits, negative for debits)
    df['amount'] = df['credit'] - df['debit']
    df['abs_amount'] = abs(df['amount'])
    
    # Extract transaction type
    df['transaction_type'] = df.apply(
        lambda row: 'Credit' if row['credit'] > 0 else 'Debit', axis=1
    )
    
    # Clean balance column if it exists
    if 'balance' in df.columns:
        df['balance'] = df['balance'].replace(r'[\$, CR]', '', regex=True).astype(float)
    else:
        # Create a simulated balance if not present
        df['balance'] = df['amount'].cumsum()
    
    # Check if category columns exist, create default if not
    if 'category' not in df.columns:
        df['category'] = 'Uncategorized'
    
    if 'category_confidence' not in df.columns:
        df['category_confidence'] = 'medium'
    
    return df

# Title and overview
st.markdown("<h1 class='main-header'>💰 AI Analyzer - Bank Statement</h1>", unsafe_allow_html=True)

# Sidebar for file upload and filters
st.sidebar.header("📁 Data Upload")

# File upload section
st.sidebar.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader(
    "Upload your transaction data (JSON format)",
    type=['json'],
    help="Upload a JSON file with transaction data in the format provided"
)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Check if file is uploaded
if uploaded_file is not None:
    try:
        # Read the uploaded file
        data = json.load(uploaded_file)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Process the data
        df = process_data(df)
        
        # Show success message
        st.sidebar.success(f"✅ Successfully loaded {len(df)} transactions")
        
        # Show data preview
        with st.sidebar.expander("📊 Data Preview", expanded=False):
            st.write(f"**Total Rows:** {len(df)}")
            st.write(f"**Date Range:** {df['date'].min().date()} to {df['date'].max().date()}")
            st.write(f"**Categories:** {len(df['category'].unique())}")
            st.dataframe(df.head(3), use_container_width=True, height=150)
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {str(e)}")
        st.error(f"Failed to process the uploaded file. Please check the format.")
        st.info("""
        **Expected JSON format:**
        ```json
        [
            {
                "date": "28 Nov 25",
                "description/particulars": "Transaction description",
                "debit": "$4.60",
                "credit": "",
                "balance": "$3,214.11 CR",
                "category": "Food & Dining",
                "category_confidence": "high"
            }
        ]
        ```
        """)
        df = pd.DataFrame()  # Empty dataframe
else:
    # Default to sample data if no file uploaded
    st.sidebar.info("📝 No file uploaded. Using sample data.")
    
    # Load default sample data from the provided JSON
    try:
        sample_data = [
            {
                "date":"28 Nov 25",
                "description/particulars":"V0103 26/11 COCACOLA EPP MORNINGSID 74564725331",
                "debit":"$4.60",
                "credit":"",
                "balance":"$3,214.11 CR",
                "category":"Food & Dining",
                "category_confidence":"high"
            },
            {
                "date":"01 Dec 25",
                "description/particulars":"V0103 26/11 MORNINGSIDE STATION MORNINGSID 74940525331",
                "debit":"$5.00",
                "credit":"",
                "balance":"$8,457.39 CR",
                "category":"Food & Dining",
                "category_confidence":"medium"
            },
            {
                "date":"01 Dec 25",
                "description/particulars":"HASHIM HILAL DECEMBER",
                "debit":"",
                "credit":"$4,657.43",
                "balance":"$7,625.44 CR",
                "category":"Income & Salary",
                "category_confidence":"high"
            }
        ]
        df = pd.DataFrame(sample_data)
        df = process_data(df)
        st.sidebar.warning("⚠️ Using sample data. Upload your own file for full analysis.")
    except:
        df = pd.DataFrame()

# Only show filters and visualizations if we have data
if not df.empty:
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
        default=all_categories[:min(5, len(all_categories))]  # Default to first 5 categories
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
            
            if not top_debits.empty:
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
            else:
                st.info("No debit transactions found in the filtered data.")
        
        # Monthly spending trend
        monthly_spending = filtered_df[filtered_df['debit'] > 0].copy()
        if not monthly_spending.empty:
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
        else:
            st.info("No debit transactions available for monthly trend analysis.")
    
    with tab3:
        st.markdown("<h3 class='sub-header'>Time-Based Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Spending by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_spending = filtered_df[filtered_df['debit'] > 0].copy()
            if not day_spending.empty:
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
            else:
                st.info("No debit transactions available for day of week analysis.")
        
        with col2:
            # Spending by day of month
            day_of_month_spending = filtered_df[filtered_df['debit'] > 0].copy()
            if not day_of_month_spending.empty:
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
            else:
                st.info("No debit transactions available for day of month analysis.")
        
        # Heatmap of spending by category and day of week
        heatmap_data = filtered_df[filtered_df['debit'] > 0].copy()
        if not heatmap_data.empty:
            heatmap_data['day_of_week'] = pd.Categorical(heatmap_data['day_of_week'], categories=day_order, ordered=True)
            
            pivot_table = pd.pivot_table(
                heatmap_data,
                values='debit',
                index='category',
                columns='day_of_week',
                aggfunc='sum',
                fill_value=0
            )
            
            if not pivot_table.empty:
                fig = px.imshow(
                    pivot_table,
                    title='Spending Heatmap: Category vs Day of Week',
                    color_continuous_scale='YlOrRd',
                    labels=dict(x="Day of Week", y="Category", color="Spending ($)")
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True, width='stretch')
            else:
                st.info("Not enough data to generate heatmap.")
    
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
            if not category_data.empty:
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
            else:
                st.info(f"No transactions found for category: {selected_category}")
        
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
        # Only include columns that exist
        columns_to_display = [col for col in columns_to_display if col in display_df_display.columns]
        display_df_display = display_df_display[columns_to_display]
        
        # Display the dataframe
        if not display_df_display.empty:
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
        else:
            st.info("No transactions match the current filters.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6B7280;'>
            <p>Financial Transaction Dashboard • Upload your own transaction data for analysis</p>
            <p>Visualizations created with Plotly and Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    # Show instructions if no data is loaded
    st.info("👈 Please upload a transaction data file in JSON format using the sidebar uploader.")
    
    # Show example of expected format
    with st.expander("📋 Example of Expected JSON Format", expanded=True):
        st.code("""
[
    {
        "date": "28 Nov 25",
        "description/particulars": "V0103 26/11 COCACOLA EPP MORNINGSID 74564725331",
        "debit": "$4.60",
        "credit": "",
        "balance": "$3,214.11 CR",
        "category": "Food & Dining",
        "category_confidence": "high"
    },
    {
        "date": "01 Dec 25",
        "description/particulars": "V0103 26/11 MORNINGSIDE STATION MORNINGSID 74940525331",
        "debit": "$5.00",
        "credit": "",
        "balance": "$8,457.39 CR",
        "category": "Food & Dining",
        "category_confidence": "medium"
    },
    {
        "date": "01 Dec 25",
        "description/particulars": "HASHIM HILAL DECEMBER",
        "debit": "",
        "credit": "$5,555.55",
        "balance": "$10,000.00 CR",
        "category": "Income & Salary",
        "category_confidence": "high"
    }
]
        """, language="json")
    
    st.markdown("""
    ### 📁 How to Use This Dashboard:
    
    1. **Upload your data**: Use the file uploader in the sidebar to upload your transaction data in JSON format
    2. **Apply filters**: Use the filters in the sidebar to focus on specific dates, categories, or transaction types
    3. **Explore visualizations**: Navigate through the tabs to see different analyses of your financial data
    4. **Download results**: Export filtered data as CSV from the Transaction Details tab
    
    ### 🔧 Features:
    - **Balance tracking**: See how your account balance changes over time
    - **Spending analysis**: Identify where your money is going by category
    - **Time patterns**: Discover spending habits by day of week or month
    - **Interactive charts**: Hover over charts to see detailed information
    - **Data export**: Download filtered transaction data for further analysis
    """)
