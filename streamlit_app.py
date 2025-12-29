# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
import base64
import re
from io import BytesIO
import tempfile
import os
from PIL import Image
from pdf2image import convert_from_path
from openai import OpenAI
from typing import Dict, List
from huggingface_hub import InferenceClient

# Set page configuration
st.set_page_config(
    page_title="AI Bank Statement Analyzer",
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
    .upload-section {
        background-color: #F0F9FF;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border: 2px dashed #3B82F6;
    }
    .status-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
    }
    .status-success {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #10B981;
    }
    .status-warning {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #F59E0B;
    }
    .status-error {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #EF4444;
    }
    .insight-section {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #3B82F6;
    }
    .insight-section-warning {
        border-left: 5px solid #F59E0B;
        background-color: #FFFBEB;
    }
    .insight-section-success {
        border-left: 5px solid #10B981;
        background-color: #ECFDF5;
    }
    .insight-section-info {
        border-left: 5px solid #3B82F6;
        background-color: #EFF6FF;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing data
if 'df_categorized' not in st.session_state:
    st.session_state.df_categorized = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None
if 'ai_processing' not in st.session_state:
    st.session_state.ai_processing = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_json_safely(text):
    """Surgically extracts the first JSON object found in a string."""
    try:
        # Look for the first '{' and the last '}'
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return None
    except json.JSONDecodeError:
        # If it's still broken (truncated), try to close the JSON manually
        try:
            return json.loads(text + ']}')
        except:
            return None

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

# ============================================================================
# PDF PROCESSING FUNCTIONS
# ============================================================================

def parse_full_statement(pdf_path, hf_client):
    """Process PDF and extract transactions using HuggingFace Vision model"""
    try:
        pages = convert_from_path(pdf_path, dpi=250) # Higher DPI for dense text
        full_data = {"bank_name": None, "account_number": None, "all_transactions": []}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, page in enumerate(pages):
            status_text.text(f"Processing page {i+1}/{len(pages)}...")
            progress_bar.progress((i + 1) / len(pages))

            buffered = BytesIO()
            page.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # NUDGE: Explicitly tell the AI to be brief to save tokens
            prompt = """
            Output ONLY raw JSON for the transaction table on this page.
            Do not explain. Do not add intro text.
            JSON structure: {"bank_name": "", "account_number": "", "year": "", "transactions": [{"date": "", "description/particulars": "", "debit": "", "credit": "", "balance": ""}]}
            """

            try:
                response = hf_client.chat_completion(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ],
                    }],
                    max_tokens=3000,
                    temperature=0.1
                )

                raw_content = response.choices[0].message.content
                page_json = extract_json_safely(raw_content)

                if page_json:
                    if page_json.get("bank_name") and not full_data["bank_name"]:
                        full_data["bank_name"] = page_json["bank_name"]
                    full_data["all_transactions"].extend(page_json.get("transactions", []))
                else:
                    st.warning(f"Could not parse JSON on Page {i+1}")

            except Exception as e:
                st.warning(f"Error on page {i+1}: {e}")

        progress_bar.empty()
        status_text.empty()
        
        return full_data
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def process_pdf_to_dataframe(pdf_file):
    """Main function to process PDF and return categorized DataFrame"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            pdf_path = tmp_file.name
        
        # Initialize HuggingFace clients
        HF_TOKEN = st.secrets["HUGGINGFACE_API_KEY"]
        
        # Initialize Vision client for PDF processing
        vision_client = InferenceClient(model="Qwen/Qwen2.5-VL-7B-Instruct", token=HF_TOKEN)
        
        # Step 1: Extract transactions from PDF
        st.info("📄 Extracting transactions from PDF...")
        result = parse_full_statement(pdf_path, vision_client)
        
        if not result or not result["all_transactions"]:
            st.error("No transactions found in the PDF")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(result["all_transactions"])
        st.session_state.df_raw = df.copy()
        
        # Step 2: Categorize transactions
        st.info("🏷️ Categorizing transactions...")
        
        # Initialize categorizer
        config = {
            "base_url": "https://router.huggingface.co/v1",
            "api_key": st.secrets["HUGGINGFACE_API_KEY"],
            "model": "openai/gpt-oss-20b",
            "max_tokens": 16000,
            "timeout": 300
        }
        
        categorizer = HuggingFaceTransactionCategorizer(config)
        
        # Process data
        df_categorized = categorizer.categorize_dataframe(df, batch_size=20)
        
        # Clean up temporary file
        os.unlink(pdf_path)
        
        return df_categorized
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# ============================================================================
# TRANSACTION CATEGORIZER CLASS
# ============================================================================

class HuggingFaceTransactionCategorizer:
    """
    Categorize bank transactions using HuggingFace models via OpenAI-compatible API
    """

    def __init__(self, config: Dict = None):
        """
        Initialize with HuggingFace configuration
        """
        if config is None:
            config = {
                "base_url": "https://router.huggingface.co/v1",
                "api_key": st.secrets["HUGGINGFACE_API_KEY"],
                "model": "openai/gpt-oss-20b",
                "max_tokens": 16000,
                "timeout": 300
            }

        self.config = config
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            timeout=config.get("timeout", 300)
        )
        self.model = config["model"]
        self.max_tokens = config.get("max_tokens", 16000)

    def create_categorization_prompt(self, transactions: List[Dict]) -> str:
        """
        Create a structured prompt for LLM to categorize transactions
        """

        # Convert transactions to simplified format
        simplified_transactions = []
        for idx, trans in enumerate(transactions):
            simplified_transactions.append({
                "id": idx,
                "description": trans.get("description/particulars", "")
            })

        transactions_json = json.dumps(simplified_transactions, indent=2)

        prompt = f"""You are a financial analyst AI that categorizes bank transactions.

TASK: Analyze each transaction description and assign the most appropriate category.

CATEGORIES:
- Income & Salary: Wages, salary payments, bonuses, tax refunds
- Groceries & Supermarkets: Supermarket purchases, grocery stores (Coles, Woolworths, Aldi)
- Food & Dining: Restaurants, cafes, fast food, Coca-Cola, food delivery
- Fuel & Transport: Gas stations, public transport, parking, tolls, ride-sharing
- Shopping & Retail: Clothing, electronics, general retail, online shopping
- Healthcare & Pharmacy: Medical services, pharmacies (Chemmart), health insurance
- Utilities & Bills: Electricity, water, gas, internet, phone bills
- Entertainment & Leisure: Movies, games, hobbies, streaming services
- Travel & Accommodation: Hotels, flights, travel bookings
- Financial Services: Bank fees, insurance, investments, transfers
- Cash Withdrawal: ATM withdrawals
- Personal Care: Hair salons, beauty services, gym memberships
- Education: School fees, courses, books
- Home & Garden: Hardware, furniture, home improvements
- Transfers & Payments: P2P transfers (NPP, PayID), bill payments, BPAY
- Subscriptions: Recurring subscription services
- Charity & Donations: Charitable contributions, mosque donations
- Other: Anything that doesn't fit above categories

TRANSACTIONS:
{transactions_json}

INSTRUCTIONS:
1. Analyze each transaction's description carefully
2. Match merchants/keywords to appropriate categories
3. Be consistent - similar merchants get the same category
4. Return ONLY valid JSON with this exact structure (no markdown, no code blocks, no explanations):

{{"categorized_transactions":[{{"id":0,"category":"Category Name","confidence":"high"}},{{"id":1,"category":"Category Name","confidence":"medium"}}]}}

Return the JSON now:"""

        return prompt

    def call_llm(self, prompt: str) -> str:
        """
        Call the HuggingFace model via OpenAI-compatible API
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst that categorizes bank transactions. Always return valid JSON only, no markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error calling HuggingFace API: {e}")
            raise

    def parse_llm_response(self, response: str) -> List[Dict]:
        """
        Parse and clean LLM response to extract categories
        """
        # Clean response - remove markdown code blocks if present
        response = response.strip()

        # Remove markdown code block syntax
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            parts = response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith('{') or part.startswith('['):
                    response = part
                    break

        response = response.strip()

        # Parse JSON
        try:
            result = json.loads(response)

            # Handle different response formats
            if isinstance(result, dict):
                if "categorized_transactions" in result:
                    return result["categorized_transactions"]
                elif "transactions" in result:
                    return result["transactions"]
                elif "categories" in result:
                    return result["categories"]
            elif isinstance(result, list):
                return result

            return []
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {e}")
            st.error(f"Response was: {response[:500]}")
            return []

    def categorize_transactions(self, transactions: List[Dict], batch_size: int = 20) -> List[Dict]:
        """
        Categorize all transactions using HuggingFace LLM
        """
        all_categorized = []
        total_batches = (len(transactions) + batch_size - 1) // batch_size

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process in batches
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            status_text.text(f"Processing batch {batch_num}/{total_batches} ({len(batch)} transactions)...")
            progress_bar.progress(batch_num / total_batches)

            try:
                # Create prompt
                prompt = self.create_categorization_prompt(batch)

                # Call LLM
                response = self.call_llm(prompt)

                # Parse response
                categorized = self.parse_llm_response(response)

                # Merge categories back into original transactions
                for j, trans in enumerate(batch):
                    if j < len(categorized):
                        cat_data = categorized[j]
                        trans["category"] = cat_data.get("category", "Other")
                        trans["category_confidence"] = cat_data.get("confidence", "medium")
                    else:
                        trans["category"] = "Other"
                        trans["category_confidence"] = "low"

                    all_categorized.append(trans)

            except Exception as e:
                st.warning(f"Error in batch {batch_num}: {e}. Using fallback categorization.")
                # Fallback: assign "Other" to all in batch
                for trans in batch:
                    trans["category"] = "Other"
                    trans["category_confidence"] = "low"
                    all_categorized.append(trans)

        progress_bar.empty()
        status_text.empty()
        
        return all_categorized

    def categorize_dataframe(self, df: pd.DataFrame, batch_size: int = 20) -> pd.DataFrame:
        """
        Categorize transactions in a DataFrame
        """
        # Convert DataFrame to list of dicts
        transactions = df.to_dict('records')

        # Categorize
        categorized_transactions = self.categorize_transactions(transactions, batch_size)

        # Convert back to DataFrame
        result_df = pd.DataFrame(categorized_transactions)

        return result_df

# ============================================================================
# AI FINANCIAL INSIGHTS AND BUDGET ADVISOR CLASS
# ============================================================================

class FinancialInsightsAdvisor:
    """
    Generate personalized financial insights and budget recommendations using LLM
    """

    def __init__(self, config: Dict = None):
        """
        Initialize with HuggingFace configuration
        """
        if config is None:
            config = {
                "base_url": "https://router.huggingface.co/v1",
                "api_key": st.secrets["HUGGINGFACE_API_KEY"],
                "model": "openai/gpt-oss-20b",
                "max_tokens": 8000,
                "timeout": 300
            }

        self.config = config
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            timeout=config.get("timeout", 300)
        )
        self.model = config["model"]
        self.max_tokens = config.get("max_tokens", 8000)

    def prepare_financial_summary(self, df: pd.DataFrame) -> Dict:
        """
        Extract key financial metrics from the DataFrame
        """
        # Ensure numeric columns
        df_clean = df.copy()
        for col in ['debit', 'credit', 'balance']:
            if col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str).str.replace(r'[\$, CR]', '', regex=True).str.replace(',', '')
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

        # Calculate key metrics
        total_income = df_clean['credit'].sum()
        total_expenses = df_clean['debit'].sum()
        net_savings = total_income - total_expenses
        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0

        # Starting and ending balance
        if 'balance' in df_clean.columns:
            start_balance = df_clean.iloc[0]['balance']
            end_balance = df_clean.iloc[-1]['balance']
            balance_change = end_balance - start_balance
        else:
            start_balance = 0
            end_balance = net_savings
            balance_change = net_savings

        # Date range
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        if df_clean['date'].notna().any():
            date_range_days = (df_clean['date'].max() - df_clean['date'].min()).days + 1
            start_date = df_clean['date'].min().strftime('%Y-%m-%d')
            end_date = df_clean['date'].max().strftime('%Y-%m-%d')
        else:
            date_range_days = len(df_clean)
            start_date = "Unknown"
            end_date = "Unknown"

        # Category breakdown (expenses only)
        spending_data = df_clean[df_clean['debit'] > 0].copy()
        if not spending_data.empty:
            category_spending = spending_data.groupby('category')['debit'].agg(['sum', 'count', 'mean']).round(2)
            category_spending.columns = ['total', 'transactions', 'avg_amount']
            category_spending['percentage'] = (category_spending['total'] / total_expenses * 100).round(2)
            category_spending = category_spending.sort_values('total', ascending=False)
            top_categories = category_spending.head(10).to_dict('index')
        else:
            top_categories = {}

        # Daily average spending
        if not spending_data.empty and 'date' in spending_data.columns and spending_data['date'].notna().any():
            daily_spending = spending_data.groupby(spending_data['date'].dt.date)['debit'].sum()
            avg_daily_spending = daily_spending.mean()
            top_spending_days = daily_spending.nlargest(3).to_dict()
        else:
            avg_daily_spending = 0
            top_spending_days = {}

        # Income sources
        income_data = df_clean[df_clean['credit'] > 0].copy()
        income_by_category = income_data.groupby('category')['credit'].sum().to_dict()

        # Transaction patterns
        transaction_count = len(df_clean)
        avg_transaction_size = spending_data['debit'].mean() if not spending_data.empty else 0

        # Day of week patterns
        if not spending_data.empty and 'date' in spending_data.columns and spending_data['date'].notna().any():
            spending_data['day_of_week'] = spending_data['date'].dt.day_name()
            spending_by_dow = spending_data.groupby('day_of_week')['debit'].sum().to_dict()
        else:
            spending_by_dow = {}

        summary = {
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "days": date_range_days
            },
            "income": {
                "total": round(total_income, 2),
                "sources": income_by_category
            },
            "expenses": {
                "total": round(total_expenses, 2),
                "daily_average": round(avg_daily_spending, 2),
                "avg_transaction_size": round(avg_transaction_size, 2)
            },
            "savings": {
                "net_amount": round(net_savings, 2),
                "savings_rate_percent": round(savings_rate, 2)
            },
            "balance": {
                "starting": round(start_balance, 2),
                "ending": round(end_balance, 2),
                "change": round(balance_change, 2)
            },
            "category_breakdown": top_categories,
            "top_spending_days": {str(k): round(v, 2) for k, v in top_spending_days.items()},
            "spending_by_day_of_week": {k: round(v, 2) for k, v in spending_by_dow.items()},
            "transaction_count": transaction_count
        }

        return summary

    def create_insights_prompt(self, financial_summary: Dict) -> str:
            """
            Create a comprehensive prompt for financial insights
            """

            summary_json = json.dumps(financial_summary, indent=2)

            prompt = f"""You are an expert financial advisor analyzing a user's bank statement data. Your role is to provide personalized, actionable financial insights and budget recommendations.

                **FINANCIAL DATA SUMMARY:**
                ```json
                {summary_json}
                ```

                **YOUR TASK:**
                Analyze this financial data comprehensively and provide:

                1. **OVERALL FINANCIAL HEALTH ASSESSMENT**
                - Evaluate their savings rate and financial position
                - Comment on income stability and expense patterns
                - Identify the overall financial health score (1-10)

                2. **SPENDING PATTERN ANALYSIS**
                - Identify top spending categories and their reasonableness
                - Highlight any unusual spending patterns or red flags
                - Compare spending against typical budget guidelines (50/30/20 rule, etc.)
                - Identify categories where spending is above average

                3. **KEY INSIGHTS & OBSERVATIONS**
                - Point out interesting patterns (day of week trends, spending spikes)
                - Identify potential waste or unnecessary expenses
                - Recognize positive financial behaviors
                - Note any concerning trends

                4. **ACTIONABLE RECOMMENDATIONS**
                - Provide 5-7 specific, actionable recommendations to improve their budget
                - Suggest realistic spending targets for each major category
                - Recommend savings goals based on their income
                - Provide tips for reducing expenses in high-spending categories
                - Suggest budget allocation percentages

                5. **BUDGET PLAN**
                - Create a recommended monthly budget breakdown
                - Set spending limits for each category
                - Define a realistic savings target
                - Suggest an emergency fund goal

                6. **QUICK WINS**
                - List 3-5 immediate actions they can take this week
                - Focus on high-impact, easy-to-implement changes

                **TONE & STYLE:**
                - Be encouraging and supportive, not judgmental
                - Use clear, simple language (avoid jargon)
                - Be specific with numbers and percentages
                - Focus on practical, achievable recommendations
                - Celebrate positive behaviors while addressing concerns

                **FORMAT YOUR RESPONSE AS:**
                Use clear headings, bullet points, and structured sections. Include specific dollar amounts and percentages. Make it easy to read and actionable.

                Provide your comprehensive financial analysis and recommendations now:"""

            return prompt

    def get_financial_insights(self, df: pd.DataFrame) -> str:
        """
        Generate comprehensive financial insights using LLM
        """
        try:
            # Prepare financial summary
            financial_summary = self.prepare_financial_summary(df)
            
            # Create prompt
            prompt = self.create_insights_prompt(financial_summary)

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial advisor providing personalized budget insights and recommendations. Be specific, actionable, and encouraging."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.7
            )

            insights = response.choices[0].message.content
            return insights

        except Exception as e:
            st.error(f"Error generating AI insights: {e}")
            return self._generate_fallback_insights(financial_summary)
        

    def _generate_fallback_insights(self, summary: Dict) -> str:
        """
        Generate basic insights if LLM call fails
        """
        insights = f"""
                # FINANCIAL ANALYSIS REPORT

                ## Overall Financial Health
                - Total Income: ${summary['income']['total']:,.2f}
                - Total Expenses: ${summary['expenses']['total']:,.2f}
                - Net Savings: ${summary['savings']['net_amount']:,.2f}
                - Savings Rate: {summary['savings']['savings_rate_percent']:.1f}%

                ## Top Spending Categories
                """
        for category, data in summary['category_breakdown'].items():
            insights += f"- {category}: ${data['total']:,.2f} ({data['percentage']:.1f}% of expenses)\n"

        insights += f"""
            ## Recommendations
            1. Review high-spending categories for potential savings
            2. Aim for a savings rate of at least 20%
            3. Track daily spending to stay within budget
            4. Set specific spending limits for each category
            5. Build an emergency fund covering 3-6 months of expenses
            """
        return insights 


def ask_financial_question(df_categorized: pd.DataFrame, question: str) -> str:
    """
    Ask specific financial questions about your data

    Args:
        df_categorized: Your categorized DataFrame
        question: Your specific question (e.g., "How can I save $500 per month?")

    Returns:
        AI-generated answer
    """
    config = {
        "base_url": "https://router.huggingface.co/v1",
        "api_key": st.secrets('HUGGINGFACE_API_KEY'),
        "model": "openai/gpt-oss-20b",
        "max_tokens": 8000,
        "timeout": 300
    }

    advisor = FinancialInsightsAdvisor(config)
    summary = advisor.prepare_financial_summary(df_categorized)

    prompt = f"""You are a financial advisor. Based on this financial data:

{json.dumps(summary, indent=2)}

Answer this specific question: {question}

Provide a detailed, actionable answer with specific recommendations and numbers:"""

    client = OpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"],
        timeout=config["timeout"]
    )

    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": "You are a helpful financial advisor providing specific, actionable advice."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config["max_tokens"],
        temperature=0.7
    )

    answer = response.choices[0].message.content
    print(f"\n💡 Answer to: '{question}'\n")
    print(answer)

    return answer

# ============================================================================
# STREAMLIT APP LAYOUT
# ============================================================================

# Title and overview
st.markdown("<h1 class='main-header'>🏦 Bank Statement Analyzer</h1>", unsafe_allow_html=True)
st.markdown("Upload your bank statement PDF and get instant financial insights with AI-powered categorization.")

# API Key
#st.markdown("---")
#st.header("⚙️ Configuration")
#Token = st.text_input("HuggingFace API Key / Token", type="password", help="Enter your HuggingFace Token")

# File upload section

uploaded_pdf = st.sidebar.file_uploader(
    "Upload your bank statement PDF",
    type=['pdf'],
    help="Upload a PDF bank statement to analyze your transactions"
)

# Process PDF button
if uploaded_pdf is not None and st.session_state.df_categorized is None:
    if st.sidebar.button("🚀 Process PDF", type="primary", use_container_width=True):
        with st.spinner("Processing your bank statement... This may take a few minutes."):
            df_categorized = process_pdf_to_dataframe(uploaded_pdf)
            if df_categorized is not None:
                st.session_state.df_categorized = df_categorized
                st.session_state.processing_status = "success"
                st.rerun()
            else:
                st.session_state.processing_status = "error"

# Clear data button
if st.session_state.df_categorized is not None:
    if st.sidebar.button("🗑️ Clear Data", use_container_width=True):
        st.session_state.df_categorized = None
        st.session_state.df_raw = None
        st.session_state.ai_insights = None
        st.session_state.processing_status = None
        st.rerun()

# Show processing status
if st.session_state.processing_status == "success":
    st.sidebar.markdown('<div class="status-box status-success">✅ PDF processed successfully!</div>', unsafe_allow_html=True)
elif st.session_state.processing_status == "error":
    st.sidebar.markdown('<div class="status-box status-error">❌ Error processing PDF</div>', unsafe_allow_html=True)

# Load data
if st.session_state.df_categorized is not None:
    df = process_data(st.session_state.df_categorized.copy())
    
    
    # Download options in sidebar
    with st.sidebar.expander("💾 Export Data", expanded=False):
        json_data = df.to_json(orient='records', indent=4, date_format='iso')
        st.download_button(
            label="📥 Download Categorized JSON",
            data=json_data,
            file_name="categorized_transactions.json",
            mime="application/json"
        )
        
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="categorized_transactions.csv",
            mime="text/csv"
        )
    
    # Apply filters only if we have data
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
        default=all_categories[:min(5, len(all_categories))]
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
    st.markdown("<h2 class='sub-header'>📈 Financial Summary</h2>", unsafe_allow_html=True)
    
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
        st.metric("Transactions", f"{num_transactions:,}")
    
    # Category distribution metrics
    st.markdown("<h3 class='sub-header'>🏷️ Category Distribution</h3>", unsafe_allow_html=True)
    
    top_categories = filtered_df[filtered_df['debit'] > 0].groupby('category')['debit'].sum().nlargest(5)
    
    cols = st.columns(len(top_categories))
    for idx, (category, amount) in enumerate(top_categories.items()):
        with cols[idx]:
            st.metric(category[:15] + "..." if len(category) > 15 else category, 
                     f"${amount:,.2f}")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "💰 Spending Analysis", 
        "📅 Time Analysis", 
        "🏷️ Category Breakdown",
        "📋 Transaction Details",
        "🤖 AI Insights"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        with col2:
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
    
    with tab3:
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
    
    with tab4:
        # Select a category to analyze
        selected_category = st.selectbox(
            "Select a category to analyze in detail:",
            options=sorted(filtered_df['category'].unique()),
            key="category_detail"
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
    
    with tab5:
        # Search and filter transactions
        search_term = st.text_input("Search in transaction descriptions:", "")
        
        if search_term:
            display_df = filtered_df[filtered_df['description/particulars'].str.contains(search_term, case=False, na=False)]
        else:
            display_df = filtered_df
        
        # Sort options
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox(
                "Sort by:",
                options=['date', 'debit', 'credit', 'category', 'balance'],
                index=0
            )
        with col2:
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
        columns_to_display = [col for col in columns_to_display if col in display_df_display.columns]
        display_df_display = display_df_display[columns_to_display]
        
        # Display the dataframe
        if not display_df_display.empty:
            st.dataframe(
                display_df_display,
                use_container_width=True,
                height=400
            )
        else:
            st.info("No transactions match the current filters.")
    
    with tab6:
        st.markdown("<h2 class='sub-header'>🤖 AI Financial Insights & Budget Advisor</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Get personalized AI-powered insights into your financial habits and receive actionable budget recommendations.
            
            **What you'll get:**
            - 📊 Overall financial health assessment
            - 💡 Key insights and observations
            - 🎯 Actionable recommendations
            - 📋 Personalized budget plan
            - ⚡ Quick wins for immediate impact
            """)
        
        with col2:
            if st.button("🚀 Generate AI Insights", type="primary", use_container_width=True):
                if st.session_state.ai_insights is None:
                    st.session_state.ai_processing = True
                    with st.spinner("🤖 AI is analyzing your financial data... This may take a minute."):
                        try:
                            config = {
                                "base_url": "https://router.huggingface.co/v1",
                                "api_key": st.secrets["HUGGINGFACE_API_KEY"],
                                "model": "openai/gpt-oss-20b",
                                "max_tokens": 8000,
                                "timeout": 300
                            }
                            
                            advisor = FinancialInsightsAdvisor(config)
                            insights = advisor.get_financial_insights(df)
                            st.session_state.ai_insights = insights
                            st.session_state.ai_processing = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating insights: {str(e)}")
                            st.session_state.ai_processing = False
                else:
                    st.info("Insights already generated. Scroll down to view them.")
        
        # Display loading indicator
        if st.session_state.ai_processing:
            st.info("AI is analyzing your financial data... Please wait.")
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate progress
                import time
                time.sleep(0.02)
                progress_bar.progress(i + 1)
        
        # Display AI insights if available
        if st.session_state.ai_insights:
            st.markdown("---")
            st.markdown("<h3 class='sub-header'>📋 Your Personalized Financial Analysis</h3>", unsafe_allow_html=True)
            
            # Format and display insights
            insights_text = st.session_state.ai_insights
            st.markdown(insights_text)
                        
            
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6B7280;'>
            <p>Bank Statement Analyzer • AI-powered transaction categorization and analysis</p>
            <p>Powered by HuggingFace AI models</p>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
        
    st.info("💡 **Note**: Processing may take 1-2 minutes depending on the size of your statement. AI models are used for both PDF extraction and categorization.")
