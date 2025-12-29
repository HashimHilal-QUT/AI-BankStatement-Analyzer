# 🏦 AI-Powered Bank Statement Analyzer
## https://ai-bank-app.streamlit.app

## 📋 Overview

**Bank Statement Analyzer** is a comprehensive financial intelligence application that transforms your bank statements into actionable insights using cutting-edge AI technology. Upload your PDF bank statements and instantly get categorized transactions, visual analytics, and personalized financial advice.

## ✨ Key Features

### 🔍 **Smart PDF Processing**
- **AI-Powered OCR**: Extracts transaction data from any bank statement PDF using HuggingFace's Qwen2.5-VL vision model
- **Universal Compatibility**: Works with Australian and international bank statement formats
- **Multi-page Support**: Handles single or multi-page statements with high accuracy

### 🏷️ **Intelligent Categorization**
- **AI Transaction Classification**: Automatically categorizes transactions into 18+ categories (Food & Dining, Groceries, Transport, etc.)
- **Confidence Scoring**: Each category comes with high/medium/low confidence indicators
- **Batch Processing**: Efficiently handles large transaction volumes

### 📊 **Interactive Visualizations**
- **Balance Tracking**: View account balance trends over time
- **Spending Analysis**: Interactive pie charts and bar graphs by category
- **Time Patterns**: Analyze spending by day of week, day of month, and time periods
- **Heatmaps**: Visualize spending patterns across categories and days

### 🤖 **AI Financial Advisor**
- **Personalized Insights**: Get custom financial health assessments (1-10 score)
- **Budget Recommendations**: Actionable advice based on your spending patterns
- **Savings Goals**: Realistic targets and strategies to improve savings
- **Quick Wins**: Immediate actions you can take to optimize finances
- **Q&A Feature**: Ask specific financial questions about your data

### 📁 **Data Management**
- **Flexible Export**: Download categorized data as JSON or CSV
- **Insights Reports**: Export AI-generated financial advice as text files
- **Filtering**: Filter by date range, categories, transaction types, and confidence levels
- **Search**: Find specific transactions quickly

## 🚀 How It Works

1. **Upload**: Drag and drop your bank statement PDF
2. **Process**: AI extracts and categorizes all transactions
3. **Analyze**: Explore interactive visualizations and insights
4. **Optimize**: Receive personalized financial recommendations
5. **Export**: Download your categorized data and insights

## 🛠️ Technology Stack

- **Frontend**: Streamlit for interactive web interface
- **AI Models**: 
  - HuggingFace Qwen2.5-VL (Vision) for PDF extraction
  - OpenAI-compatible GPT models for categorization and insights
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualization**: Plotly, Matplotlib for interactive charts
- **PDF Processing**: pdf2image, Poppler

## 📈 Use Cases
### 🏠 Personal Finance Management
- Track spending habits and identify saving opportunities
- Create and stick to personalized budgets
- Monitor financial health over time

### 💼 Small Business Owners
- Analyze business expenses and cash flow
- Categorize business transactions automatically
- Generate financial reports for tax purposes

### 🎓 Students & Young Professionals
- Learn budgeting skills with AI guidance
= Track daily expenses and subscription costs
- Set and achieve savings goals

### 👨‍👩‍👧‍👦 Families
- Manage household budgets
- Track shared expenses and subscriptions
- Plan for future financial goals

##🎯 Key Benefits
- **Time-Saving**: Process months of statements in minutes instead of hours
- **Insightful**: Discover spending patterns you might have missed
- **Accessible**: No financial expertise needed - AI explains everything
- **Private**: All processing happens locally/through secure APIs
= **Actionable**: Get specific, personalized recommendations

## 📄 Supported PDF Formats
- Standard bank statement layouts (tables of transactions)
- Most Australian banks (Commonwealth, ANZ, Westpac, NAB, etc.)
- International bank statements
- Single or multi-page PDFs
- Scanned statements with clear text

## 🚨 Limitations
- Requires clear PDF text (scans should be high-quality)
- Processing time varies with statement size (1-3 minutes typical)
- Internet connection required for AI API calls
- Large statements (>50 pages) may require batch processing


## **Features**:
📊 Overview - Balance trends and transaction counts

💰 Spending Analysis - Category breakdowns and top transactions

📅 Time Analysis - Daily, weekly, and monthly patterns

🏷️ Category Breakdown - Detailed category analysis

📋 Transaction Details - Searchable transaction table

🤖 AI Insights - Personalized financial advice and Q&A



### 🎥 Demo Screenshots

<img width="1900" height="1000" alt="FirstPage" src="https://github.com/user-attachments/assets/628772b1-2d52-4233-930b-15f27998ad93" />
<img width="1900" height="1000" alt="Time Analysis" src="https://github.com/user-attachments/assets/918e91ed-00dc-4e05-88be-117c30bf1998" />
<img width="1900" height="1000" alt="Spend Analysis" src="https://github.com/user-attachments/assets/c31c433e-03ea-4ab0-a886-f3331bb8e96c" />
<img width="1900" height="1000" alt="Category Breakdown" src="https://github.com/user-attachments/assets/ea549e4c-9e5b-4563-aa1c-9d2114bb5952" />
<img width="1900" height="1000" alt="AI Based Insights and Recommendations" src="https://github.com/user-attachments/assets/b7130c64-f92a-4e1b-a441-f48f20effd91" />
<img width="1900" height="1000" alt="AI based Budget Plan" src="https://github.com/user-attachments/assets/ea7c07a3-437d-4f67-a563-70144a2defb5" />

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **PDF processing failed** | Ensure Poppler is installed and in PATH |
| **API key error** | Check valid HuggingFace API key |
| **Slow processing** | Reduce PDF DPI in code or split large statements into smaller files |
| **No transactions found** | Ensure PDF has clear text/table format, try higher DPI setting |
| **Memory error** | Reduce batch size in categorization settings |
| **Model timeout** | Increase timeout duration in configuration |
| **Incorrect categorization** | Review confidence scores, manually adjust in Transaction Details tab |
| **Streamlit connection issues** | Restart Streamlit server, check network connection |
| **Plotly charts not displaying** | Update Plotly and Streamlit to latest versions |
| **Import errors** | Verify all dependencies in requirements.txt are installed |
| **File upload issues** | Check file size (max 200MB recommended), ensure PDF is not password protected |
| **AI insights not generating** | Verify HuggingFace API key has access to required models |
| **Export/download problems** | Check browser permissions, try different browser |
| **Filter not working** | Clear browser cache, refresh the page |
| **Date parsing errors** | Ensure date format in PDF matches expected format |
| **Balance calculation issues** | Verify balance column exists in extracted data |
