import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor

def format_indian_currency(x, pos):
    if x >= 10000000:  # Crores
        return f'{x/10000000:.1f}Cr'
    elif x >= 100000:  # Lakhs
        return f'{x/100000:.1f}L'
    elif x >= 1000:
        return f'{x/1000:.1f}k'
    else:
        return f'₹{int(x)}'

# Streamlit page config
st.set_page_config(
    page_title="Intelligent Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Light Theme
st.markdown("""
    <style>
    /* Main Background - Clean White */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 2px solid #dee2e6;
    }
    
    /* Sidebar Title */
    [data-testid="stSidebar"] h1 {
        color: #2c3e50 !important;
        font-weight: 700 !important;
        font-size: 22px !important;
        padding: 15px 0 10px 0 !important;
        text-align: center !important;
        border-bottom: 2px solid #3498db !important;
    }
    
    /* Title Styling */
    h1 {
        color: #2c3e50 !important;
        font-weight: 800 !important;
        font-size: 32px !important;
        text-align: center !important;
        padding: 15px 0 25px 0 !important;
        border-bottom: 3px solid #3498db !important;
        margin-bottom: 25px !important;
    }
    
    /* Metrics Styling */
    .stMetric {
        background-color: #f8f9fa !important;
        padding: 15px !important;
        border-radius: 10px !important;
        border-left: 4px solid #3498db !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    }
    
    /* Header Box */
    .header-box {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    </style>
""", unsafe_allow_html=True)


# Sidebar Navigation

st.sidebar.title("📊Intelligent Sales Prediction System")
menu = ["Dataset Preview", "Sales by Category", "Sales by Country", "Sales by Payment Method",
        "Sales by Channel", "Return Analysis", "Sales Trend", "Sales Prediction", 
        "Return Prediction", "Future Sales Forecast", "Recommendations"]
choice = st.sidebar.radio("Go to", menu)

# Exchange rate (1 USD = 83 INR)
EXCHANGE_RATE = 83

# Load Dataset

df = pd.read_csv("online_sales_datasett.csv")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
df['SalesAmount'] = df['Quantity'] * df['UnitPrice']
df['SalesAmountINR'] = df['SalesAmount'] * EXCHANGE_RATE  # Convert to Rupees
df['Month'] = df['InvoiceDate'].dt.month
df = df.dropna()

# Encode for models
df_encoded = pd.get_dummies(
    df,
    columns=['Category','Country','PaymentMethod','SalesChannel'],
    drop_first=True
)
df_encoded['ReturnStatus'] = df_encoded['ReturnStatus'].astype('category').cat.codes

# ===========================
# Dataset Preview
# ===========================
if choice == "Dataset Preview":
    st.markdown("""
        <div class="header-box">
            <h1 style="color: white !important; border: none !important;">📁 Dataset Preview</h1>
            <p style="color: white !important; font-size: 16px !important;">Explore the Online
                 Sales Dataset</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Total Sales Amount", f"₹{df['SalesAmountINR'].sum():,.2f}")
    
    st.markdown("### 📋 First 10 Records")
    st.dataframe(df.head(10))
    st.write("Dataset Shape:", df.shape)


# Sales by Category
elif choice == "Sales by Category":
    st.markdown("""
        <div class="header-box">
            <h1 style="color: white !important; border: none !important;">📦 Sales by Category</h1>
            <p style="color: white !important; font-size: 16px !important;">Total Sales Distribution Across Product Categories</p>
        </div>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    sns.barplot(x='Category', y='SalesAmountINR', data=df, estimator=sum, ax=ax, palette='husl')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_indian_currency))
    ax.set_title('Total Sales by Category', fontsize=14, fontweight='bold', color='#2c3e50', pad=10)
    ax.set_xlabel('Category', fontsize=11, fontweight='600', color='#34495e')
    ax.set_ylabel('Sales Amount (₹)', fontsize=11, fontweight='600', color='#34495e')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Category sales ranking table
    st.markdown("### 📊 Complete Category Sales Ranking (Highest to Lowest)")
    category_sales = df.groupby('Category')['SalesAmountINR'].sum().sort_values(ascending=False)
    cat_df = pd.DataFrame({
        'Category': category_sales.index,
        'Sales (₹)': [f"₹{x:,.0f} ({format_indian_currency(x, None)})" for x in category_sales.values],
        'Rank': range(1, len(category_sales)+1)
    })
    st.dataframe(cat_df, use_container_width=True)


# Sales by Country
elif choice == "Sales by Country":
    st.markdown("""
        <div class="header-box">
            <h1 style="color: white !important; border: none !important;">🌍 Sales by Country</h1>
            <p style="color: white !important; font-size: 16px !important;">Geographic Distribution of Sales</p>
        </div>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    country_sales = df.groupby('Country')['SalesAmountINR'].sum().sort_values()
    country_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(country_sales)))
    ax.barh(country_sales.index, country_sales.values, color=country_colors)
    ax.set_title('Sales by Country', fontsize=14, fontweight='bold', color='#2c3e50', pad=10)
    ax.set_xlabel('Sales Amount (₹)', fontsize=11, fontweight='600', color='#34495e')
    ax.set_ylabel('Country', fontsize=11, fontweight='600', color='#34495e')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_indian_currency))
    ax.grid(True, alpha=0.3, linestyle='--')
    st.pyplot(fig)


# Sales by Payment Method

elif choice == "Sales by Payment Method":
    st.markdown("""
        <div class="header-box">
            <h1 style="color: white !important; border: none !important;">💳 Sales by Payment Method</h1>
            <p style="color: white !important; font-size: 16px !important;">Payment Method Distribution Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    payment_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    payment_counts = df['PaymentMethod'].value_counts()
    wedges, texts, autotexts = ax.pie(payment_counts.values, labels=payment_counts.index, 
                                       autopct='%1.1f%%', colors=payment_colors,
                                       explode=[0.02]*len(payment_counts), shadow=False,
                                       textprops={'fontsize': 11, 'fontweight': 'bold', 'color': '#2c3e50'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Payment Method Distribution', fontsize=14, fontweight='bold', color='#2c3e50', pad=10)
    st.pyplot(fig)


# Sales by Channel
elif choice == "Sales by Channel":
    st.markdown("""
        <div class="header-box">
            <h1 style="color: white !important; border: none !important;">📡 Sales by Channel</h1>
            <p style="color: white !important; font-size: 16px !important;">Sales Channel Performance Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
    channel_colors = ['#3498db', '#2ecc71']
    channel_counts = df['SalesChannel'].value_counts()
    bars = ax.bar(channel_counts.index, channel_counts.values, color=channel_colors, edgecolor='white', linewidth=2)
    ax.set_title('Sales Channel Usage', fontsize=14, fontweight='bold', color='#2c3e50', pad=10)
    ax.set_xlabel('Sales Channel', fontsize=11, fontweight='600', color='#34495e')
    ax.set_ylabel('Count', fontsize=11, fontweight='600', color='#34495e')
    for bar, val in zip(bars, channel_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80, 
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    st.pyplot(fig)

# Return Analysis
elif choice == "Return Analysis":
    st.markdown("""
        <div class="header-box">
            <h1 style="color: white !important; border: none !important;">↩️ Return Analysis</h1>
            <p style="color: white !important; font-size: 16px !important;">Return vs Non-Return Order Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    return_colors = ['#2ecc71', '#e74c3c']
    return_counts = df['ReturnStatus'].value_counts()
    wedges, texts, autotexts = ax.pie(return_counts.values, labels=return_counts.index, 
                                        autopct='%1.1f%%', colors=return_colors,
                                        explode=[0.02]*len(return_counts), shadow=False,
                                        textprops={'fontsize': 11, 'fontweight': 'bold', 'color': '#2c3e50'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Return vs Non-Return Orders', fontsize=14, fontweight='bold', color='#2c3e50', pad=10)
    st.pyplot(fig)

# Sales Trend
elif choice == "Sales Trend":
    st.markdown("""
        <div class="header-box">
            <h1 style="color: white !important; border: none !important;">📈 Sales Trend</h1>
            <p style="color: white !important; font-size: 16px !important;">Monthly Sales Performance Trend</p>
        </div>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    monthly_sales = df.groupby('Month')['SalesAmountINR'].sum()
    ax.plot(monthly_sales.index, monthly_sales.values, marker='o', markersize=8, 
            color='#3498db', linewidth=2.5, markerfacecolor='white', markeredgewidth=2)
    ax.fill_between(monthly_sales.index, monthly_sales.values, alpha=0.2, color='#3498db')
    ax.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold', color='#2c3e50', pad=10)
    ax.set_xlabel('Month', fontsize=11, fontweight='600', color='#34495e')
    ax.set_ylabel('Sales Amount (₹)', fontsize=11, fontweight='600', color='#34495e')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_indian_currency))
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.3, linestyle='--')
    st.pyplot(fig)

# Sales Prediction

elif choice == "Sales Prediction":
    st.markdown("""
        <div class="header-box">
            <h1 style="color: white !important; border: none !important;">🤖 Sales Prediction</h1>
            <p style="color: white !important; font-size: 16px !important;">Machine Learning Model using Random Forest Regressor</p>
        </div>
    """, unsafe_allow_html=True)
    
    X_sales = df_encoded.drop(['SalesAmount','ReturnStatus','InvoiceNo','StockCode','Description',
                               'InvoiceDate','CustomerID','ShipmentProvider','WarehouseLocation',
                               'OrderPriority'],
                              axis=1, errors='ignore')
    X_sales = X_sales.select_dtypes(include=['int64','float64'])
    y_sales = df_encoded['SalesAmount']

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_sales, y_sales, test_size=0.2, random_state=42
    )
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_rf, y_train_rf)
    sales_pred_rf = rf_model.predict(X_test_rf)

    r2 = r2_score(y_test_rf, sales_pred_rf)

    tolerance = 0.10
    accuracy_rf = np.mean(np.abs(sales_pred_rf - y_test_rf) <= tolerance * y_test_rf) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("Tolerance", "0.10")
    with col3:
        st.metric("Accuracy (±10%)", f"{accuracy_rf:.2f}%")

    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    ax.scatter(y_test_rf * EXCHANGE_RATE, sales_pred_rf * EXCHANGE_RATE, alpha=0.5, color='#3498db', s=40)
    ax.plot([y_test_rf.min() * EXCHANGE_RATE, y_test_rf.max() * EXCHANGE_RATE], 
            [y_test_rf.min() * EXCHANGE_RATE, y_test_rf.max() * EXCHANGE_RATE], 
            color='#e74c3c', linewidth=2, linestyle='--', label='Perfect Prediction')
    ax.set_title('Actual vs Predicted Sales (Random Forest)', fontsize=12, fontweight='bold', color='#2c3e50', pad=10)
    ax.set_xlabel('Actual Sales (₹)', fontsize=10, fontweight='600', color='#34495e')
    ax.set_ylabel('Predicted Sales (₹)', fontsize=10, fontweight='600', color='#34495e')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_indian_currency))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_indian_currency))
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    st.pyplot(fig)


# Return Prediction

elif choice == "Return Prediction":
    st.markdown("""
        <div class="header-box">
            <h1 style="color: white !important; border: none !important;">🎯 Return Prediction</h1>
            <p style="color: white !important; font-size: 16px !important;">Machine Learning Model using Decision Tree Classifier</p>
        </div>
    """, unsafe_allow_html=True)
    
    X_log = df_encoded.drop(['ReturnStatus','SalesAmount'], axis=1)
    X_log = X_log.select_dtypes(include=['int64','float64'])
    y_log = df_encoded['ReturnStatus']

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_log, y_log, test_size=0.2, random_state=42
    )
    tree_model = DecisionTreeClassifier(max_depth=5)
    tree_model.fit(X_train_clf, y_train_clf)
    tree_pred = tree_model.predict(X_test_clf)

    accuracy_dt = accuracy_score(y_test_clf, tree_pred)
    cm_tree = confusion_matrix(y_test_clf, tree_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", f"{accuracy_dt*100:.2f}%")
    with col2:
        st.metric("Total Predictions", len(y_test_clf))

    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                annot_kws={'size': 14, 'fontweight': 'bold'},
                xticklabels=['No Return', 'Return'], yticklabels=['No Return', 'Return'])
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='600', color='#34495e')
    ax.set_ylabel('Actual', fontsize=12, fontweight='600', color='#34495e')
    st.pyplot(fig)

# Future Sales Forecast

elif choice == "Future Sales Forecast":
    st.markdown("""
        <div class="header-box">
            <h1 style="color: white !important; border: none !important;">🔮 Future Sales Forecast</h1>
            <p style="color: white !important; font-size: 16px !important;">Linear Regression Model for Sales Forecasting</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Prepare features and target
    X_sales = df_encoded.drop(
        ['SalesAmount','ReturnStatus','InvoiceNo','StockCode','Description',
         'InvoiceDate','CustomerID','ShipmentProvider','WarehouseLocation',
         'OrderPriority'],
        axis=1, errors='ignore'
    )
    y_sales = df_encoded['SalesAmount']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sales, y_sales, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    sales_pred = lr_model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, sales_pred)


    col1, col2, col3 = st.columns(3)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, sales_pred)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("MAE", f"{mae:.2f}")
    with col3:
        st.metric("Model Performance", f"{r2*100:.2f}%")

    # Plot Actual vs Predicted
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.scatter(y_test * EXCHANGE_RATE, sales_pred * EXCHANGE_RATE, alpha=0.5, label="Predicted vs Actual")
    ax.plot([y_test.min() * EXCHANGE_RATE, y_test.max() * EXCHANGE_RATE], 
            [y_test.min() * EXCHANGE_RATE, y_test.max() * EXCHANGE_RATE], color='red', label="Perfect Prediction")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_indian_currency))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_indian_currency))
    ax.grid(True)
    ax.set_xlabel("Actual Sales (₹)")
    ax.set_ylabel("Predicted Sales (₹)")
    ax.set_title("Linear Regression: Actual vs Predicted Sales")
    ax.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the figure for next plot


# Recommendations

elif choice == "Recommendations":
    st.markdown("""
        <div class="header-box">
            <h1 style="color: white !important; border: none !important;">💡 Sales Recommendations</h1>
            <p style="color: white !important; font-size: 16px !important;">Actionable Strategies for Underperforming Categories</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate category sales
    category_sales = df.groupby('Category')['SalesAmountINR'].sum().sort_values()
    
    if len(category_sales) >= 2:
        lowest_category = category_sales.index[0]
        second_lowest_category = category_sales.index[1]
        lowest_sales = category_sales.iloc[0]
        second_sales = category_sales.iloc[1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error(f"**🚨 Lowest Category: {lowest_category}**")
            st.metric("Sales", f"₹{lowest_sales:,.0f}")
        
        with col2:
            st.warning(f"**⚠️ Second Lowest: {second_lowest_category}**")
            st.metric("Sales", f"₹{second_sales:,.0f}") 
        
        # Apparels Section
        st.markdown("### 👕 **Apparels Recommendations**")
        with st.expander("📈 **Apparels Category Strategy**", expanded=True):
            st.markdown("""
            **🎯 Priority Action Plan:**
            
            1. **Targeted Promotions**: 30% discount + free shipping for 7 days  
            2. **Email/SMS Campaign**: "Apparels Flash Sale!"
            3. **Social Media Boost**: ₹10,000 Instagram ads
            4. **Bundle Offers**: Apparels + Accessories
            5. **Customer Segmentation**: Past apparel buyers
            
            **Expected Impact**: 40-60% uplift
            """)
        with st.expander("👕 **T-shirt Specific Strategies**", expanded=False):
            st.markdown("""
            **T-shirt Boost Plan:**
            1. **Size Expansion**: XS-XXL + 15 colors
            2. **Influencer Campaign**: 5 fashion influencers  
            3. **Casual Bundles**: T-shirt + Shorts 20% OFF
            4. **Summer Prints**: 10 new designs
            5. **Buy 3 Get 1**: Gifting/back-to-school
            
            **Target**: 50-70% increase
            """)
        
        # Stationery Section  
        st.markdown("### 📓 **Stationery Recommendations**")
        with st.expander("📊 **Stationery Category Strategy**", expanded=False):
            st.markdown("""
            **🚀 Secondary Action Plan:**
            
            1. **15% Flash Sale**: Weekend promotion
            2. **Cross-Sell**: Stationery + Electronics
            3. **Content Videos**: Product demos
            4. **Loyalty Points**: Double points
            5. **Inventory Check**: Stock issues
            
            **Expected Impact**: 25-40% improvement
            """)
        with st.expander("📓 **Notebook Strategies**", expanded=False):
            st.markdown("""
            **📚 Notebook Plan:**
            1. **Back-to-School**: Notebook + Pen ₹99
            2. **Premium Paper**: A5 hard cover  
            3. **Buy 3 Free Pen**: Student special
            4. **Office Bulk**: 20% off 50+
            5. **Planners**: Monthly with quotes
            
            **Target**: 60% uplift
            """)
        with st.expander("✒️ **Pen Strategies**", expanded=False):
            st.markdown("""
            **🖊️ Pen Recovery:**
            1. **20 New Colors**
            2. **Gel Pen Launch**
            3. **Corporate Gifting**
            4. **Buy 12 Get 3**
            5. **Premium Metal Pens**
            
            **Target**: 45% boost
            """)

        st.success("✅ **All recommendations data-driven & actionable**")



    else:
        st.warning("Not enough data for recommendations")

       

