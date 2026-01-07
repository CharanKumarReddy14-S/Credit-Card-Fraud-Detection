"""
Streamlit Frontend for Credit Card Fraud Detection
Professional UI for model interaction and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .fraud-alert {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .safe-alert {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models and preprocessor"""
    try:
        # Load best model
        if os.path.exists('models/best_model.pkl'):
            model = joblib.load('models/best_model.pkl')
            model_type = 'sklearn'
        elif os.path.exists('models/best_model.keras'):
            from tensorflow import keras
            model = keras.models.load_model('models/best_model.keras')
            model_type = 'keras'
        else:
            model = joblib.load('models/xgboost.pkl')
            model_type = 'sklearn'
        
        # Load preprocessor
        preprocessor = joblib.load('models/preprocessor.pkl')
        
        # Load training results
        with open('models/training_results.json', 'r') as f:
            results = json.load(f)
        
        return model, preprocessor, results, model_type
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run train_model.py first to train the models")
        return None, None, None, None


def preprocess_data(df, preprocessor):
    """Preprocess data for prediction"""
    scaler = preprocessor['scaler']
    feature_columns = preprocessor['feature_columns']
    
    # Feature engineering
    if 'Time' in df.columns:
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Day'] = (df['Time'] / 86400).astype(int)
        df['Time_of_Day'] = pd.cut(df['Hour'], 
                                   bins=[0, 6, 12, 18, 24],
                                   labels=[0, 1, 2, 3])
        df['Time_of_Day'] = df['Time_of_Day'].cat.codes
    
    if 'Amount' in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_Category'] = pd.cut(df['Amount'],
                                      bins=[-np.inf, 10, 100, 500, np.inf],
                                      labels=[0, 1, 2, 3])
        df['Amount_Category'] = df['Amount_Category'].astype(int)
    
    # Ensure all features present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_columns]
    df_scaled = scaler.transform(df)
    
    return pd.DataFrame(df_scaled, columns=feature_columns)


def make_predictions(model, data, model_type):
    """Make predictions with the model"""
    if model_type == 'keras':
        probs = model.predict(data, verbose=0).flatten()
    else:
        probs = model.predict_proba(data)[:, 1]
    
    predictions = (probs > 0.5).astype(int)
    
    return predictions, probs


def plot_fraud_distribution(df, predictions):
    """Create fraud distribution pie chart"""
    fraud_counts = pd.Series(predictions).value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Legitimate', 'Fraudulent'],
        values=[fraud_counts.get(0, 0), fraud_counts.get(1, 0)],
        marker=dict(colors=['#00C851', '#ff4444']),
        hole=0.4,
        textinfo='label+percent+value',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title="Transaction Distribution",
        height=400,
        showlegend=True
    )
    
    return fig


def plot_amount_distribution(df, predictions):
    """Create amount distribution by fraud status"""
    df_plot = df.copy()
    df_plot['Fraud'] = predictions
    df_plot['Status'] = df_plot['Fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig = px.box(
        df_plot, 
        x='Status', 
        y='Amount',
        color='Status',
        color_discrete_map={'Legitimate': '#00C851', 'Fraudulent': '#ff4444'},
        title="Transaction Amount by Status"
    )
    
    fig.update_layout(height=400)
    
    return fig


def plot_probability_distribution(probabilities):
    """Create fraud probability distribution"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=probabilities,
        nbinsx=50,
        marker=dict(
            color=probabilities,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Probability")
        ),
        name='Fraud Probability'
    ))
    
    fig.update_layout(
        title="Fraud Probability Distribution",
        xaxis_title="Fraud Probability",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )
    
    return fig


def plot_time_series(df, predictions):
    """Create time series plot of transactions"""
    if 'Time' not in df.columns:
        return None
    
    df_plot = df.copy()
    df_plot['Fraud'] = predictions
    df_plot['Hour'] = (df_plot['Time'] / 3600) % 24
    
    # Aggregate by hour
    hourly_stats = df_plot.groupby('Hour').agg({
        'Fraud': ['sum', 'count']
    }).reset_index()
    hourly_stats.columns = ['Hour', 'Fraud_Count', 'Total_Count']
    hourly_stats['Fraud_Rate'] = (hourly_stats['Fraud_Count'] / hourly_stats['Total_Count']) * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=hourly_stats['Hour'], y=hourly_stats['Total_Count'], 
               name='Total Transactions', marker_color='lightblue'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=hourly_stats['Hour'], y=hourly_stats['Fraud_Rate'], 
                   name='Fraud Rate (%)', marker_color='red', mode='lines+markers'),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Hour of Day")
    fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
    fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)
    fig.update_layout(title="Transactions and Fraud Rate by Hour", height=400)
    
    return fig


def main():
    # Header
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("### AI-Powered Transaction Monitoring & Analysis")
    
    # Load models
    model, preprocessor, results, model_type = load_models()
    
    if model is None:
        st.error("⚠️ Models not found. Please train the models first by running: `python train_model.py`")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model info
        st.subheader("Model Information")
        if results:
            best_model = results.get('best_model', 'Unknown')
            st.info(f"**Active Model:** {best_model.upper()}")
            
            if 'models' in results and best_model in results['models']:
                metrics = results['models'][best_model]
                st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                st.metric("Precision", f"{metrics['precision']:.4f}")
                st.metric("Recall", f"{metrics['recall']:.4f}")
        
        st.markdown("---")
        
        # Analysis mode
        st.subheader("Analysis Mode")
        mode = st.radio(
            "Select mode:",
            ["Single Transaction", "Batch Analysis"],
            help="Analyze one transaction or upload a CSV file"
        )
        
        st.markdown("---")
        st.caption("Built with ❤️ using ML & Deep Learning")
    
    # Main content
    if mode == "Single Transaction":
        st.header("🔍 Single Transaction Analysis")
        
        with st.expander("ℹ️ How to use", expanded=False):
            st.write("""
            1. Enter transaction details below
            2. Click 'Analyze Transaction'
            3. View fraud prediction and confidence level
            
            **Note:** V1-V28 are PCA-transformed features from the original dataset
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input("Time (seconds)", value=0.0, min_value=0.0)
            amount = st.number_input("Amount ($)", value=100.0, min_value=0.0, step=10.0)
        
        with col2:
            st.write("**PCA Features (V1-V28)**")
            st.caption("Use example values or modify as needed")
        
        # Create input fields for V1-V28
        v_features = {}
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i-1) % 4]:
                v_features[f'V{i}'] = st.number_input(
                    f'V{i}', 
                    value=0.0, 
                    format="%.6f",
                    key=f'v{i}'
                )
        
        if st.button("🔍 Analyze Transaction", type="primary"):
            # Create transaction data
            transaction = {
                'Time': time,
                'Amount': amount,
                **v_features
            }
            
            # Preprocess
            df = pd.DataFrame([transaction])
            processed = preprocess_data(df, preprocessor)
            
            # Predict
            prediction, probability = make_predictions(model, processed, model_type)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction[0] == 1:
                    st.markdown('<div class="fraud-alert">⚠️ FRAUDULENT</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-alert">✅ LEGITIMATE</div>', 
                              unsafe_allow_html=True)
            
            with col2:
                st.metric("Fraud Probability", f"{probability[0]:.2%}")
            
            with col3:
                if probability[0] > 0.8:
                    risk = "🔴 CRITICAL"
                elif probability[0] > 0.6:
                    risk = "🟠 HIGH"
                elif probability[0] > 0.3:
                    risk = "🟡 MEDIUM"
                else:
                    risk = "🟢 LOW"
                st.metric("Risk Level", risk)
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[0] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Confidence"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if probability[0] > 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Batch Analysis
        st.header("📁 Batch Transaction Analysis")
        
        with st.expander("ℹ️ File Requirements", expanded=False):
            st.write("""
            **Required columns:** Time, V1-V28, Amount
            
            - CSV format
            - One transaction per row
            - All 30 features must be present
            
            [Download sample CSV](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
            """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                # Read file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File loaded: {len(df)} transactions")
                
                # Show preview
                with st.expander("📄 Data Preview", expanded=False):
                    st.dataframe(df.head(10))
                
                # Process button
                if st.button("🚀 Analyze All Transactions", type="primary"):
                    with st.spinner("Analyzing transactions..."):
                        # Preprocess
                        df_processed = preprocess_data(df.copy(), preprocessor)
                        
                        # Predict
                        predictions, probabilities = make_predictions(
                            model, df_processed, model_type
                        )
                        
                        # Add results to dataframe
                        df['Fraud_Prediction'] = predictions
                        df['Fraud_Probability'] = probabilities
                        df['Risk_Level'] = pd.cut(
                            probabilities,
                            bins=[0, 0.3, 0.6, 0.8, 1.0],
                            labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                        )
                    
                    st.success("✅ Analysis complete!")
                    
                    # Summary metrics
                    st.subheader("📊 Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    fraud_count = predictions.sum()
                    legit_count = len(predictions) - fraud_count
                    fraud_pct = (fraud_count / len(predictions)) * 100
                    avg_fraud_amount = df[df['Fraud_Prediction'] == 1]['Amount'].mean()
                    
                    col1.metric("Total Transactions", f"{len(df):,}")
                    col2.metric("Fraudulent", f"{fraud_count:,}", 
                              delta=f"{fraud_pct:.1f}%")
                    col3.metric("Legitimate", f"{legit_count:,}")
                    col4.metric("Avg Fraud Amount", f"${avg_fraud_amount:.2f}")
                    
                    # Visualizations
                    st.subheader("📈 Visual Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = plot_fraud_distribution(df, predictions)
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        fig3 = plot_probability_distribution(probabilities)
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    with col2:
                        fig2 = plot_amount_distribution(df, predictions)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        fig4 = plot_time_series(df, predictions)
                        if fig4:
                            st.plotly_chart(fig4, use_container_width=True)
                    
                    # Detailed results table
                    st.subheader("📋 Detailed Results")
                    
                    # Filter options
                    filter_col1, filter_col2 = st.columns(2)
                    with filter_col1:
                        filter_option = st.selectbox(
                            "Filter by:",
                            ["All", "Fraudulent Only", "Legitimate Only"]
                        )
                    
                    with filter_col2:
                        sort_option = st.selectbox(
                            "Sort by:",
                            ["Fraud Probability", "Amount", "Time"]
                        )
                    
                    # Apply filters
                    df_display = df.copy()
                    if filter_option == "Fraudulent Only":
                        df_display = df_display[df_display['Fraud_Prediction'] == 1]
                    elif filter_option == "Legitimate Only":
                        df_display = df_display[df_display['Fraud_Prediction'] == 0]
                    
                    # Sort
                    if sort_option == "Fraud Probability":
                        df_display = df_display.sort_values('Fraud_Probability', ascending=False)
                    elif sort_option == "Amount":
                        df_display = df_display.sort_values('Amount', ascending=False)
                    else:
                        df_display = df_display.sort_values('Time')
                    
                    # Display
                    st.dataframe(
                        df_display[['Time', 'Amount', 'Fraud_Prediction', 
                                   'Fraud_Probability', 'Risk_Level']],
                        use_container_width=True
                    )
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.info("Please ensure your CSV has the required format")


if __name__ == "__main__":
    main()