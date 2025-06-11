
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AI Forecast Bot", layout="wide")
st.title("AI Forecast Bot: Lifting Quantity Prediction (Part-wise)")

st.markdown("Upload your enriched Excel file (with features like lags, rolling averages, seasonality, etc.)")
uploaded_file = st.file_uploader("Choose a file", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Month'] = pd.to_datetime(df['Month'])
    df.sort_values(['Part No', 'Month'], inplace=True)

    search_input = st.text_input("🔍 Search for Part No")
    filtered_parts = df['Part No'].astype(str).unique()
    if search_input:
        filtered_parts = [p for p in filtered_parts if search_input.lower() in str(p).lower()]

    part_selected = st.selectbox("Select a part to forecast", filtered_parts)
    df_part = df[df['Part No'].astype(str) == str(part_selected)].copy()

    # Feature list based on enriched data
    features = [
        'Firm Schedule Qty', 'Firm_Lag1', 'Actual_Lag1', 'Month_Num', 'Year', 'Quarter',
        'Avg_Gap_Percent', 'Seasonal_Firm_Avg', 'Holiday_Impact',
        'Rolling_6mo_Firm', 'Rolling_6mo_Actual'
    ]
    target = 'Actual Lifting Qty'

    df_model = df_part.dropna(subset=features + [target])

    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=4)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction and evaluation
    y_pred = model.predict(X_test)
    test_results = X_test.copy()
    test_results['Month'] = df_model['Month'].iloc[-4:].values
    test_results['Actual'] = y_test.values
    test_results['Predicted'] = y_pred
    test_results['Error'] = test_results['Actual'] - test_results['Predicted']

    st.subheader("📊 Forecast Results (Last 4 Months)")
    st.dataframe(test_results[['Month', 'Actual', 'Predicted', 'Error']].round(2))

    # Line Chart
    st.line_chart(test_results.set_index('Month')[['Actual', 'Predicted']])

    # Upload for future prediction
    st.subheader("📥 Upload Next Month's Firm Schedule for Forecast")
    future_file = st.file_uploader("Upload future schedule (same enriched format)", type="xlsx", key="future")

    if future_file:
        future_df = pd.read_excel(future_file)
        future_df['Month'] = pd.to_datetime(future_df['Month'])
        future_df = future_df[future_df['Part No'].astype(str) == str(part_selected)].copy()

        future_features = future_df[features].dropna()
        future_df = future_df.loc[future_features.index]
        future_df['Predicted Lifting'] = model.predict(future_features)

        st.subheader("🔮 Predicted Actual Lifting")
        st.dataframe(future_df[['Month', 'Part No', 'Firm Schedule Qty', 'Predicted Lifting']].round(2))

        # Line chart of forecast
        fig, ax = plt.subplots()
        ax.plot(future_df['Month'], future_df['Firm Schedule Qty'], label='Firm Schedule')
        ax.plot(future_df['Month'], future_df['Predicted Lifting'], label='Predicted Lifting')
        ax.set_title("Firm Schedule vs Predicted Lifting")
        ax.set_xlabel("Month")
        ax.set_ylabel("Quantity")
        ax.legend()
        st.pyplot(fig)

        # Downloadable output
        output = BytesIO()
        future_df.to_excel(output, index=False)
        output.seek(0)
        st.download_button(
            label="📥 Download Forecast Data",
            data=output,
            file_name=f"forecast_{part_selected}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
