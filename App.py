import streamlit as st
import joblib
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd

# --- Custom CSS for Clean UI ---
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f9ff;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            color: #003366;
            margin-top: 0;
            font-size: 36px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #007acc;
            color: white;
            border-radius: 6px;
            padding: 0.6em 1.2em;
            border: none;
            font-weight: bold;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #005a99;
        }
        .feature-table {
            background-color: white;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #ccc;
            font-size: 14px;
            margin-top: 10px;
        }
        .dataframe {
            background-color: #ffffff;
            border-radius: 10px;
            border: 1px solid #ccc;
        }
        .dataframe thead th {
            background-color: #007acc;
            color: white;
        }
        .dataframe tbody tr:hover {
            background-color: #e0efff;
        }
        .clock {
            font-size: 30px;
            font-weight: bold;
            color: #003366;
            text-align: center;
        }
        footer {
            position: relative;
            bottom: 0;
            width: 100%;
            margin-top: 50px;
            padding: 20px 10px;
            background-color: #e1eaf5;
            color: #333;
            font-size: 14px;
            text-align: center;
        }
        .risk-message {
            font-weight: bold;
            font-size: 20px;
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .no-risk {
            color: #28a745;
            background-color: #e6f7e6;
            border-left: 5px solid #28a745;
        }
        .high-risk {
            color: #dc3545;
            background-color: #ffebee;
            border-left: 5px solid #dc3545;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Digital Clock ---
clock_placeholder = st.sidebar.empty()
def update_clock():
    clock_placeholder.markdown(f'<div class="clock">⏰ **Current Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
update_clock()

# --- Load Model ---
model = joblib.load("behavioral_scoring_model.pkl")
PREDICTION_CSV = "predictions.csv"
FEEDBACK_CSV = "feedback.csv"

# --- Helper Functions ---
def map_input_to_numeric(value, category_type):
    mapping = {
        "Frequent_Product_Category": {
            "Agriculture": 0, "Technology & Gadgets": 1,
            "Trade & Commerce": 2, "Transportation": 3, "Communication & Telecom": 4
        },
        "Loan_Reapplication_Frequency": {"Low": 0, "Medium": 1, "High": 2},
        "Preferred_Payment_Method": {"Cash": 0, "Mobile Money": 1, "Bank Transfer": 2}
    }
    return mapping.get(category_type, {}).get(value, -1)

def save_to_csv(data_row, file_path, headers=None):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists and headers:
            writer.writerow(headers)
        writer.writerow(data_row)

# --- App Title ---
st.markdown("<h1 class='main-title'>🧠 Behavioral Credit Scoring System</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction History", "Feedback Records"])

# --- Layout ---
if page == "Home":
    col1, col2 = st.columns([2, 1])

    with col1:
        # Add empty container for the risk message (will be filled after prediction)
        risk_message_placeholder = st.empty()
        
        with st.form("prediction_form"):
            st.subheader("📋 Customer Loan Information")

            loan_cycle = st.number_input("Loan Cycle", min_value=0, value=st.session_state.get('loan_cycle', 1), key='loan_cycle')
            repayment_delay = st.number_input("Avg. Repayment Delay (Days)", min_value=0.0, value=st.session_state.get('repayment_delay', 0.0), key='repayment_delay')
            total_loans = st.number_input("Total Loans Taken", min_value=0, value=st.session_state.get('total_loans', 1), key='total_loans')
            default_rate = st.number_input("Default Rate (%)", min_value=0.0, value=st.session_state.get('default_rate', 0.0), key='default_rate')

            category = st.selectbox("Frequent Product Category", [
                "Agriculture", "Technology & Gadgets", "Trade & Commerce", "Transportation", "Communication & Telecom"
            ], key='category')

            last_loan_amount = st.number_input("Last Loan Amount (#)", min_value=0.0, value=st.session_state.get('last_loan_amount', 1000.0), key='last_loan_amount')
            loan_freq = st.selectbox("Loan Reapplication Frequency", ["Low", "Medium", "High"], key='loan_freq')
            payment_method = st.selectbox("Preferred Payment Method", ["Cash", "Mobile Money", "Bank Transfer"], key='payment_method')

            col_a, col_b = st.columns(2)
            submitted = col_a.form_submit_button("🔍 Predict Loan Status")
            reset = col_b.form_submit_button("🔄 Clear Form")

    with col2:
        st.subheader("📘 Feature Descriptions")
        st.markdown("""<div class="feature-table">
        <table>
            <tr><th>Feature</th><th>Notes</th></tr>
            <tr><td><code>Loan Cycle</code></td><td>Times borrower has taken a loan</td></tr>
            <tr><td><code>Repayment Delay</code></td><td>Avg. days of repayment delay</td></tr>
            <tr><td><code>Total Loans</code></td><td>Total historical loans</td></tr>
            <tr><td><code>Default Rate (%)</code></td><td>Historical default %</td></tr>
            <tr><td><code>Product Category</code></td><td>Frequent loan category</td></tr>
            <tr><td><code>Last Loan Amount</code></td><td>Amount of most recent loan</td></tr>
            <tr><td><code>Reapplication Freq</code></td><td>How often borrower reapplies</td></tr>
            <tr><td><code>Payment Method</code></td><td>Cash, Mobile Money, Bank Transfer</td></tr>
        </table></div>""", unsafe_allow_html=True)

    # --- Prediction Logic ---
    if submitted:
        try:
            raw_inputs = [loan_cycle, repayment_delay, total_loans, default_rate, category,
                          last_loan_amount, loan_freq, payment_method]

            model_input = [
                loan_cycle,
                repayment_delay,
                total_loans,
                default_rate,
                map_input_to_numeric(category, "Frequent_Product_Category"),
                last_loan_amount,
                map_input_to_numeric(loan_freq, "Loan_Reapplication_Frequency"),
                map_input_to_numeric(payment_method, "Preferred_Payment_Method")
            ]

            prediction = model.predict([model_input])[0]
            status = ["Paid", "Active", "Delinquent"][prediction]
            remarks = {
                "Paid": "✅ Low risk: Good repayment history.",
                "Active": "🟡 Medium risk: Currently repaying.",
                "Delinquent": "🔴 High risk: Previous default."
            }

            # Display the risk message in the placeholder
            if status in ["Paid", "Active"]:
                risk_message_placeholder.markdown(
                    '<div class="risk-message no-risk">✓ CUSTOMER WILL NOT DEFAULT</div>', 
                    unsafe_allow_html=True
                )
            else:
                risk_message_placeholder.markdown(
                    '<div class="risk-message high-risk">✗ CUSTOMER WILL DEFAULT</div>', 
                    unsafe_allow_html=True
                )

            st.success(f"📊 Predicted Loan Status: **{status}**")
            st.info(f"{remarks[status]}")

            # Pie Chart
            labels = ["Paid", "Active", "Delinquent"]
            sizes = [1 if lbl == status else 0 for lbl in labels]
            colors = ['#28a745', '#ffc107', '#dc3545']

            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, colors=colors,
                   autopct=lambda p: f'{int(p)}%' if p > 0 else '',
                   startangle=140, textprops=dict(color="black", fontsize=12))
            ax.axis('equal')
            st.pyplot(fig)

            # Save prediction
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            save_to_csv([timestamp] + raw_inputs + [status, remarks[status]],
                        PREDICTION_CSV,
                        headers=["Timestamp", "Loan_Cycle", "Repayment_Delay", "Total_Loans",
                                 "Default_Rate", "Frequent_Product_Category", "Last_Loan_Amount",
                                 "Loan_Reapplication_Frequency", "Preferred_Payment_Method", "Predicted_Status", "Remarks"])
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {str(e)}")

    if reset:
        # Clear the risk message when form is reset
        risk_message_placeholder.empty()
        for key in ['loan_cycle', 'repayment_delay', 'total_loans', 'default_rate',
                    'category', 'last_loan_amount', 'loan_freq', 'payment_method']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # --- Feedback Section ---
    st.markdown("---")
    st.subheader("💬 Feedback")
    with st.form("feedback_form"):
        name = st.text_input("Name")
        feedback = st.text_area("Your feedback or suggestions...")
        feedback_submit = st.form_submit_button("📩 Submit Feedback")
        if feedback_submit:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            save_to_csv([timestamp, name, feedback], FEEDBACK_CSV, headers=["Timestamp", "Name", "Feedback"])
            st.success("✅ Thank you for your feedback!")

elif page == "Prediction History":
    st.subheader("📁 Prediction History")

    if os.path.exists(PREDICTION_CSV):
        df = pd.read_csv(PREDICTION_CSV, encoding='utf-8')
        if not df.empty:
            st.dataframe(df, use_container_width=True)

            # Summary
            st.subheader("📈 Prediction Status Summary")
            status_counts = df["Predicted_Status"].value_counts()
            fig, ax = plt.subplots()
            status_counts.plot(kind='bar', color=['#28a745', '#ffc107', '#dc3545'], ax=ax)
            ax.set_ylabel("Count")
            ax.set_title("Number of Predictions per Status")
            st.pyplot(fig)

            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download History as CSV", data=csv_download,
                               file_name='prediction_history.csv', mime='text/csv')

            if st.button("🧹 Clear History"):
                with open(PREDICTION_CSV, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Loan_Cycle", "Repayment_Delay", "Total_Loans",
                                     "Default_Rate", "Frequent_Product_Category", "Last_Loan_Amount",
                                     "Loan_Reapplication_Frequency", "Preferred_Payment_Method", "Predicted_Status", "Remarks"])
                st.success("🧼 Prediction history cleared.")

elif page == "Feedback Records":
    st.subheader("📥 Submitted Feedback")

    if os.path.exists(FEEDBACK_CSV):
        df_fb = pd.read_csv(FEEDBACK_CSV, encoding='utf-8')
        if not df_fb.empty:
            st.dataframe(df_fb, use_container_width=True)
            
            # Create columns for buttons to align them horizontally
            col1, col2 = st.columns(2)
            
            with col1:
                feedback_csv = df_fb.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Feedback as CSV", 
                    data=feedback_csv,
                    file_name='feedback_records.csv', 
                    mime='text/csv'
                )
            
            with col2:
                if st.button("🧹 Clear Feedback History", key="clear_feedback"):
                    try:
                        # Create empty file with just headers
                        with open(FEEDBACK_CSV, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(["Timestamp", "Name", "Feedback"])
                        st.success("✅ Feedback history cleared successfully!")
                        st.rerun()  # Refresh to show empty state
                    except Exception as e:
                        st.error(f"⚠️ Error clearing feedback: {str(e)}")
        else:
            st.info("No feedback has been submitted yet.")
    else:
        st.info("Feedback file not found.")
        
          #streamlit run App.py