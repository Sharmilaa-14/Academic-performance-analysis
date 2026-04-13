import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Academic Performance Analysis", layout="wide")

# ---------------- SESSION STATE INIT ----------------
if "data" not in st.session_state:
    st.session_state.data = None
if "show_module2" not in st.session_state:
    st.session_state.show_module2 = False

# ---------------- LEFT SIDEBAR ----------------
st.sidebar.header("📌 Step Navigation")
st.sidebar.markdown("""
1. Upload Dataset & Integrate  
2. Perform Academic Analysis
""")
st.sidebar.markdown("---")
st.sidebar.header("📝 Quick Instructions")
st.sidebar.markdown("""
- Use CSV with headers: `Attendance`, `Internal_Marks`, `Assignment_Score`, `Study_Hours`, `Final_Result`  
- Step 2 will be enabled only after Step 1
""")

# ---------------- MAIN CONTENT + RIGHT COLUMN ----------------
main_col, right_col = st.columns([3, 1])  # Main content 3:1 right panel

# ---------------- RIGHT COLUMN ----------------
with right_col:
    st.markdown(
        '<div style="background-color:#e3f2fd;padding:10px;border-radius:10px">'
        '<h4>🔹 Project Steps / Workflow</h4>'
        '<ul>'
        '<li>Data Warehouse Design & Integration 🏗️</li>'
        '<li>Decision Tree Analysis 🌳</li>'
        '</ul>'
        '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div style="background-color:#fff3e0;padding:10px;border-radius:10px">'
        '<h4>💡 Tips & Chart Legend</h4>'
        '<ul>'
        '<li>🟢 Green: Correct Predictions</li>'
        '<li>🔴 Red: Misclassifications</li>'
        '<li>🔵 Blue: Predicted Values</li>'
        '<li>Use sliders for individual predictions</li>'
        '</ul>'
        '</div>', unsafe_allow_html=True)

# ---------------- MAIN COLUMN ----------------
with main_col:
    st.title("🎓 Academic Performance Analysis")
    st.caption("Using Data Warehousing and Decision Trees")

    # ================= MODULE 1 =================
    st.header("📦 Module 1: Data Warehouse & Academic Data Integration")
    uploaded_file = st.file_uploader("Upload Academic Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("✔ Dataset stored in Data Warehouse")
        st.write("")  # spacing

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Dataset Preview")
            st.dataframe(st.session_state.data.head(), height=300)
        with col2:
            st.subheader("📈 Dataset Summary")
            st.dataframe(st.session_state.data.describe(), height=300)

        st.write("")  # spacing
        if st.button("➡ Proceed to Module 2"):
            st.session_state.show_module2 = True

    # ================= MODULE 2 =================
    if st.session_state.show_module2:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("🌳 Module 2: Academic Performance Analysis")
        st.write("")  # spacing

        data = st.session_state.data
        X = data[['Attendance', 'Internal_Marks', 'Assignment_Score', 'Study_Hours']]
        y = data['Final_Result']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = DecisionTreeClassifier(criterion="entropy")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        analysis_df = X_test.copy()
        analysis_df["Actual"] = y_test.values
        analysis_df["Predicted"] = y_pred

        # --------- Decision Tree ---------
        st.subheader("🌲 Decision Tree")
        fig2, ax2 = plt.subplots(figsize=(12,8))
        plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, ax=ax2)
        st.pyplot(fig2)

        st.write("")  # spacing

        # --------- Confusion Matrix ---------
        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(13,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar=True, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        st.success(f"Overall Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

        st.write("")  # spacing

        # --------- Sample Predictions ---------
        with st.expander("📑 Sample Predictions"):
            st.dataframe(analysis_df.head(10))

        # --------- Individual Prediction ---------
        with st.expander("🔍 Individual Student Prediction"):
            att, im, ass, hrs = st.columns(4)
            attendance = att.slider("Attendance (%)", 0, 100, 75)
            internal = im.slider("Internal Marks", 0, 100, 70)
            assignment = ass.slider("Assignment Score", 0, 100, 65)
            hours = hrs.slider("Study Hours per Day", 0, 10, 3)

            if st.button("Predict Performance"):
                result = model.predict([[attendance, internal, assignment, hours]])
                st.success(f"Predicted Performance: **{result[0]}**")
