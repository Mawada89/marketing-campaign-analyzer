import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="محلل الحملات التسويقية",
    page_icon="📊",
    layout="wide"
)

st.title("📊 محلل الحملات التسويقية")
st.write("ارفعي ملف Excel أو CSV لتحليل أداء حملاتك التسويقية.")

uploaded_file = st.file_uploader(
    "اختاري ملف الحملات",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📋 معاينة البيانات")
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 تحليل الأداء")

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            selected_column = st.selectbox(
                "اختاري المؤشر المراد تحليله",
                numeric_columns
            )

            st.bar_chart(df[selected_column])

            st.metric(
                "إجمالي المؤشر",
                f"{df[selected_column].sum():,.2f}"
            )

            st.subheader("📥 تنزيل التقرير")
            csv_data = df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "تحميل التقرير بصيغة CSV",
                data=csv_data,
                file_name="marketing_report.csv",
                mime="text/csv"
            )
        else:
            st.warning(
                "لا توجد أعمدة رقمية في الملف لتحليلها."
            )

    except Exception as e:
        st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
else:
    st.info(
        "ابدئي برفع ملف يحتوي على بيانات حملاتك."
    )
