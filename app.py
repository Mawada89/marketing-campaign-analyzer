
import streamlit as st
import pandas as pd

st.title("تحليل الحملات التسويقية")
st.write("مرحباً بك في برنامج تحليل الحملات التسويقية")

data = {
    "الحملة": ["حملة 1", "حملة 2", "حملة 3"],
    "التكلفة": [100, 150, 200],
    "المبيعات": [300, 450, 500]
}

df = pd.DataFrame(data)
df["الربح"] = df["المبيعات"] - df["التكلفة"]

st.subheader("تقرير الحملات")
st.dataframe(df)

st.bar_chart(df.set_index("الحملة")[["التكلفة", "المبيعات"]])
