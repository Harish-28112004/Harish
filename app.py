import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Advanced Spam Email Detector")

# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv("spam.csv")
df.columns = ["label", "text"]
df["label"] = df["label"].map({"spam":1, "ham":0})

# ------------------------------
# TRAIN MODEL
# ------------------------------
cv = CountVectorizer()
X = cv.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ------------------------------
# METRICS
# ------------------------------
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)

# ------------------------------
# USER INPUT
# ------------------------------
st.subheader("✍️ Enter Email Text")

user_input = st.text_area("Type your message here:")

if st.button("Analyze"):
    if user_input.strip() != "":
        data = cv.transform([user_input]).toarray()
        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0]

        if prediction == 1:
            st.error(f"🚨 SPAM (Confidence: {round(max(prob)*100,2)}%)")
        else:
            st.success(f"✅ NOT SPAM (Confidence: {round(max(prob)*100,2)}%)")
    else:
        st.warning("Please enter some text")

# ------------------------------
# METRICS DISPLAY
# ------------------------------
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", round(accuracy,2))
col2.metric("Precision", round(precision,2))
col3.metric("Recall", round(recall,2))
col4.metric("F1 Score", round(f1,2))

# ------------------------------
# CONFUSION MATRIX (COLOR)
# ------------------------------
st.subheader("📉 Confusion Matrix")

fig, ax = plt.subplots()
cax = ax.matshow(cm)

for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax.text(j, i, cm[i][j], ha='center', va='center')
    
    st.pyplot(fig)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar(cax)

st.pyplot(fig)
