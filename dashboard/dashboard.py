
# import streamlit as st
# import pandas as pd
# import json
# import matplotlib.pyplot as plt

# # Load log data
# def load_data(file_path="chat_log.jsonl"):
#     records = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 records.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue
#     return pd.DataFrame(records)

# # Load and clean
# df = load_data()
# st.title("📊 ESPRIT Chatbot Dashboard (Context Analysis)")

# if df.empty:
#     st.warning("No data found. Interact with the chatbot first.")
# else:
#     # Count top used context chunks
#     st.subheader("Most Frequently Retrieved Contexts")
#     df["short_context"] = df["context"].apply(lambda x: str(x)[:200] + "..." if pd.notnull(x) and len(str(x)) > 200 else str(x))

#     context_counts = df["short_context"].value_counts().head(10)

#     fig, ax = plt.subplots(figsize=(12, 6))
#     context_counts.sort_values().plot.barh(ax=ax)
#     ax.set_xlabel("Times Retrieved")
#     ax.set_ylabel("Context Snippet")
#     ax.set_title("Top Retrieved Knowledge Base Chunks")
#     st.pyplot(fig)

#     # Optional: Display raw data table
#     with st.expander("Show Raw Log Table"):
#         st.dataframe(df[["timestamp", "question", "short_context", "answer"]])

import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import random

# Load chat log
def load_data(file_path="chat_log.jsonl"):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)

# Load and clean
df = load_data()
st.title("📊 ESPRIT Chatbot – Grouped by Retrieved Context")

if df.empty:
    st.warning("No data found. Ask the chatbot something first.")
else:
    # Filter out empty context rows
    df = df[df["context"].notnull()]

    # Group by exact context text
    grouped = df.groupby("context")

    # Create a summarized DataFrame with representative question
    context_groups = []
    for context_text, group in grouped:
        representative_question = random.choice(group["question"].tolist())
        context_preview = context_text[:200] + "..." if len(context_text) > 200 else context_text
        context_groups.append({
            "context": context_text,
            "short_context": context_preview,
            "question": representative_question,
            "count": len(group)
        })

    rep_df = pd.DataFrame(context_groups).sort_values(by="count", ascending=True)

    # Horizontal bar chart
    st.subheader("Top Used Contexts (with Representative Questions)")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(rep_df["question"], rep_df["count"])
    ax.set_xlabel("Number of Questions Using the Same Context")
    ax.set_title("Most Frequently Retrieved Contexts (Grouped by Context Text)")
    st.pyplot(fig)

    # Optional detail
    with st.expander("📋 Show full data with context previews"):
        st.dataframe(rep_df[["question", "count", "short_context"]])
