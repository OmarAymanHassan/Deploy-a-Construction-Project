import streamlit as st
from graph import graph

st.set_page_config(page_title="Construction Company Evaluator", layout="wide")

st.title("🏗️ Construction Company Evaluation (LangGraph + Gemini)")

# ---------------------------
# Inputs
# ---------------------------

extraction_info = st.text_area(
    "📄 Project Requirements & Protocols",
    height=200
)

company_name = st.text_area(
    "🏢 Company Name to get the Information (History, Projects, Reputation)",
    height=200
)

#links = st.text_area(
 #   "🔗 Sources (URLs)",
  #  height=100
#)

# ---------------------------
# Run Graph
# ---------------------------

if st.button("Evaluate Company"):
    if not extraction_info or not company_name:
        st.warning("Please fill all required fields.")
    else:
        with st.spinner("Evaluating..."):
            result = graph.invoke({
                "extraction_info": extraction_info,
                "company_name": company_name,
                        })

        #output = result["result"]
        output = result["final_extraction_with_score"]

        # ---------------------------
        # Display Results
        # ---------------------------

        st.subheader("📊 Evaluation Result")

        st.write("### Key Signals")
        st.json(output.external_company_insights.model_dump_json())

        st.write("### Confidence Score")
        st.metric(
            "Overall Confidence",
            output.overall_confidence.overall_confidence
        )

        st.write("**Why this confidence?**")
        st.info(output.overall_confidence.explanation)

        st.write("### Sources")
        for src in output.sources:
            st.markdown(f"- {src}")
