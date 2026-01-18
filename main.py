import streamlit as st
import pandas as pd
import os
from langsmith import evaluate
from langsmith import Client
import uuid
import ragapp
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
  
#langsmith
os.environ["LANGCHAIN_API_KEY"]=str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"

# Evaluation functions
def answer_hallucination_evaluator(run, example) -> dict:
    """A simple evaluator for generation hallucination"""
    grade_prompt_hallucinations = hub.pull("langchain-ai/rag-answer-hallucination")

    # RAG inputs
    input_question = example.inputs["question"]
    contexts = run.outputs["contexts"]
        
    # RAG answer
    prediction = run.outputs["answer"]

    # LLM grader
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3,max_tokens=500)

    # Structured prompt
    answer_grader = grade_prompt_hallucinations | llm

    # Get score
    score = answer_grader.invoke({"documents": contexts,"student_answer": prediction})
    final_score = score["Score"]
    return {"key": "answer_hallucination", "score": final_score}

def docs_relevance_evaluator(run, example) -> dict:
    """A simple evaluator for document relevance"""
    grade_prompt_doc_relevance = hub.pull("langchain-ai/rag-document-relevance")

    # RAG inputs
    input_question = example.inputs["question"]
    contexts = run.outputs["contexts"]
        
    # RAG answer
    prediction = run.outputs["answer"]

    # LLM grader
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3,max_tokens=500)

    # Structured prompt
    answer_grader = grade_prompt_doc_relevance | llm

    # Get score
    score = answer_grader.invoke({"question":input_question,"documents":contexts})
    final_score = score["Score"]
    return {"key": "document_relevance", "score": final_score}
  
def answer_evaluator(run, example) -> dict:
    """A simple evaluator for RAG answer accuracy"""

    grade_prompt_answer_accuracy = hub.pull("langchain-ai/rag-answer-vs-reference")

    # Get summary
    input_question = example.inputs["question"]
    reference = example.outputs["ground_truth"]
    prediction = run.outputs["answer"]

    # LLM grader
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3,max_tokens=500)

    # Structured prompt
    answer_grader = grade_prompt_answer_accuracy | llm

    # Get score
    score = answer_grader.invoke({"question": input_question,"correct_answer": reference,"student_answer": prediction})
    final_score = score["Score"]
    return {"key": "answer_score", "score": final_score}

def predict_rag_answer_with_context(example: dict):
    response = ragapp.get_answer(example["question"])
    return {"answer": response["answer"], "contexts": response["contexts"]}

# Streamlit UI
def main():
    st.title("LLM Evaluation Tool")
    gt_file = st.file_uploader("Upload Ground Truth CSV", type=["csv"])
    llm_file = st.file_uploader("Upload LLM application to evaluate",type=[".py"])
    metric = st.selectbox("Select Evaluation Metric", ["hallucination","relevance", "answer accuracy"])
    
    if gt_file is not None:
        df = pd.read_csv(gt_file)
        st.write("Ground Truth CSV Preview:")
        st.dataframe(df.head())
        question_col = st.selectbox("Select Question Column", df.columns)
        ground_truth_col = st.selectbox("Select Ground Truth Column", df.columns)

        if st.button("Create Langsmith Dataset"):
            # Prepare data for Langsmith dataset
            dataset_name = f"{metric}-dataset-{uuid.uuid4()}" # Generate uniquedataset name
            st.session_state.dataset=dataset_name
            dataset_rows = []
            for _, row in df.iterrows():
                dataset_rows.append({
                        "inputs": {"question": row[question_col]},
                        "outputs": {"ground_truth": row[ground_truth_col]}
                        })
                
            # Create Langsmith dataset
            client = Client()
            dataset = client.create_dataset(dataset_name=dataset_name)

            # Add rows to the dataset
            for row in dataset_rows:
                client.create_example(inputs=row["inputs"],outputs=row["outputs"], dataset_id=dataset.id)
            st.success(f"Dataset '{dataset_name}' created successfully in Langsmith!")
            langsmith_url = f"https://smith.langchain.com/o/a87e4dfa-61d0-4714-8c72/datasets/{dataset.id}"
            st.markdown(f"[View Dataset in Langsmith]({langsmith_url})")
                
    if gt_file and llm_file and st.button("Evaluate"):
        if metric == "hallucination":
            st.write("You have selected the hallucination evaluation metric.")
            results = []
            experiment_results = evaluate(
                predict_rag_answer_with_context,
                data= st.session_state.dataset,
                evaluators=[answer_hallucination_evaluator],
                experiment_prefix="rag-qa-gemini-hallucination",
                metadata={
                        "variant": "Vit-min context, gemini-1.5-flash",
                },
            )
            st.success("Evaluation Completed! Check the results below:")
            st.write(experiment_results)
            eval_id=str(experiment_results._results[0]['example'].dataset_id)
            langsmith_eval_url = f"https://smith.langchain.com/o/a87e4dfa-61d0-4714-8c72/datasets/{eval_id}"
            st.markdown(f"[View Evaluation in Langsmith]({langsmith_eval_url})")

        elif metric == "relevance":
            st.write("You have selected the relevance evaluation metric.")
            results = []
            experiment_results = evaluate(
                predict_rag_answer_with_context,
                data= st.session_state.dataset,
                evaluators=[docs_relevance_evaluator],
                experiment_prefix="rag-qa-gemini-relevance",
                metadata={
                        "variant": "Vit-min context, gemini-1.5-flash",
                },
            )
            st.success("Evaluation Completed! Check the results below:")
            st.write(experiment_results)
            eval_id=str(experiment_results._results[0]['example'].dataset_id)
            langsmith_eval_url = f"https://smith.langchain.com/o/a87e4dfa-61d0-4714-8c72/datasets/{eval_id}"
            st.markdown(f"[View Evaluation in Langsmith]({langsmith_eval_url})")

        elif metric=="answer accuracy":
            st.write("You have selected the answer accuracy evaluation metric.")
            results = []
            experiment_results = evaluate(
                predict_rag_answer_with_context,
                data= st.session_state.dataset,
                evaluators=[answer_evaluator],
                experiment_prefix="rag-qa-gemini-answer-accuracy",
                metadata={
                        "variant": "Vit-min context, gemini-1.5-flash",
                },
            )
            st.success("Evaluation Completed! Check the results below:")
            st.write(experiment_results)
            eval_id=str(experiment_results._results[0]['example'].dataset_id)
            langsmith_eval_url = f"https://smith.langchain.com/o/a87e4dfa-61d0-4714-8c72/datasets/{eval_id}"
            st.markdown(f"[View Evaluation in Langsmith]({langsmith_eval_url})")
                 
if __name__ == "__main__":
      main()
