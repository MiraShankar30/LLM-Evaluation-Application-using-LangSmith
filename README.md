# LLM-Evaluation-Application-using-LangSmith
This project is a Streamlit-based application for evaluating LLM applications using LangSmith. It allows users to upload ground truth data, select evaluation metrics, and run automated evaluations with results published to LangSmith.

## What the Application Does
- Accepts ground truth data in CSV format  
- Accepts an LLM application in Python (.py) format  
- Allows selection of evaluation metrics: hallucination, relevance, and answer accuracy  
- Creates a LangSmith dataset from the uploaded CSV  
- Runs evaluation on the selected metrics  
- Provides a link to view evaluation results in LangSmith  

## Ground Truth CSV
The CSV file should contain:  
- question: input query  
- ground_truth: expected correct answer  
This data is used to create a LangSmith dataset automatically.

## LLM Application Under Test
ragapp.py is provided as a sample LLM application.  
It answers questions strictly based on content retrieved from a given URL and returns both the answer and supporting context.

## Evaluation Metrics
- Hallucination: Checks whether the generated answer is supported by the retrieved context.  
- Relevance: Measures how relevant the retrieved documents are to the input question.  
- Answer Accuracy: Compares the generated answer against the provided ground truth.  

## Evaluation Flow
- Ground truth CSV is uploaded and converted into a LangSmith dataset
- Each question is executed against the LLM application
- Selected evaluators score the responses using an LLM-based judge
- Results are uploaded to LangSmith
- A link is generated to view the evaluation details

## Outcome
The application provides a simple and repeatable way to evaluate LLM outputs, track quality metrics, and analyze results using LangSmith.  
