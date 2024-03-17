# FastAPI backend
from fastapi import FastAPI, File, HTTPException
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch

app = FastAPI()

tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

@app.post("/qa")
async def qa(question: str, text_file: str = File(...)):
    print("Received question:", question)
    print("Received text file:", text_file[:100])  # Print the first 100 characters of the text file
    
    if not text_file:
        raise HTTPException(status_code=400, detail="Missing text file")

    inputs_question = tokenizer(question, return_tensors="pt")
    inputs_text = tokenizer(text_file, return_tensors="pt")

    # Generate embeddings for the question and text
    with torch.no_grad():
        embeddings_question = model(**inputs_question).pooler_output
        embeddings_text = model(**inputs_text).pooler_output

    # Find the most similar passage to the question (rest of the logic remains the same)
    similarity_scores = torch.matmul(embeddings_question, embeddings_text.T)
    closest_idx = torch.argmax(similarity_scores)
    passages = text_file.split("\n\n")  # Assuming paragraphs are separated by double newlines
    closest_passage = passages[closest_idx]

    # Implement answer generation using a simple approach (replace with more sophisticated techniques)
    answer = closest_passage.split(". ")[0]  # Extract first sentence as a simple answer

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
