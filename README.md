
#PDFQeuryAnalyzer

Qeury Analyzer works by taking in a txt file and a query and giving out the result. 
The model used here is dense passage retrieval(DPR). The model works by having both content and query in text. The model is taken from hugging face.

https://huggingface.co/facebook/dpr-question_encoder-single-nq-base


## Documentation

facebook/dpr-question_encoder-single-nq-base

https://huggingface.co/facebook/dpr-question_encoder-single-nq-base

Example use

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
embeddings = model(input_ids).pooler_output

## Deployment

To run this project locally

```bash
  streamlit run frontend.py
```
```bash
  python backend.py
```

