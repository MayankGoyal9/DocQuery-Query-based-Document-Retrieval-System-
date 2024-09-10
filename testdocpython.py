import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as  np
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

def fetcher(pdf_path):
    with open(pdf_path,'rb') as file:
        pdf_reader =PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

pdf_text= fetcher("C:\\Users\\mayan\\Desktop\\knowledge_doc.pdf")

def trim(text,max_tokens=1024):
    return text[:max_tokens]

model = SentenceTransformer('all-MiniLM-L6-v2')
document_embedding= model.encode([pdf_text])
index = faiss.IndexFlatL2(document_embedding.shape[1])
index.add(np.array(document_embedding))

def pull(query):
    query_embedding= model.encode([query])
    _, indices = index.search(np.array(query_embedding), 1)
    return trim(pdf_text)

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

def reply(context, query):
    input_text= f"Question: {query}\nAnswer:"
    response =generator(input_text, max_new_tokens=100, do_sample=True)
    generated_text = response[0]['generated_text']
    generated_text = generated_text.replace(f"Question: {query}", "").replace("Answer:", "").strip()
    return generated_text

def core(query):
    context = pull(query)
    response = reply(context, query)
    return response

query = "What are Emergency calls? "
answer = core(query)
print(query)
print(answer)

