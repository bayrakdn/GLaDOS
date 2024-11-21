import locale
locale.getpreferredencoding = lambda: "UTF-8"


!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --upgrade
!pip install -i https://pypi.pypi.org/simple/ bitsandbytes


!pip install langchain einops accelerate transformers bitsandbytes scipy
!pip install --upgrade langchain==0.1.13  # Replace with a suitable version
from langchain_core.caches import BaseCache


!pip install llama-index llama_hub
!pip install transformers
!pip install sentence_transformers
!pip install pydantic


!pip install llama-index==0.7.21 llama_hub==0.0.19 #try ignoring this one if a problem occurs


!pip install torch==2.2.1


# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# Import torch for datatype attributes
import torch


# Define variable to hold llama2 weights naming
name = "meta-llama/Llama-2-13b-chat-hf"
# Set auth token variable from hugging face
auth_token = "Your Own Unique Token Right Here" #Do not share your unique Token with anyone.


# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(name,
    cache_dir='./model/', use_auth_token=auth_token)



# Create model
model = AutoModelForCausalLM.from_pretrained(name,
    cache_dir='./model/', use_auth_token=auth_token, torch_dtype=torch.float16,
    rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True)




# Setup a user query prompt with the bot name as GLaDOS
user_query = "### User: What does BBC stands for?  \
          \
          ### GLaDOS: "

# Pass the user query prompt to the tokenizer
inputs = tokenizer(user_query, return_tensors="pt").to(model.device)

# Setup the text streamer
streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)

# Generate text based on the user query
output = model.generate(**inputs, streamer=streamer, use_cache=True, max_length=100)




!pip install llama-index-core
!pip install llam-index-legacy



# Actually run the thing
output = model.generate(**inputs, streamer=streamer,
                        use_cache=True, max_length=100)  #max_new_tokens=float('inf')



# Covert the output tokens back to text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)



# Install the correct prompt wrapper
!pip install llama-index

# Import the correct prompt wrapper
from llama_index.prompts import SimpleInputPrompt

# Create a system prompt
system_prompt = """<s>[INST] <<SYS>>
Talk English always.<</SYS>>"""

# Create a query wrapper prompt
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")



# Import the prompt wrapper...but for llama index
!pip install llama-index-llms-huggingface
#from llama_index.core.prompts import SimpleInputPrompt

# Create a system prompt
system_prompt = """<s>[INST] <<SYS>>
Talk English always.<</SYS>>"""

from llama_index.core.prompts import SimpleInputPrompt
# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")



# Complete the query prompt
query_wrapper_prompt.format(query_str='hello') ##oneup




# Import the llama index HF Wrapper
from llama_index.llms.huggingface import HuggingFaceLLM
# Create a HF LLM using the llama index wrapper
llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)




# Bring in embeddings wrapper
from llama_index.embeddings import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings




# Create and dl embeddings instance
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)




# Download PDF Loader

from llama_index.core import VectorStoreIndex, download_loader
from pathlib import Path
PyMuPDFReader = download_loader("PyMuPDFReader")
import fitz  # PyMuPDF uses 'fitz' as the import name

def load_pdf(file_path):
    doc = fitz.open(file_path)
    for page in doc:
        # Process each page
        print(page.get_text())

file_path = '/content/A.pdf'
load_pdf(file_path)

documents = PyMuPDFReader().load(file_path=Path('/content/A.pdf'), metadata=True)



# Bring in stuff to change service context
from llama_index import set_global_service_context
from llama_index import ServiceContext




# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
# And set the service context
set_global_service_context(service_context)



# Create an index - we'll be able to query this in a sec
index = VectorStoreIndex.from_documents(documents)



# Setup index query engine using LLM
query_engine = index.as_query_engine()


# Test out a query in natural
response = query_engine.query("what was the FY2022 return on equity?") #You can ask questions about a file you uploaded earlier.


















