import streamlit as st

# Define Streamlit app
st.title("Document Search App")
from llama_index.core import StorageContext, load_index_from_storage
import torch

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
llm = LlamaCPP(
    #model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    model_path='/home/charles/phd_project/LLM_H/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False,
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

from llama_index.core.settings import Settings

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="thenlper/gte-large")
)

from llama_index.core.settings import Settings
from llama_index.core.callbacks import CallbackManager
callback_manager = CallbackManager()
Settings.llm = llm
Settings.embed_model = embed_model
Settings.callback_manager = callback_manager
storage_context = StorageContext.from_defaults(persist_dir="/home/charles/phd_project/LLM_H/vindex2")
index = load_index_from_storage(storage_context)
# Add search input
query = st.text_input("Enter your query:")
query_engine = index.as_query_engine()
# Perform search when the user submits the query
if st.button("Search"):
    results = query_engine.query(query)
    st.write(results)

