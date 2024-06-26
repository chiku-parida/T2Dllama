#################################################################
#  Load your index and embeddings from the directory            #
#  initialize the query engine                                  #
#  Authors: Chiku Parida and Martin H. Peterson                 #
#################################################################



import torch
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.settings import Settings
from llama_index.core.settings import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core import StorageContext, load_index_from_storage



import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#the LLM model using llama CPP
llm = LlamaCPP(
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    model_path=None, #if you have downloaded the model then give the model path and comment the above tag.
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900, #default 4096 recommended to use a less value to make it faster
    generate_kwargs={},
    # set to at least 1 to use GPU as CPUs are very slow
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)


#create necessary embeddings using langchain and huggingface
embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="thenlper/gte-large")
)

#declare some global settings
callback_manager = CallbackManager()
Settings.llm = llm
Settings.embed_model = embed_model
Settings.callback_manager = callback_manager

#load the vector index database from the directory
storage_context = StorageContext.from_defaults(persist_dir="/home/charles/phd_project/LLM_H/vindex2")
index = load_index_from_storage(storage_context)

#initialize your query engine
query_engine = index.as_query_engine()
response = query_engine.query("How to calculate open circuit voltage?")
#print the response
print(response)
