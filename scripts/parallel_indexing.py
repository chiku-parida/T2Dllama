#################################################################
#  Parallel Indexing script if your are using CPU for indexing  #
#  For fast indexing GPU recomended over CPU                    #
#################################################################
import multiprocessing
from llama_index.core.settings import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext


def index_documents(documents):
    # Initialize settings
    llm = Settings.llm
    embed_model = Settings.embed_model
    callback_manager = CallbackManager()
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.callback_manager = callback_manager

    index = VectorStoreIndex.from_documents(documents)
    return index

def parallel_indexing(documents, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
        print(f"Using {num_processes} processes for parallel indexing.")   
    # Split documents into chunks for parallel processing
    chunk_size = len(documents) // num_processes
    document_chunks = [documents[i:i+chunk_size] for i in range(0, len(documents), chunk_size)]
    
    # Create a multiprocessing pool and index documents in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        indexes = pool.map(index_documents, document_chunks)
    
    # Merge indexes from different processes
    merged_index = merge_indexes(indexes)
    
    return merged_index

def merge_indexes(indexes):
    merged_index = []
    for index in indexes:
        merged_index.extend(index)
    
    return merged_index
