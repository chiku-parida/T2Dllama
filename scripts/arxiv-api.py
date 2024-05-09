'''
This script will download the PDFs of the top N papers from the search results 
and save them to a directory. It will then read the PDFs and return a list of 
Document objects, which can be used to build a VectorStoreIndex.
'''


import os
import hashlib
from typing import List, Optional, Tuple

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import requests

def _hacky_hash(some_string):
  return hashlib.md5(some_string.encode("utf-8")).hexdigest()
def load_data(
    search_query,
    papers_dir = ".papers",
    max_results = 10,
):
    """Search for a topic on Arxiv, download the PDFs of the top results locally, then read them.

    Args:
        search_query (str): A topic to search for (e.g. "Artificial Intelligence").
        papers_dir (Optional[str]): Locally directory to store the papers
        max_results (Optional[int]): Maximum number of papers to fetch.

    Returns:
        List[Document]: A list of Document objects.
    """
    import arxiv

    arxiv_search = arxiv.Search(
        query=search_query,
        id_list=[],
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    search_results = list(arxiv_search.results())
    logging.debug(f"> Successfully fetched {len(search_results)} paperes")

    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir)

    paper_lookup = {}
    for paper in search_results:
        # Hash filename to avoid bad characters in file path
        filename = f"{_hacky_hash(paper.title)}.pdf"
        paper_lookup[filename] = {
            "Title of this paper": paper.title,
            "Authors": (", ").join([a.name for a in paper.authors]),
            "Date published": paper.published.strftime("%m/%d/%Y"),
            "URL": requests.get(paper.entry_id),
            # "summary": paper.summary
        }
        try:
            paper.download_pdf(dirpath=papers_dir, filename=filename)
            logging.debug(f"> Downloading {filename}...")
        except requests.exceptions.HTTPError as e:
            if e.code == 403:
                logging.warning(f"HTTP Error 403 Forbidden occurred for {filename}. Skipping.")
            elif e.code == 404:
                logging.warning(f"HTTP Error 404 Not Found occurred for {filename}. Skipping.")
            else:
                raise  # Re-raise the exception if it's not a 403 error


    def get_paper_metadata(filename):
        return paper_lookup[os.path.basename(filename)]

    arxiv_documents = SimpleDirectoryReader(
        papers_dir,
        file_metadata=get_paper_metadata,
        exclude_hidden=False,  # default directory is hidden ".papers"
    ).load_data()
    # Include extra documents containing the abstracts
    abstract_documents = []
    for paper in search_results:
        d = (
            f"The following is a summary of the paper: {paper.title}\n\nSummary:"
            f" {paper.summary}"
        )
        abstract_documents.append(Document(text=d))
    return arxiv_documents + abstract_documents

documents =load_data('DFT', papers_dir = '.papers', max_results = 1000)
