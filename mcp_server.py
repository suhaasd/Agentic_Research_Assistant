import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "notebook"))

from fastmcp import FastMCP
from drive_ingestion import ingest_folder, fetch_and_ingest

mcp = FastMCP("Research Paper Ingester")


@mcp.tool()
def ingest_papers(folder_id: str) -> dict:
    """Download all PDFs from a public Google Drive folder, chunk them, embed them, and store in the vector database."""
    return ingest_folder(folder_id)


@mcp.tool()
def fetch_paper(url: str) -> dict:
    """Fetch a single PDF from an arXiv URL, DOI, or direct PDF link, chunk it, embed it, and store in the vector database."""
    return fetch_and_ingest(url)


if __name__ == "__main__":
    mcp.run()
