import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "notebook"))

from fastmcp import FastMCP
from drive_ingestion import ingest_folder

mcp = FastMCP("Research Paper Drive Ingester")


@mcp.tool()
def ingest_papers(folder_id: str) -> dict:
    """Download all PDFs from a public Google Drive folder, chunk them, embed them, and store in the vector database."""
    return ingest_folder(folder_id)


if __name__ == "__main__":
    mcp.run()
