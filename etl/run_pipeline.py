"""
ETL Pipeline Main Entry Point
Run this script to extract, embed, and store the PDF data.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.vector_store import run_etl_pipeline


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(
        project_root,
        "State-of-the-Cyber-Security-Sector-in-Ireland-2022-Report.pdf",
    )

    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        sys.exit(1)

    persist_dir = os.path.join(project_root, "chroma_db")
    data_dir = os.path.join(project_root, "data")

    run_etl_pipeline(
        pdf_path=pdf_path,
        persist_dir=persist_dir,
        data_dir=data_dir,
    )


if __name__ == "__main__":
    main()
