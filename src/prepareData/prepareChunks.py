import re
from typing import Any, Dict, List
from langchain_text_splitters import MarkdownHeaderTextSplitter
import os


def parse_policy(file_path: str = "Data\\Returns and Shipping Policy.txt") -> List[Dict[str, Any]]:
    """
    Splits policy by markdown headers so each section is an isolated chunk.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # FIX: Ensure there is a space after any markdown hashes 
    # Transforms "#Shipping Information" into "# Shipping Information"
    content = re.sub(r'^(#+)(?=[^\s#])', r'\1 ', content, flags=re.MULTILINE)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Section"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    splits = markdown_splitter.split_text(content)

    chunks = []
    for split in splits:
        cleaned_text = split.page_content.strip()
    
    # Skip empty chunks or standalone titles
        if len(cleaned_text) < 30:
            continue

        section_hierarchy = [
             val for key, val in split.metadata.items() if key.startswith("Header") or key == "Section"
         ]
        section_name = " > ".join(section_hierarchy) if section_hierarchy else "General Policy"

        chunks.append({
            "text": split.page_content.strip(),
            "source": file_path,
            "section": section_name,
            "type": "policy"
        })
    return chunks


def parse_help_articles(file_path: str = "Data\\Help Articles.txt") -> List[Dict[str, Any]]:
    """
    Parses help articles formatted as '# Title: Content' on individual lines.
    """
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = []
    
    for line in content.splitlines():
        line = line.strip()
        
        # Skip empty lines or lines that don't start with '#'
        if not line or not line.startswith("# "):
            continue
            
        # Remove the leading "# "
        line_content = line[2:].strip()
        
        # Split at the first colon to separate the title from the body
        if ":" in line_content:
            title, body = line_content.split(":", 1)
            title = title.strip()
            
            chunks.append({
                "text": line_content, 
                "source": file_path,
                "section": title,
                "type": "article"
            })
        else:
            # Fallback just in case a line is missing a colon
            chunks.append({
                "text": line_content,
                "source": file_path,
                "section": line_content[:50],
                "type": "article"
            })
            
    return chunks

def parse_reviews(file_path: str = "Data\\customer_reviews_database.md") -> List[Dict[str, Any]]:
    """
    Parses a markdown file of reviews structured with product headers and multi-line bullet points.
    """
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = []
    
    # Split the document by markdown headers (e.g., '## Product P01')
    sections = re.split(r'(?m)^##\s+', content)
    
    for section in sections:
        # Skip empty sections or the top-level "# Customer Reviews Database"
        if not section.strip() or section.startswith("# Customer"):
            continue
            
        lines = section.split('\n')
        current_product = lines[0].strip()
        
        current_meta = ""
        current_text = []
        
        # Loop through the remaining lines in this product section
        for line in lines[1:]:
            if line.startswith('* **'):
                # Save the previous review before starting a new one
                if current_meta:
                    chunks.append({
                        "text": f"Product: {current_product} | {current_meta.strip('* ')}: {' '.join(current_text).strip()}",
                        "source": file_path,
                        "section": current_product,
                        "type": "review"
                    })
                
                # Start tracking the new review
                current_meta = line.strip()
                current_text = []
            elif line.strip() and current_meta:
                # Append the review description lines
                current_text.append(line.strip())
                
        # Save the very last review in the section
        if current_meta:
            chunks.append({
                "text": f"Product: {current_product} | {current_meta.strip('* ')}: {' '.join(current_text).strip()}",
                "source": file_path,
                "section": current_product,
                "type": "review"
            })
            
    return chunks

def get_all_chunks() -> List[Dict[str, Any]]:
    """Loads, parses, and returns all validated document chunks."""
    all_chunks = []
    
    # 1. Parse all three files
    all_chunks.extend(parse_policy("Data\\Returns and Shipping Policy.txt"))
    all_chunks.extend(parse_reviews("Data\\customer_reviews_database.md"))
    all_chunks.extend(parse_help_articles("Data\\Help Articles.txt"))
    
    # 2. Filter out the empty preamble chunks (like the policy title)
    final_chunks = [
        c for c in all_chunks 
        if len(c['text'].strip()) >= 30 and c['text'] != '# AuraTech Returns & Shipping Policy'
    ]
    return final_chunks
