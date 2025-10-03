# TTRPG PDF Text Extraction Script
# Extract text from TTRPG PDFs and prepare for LLM training

import fitz  # PyMuPDF
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using PyMuPDF for better formatting retention
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as string
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            # Add page separator for context
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def clean_ttrpg_text(text: str) -> str:
    """
    Clean extracted text for better training quality
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Split into lines for processing
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip likely headers/footers (very short lines, just numbers, etc.)
        if len(line) < 3:
            continue
            
        # Skip lines that are just page numbers
        if line.isdigit() and len(line) < 4:
            continue
            
        # Skip common PDF artifacts
        if line.lower() in ['table of contents', 'index', 'appendix']:
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def split_into_sections(text: str) -> List[str]:
    """
    Split text into meaningful sections for training
    
    Args:
        text: Cleaned text
        
    Returns:
        List of text sections
    """
    # Common TTRPG section markers
    section_markers = [
        r'chapter \d+',
        r'section \d+',
        r'\d+\.\d+',
        r'rule:',
        r'mechanics:',
        r'combat',
        r'spells',
        r'equipment',
        r'character creation',
        r'game master',
        r'abilities',
        r'skills'
    ]
    
    sections = []
    current_section = ""
    
    for line in text.split('\n'):
        # Check if this line starts a new section
        is_new_section = any(re.search(marker, line.lower()) for marker in section_markers)
        
        if is_new_section and current_section:
            # Save previous section if it has substantial content
            if len(current_section.strip()) > 100:
                sections.append(current_section.strip())
            current_section = line + "\n"
        else:
            current_section += line + "\n"
    
    # Add the last section
    if len(current_section.strip()) > 100:
        sections.append(current_section.strip())
    
    return sections

def create_training_examples(text: str, source_name: str) -> List[Dict[str, Any]]:
    """
    Convert text into instruction-following format for GM LLM
    
    Args:
        text: Cleaned text from PDF
        source_name: Name of the source PDF/rulebook
        
    Returns:
        List of training examples
    """
    examples = []
    sections = split_into_sections(text)
    
    # GM-specific prompts for TTRPG context
    gm_prompts = [
        "Explain this rule to a new player:",
        "How should I interpret this rule as a GM?",
        "What's the mechanical effect of this rule?",
        "How do I handle this situation in-game?",
        "What are the key points of this rule?",
        "How does this rule affect gameplay?",
        "What should players know about this?",
        "As a GM, how do I implement this rule?"
    ]
    
    for i, section in enumerate(sections):
        if len(section) > 150:  # Only use substantial sections
            # Create multiple examples per section with different prompts
            for prompt in gm_prompts[:3]:  # Use first 3 prompts to avoid too much repetition
                example = {
                    "instruction": prompt,
                    "input": f"From {source_name}: {section[:300]}...",  # First 300 chars as context
                    "output": section,
                    "source": source_name,
                    "section_id": i
                }
                examples.append(example)
    
    return examples

def create_conversational_format(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert examples to conversational format for chat-based training
    
    Args:
        examples: List of instruction examples
        
    Returns:
        List of conversational examples
    """
    conversational_examples = []
    
    for example in examples:
        conv_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Game Master for tabletop RPGs. You have deep knowledge of game rules, mechanics, and how to create engaging gameplay experiences. Provide clear, helpful explanations of rules and practical GM advice."
                },
                {
                    "role": "user",
                    "content": f"{example['instruction']}\n\n{example['input']}"
                },
                {
                    "role": "assistant",
                    "content": example['output']
                }
            ],
            "source": example['source'],
            "section_id": example.get('section_id', 0)
        }
        conversational_examples.append(conv_example)
    
    return conversational_examples

def process_ttrpg_pdfs(pdf_directory: str, output_file: str) -> List[Dict[str, Any]]:
    """
    Process all PDFs in directory and create training dataset
    
    Args:
        pdf_directory: Directory containing PDF files
        output_file: Output JSON file path
        
    Returns:
        List of training examples
    """
    if not os.path.exists(pdf_directory):
        print(f"Directory {pdf_directory} does not exist!")
        return []
    
    all_training_data = []
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file.name}...")
        
        # Extract text from PDF
        raw_text = extract_text_from_pdf(str(pdf_file))
        
        if not raw_text:
            print(f"  Warning: No text extracted from {pdf_file.name}")
            continue
        
        # Clean the text
        cleaned_text = clean_ttrpg_text(raw_text)
        print(f"  Extracted {len(cleaned_text)} characters of text")
        
        # Create training examples
        examples = create_training_examples(cleaned_text, pdf_file.stem)
        print(f"  Created {len(examples)} training examples")
        
        all_training_data.extend(examples)
    
    # Convert to conversational format
    conversational_data = create_conversational_format(all_training_data)
    
    # Save both formats
    print(f"\nSaving {len(all_training_data)} instruction examples...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_training_data, f, indent=2, ensure_ascii=False)
    
    conversational_output = output_file.replace('.json', '_conversational.json')
    print(f"Saving {len(conversational_data)} conversational examples...")
    with open(conversational_output, 'w', encoding='utf-8') as f:
        json.dump(conversational_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset creation complete!")
    print(f"Total examples: {len(all_training_data)}")
    print(f"Files saved: {output_file} and {conversational_output}")
    
    return all_training_data

def main():
    """Main function to run PDF processing"""
    # Configuration
    pdf_directory = "./ttrpg_pdfs"  # Change this to your PDF directory
    output_file = "./datasets/ttrpg_training_data.json"
    
    print("TTRPG PDF Processing Script")
    print("=" * 40)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Process PDFs
    training_data = process_ttrpg_pdfs(pdf_directory, output_file)
    
    if training_data:
        print("\nNext steps:")
        print("1. Review the generated JSON files")
        print("2. Run the training script with these datasets")
        print("3. Adjust parameters based on your GPU memory")
    else:
        print("\nNo training data was created. Please check:")
        print(f"1. PDF directory exists: {pdf_directory}")
        print("2. PDF files are readable")
        print("3. PDFs contain extractable text")

if __name__ == "__main__":
    main()
