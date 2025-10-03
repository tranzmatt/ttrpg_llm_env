# TTRPG LLM Project

## System Information
- Python: python3.12 (3.12)
- Environment: ttrpg_llm_env

## Directory Structure
- `ttrpg_pdfs/` - Place your TTRPG PDF files here
- `trained_models/` - Trained models are saved here
- `datasets/` - Generated training datasets
- `logs/` - Training logs and outputs
- `exports/` - Exported models in different formats

## Quick Start
```bash
# 1. Setup environment (if not done already)
python3 setup_environment_updated.py
source ttrpg_llm_env/bin/activate

# 2. Place PDFs in ttrpg_pdfs/ directory

# 3. Run pipeline
python pdf_extractor.py
python train_ttrpg_llm.py
python gm_inference.py
```

## Scripts
- `setup_environment_updated.py` - Environment setup (run first)
- `pdf_extractor.py` - Extract text from PDFs
- `train_ttrpg_llm.py` - Train the LLM
- `gm_inference.py` - Use the trained model

## Troubleshooting
- If you get CUDA out of memory errors, reduce batch_size in train_ttrpg_llm.py
- For Ubuntu 24.04: Uses Python 3.12
- For Ubuntu 22.04: Uses Python 3.10
- Monitor GPU with: nvidia-smi
