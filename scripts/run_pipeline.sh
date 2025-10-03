#!/bin/bash

# TTRPG LLM Training - Quick Start Script
# This script runs the complete pipeline from PDF processing to model training

echo "🎲 TTRPG LLM Training Pipeline"
echo "=============================="

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "❌ Virtual environment not activated!"
    echo "Please run: source ttrpg_llm_env/bin/activate"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Check for PDFs
if [ ! -d "./ttrpg_pdfs" ] || [ -z "$(ls -A ./ttrpg_pdfs/*.pdf 2>/dev/null)" ]; then
    echo "❌ No PDF files found in ./ttrpg_pdfs/"
    echo "Please place your TTRPG PDF files in the ./ttrpg_pdfs/ directory"
    exit 1
fi

PDF_COUNT=$(ls -1 ./ttrpg_pdfs/*.pdf 2>/dev/null | wc -l)
echo "✅ Found $PDF_COUNT PDF files to process"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🔥 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  nvidia-smi not found. Training will use CPU (much slower)"
fi

echo ""
echo "Pipeline Steps:"
echo "1. Extract text from PDFs"
echo "2. Train TTRPG GM model"
echo "3. Test the trained model"
echo ""

# Ask for confirmation
read -p "Continue with training pipeline? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Pipeline cancelled."
    exit 0
fi

echo ""
echo "🔄 Starting pipeline..."

# Step 1: Extract text from PDFs
echo ""
echo "📖 Step 1: Extracting text from PDFs..."
python pdf_extractor.py

if [ $? -ne 0 ]; then
    echo "❌ PDF extraction failed!"
    exit 1
fi

# Check if training data was created
if [ ! -f "./datasets/ttrpg_training_data_conversational.json" ]; then
    echo "❌ Training data not found!"
    exit 1
fi

echo "✅ PDF extraction completed"

# Step 2: Train the model
echo ""
echo "🧠 Step 2: Training TTRPG GM model..."
echo "⚠️  This may take several hours depending on your dataset size and GPU"
echo "⚠️  Monitor with nvidia-smi in another terminal"

python train_ttrpg_llm.py

if [ $? -ne 0 ]; then
    echo "❌ Model training failed!"
    echo ""
    echo "Troubleshooting tips:"
    echo "- Check GPU memory with nvidia-smi"
    echo "- Reduce batch_size or max_seq_length in train_ttrpg_llm.py"
    echo "- Try a smaller model like phi3-mini"
    exit 1
fi

echo "✅ Model training completed"

# Step 3: Test the model
echo ""
echo "🧪 Step 3: Testing the trained model..."

if [ -d "./trained_ttrpg_gm" ]; then
    echo "Model found at: ./trained_ttrpg_gm"
    echo ""
    echo "Would you like to test the model now? (y/n)"
    read -p "> " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python gm_inference.py
    fi
else
    echo "⚠️  Trained model directory not found"
fi

echo ""
echo "🎉 Pipeline completed!"
echo ""
echo "Your TTRPG GM model is ready!"
echo "To use it: python gm_inference.py"
echo ""
echo "Files created:"
echo "- ./ttrpg_training_data.json (instruction format)"
echo "- ./ttrpg_training_data_conversational.json (chat format)"
echo "- ./trained_ttrpg_gm/ (your trained model)"
echo ""
echo "Enjoy your AI Game Master! 🎲"
