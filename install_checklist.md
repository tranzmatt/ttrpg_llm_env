# Drop-in Replacement Installation Checklist

## Step 1: Backup Your Current Files (Optional but Recommended)
```bash
cd your_project_directory
cp scripts/pdf_extractor.py scripts/pdf_extractor.py.backup
cp scripts/train_ttrpg_llm.py scripts/train_ttrpg_llm.py.backup
```

## Step 2: Replace the Files

### Option A: Copy from Claude Artifacts
1. Open the artifact "pdf_extractor.py (Fixed - Drop-in)"
2. Copy all the code
3. Replace the content of `scripts/pdf_extractor.py`
4. Open the artifact "train_ttrpg_llm.py (Fixed - Drop-in)"  
5. Copy all the code
6. Replace the content of `scripts/train_ttrpg_llm.py`

### Option B: Direct Copy (if files are accessible)
```bash
# After saving the artifact code to files:
cp /path/to/downloaded/pdf_extractor.py scripts/
cp /path/to/downloaded/train_ttrpg_llm.py scripts/
```

## Step 3: Verify the Changes
```bash
# Check that validation function exists in pdf_extractor.py
grep -n "validate_training_data" scripts/pdf_extractor.py

# Check that val_split parameter exists in train script
grep -n "val_split" scripts/train_ttrpg_llm.py
```

Expected output:
```
# pdf_extractor.py should show function definition around line 180
# train_ttrpg_llm.py should show multiple occurrences
```

## Step 4: Test with a Small Dataset (Recommended)
```bash
# 1. Activate your environment
source ttrpg_llm_env/bin/activate

# 2. If you have PDFs, run extraction
python scripts/pdf_extractor.py

# You should now see data validation output:
# ==================================================
# Validating Data Quality
# ==================================================
# Total examples: XXX
# ✓ Data structure validation passed
```

## Step 5: Test Training with Small Sample
Edit `scripts/train_ttrpg_llm.py` temporarily:
```python
# Around line 325, modify config for testing:
config = {
    # ... existing settings ...
    "max_samples": 50,   # Small test
    "val_split": 0.2,    # 20% validation
    "test_model_after_training": True
}
```

Run training test:
```bash
python scripts/train_ttrpg_llm.py
```

Expected output should include:
```
Training samples: 40
Validation samples: 10

[... training progress with evaluation ...]

✓ Training completed successfully
Best validation loss: X.XXXX
```

## Step 6: Run Full Training
Once the test works, restore config:
```python
config = {
    "model_name": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "data_file": "./datasets/ttrpg_training_data_conversational.json",
    "output_dir": "./trained_ttrpg_gm",
    "max_seq_length": 2048,
    "max_samples": None,      # Use all data
    "val_split": 0.1,         # 10% validation
    "test_model_after_training": True
}
```

Then run full training:
```bash
python scripts/train_ttrpg_llm.py
```

## Step 7: Verify Everything Still Works
```bash
# Your Makefile should still work:
make train

# Your inference should still work:
python scripts/gm_inference.py
```

## Troubleshooting

### If you see: "NameError: name 'validate_training_data' is not defined"
- The pdf_extractor.py wasn't updated correctly
- Re-copy the entire file from the artifact

### If you see: "TypeError: prepare_dataset() got an unexpected keyword argument 'val_split'"
- The train_ttrpg_llm.py wasn't updated correctly
- Re-copy the entire file from the artifact

### If training crashes with "eval_dataset" error
- Your datasets variable is not a dict
- Verify train_ttrpg_llm.py was fully replaced

### If you want to revert
```bash
cp scripts/pdf_extractor.py.backup scripts/pdf_extractor.py
cp scripts/train_ttrpg_llm.py.backup scripts/train_ttrpg_llm.py
```

## What Changed Summary

### pdf_extractor.py
- ✅ Added: `validate_training_data()` function
- ✅ Modified: `process_ttrpg_pdfs()` to call validation
- ✅ Everything else: UNCHANGED

### train_ttrpg_llm.py  
- ✅ Modified: `prepare_dataset()` - now returns dict with train/val
- ✅ Modified: `train_ttrpg_model()` - accepts dict, supports validation
- ✅ Modified: `export_model()` - better error handling
- ✅ Added: `cleanup_memory()` function
- ✅ Modified: `main()` - better error handling
- ✅ Everything else: UNCHANGED

## Verification Checklist

After installation, verify:
- [ ] PDF extraction shows data validation output
- [ ] Training shows "Training samples: X, Validation samples: Y"
- [ ] Training shows evaluation losses during training
- [ ] Training shows "Best validation loss: X.XXXX" at end
- [ ] Export handles errors gracefully
- [ ] Inference still works with trained model
- [ ] Makefile commands still work

## You're Done! 🎉

Your code now has:
- ✅ Data quality validation before training
- ✅ Train/validation split to detect overfitting
- ✅ Better error messages and handling
- ✅ Robust model export
- ✅ Memory cleanup

And everything else works exactly as before!
