# 🎬 UCF50 Video Classification Benchmark - Complete Guide

## 📋 Overview

This comprehensive benchmark suite trains and evaluates **5 different RNN-based architectures** for video classification on the UCF50 dataset, generating beautiful visualizations and detailed reports.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 2. Update Configuration

Edit the script and set your data path:

```python
DATA_PATH = '/path/to/UCF50'  # Update this!
```

### 3. Run Benchmark

**Option A: Single Class Range (10 classes)**
```bash
python benchmark.py --mode single --class_start 0 --class_end 10
```

**Option B: All 5 Class Ranges (Complete benchmark)**
```bash
python benchmark.py --mode all
```

**Option C: Specific Models Only**
```bash
python benchmark.py --mode single --models cnn_lstm conv_lstm late_fusion
```

---

## 🏗️ Architecture Options

| Model | Description | Strengths |
|-------|-------------|-----------|
| **Single Frame** | Classify each frame independently, average results | Fast, simple baseline |
| **Early Fusion** | Concatenate all frames in channel dimension | Learns spatial-temporal features together |
| **Late Fusion** | Average CNN features before classification | Balanced approach |
| **CNN+LSTM** ⭐ | Extract CNN features then process with LSTM | Best performance, captures temporal dynamics |
| **ConvLSTM** | LSTM with convolutional operations inside | Preserves spatial structure throughout |

---

## 📊 What You Get

### Automatic Outputs

For each run, the benchmark generates:

#### 📁 Individual Model Results
- ✅ Training history plots (loss & accuracy curves)
- ✅ Confusion matrices
- ✅ Best model checkpoints (`.pth` files)

#### 📈 Comparison Visualizations
- ✅ Accuracy comparison bar charts
- ✅ Training time analysis
- ✅ Parameters vs Accuracy scatter plot
- ✅ Efficiency score rankings
- ✅ Performance radar charts
- ✅ Beautiful summary tables

#### 📄 Reports
- ✅ CSV file with all metrics
- ✅ JSON file with detailed results
- ✅ **Interactive HTML dashboard** 🎨

### Example Output Structure

```
benchmark_results/
├── run_0_10_20241107_153045/
│   ├── single_frame_best.pth
│   ├── single_frame_history.png
│   ├── single_frame_confusion.png
│   ├── early_fusion_best.pth
│   ├── ... (same for all models)
│   ├── comparison_bars.png
│   ├── efficiency_score.png
│   ├── params_vs_accuracy.png
│   ├── radar_chart.png
│   ├── summary_table.png
│   ├── benchmark_results.csv
│   ├── detailed_results.json
│   └── benchmark_report.html ⭐
└── aggregate_20241107_160245/
    ├── aggregate_results.csv
    └── aggregate_comparison.png
```

---

## 🎯 Key Features

### ✅ Follows All Requirements
- Random seed = 42 (reproducible results)
- Uses first 20 frames only
- Processes 10 classes at a time
- Single codebase for all runs
- Target: ~75% accuracy

### 🎨 Beautiful Visualizations
- Professional matplotlib/seaborn plots
- Color-coded comparisons
- Interactive HTML reports
- Progress bars and animations

### 📊 Comprehensive Metrics
- Validation & Training Accuracy
- Training Time
- Model Parameters
- Efficiency Score (Acc/Time)
- Per-class confusion matrices
- Convergence analysis

### 💾 Memory Efficient
- Works on Google Colab (free tier)
- Batch processing
- Automatic garbage collection
- Configurable batch size

---

## ⚙️ Configuration Options

```python
class Config:
    # Data
    DATA_PATH = 'path/to/UCF50'
    NUM_FRAMES = 20
    IMG_SIZE = (112, 112)
    
    # Class range (change for each run)
    CLASS_START = 0
    CLASS_END = 10
    
    # Training
    BATCH_SIZE = 8  # Reduce if OOM
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Models to train
    MODELS_TO_TRAIN = [
        'single_frame',
        'early_fusion', 
        'late_fusion',
        'cnn_lstm',
        'conv_lstm'
    ]
```

---

## 📝 Running the 5 Required Runs

As per assignment requirements, run the code 5 times with different class ranges:

```bash
# Run 1: Classes 0-9
python benchmark.py --mode single --class_start 0 --class_end 10

# Run 2: Classes 10-19
python benchmark.py --mode single --class_start 10 --class_end 20

# Run 3: Classes 20-29
python benchmark.py --mode single --class_start 20 --class_end 30

# Run 4: Classes 30-39
python benchmark.py --mode single --class_start 30 --class_end 40

# Run 5: Classes 40-49
python benchmark.py --mode single --class_start 40 --class_end 50
```

**OR** simply run everything at once:

```bash
python benchmark.py --mode all
```

This will automatically run all 5 ranges and generate an aggregate report!

---

## 🏆 Expected Results

Based on the architectures, you should expect:

| Model | Expected Accuracy | Training Time | Parameters |
|-------|------------------|---------------|------------|
| CNN+LSTM | **75-80%** ⭐ | ~45 min | ~12M |
| ConvLSTM | **73-78%** | ~50 min | ~15M |
| Late Fusion | **72-76%** | ~38 min | ~12M |
| Early Fusion | **68-74%** | ~42 min | ~25M |
| Single Frame | **65-70%** | ~32 min | ~12M |

---

## 💡 Pro Tips

### For Best Results:
1. **Use CNN+LSTM** - Usually best performance
2. **Enable data augmentation** - Already included
3. **Use transfer learning** - ResNet18 backbone included
4. **Monitor overfitting** - Early stopping implemented

### For Faster Training:
1. Reduce `EPOCHS` to 30-40
2. Reduce `BATCH_SIZE` to 4
3. Use fewer frames (but assignment requires 20)
4. Train only specific models

### For Better Memory Management:
1. Close plots after saving: `plt.close()`
2. Delete unused models: `del model; torch.cuda.empty_cache()`
3. Use smaller `IMG_SIZE` like (96, 96)
4. Process fewer classes at once

---

## 🎨 Viewing Results

### HTML Report (Best!)
Open the generated `benchmark_report.html` in any browser for:
- Interactive tables
- Beautiful visualizations
- Complete summary
- Professional formatting

### Jupyter Notebook
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('benchmark_results/run_0_10_*/benchmark_results.csv')
print(df)

# View images
from IPython.display import Image, display
display(Image('benchmark_results/run_0_10_*/comparison_bars.png'))
```

### Command Line
```bash
# View CSV
cat benchmark_results/run_0_10_*/benchmark_results.csv

# View images (Linux/Mac)
open benchmark_results/run_0_10_*/comparison_bars.png
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
config.BATCH_SIZE = 4  # Reduce from 8
config.IMG_SIZE = (96, 96)  # Reduce from (112, 112)
```

### Video Loading Errors
Make sure videos are in `.avi` format and organized as:
```
UCF50/
├── class_0/
│   ├── video1.avi
│   ├── video2.avi
├── class_1/
...
```

### Slow Training
- Use GPU (CUDA)
- Reduce epochs
- Train fewer models
- Use smaller images

---

## 📚 Assignment Submission

### What to Submit:

1. **Code Files**
   - `benchmark.py` (main script)
   - Any helper scripts

2. **Results** (for each of 5 runs)
   - `benchmark_results.csv`
   - `benchmark_report.html`
   - Key visualization PNGs

3. **Video Explanation**
   - Explain your code
   - Show results
   - Discuss findings

4. **Bonus** (15% extra credit)
   - All 5 architectures implemented ✅
   - Comprehensive comparison ✅
   - Beautiful visualizations ✅

---

## 🎓 Understanding the Code

### Training Pipeline
```
Load Data → Split Train/Val → Create DataLoaders → 
Train Each Model → Save Best Checkpoint → 
Generate Visualizations → Create Reports
```

### Key Components
- **Dataset Class**: Loads videos frame-by-frame
- **Model Classes**: 5 different architectures
- **Training Loop**: Standard PyTorch training
- **Evaluation**: Metrics & confusion matrices
- **Visualization**: Matplotlib/Seaborn plots

---

## 📞 Need Help?

Check these first:
1. GPU available? `torch.cuda.is_available()`
2. Data path correct? Check file structure
3. Dependencies installed? Run pip install again
4. Memory issues? Reduce batch size

---

## 🌟 Bonus Features

This implementation includes:
- ✨ Transfer learning (pretrained ResNet18)
- ✨ Data augmentation
- ✨ Learning rate scheduling
- ✨ Early stopping
- ✨ Confusion matrices
- ✨ HTML dashboard
- ✨ Aggregate analysis (all 5 runs)
- ✨ Progress bars
- ✨ Automatic checkpoint saving

---

## 🎉 Happy Benchmarking!

You now have a complete, professional-grade video classification benchmark suite that will impress your professor and earn you that 15% bonus! 🚀

**Remember**: The code handles everything automatically - just update the data path and run!