# Quick Reference: SambaNova Homework

## 🎯 Goal
Compare GPT-OSS-120B performance on Metis (SambaNova) vs Sophia (GPU)

## 📁 Files Created
- `compare_metis_sophia.py` - Benchmark script
- `Homework5_README.md` - Detailed instructions
- `Homework5_Analysis_TEMPLATE.md` - Template for your analysis
- `run_homework.sh` - Quick start script

## 🚀 Quick Start (3 commands)

```bash
cd /home/clarexie/2025/advanced-ai-science-training-series/05-AITestbed/SambaNova
./run_homework.sh
# Follow prompts, then wait 2-3 minutes
```

## 📊 What It Does

1. Tests 5 scientific Q&A prompts
2. Measures latency and throughput on both platforms
3. Generates comparison report
4. Saves results to JSON file

## 📝 After Running

1. Review console output
2. Open generated JSON file
3. Copy `Homework5_Analysis_TEMPLATE.md` to `Homework5_Analysis.md`
4. Fill in your results and analysis
5. Commit to GitHub

## 🔧 Manual Run

```bash
cd /home/clarexie/2025/advanced-ai-science-training-series/05-AITestbed/SambaNova
python compare_metis_sophia.py
```

## 📈 Expected Results

- **Metis**: Lower latency, higher throughput (inference-optimized)
- **Sophia**: Slightly higher latency (general-purpose GPUs)
- **Speedup**: Metis typically 1.2x - 2.0x faster

## 🐛 Troubleshooting

**Token expired?**
```bash
cd /home/clarexie/2025/advanced-ai-science-training-series/05-AITestbed
python inference_auth_token.py authenticate --force
```

**Import errors?**
```bash
# Make sure you're in the right directory
cd /home/clarexie/2025/advanced-ai-science-training-series/05-AITestbed/SambaNova
pwd  # Should show .../05-AITestbed/SambaNova
```

**Connection timeout?**
- Check network connection
- Try reducing prompts in script
- Add longer delays

## 📚 Key Concepts

**Metis (SambaNova)**
- RDU (Reconfigurable Dataflow Unit) processors
- Dataflow architecture
- Inference-optimized

**Sophia (GPU)**
- NVIDIA A100 GPUs  
- CUDA architecture
- Training + Inference

## 💡 Tips

- Run quick test first (5 prompts)
- For full analysis, increase to 15 prompts
- Compare actual responses for quality
- Document observations in analysis

## ✅ Checklist

- [ ] Run benchmark script
- [ ] Review results
- [ ] Fill in analysis template
- [ ] Add sample responses
- [ ] Explain performance differences
- [ ] Make recommendations
- [ ] Commit to GitHub

## 📞 Need Help?

Check `Homework5_README.md` for detailed instructions!
