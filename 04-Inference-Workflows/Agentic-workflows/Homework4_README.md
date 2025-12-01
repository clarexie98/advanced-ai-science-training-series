# Homework -- ClareXie Multi-Agent System 
## Overview

This homework implements a **Scientific Research Multi-Agent System** that demonstrates:
- Custom tools for scientific data gathering (arXiv, PubChem, UniProt)
- Multi-agent architecture 
- Integration with ALCF Inference Endpoints
- Structured workflow using LangGraph

### **Two Agents:**
1. **Research Agent** - Gathers information using various tools
2. **Synthesis Agent** - Formats findings into structured summaries

### **Four Custom Tools:**
1. `search_arxiv` - Search scientific papers on arXiv
2. `lookup_molecular_properties` - Query PubChem for molecular data
3. `search_protein_database` - Search UniProt for protein information
4. `calculate_simple_statistics` - Perform statistical calculations

## Files 

```
clarexie_tools.py         # Custom tool implementations
clarexie_multi_agent.py   # Multi-agent workflow
requirements.txt          # Python dependencies
```

## Step-by-Step Execution Instructions

### Step 1: Navigate to Directory

```bash
cd /home/clarexie/2025/advanced-ai-science-training-series/04-Inference-Workflows/Agentic-workflows
```

### Step 2: Ensure Environment is Activated

```bash
source 0_activate_env.sh
```

### Step 3: Install Additional Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `arxiv` - For arXiv paper search
- `pubchempy` - For molecular property lookup
- `requests` - For HTTP requests to UniProt

### Step 4: Verify Authentication Token

Make sure you have authenticated with ALCF endpoints (you should have done this already):

```bash
# Check if inference_auth_token.py exists
ls -la inference_auth_token.py

# If needed, re-authenticate
python inference_auth_token.py --force authenticate
```

### Step 5: Run the Multi-Agent System

```bash
python clarexie_multi_agent.py
```

You should see:

1. **Initialization messages:**
   ```
   🔐 Authenticating with ALCF Inference Endpoints...
   🤖 Initializing language model...
   🏗️  Building multi-agent workflow...
   ```

2. **For each query, you'll see:**
   - Human message (the query)
   - AI messages showing tool calls
   - Tool messages showing results
   - Final synthesis message with structured summary

3. **Three example queries are included:**
   - Query 1: Papers about ML for drug discovery + aspirin properties
   - Query 2: Hemoglobin protein info + statistics calculation
   - Query 3: Caffeine properties + brain effect papers


### Workflow Flow:

```
START 
  ↓
research_agent (decides which tools to call)
  ↓
  ├─→ tools (if tool calls needed)
  │     ↓
  │   research_agent (uses tool results)
  │     ↓
  └─→ synthesis_agent (formats final output)
        ↓
      END
```

### Example:

For query "Find papers about ML for drug discovery and tell me about aspirin":

1. `research_agent` analyzes the query
2. Calls `search_arxiv("machine learning drug discovery")`
3. Calls `lookup_molecular_properties("aspirin")`
4. `research_agent` receives tool results
5. `synthesis_agent` formats everything into a structured summary

