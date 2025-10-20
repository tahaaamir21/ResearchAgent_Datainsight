# 📋 Project Overview - Research Intelligence Platform

## 🎯 What is This Project?

This is a **Multi-Agent AI Research Platform** that autonomously discovers, validates, synthesizes, and analyzes academic research from multiple sources. Think of it as your AI research assistant that can:

1. **Search** ArXiv, Semantic Scholar, and the web simultaneously
2. **Validate** sources for credibility and relevance
3. **Synthesize** findings into knowledge graphs
4. **Analyze** using ML (topic modeling, clustering, predictions)
5. **Generate** comprehensive research reports
6. **Monitor** emerging trends and new research

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                            │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Streamlit UI    │   OR    │  Command Line    │          │
│  │  (Web Browser)   │         │  (Terminal)      │          │
│  └────────┬─────────┘         └────────┬─────────┘          │
└───────────┼──────────────────────────┬─┼────────────────────┘
            │                          │ │
            └──────────────┬───────────┘ │
                           ▼             ▼
┌──────────────────────────────────────────────────────────────┐
│              LANGGRAPH ORCHESTRATOR                           │
│  (Coordinates all agents via state machine)                  │
└──────────────────┬───────────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌─────────┐  ┌─────────┐  ┌──────────┐
│Discovery│  │Validation│ │   RAG    │
│ Agent   │→ │  Agent   │→│  Agent   │
└─────────┘  └─────────┘  └──────────┘
                               │
    ┌──────────────────────────┴──────────┐
    ▼              ▼              ▼        ▼
┌──────────┐  ┌────────┐  ┌─────────┐  ┌──────────┐
│Synthesis │  │   ML   │  │Reporter │  │Monitoring│
│  Agent   │→ │ Agent  │→ │  Agent  │→ │  Agent   │
└──────────┘  └────────┘  └─────────┘  └──────────┘
     │            │            │             │
     └────────────┴────────────┴─────────────┘
                    ▼
        ┌───────────────────────┐
        │  OUTPUTS & STORAGE    │
        ├───────────────────────┤
        │ • Research Reports    │
        │ • Knowledge Graphs    │
        │ • Vector Database     │
        │ • ML Analysis         │
        └───────────────────────┘
```

## 📁 Directory Structure Explained

```
datascienceproj/
│
├── 1_Documentation/          📚 Documentation & problem statements
│   └── Your project docs go here
│
├── 2_Data/                   💾 Raw and processed data
│   ├── raw/                  Raw downloaded research data
│   └── processed/            Validated, cleaned sources
│
├── 3_Notebooks/              📓 Jupyter notebooks for experiments
│   └── Testing and analysis notebooks
│
├── 4_Src_Code/               🔧 Main source code
│   ├── agents/               Individual agent modules
│   │   ├── __init__.py
│   │   └── state.py          Shared state definition
│   ├── agentic_ai_pipeline.py  ⭐ MAIN PIPELINE (all agents)
│   └── README.md             Code documentation
│
├── 5_Pipeline/               🔄 Airflow orchestration
│   ├── airflow_dag.py        Production DAG
│   └── README.md             Airflow setup guide
│
├── 6_Models/                 🤖 ML models & vector stores
│   └── vectorstore/          ChromaDB vector database
│
├── 7_Results/                📊 Generated outputs
│   ├── *.png                 Knowledge graph visualizations
│   └── *.txt                 Research reports
│
├── 8_Demo/                   🎬 Demo materials
│   ├── screenshots/          UI screenshots
│   └── demo_video.mp4        (You'll record this)
│
├── 9_Deployment/             🚀 Deployment files
│   ├── app.py                ⭐ STREAMLIT WEB UI
│   └── README.md             Deployment guide
│
├── requirements.txt          📦 Python dependencies
├── README.md                 📖 Main project documentation
├── PROJECT_OVERVIEW.md       📋 This file
├── run_streamlit.bat         ▶️  Windows launch script
├── run_streamlit.sh          ▶️  Linux/Mac launch script
│
└── (Airflow files - if using Docker orchestration)
    ├── docker-compose-airflow.yml
    ├── AIRFLOW_README.md
    └── setup_airflow.bat
```

## 🚀 How to Use

### Quick Start (3 Steps)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Keys**
   Create `.env` file:
   ```env
   GROQ_API_KEY=your_key_here
   SERPAPI_KEY=your_optional_key
   ```

3. **Launch UI**
   ```bash
   # Windows
   run_streamlit.bat
   
   # Linux/Mac
   chmod +x run_streamlit.sh
   ./run_streamlit.sh
   ```

### Usage Modes

#### Mode 1: Web UI (Recommended ⭐)
```bash
streamlit run 9_Deployment/app.py
```
- **Best for**: Interactive research, visualizations
- **Port**: http://localhost:8501
- **Features**: Real-time progress, downloadable reports

#### Mode 2: Command Line
```bash
python 4_Src_Code/agentic_ai_pipeline.py
```
- **Best for**: Automated runs, scripting
- **Output**: Terminal output + saved files

#### Mode 3: Airflow (Production)
```bash
docker-compose -f docker-compose-airflow.yml up -d
```
- **Best for**: Scheduled runs, production deployment
- **Port**: http://localhost:8080
- **Features**: Scheduling, monitoring, alerts

## 🧠 How It Works (Step-by-Step)

### Step 1: User Input
- Enter research query (e.g., "Transformer models for computer vision")
- Select depth (quick/standard/deep)

### Step 2: Discovery Agent 🔍
- Searches ArXiv for academic papers
- Queries Semantic Scholar for peer-reviewed research
- Searches web via SERPAPI (if configured)
- Deduplicates results

### Step 3: Validation Agent ✅
- Scores each source (0-100) based on:
  - Source type (peer-reviewed vs preprint)
  - Citation count
  - Recency
  - Content quality
- Checks relevance using LLM
- Filters out low-quality sources

### Step 4: RAG Agent 🧠
- Creates vector embeddings using HuggingFace
- Stores in ChromaDB vector database
- Enables semantic search across sources

### Step 5: Synthesis Agent 🧬
- Extracts key concepts
- Identifies consensus findings
- Detects contradictions
- Finds research gaps
- Builds knowledge graph (nodes + edges)

### Step 6: ML Agent 🤖
- **Topic Modeling**: Discovers hidden topics using LDA
- **Clustering**: Groups similar papers using K-means
- **Citation Prediction**: Predicts future impact
- **Quality Scoring**: ML-based assessment

### Step 7: Reporter Agent 📊
- Generates executive summary
- Creates detailed research report
- Builds citation map
- Prepares visualizations

### Step 8: Monitoring Agent 📡
- Sets up alert triggers
- Analyzes trends
- Tracks emerging topics
- Identifies key authors

### Step 9: Output 📁
- **Text Report**: Comprehensive analysis (saved to `7_Results/`)
- **Knowledge Graph**: Visual PNG (saved to `7_Results/`)
- **Vector DB**: Semantic search database (`6_Models/vectorstore/`)
- **JSON Data**: Structured data for further processing

## 🎨 Streamlit UI Features

### New Research Tab 🚀
- Enter research query
- Select depth
- Real-time progress tracking
- Agent execution visualization

### Results Tab 📊
- Executive summary card
- Key metrics dashboard
- Interactive knowledge graph display
- Key concepts (as badges)
- Consensus findings
- Research gaps (expandable)
- ML analysis:
  - Topics discovered (LDA)
  - Paper clusters (K-means)
  - Citation predictions
- Top sources with citations
- Download buttons (TXT, JSON, PNG)

### History Tab 📚
- View past research sessions
- Access previous reports
- Browse generated graphs

## 🔧 Customization

### Change LLM Model
Edit `4_Src_Code/agentic_ai_pipeline.py`:
```python
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant",  # Change this
    temperature=0.3
)
```

### Modify UI Theme
Edit `9_Deployment/app.py`:
```python
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(120deg, #YOUR_COLOR, #YOUR_COLOR);
    }
</style>
""", unsafe_allow_html=True)
```

### Adjust Search Parameters
In `4_Src_Code/agentic_ai_pipeline.py`:
```python
# Discovery Agent
all_sources.extend(self.search_arxiv(query, max_results=8))  # Change this
```

### Add New Agent
1. Create agent class
2. Add to workflow in `create_research_workflow()`
3. Define edges (execution order)

## 📊 Output Files Explained

### Research Report (`report_*.txt`)
```
- Executive summary
- Key findings
- Notable insights
- Research gaps
- Conflicting evidence
- Conclusion
- Source citations
```

### Knowledge Graph (`kg_*.png`)
- **Blue circles**: Key concepts
- **Red circles**: Research sources
- **Lines**: Relationships/connections
- **Size**: Importance/centrality

### Vector Database (`chroma_db/`)
- Embeddings for all validated sources
- Enables semantic similarity search
- Used by RAG agent

## 🐛 Troubleshooting

### Streamlit won't start
```bash
# Check if port is in use
streamlit run 9_Deployment/app.py --server.port 8502
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### API key issues
- Check `.env` file exists in project root
- Verify keys are correct (no quotes needed)
- Or enter keys in Streamlit sidebar

### No knowledge graph generated
- Check `7_Results/` folder
- Ensure matplotlib is installed
- Look for errors in console output

## 📈 Performance Tips

### Faster Execution
- Use "quick" depth for testing
- Reduce `max_results` in search methods
- Use smaller LLM model

### Better Results
- Use "deep" depth for comprehensive research
- Add SERPAPI key for web search
- Use larger LLM model (llama-3.1-70b-versatile)

### Save Costs
- Use cached results when re-running
- Disable web search if not needed
- Use "quick" depth as default

## 🎓 Learning Resources

### Understanding the Code
1. Start with `4_Src_Code/README.md`
2. Read each agent's docstrings
3. Check `ResearchState` in `agents/state.py`

### LangGraph Concepts
- **Nodes**: Individual agents (functions)
- **Edges**: Execution flow between agents
- **State**: Shared data structure (TypedDict)
- **Workflow**: State machine that orchestrates agents

### RAG Explained
- **R**etrieval: Find relevant documents
- **A**ugmented: Add to LLM context
- **G**eneration: LLM generates answer with context

## 🌟 Next Steps

1. **Try it out**: Run a simple query
2. **Explore UI**: Check all tabs and features
3. **Read reports**: Analyze generated outputs
4. **Customize**: Modify agents for your needs
5. **Deploy**: Use Streamlit Cloud or Docker
6. **Extend**: Add new agents or features

## 📞 Support

If you encounter issues:
1. Check the relevant README:
   - `README.md` - Main docs
   - `4_Src_Code/README.md` - Code details
   - `9_Deployment/README.md` - UI/deployment
   - `5_Pipeline/README.md` - Airflow setup
2. Look for error messages in console
3. Check file paths and dependencies
4. Verify API keys are set correctly

## 🎉 Congratulations!

You now have a powerful AI research assistant that can:
- ✅ Search multiple academic databases automatically
- ✅ Validate and score sources for quality
- ✅ Synthesize findings across papers
- ✅ Generate knowledge graphs
- ✅ Perform ML analysis
- ✅ Create comprehensive reports
- ✅ Monitor research trends

**Happy Researching! 🔬**

---

© 2025 | Built with LangGraph, Groq LLM, ChromaDB & Streamlit


