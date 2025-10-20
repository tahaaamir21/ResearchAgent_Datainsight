# 4_Src_Code - Source Code

This directory contains the core implementation of the Multi-Agent Research Intelligence Platform.

## 📁 Structure

```
4_Src_Code/
├── agents/                       # Individual agent modules (future modular structure)
│   ├── __init__.py
│   └── state.py                  # Shared state definition
└── agentic_ai_pipeline.py       # Main pipeline (all agents integrated)
```

## 🤖 Agents

The platform consists of 7 specialized AI agents:

### 1. Discovery Agent
- **Purpose**: Multi-source search across ArXiv, Semantic Scholar, and Web
- **Methods**: 
  - `search_arxiv()`: Search academic papers
  - `search_semantic_scholar()`: Search peer-reviewed research
  - `search_web()`: Google search via SERPAPI
  - `reformulate_query()`: Generate alternative queries
  - `deduplicate_sources()`: Remove duplicate results

### 2. Validation Agent
- **Purpose**: Score source credibility and relevance
- **Scoring Factors**:
  - Source type (30 points)
  - Citation count (25 points)
  - Recency (20 points)
  - Content quality (25 points)
- **Methods**:
  - `calculate_source_score()`: Compute credibility score
  - `check_relevance()`: LLM-powered relevance check

### 3. RAG Agent
- **Purpose**: Create vector embeddings for semantic search
- **Technology**: ChromaDB + HuggingFace embeddings
- **Methods**:
  - `create_vector_store()`: Build vector database
  - `query_rag()`: Semantic similarity search

### 4. Synthesis Agent
- **Purpose**: Build knowledge graph and synthesize findings
- **Outputs**:
  - Knowledge graph (nodes + edges)
  - Consensus findings
  - Research gaps
  - Contradictions
- **Methods**:
  - `extract_key_concepts()`: Extract main concepts
  - `find_consensus()`: Identify agreement across sources
  - `detect_contradictions()`: Find conflicting evidence
  - `build_knowledge_graph()`: Create graph structure

### 5. ML Agent
- **Purpose**: Advanced machine learning analysis
- **Techniques**:
  - **LDA Topic Modeling**: Discover hidden topics
  - **K-means Clustering**: Group similar papers
  - **Citation Prediction**: Predict future impact
  - **Quality Scoring**: ML-based paper assessment
- **Methods**:
  - `topic_modeling()`: Extract topics using LDA
  - `cluster_papers()`: K-means clustering
  - `predict_citations()`: Predict future citations
  - `ml_quality_scoring()`: ML-based quality assessment

### 6. Reporter Agent
- **Purpose**: Generate comprehensive research reports
- **Outputs**:
  - Executive summary
  - Detailed report
  - Citation map
  - Visualization metadata
- **Methods**:
  - `generate_executive_summary()`: 3-4 sentence summary
  - `generate_detailed_report()`: Full analysis
  - `create_citation_map()`: Build citation network

### 7. Monitoring Agent
- **Purpose**: Set up monitoring and trend analysis
- **Features**:
  - Alert triggers
  - Trend analysis
  - Author networks
  - Publication velocity tracking
- **Methods**:
  - `create_alert_triggers()`: Define monitoring rules
  - `analyze_trends()`: Identify emerging topics

## 🔄 Workflow Orchestration

The agents are orchestrated using **LangGraph**:

```python
workflow = StateGraph(ResearchState)

# Add all agents as nodes
workflow.add_node("discovery", discovery_node)
workflow.add_node("validation", validation_node)
# ... etc

# Define execution sequence
workflow.set_entry_point("discovery")
workflow.add_edge("discovery", "validation")
workflow.add_edge("validation", "rag")
# ... etc
workflow.add_edge("monitoring", END)
```

## 📊 State Management

All agents share a common `ResearchState` (TypedDict) containing:

```python
class ResearchState(TypedDict):
    query: str
    research_depth: str
    raw_sources: List[Dict]
    validated_sources: List[Dict]
    knowledge_graph: Dict
    ml_topics: List[Dict]
    executive_summary: str
    # ... and more
```

## 🚀 Usage

### As a Module

```python
from agentic_ai_pipeline import run_research_pipeline

result = run_research_pipeline(
    query="Your research question",
    research_depth="standard"  # quick | standard | deep
)
```

### Direct Execution

```bash
python 4_Src_Code/agentic_ai_pipeline.py
```

### From Streamlit UI

```bash
streamlit run 9_Deployment/app.py
```

### From Airflow

```python
# See 5_Pipeline/airflow_dag.py
```

## ⚙️ Configuration

### LLM Model

Change the model in `agentic_ai_pipeline.py`:
```python
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant",  # Change this
    temperature=0.3
)
```

Available Groq models:
- `llama-3.1-8b-instant` (default, fast)
- `mixtral-8x7b-32768` (larger context)
- `llama-3.1-70b-versatile` (more capable)

### Search Parameters

Modify in each agent's `__init__` or methods:
```python
self.search_arxiv(query, max_results=8)  # Change max_results
```

### Embedding Model

Change in `RAGAgent`:
```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Change this
)
```

## 🧪 Testing

Individual agent testing:
```python
from agentic_ai_pipeline import DiscoveryAgent, ChatGroq

llm = ChatGroq(api_key="your_key", model="llama-3.1-8b-instant")
agent = DiscoveryAgent(llm)

results = agent.search_arxiv("machine learning", max_results=5)
```

## 📝 Adding New Agents

1. Create agent class with required methods
2. Add node function in `create_research_workflow()`
3. Add node to graph with `workflow.add_node()`
4. Define edge connections with `workflow.add_edge()`
5. Update `ResearchState` if needed

## 🐛 Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check state at each step:
```python
def debug_node(state):
    print(f"Current agent: {state['current_agent']}")
    print(f"Sources: {len(state.get('raw_sources', []))}")
    return state
```

---

For more details, see the main project README.md

