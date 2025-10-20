# 🏗️ Airflow Architecture - Research Intelligence Platform

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    APACHE AIRFLOW ORCHESTRATION                 │
│                                                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Scheduler  │  │  Webserver   │  │   Workers    │           │
│  │             │  │   (UI)       │  │              │           │
│  └──────┬──────┘  └──────────────┘  └──────┬───────┘           │
│         │                                    │                   │
│         │           ┌──────────────┐        │                   │
│         └──────────►│  PostgreSQL  │◄───────┘                   │
│                     │   Metadata   │                            │
│                     └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH PIPELINE DAG                         │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task 1: Initialize State                                │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task 2: Discovery Agent                                 │   │
│  │  • ArXiv API          • Web Search                       │   │
│  │  • Semantic Scholar   • Query Reformulation              │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task 3: Validation Agent                                │   │
│  │  • Credibility Scoring  • Relevance Check                │   │
│  │  • Quality Filtering    • LLM Validation                 │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task 4: RAG Agent                                       │   │
│  │  • Vector Embeddings   • ChromaDB                        │   │
│  │  • Semantic Search     • Document Chunking               │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task 5: Synthesis Agent                                 │   │
│  │  • Concept Extraction  • Consensus Finding               │   │
│  │  • Gap Analysis        • Knowledge Graph                 │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task 6: ML Agent                                        │   │
│  │  • Topic Modeling (LDA) • K-means Clustering             │   │
│  │  • Citation Prediction  • Quality Scoring                │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task 7: Reporter Agent                                  │   │
│  │  • Executive Summary   • Citation Mapping                │   │
│  │  • Detailed Report     • Visualizations                  │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task 8: Monitoring Agent                                │   │
│  │  • Alert Triggers      • Trend Analysis                  │   │
│  │  • Author Networks     • Publication Velocity            │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task 9: Generate Summary                                │   │
│  │  • Email Report        • Metrics Collection              │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task 10: Cleanup                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌─────────────┐
│ User Input  │  Query via Airflow Variables
└──────┬──────┘  research_query = "AI in Healthcare"
       │         research_depth = "standard"
       ▼
┌─────────────┐
│   XCom      │  State object shared between tasks
│  Storage    │  {query, sources, concepts, reports, ...}
└──────┬──────┘
       │
       ├─► Task 1: Initialize ──► state['query'] = "AI in Healthcare"
       │
       ├─► Task 2: Discovery  ──► state['raw_sources'] = [20 papers]
       │
       ├─► Task 3: Validation ──► state['validated_sources'] = [15 papers]
       │                          state['source_quality_avg'] = 78.5
       │
       ├─► Task 4: RAG        ──► state['vector_store_id'] = "research_abc123"
       │                          state['rag_ready'] = True
       │
       ├─► Task 5: Synthesis  ──► state['key_concepts'] = [12 concepts]
       │                          state['knowledge_graph'] = {nodes, edges}
       │
       ├─► Task 6: ML         ──► state['ml_topics'] = [5 topics]
       │                          state['paper_clusters'] = {3 clusters}
       │
       ├─► Task 7: Reporter   ──► state['report_file'] = "/tmp/report.txt"
       │                          state['kg_file'] = "/tmp/kg.png"
       │
       ├─► Task 8: Monitoring ──► state['alert_triggers'] = [8 triggers]
       │
       └─► Task 9: Summary    ──► Email with metrics
```

## Parallel Execution DAG

```
Initialize
    │
    ▼
Discovery
    │
    ▼
Validation
    │
    ├──────────┬──────────┐
    ▼          ▼          ▼
  RAG     Synthesis   (Independent tasks run in parallel)
    │          │
    └────┬─────┘
         ▼
     ML Agent
         │
         ▼
     Reporter
         │
         ▼
    Monitoring
```

**Benefits:**
- 30% faster execution
- Better resource utilization
- Independent failure handling

## Component Integration

```
┌─────────────────────────────────────────────────────────┐
│                  DOCKER CONTAINERS                       │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Airflow    │  │   Airflow    │  │  PostgreSQL  │  │
│  │  Webserver   │  │  Scheduler   │  │              │  │
│  │  Port 8080   │  │              │  │  Port 5432   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
         │                    │                  │
         └────────────────────┴──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                   SHARED VOLUMES                         │
│                                                          │
│  ./airflow_research_dag.py  →  /opt/airflow/dags/       │
│  ./main.py                  →  /opt/airflow/dags/       │
│  ./logs/                    →  /opt/airflow/logs/       │
│  ./chroma_db/               →  /opt/airflow/chroma_db/  │
│  ./.env                     →  /opt/airflow/.env        │
│                                                          │
└─────────────────────────────────────────────────────────┘
         │                    │                  │
         └────────────────────┴──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                   EXTERNAL APIS                          │
│                                                          │
│  • Groq API (LLM)          • ArXiv                      │
│  • SerpAPI (Web Search)    • Semantic Scholar           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Monitoring & Alerts

```
┌────────────────────────────────────────┐
│          AIRFLOW MONITORING            │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │  Task Instance Logs              │ │
│  │  • Real-time streaming           │ │
│  │  • Historical logs               │ │
│  └──────────────────────────────────┘ │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │  Metrics (XCom)                  │ │
│  │  • sources_found: 20             │ │
│  │  • sources_validated: 15         │ │
│  │  • avg_quality: 78.5             │ │
│  │  • concepts_found: 12            │ │
│  │  • ml_topics: 5                  │ │
│  └──────────────────────────────────┘ │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │  Email Notifications             │ │
│  │  • On failure                    │ │
│  │  • On retry                      │ │
│  │  • On success (optional)         │ │
│  └──────────────────────────────────┘ │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │  Retry Logic                     │ │
│  │  • Max retries: 2                │ │
│  │  • Retry delay: 5 min            │ │
│  │  • Timeout: 2 hours              │ │
│  └──────────────────────────────────┘ │
└────────────────────────────────────────┘
```

## Scheduling Options

| Schedule | Cron Expression | Use Case |
|----------|----------------|----------|
| Daily 2 AM | `0 2 * * *` | Default - overnight processing |
| Every 6 hours | `0 */6 * * *` | Active research monitoring |
| Weekdays 9 AM | `0 9 * * 1-5` | Business hours only |
| Weekly Sunday | `0 0 * * 0` | Weekly summaries |
| Monthly 1st | `0 0 1 * *` | Monthly reports |
| Manual only | `None` | On-demand execution |

## Security Architecture

```
┌─────────────────────────────────────────┐
│          SECURITY LAYERS                │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Authentication                   │ │
│  │  • Basic Auth (default)           │ │
│  │  • OAuth (configurable)           │ │
│  │  • LDAP (enterprise)              │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Secrets Management               │ │
│  │  • Environment variables (.env)   │ │
│  │  • Airflow Connections            │ │
│  │  • Airflow Variables              │ │
│  │  • External secrets backend       │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Network Security                 │ │
│  │  • Internal Docker network        │ │
│  │  • Exposed ports: 8080 (web only) │ │
│  │  • HTTPS (production)             │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Scalability

```
Development (Current)
└─ Single machine
   └─ LocalExecutor
      └─ Tasks run sequentially or limited parallel

Production (Scalable)
├─ CeleryExecutor
│  └─ Multiple workers
│     └─ Distributed task execution
├─ KubernetesExecutor
│  └─ Dynamic pod creation
│     └─ Auto-scaling workers
└─ Cloud providers
   ├─ AWS MWAA (Managed Airflow)
   ├─ Google Cloud Composer
   └─ Azure Data Factory
```

## File Output Structure

```
Project Root
│
├─ /tmp/                    # Airflow outputs
│  ├─ airflow_report_*.txt  # Research reports
│  └─ airflow_kg_*.png      # Knowledge graphs
│
├─ logs/                    # Airflow logs
│  └─ dag_id/
│     └─ task_id/
│        └─ execution_date/
│           └─ attempt.log
│
├─ chroma_db/               # Vector database
│  └─ research_*/
│     ├─ index/
│     └─ metadata/
│
└─ plugins/                 # Custom Airflow plugins (optional)
```

## Key Features

### ✅ **Automation**
- Scheduled execution (daily, weekly, custom)
- No manual intervention required
- Automatic retries on failure

### ✅ **Monitoring**
- Real-time task status
- Historical run data
- Performance metrics
- Email alerts

### ✅ **Reliability**
- Failure recovery
- Task dependencies
- Transaction management
- State persistence

### ✅ **Scalability**
- Parallel task execution
- Distributed workers
- Resource management
- Queue prioritization

### ✅ **Observability**
- Detailed logging
- Metrics collection
- Visual DAG representation
- Debugging tools

## Comparison: Standalone vs Airflow

| Feature | Standalone (main.py) | Airflow DAG |
|---------|---------------------|-------------|
| **Execution** | Manual | Automated/Scheduled |
| **Monitoring** | Console output | Web UI + Logs |
| **Retries** | Manual | Automatic |
| **Scheduling** | None | Cron-based |
| **Notifications** | None | Email/Slack |
| **Parallel** | Limited | Full support |
| **History** | None | Full history |
| **Best For** | Ad-hoc research | Production pipelines |

## Next Steps

1. **Development**: Test with standalone `python main.py`
2. **Staging**: Deploy to Airflow with manual triggers
3. **Production**: Enable scheduling and monitoring
4. **Scale**: Move to CeleryExecutor or Kubernetes

---

**Built with:** Apache Airflow 2.7.3 | Python 3.11 | Docker | PostgreSQL

