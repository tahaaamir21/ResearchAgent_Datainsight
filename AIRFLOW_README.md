# 🚀 Apache Airflow Integration - Research Intelligence Pipeline

## Overview

This project now includes **Apache Airflow** for production-grade orchestration of the multi-agent research system. Airflow provides:

- ✅ **Scheduled Execution**: Run research automatically (daily, weekly, etc.)
- ✅ **Task Monitoring**: Track each agent's progress
- ✅ **Retry Logic**: Automatic retries on failures
- ✅ **Parallel Execution**: Run independent tasks simultaneously
- ✅ **Email Alerts**: Get notified on success/failure
- ✅ **Visual DAG**: See the workflow in a web UI

---

## 🏗️ Architecture

### **Sequential DAG (research_intelligence_pipeline)**
```
Initialize → Discovery → Validation → RAG → Synthesis → ML → Reporter → Monitoring → Summary → Cleanup
```

### **Parallel DAG (research_intelligence_pipeline_parallel)**
```
Initialize → Discovery → Validation
                          ├─→ RAG ──┐
                          └─→ Synthesis ─→ ML → Reporter → Monitoring
```

---

## 📦 Installation

### **Option 1: Docker (Recommended)**

1. **Ensure Docker is installed**
   ```bash
   docker --version
   docker-compose --version
   ```

2. **Set environment variables**
   Create a `.env` file:
   ```bash
   GROQ_API_KEY=your_groq_key_here
   SERPAPI_KEY=your_serpapi_key_here (optional)
   
   # Airflow credentials
   _AIRFLOW_WWW_USER_USERNAME=airflow
   _AIRFLOW_WWW_USER_PASSWORD=airflow
   AIRFLOW_UID=50000
   ```

3. **Start Airflow**
   ```bash
   # Initialize database
   docker-compose -f docker-compose-airflow.yml up airflow-init
   
   # Start services
   docker-compose -f docker-compose-airflow.yml up -d
   ```

4. **Access Airflow UI**
   - URL: http://localhost:8080
   - Username: `airflow`
   - Password: `airflow`

### **Option 2: Local Installation**

1. **Install Airflow**
   ```bash
   pip install apache-airflow==2.7.3
   pip install apache-airflow-providers-postgres
   ```

2. **Initialize Airflow**
   ```bash
   export AIRFLOW_HOME=~/airflow
   airflow db init
   airflow users create \
       --username admin \
       --password admin \
       --firstname Admin \
       --lastname User \
       --role Admin \
       --email admin@example.com
   ```

3. **Copy DAG file**
   ```bash
   cp airflow_research_dag.py ~/airflow/dags/
   cp main.py ~/airflow/dags/
   ```

4. **Start Airflow**
   ```bash
   # Terminal 1: Start webserver
   airflow webserver --port 8080
   
   # Terminal 2: Start scheduler
   airflow scheduler
   ```

---

## 🎯 Usage

### **1. Set Research Query**

In Airflow UI, go to **Admin → Variables** and set:

| Key | Value | Description |
|-----|-------|-------------|
| `research_query` | "AI in healthcare" | Your research question |
| `research_depth` | "standard" | quick/standard/deep |

### **2. Trigger DAG**

**Option A: Manual Trigger**
1. Go to http://localhost:8080
2. Find `research_intelligence_pipeline`
3. Click the ▶️ play button

**Option B: CLI Trigger**
```bash
airflow dags trigger research_intelligence_pipeline
```

**Option C: Scheduled Execution**
- DAG runs automatically daily at 2 AM (configurable)

### **3. Monitor Progress**

1. Click on DAG name
2. Select "Graph" view to see visual flow
3. Click on tasks to see logs
4. View metrics in XCom tab

### **4. View Results**

Check `/tmp/` for output files:
```bash
# Report
cat /tmp/airflow_report_*.txt

# Knowledge graph
open /tmp/airflow_kg_*.png
```

---

## 📊 DAG Features

### **Task Breakdown**

| Task | Agent | Description | Retry |
|------|-------|-------------|-------|
| `initialize_state` | - | Load configuration | 2 |
| `discovery` | DiscoveryAgent | Search ArXiv, Scholar, Web | 2 |
| `validation` | ValidationAgent | Score & filter sources | 2 |
| `rag` | RAGAgent | Create vector database | 2 |
| `synthesis` | SynthesisAgent | Extract concepts, find gaps | 2 |
| `ml_analysis` | MLAgent | Topic modeling, clustering | 2 |
| `reporter` | ReporterAgent | Generate report | 2 |
| `monitoring` | MonitoringAgent | Setup alerts | 2 |
| `generate_summary` | - | Create email summary | 2 |
| `cleanup` | - | Final cleanup | 1 |

### **XCom Data Sharing**

Tasks communicate via XCom (cross-communication):

```python
# Discovery pushes
sources_found: int

# Validation pushes
sources_validated: int
avg_quality: float

# Synthesis pushes
concepts_found: int
consensus_findings: int

# ML pushes
ml_topics: int
clusters: int

# Reporter pushes
report_file: str
kg_file: str
```

### **Email Notifications**

Configure in `default_args`:
```python
'email': ['your-email@example.com'],
'email_on_failure': True,
'email_on_success': True,
```

Requires SMTP configuration in `airflow.cfg`:
```ini
[email]
email_backend = airflow.utils.email.send_email_smtp

[smtp]
smtp_host = smtp.gmail.com
smtp_starttls = True
smtp_ssl = False
smtp_user = your-email@gmail.com
smtp_password = your-app-password
smtp_port = 587
smtp_mail_from = your-email@gmail.com
```

---

## ⚙️ Configuration

### **Schedule Intervals**

Edit in `airflow_research_dag.py`:

```python
'schedule_interval': '0 2 * * *'  # Daily at 2 AM
```

Common schedules:
- `'@daily'` - Once per day at midnight
- `'@hourly'` - Every hour
- `'0 */6 * * *'` - Every 6 hours
- `'0 9 * * 1-5'` - Weekdays at 9 AM
- `None` - Manual trigger only

### **Retries & Timeouts**

```python
'retries': 2,
'retry_delay': timedelta(minutes=5),
'execution_timeout': timedelta(hours=2),
```

### **Concurrency**

```python
'max_active_runs': 1,  # Max concurrent DAG runs
'max_active_tasks': 16,  # Max tasks per DAG run
```

---

## 🔍 Monitoring & Debugging

### **View Logs**

**In UI:**
1. Click on task
2. Click "Log" button
3. View real-time logs

**CLI:**
```bash
airflow tasks logs research_intelligence_pipeline discovery 2024-01-15
```

### **Check Task Status**

```bash
# List DAG runs
airflow dags list-runs -d research_intelligence_pipeline

# Check task instances
airflow tasks states-for-dag-run research_intelligence_pipeline <run_id>
```

### **Common Issues**

| Issue | Solution |
|-------|----------|
| Import errors | Check DAG file in `~/airflow/dags/` |
| API key errors | Set in Airflow Variables or .env |
| Timeout errors | Increase `execution_timeout` |
| Memory errors | Reduce batch sizes in agents |

---

## 📈 Advanced Features

### **Dynamic DAG Generation**

Create DAGs programmatically:

```python
for query in ["AI in healthcare", "Quantum computing", "Climate change ML"]:
    dag_id = f"research_{query.replace(' ', '_')}"
    # Create DAG with this query
```

### **Sensors**

Wait for external triggers:

```python
from airflow.sensors.filesystem import FileSensor

wait_for_query = FileSensor(
    task_id='wait_for_query_file',
    filepath='/tmp/new_query.txt',
    poke_interval=60,
)
```

### **Branching**

Conditional task execution:

```python
from airflow.operators.python import BranchPythonOperator

def choose_depth(**context):
    sources = context['ti'].xcom_pull(key='sources_found')
    if sources > 20:
        return 'deep_analysis'
    else:
        return 'quick_analysis'

branching = BranchPythonOperator(
    task_id='choose_path',
    python_callable=choose_depth,
)
```

### **Task Groups**

Organize related tasks:

```python
from airflow.utils.task_group import TaskGroup

with TaskGroup("data_collection") as collection:
    discovery >> validation
```

---

## 🐳 Docker Commands

```bash
# Start services
docker-compose -f docker-compose-airflow.yml up -d

# Stop services
docker-compose -f docker-compose-airflow.yml down

# View logs
docker-compose -f docker-compose-airflow.yml logs -f airflow-scheduler

# Restart scheduler
docker-compose -f docker-compose-airflow.yml restart airflow-scheduler

# Access Airflow CLI
docker-compose -f docker-compose-airflow.yml run airflow-cli bash
```

---

## 🔒 Security Best Practices

1. **Change default passwords**
   ```bash
   docker-compose -f docker-compose-airflow.yml run airflow-cli \
       airflow users create --role Admin --username admin --password <new-password>
   ```

2. **Use Airflow Connections for API keys**
   - UI: Admin → Connections
   - CLI: `airflow connections add groq_api --conn-type http --conn-host api.groq.com --conn-password <key>`

3. **Enable authentication**
   - Set `AIRFLOW__API__AUTH_BACKEND` in docker-compose

4. **Use secrets backend**
   - AWS Secrets Manager
   - Google Secret Manager
   - HashiCorp Vault

---

## 📊 Metrics & Monitoring

### **Airflow Metrics**

Monitor via Prometheus/Grafana:

```yaml
# In airflow.cfg
[metrics]
statsd_on = True
statsd_host = localhost
statsd_port = 8125
```

### **Custom Metrics**

Push metrics to external systems:

```python
from airflow.stats import Stats

Stats.incr('research_pipeline.sources_found', count=sources)
Stats.timing('research_pipeline.discovery_time', duration)
```

---

## 🎓 Learning Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [DAG Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Airflow Tutorials](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html)

---

## 🤝 Integration with Existing System

The Airflow DAG wraps your existing `main.py` system:

```
main.py (7 Agents) → airflow_research_dag.py (Orchestration) → Production Pipeline
```

You can still run `main.py` directly for ad-hoc research:
```bash
python main.py
```

Or use Airflow for scheduled, monitored, production workflows.

---

## 🚀 Quick Start Commands

```bash
# 1. Setup environment
echo "GROQ_API_KEY=your_key" > .env

# 2. Start Airflow
docker-compose -f docker-compose-airflow.yml up -d

# 3. Set research query
# Via UI: http://localhost:8080 → Admin → Variables

# 4. Trigger pipeline
curl -X POST http://localhost:8080/api/v1/dags/research_intelligence_pipeline/dagRuns \
  -H "Content-Type: application/json" \
  -u airflow:airflow \
  -d '{"conf":{}}'

# 5. Monitor
# UI: http://localhost:8080

# 6. View results
ls -l /tmp/airflow_*
```

---

**Congratulations!** 🎉 You now have a production-grade, orchestrated research intelligence platform!

