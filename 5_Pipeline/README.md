# 5_Pipeline - Apache Airflow Orchestration

This directory contains Apache Airflow DAG for production orchestration of the research pipeline.

## 📁 Contents

```
5_Pipeline/
├── airflow_dag.py           # Main Airflow DAG definition
└── README.md                # This file
```

## 🔄 What is the Airflow Pipeline?

The Airflow DAG orchestrates the entire multi-agent research workflow as a production-ready pipeline with:

- **Scheduled Execution**: Run research queries on a schedule
- **Error Handling**: Retry logic and failure notifications
- **Monitoring**: Track execution metrics and performance
- **Scalability**: Distribute tasks across workers
- **XCom**: Pass data between tasks efficiently

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

See the main project `AIRFLOW_README.md` for full setup instructions.

```bash
# From project root
docker-compose -f docker-compose-airflow.yml up -d

# Access UI
open http://localhost:8080
# Username: airflow
# Password: airflow
```

### Option 2: Local Airflow

```bash
# Install Airflow
pip install apache-airflow==2.7.3

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start webserver and scheduler
airflow webserver --port 8080 &
airflow scheduler &
```

## 📊 DAG Structure

### DAG Definition

```python
dag = DAG(
    'research_intelligence_pipeline',
    default_args=default_args,
    schedule_interval='@daily',  # Run daily
    catchup=False
)
```

### Tasks (in order)

1. **`discovery_task`**: Multi-source search
2. **`validation_task`**: Source scoring
3. **`rag_task`**: Vector embedding
4. **`synthesis_task`**: Knowledge graph building
5. **`ml_task`**: ML analysis
6. **`reporter_task`**: Report generation
7. **`monitoring_task`**: Trend analysis

### Data Flow (XCom)

Data is passed between tasks using XCom:
```python
# Task 1: Push data
context['ti'].xcom_push(key='raw_sources', value=sources)

# Task 2: Pull data
sources = context['ti'].xcom_pull(
    task_ids='discovery_task',
    key='raw_sources'
)
```

## ⚙️ Configuration

### DAG Parameters

Edit `airflow_dag.py`:

```python
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,  # Enable email alerts
    'email_on_retry': False,
    'retries': 2,              # Retry failed tasks
    'retry_delay': timedelta(minutes=5)
}
```

### Schedule

Change the schedule interval:
```python
schedule_interval='@daily'      # Daily
schedule_interval='@weekly'     # Weekly
schedule_interval='0 */6 * * *' # Every 6 hours
schedule_interval=None          # Manual trigger only
```

### Variables

Set Airflow variables via UI or CLI:
```bash
# Via CLI
airflow variables set GROQ_API_KEY "your_key_here"
airflow variables set research_query "machine learning"

# Via UI
Admin > Variables > Add Variable
```

## 🎯 Usage

### Trigger DAG Manually

**Via UI:**
1. Go to http://localhost:8080
2. Find `research_intelligence_pipeline`
3. Click the "Play" button
4. Configure parameters if needed

**Via CLI:**
```bash
airflow dags trigger research_intelligence_pipeline \
    --conf '{"query": "quantum computing"}'
```

### Monitor Execution

**Via UI:**
- **Graph View**: Visualize task dependencies
- **Tree View**: See execution history
- **Gantt View**: Analyze task duration
- **Logs**: View task-specific logs

**Via CLI:**
```bash
# List DAG runs
airflow dags list-runs -d research_intelligence_pipeline

# View task logs
airflow tasks logs research_intelligence_pipeline discovery_task 2025-01-01
```

### Pause/Unpause DAG

```bash
# Pause (stop scheduled runs)
airflow dags pause research_intelligence_pipeline

# Unpause (resume)
airflow dags unpause research_intelligence_pipeline
```

## 📈 Monitoring & Alerts

### Email Notifications

Configure in `airflow.cfg`:
```ini
[smtp]
smtp_host = smtp.gmail.com
smtp_starttls = True
smtp_ssl = False
smtp_user = your_email@gmail.com
smtp_password = your_app_password
smtp_port = 587
smtp_mail_from = your_email@gmail.com
```

### Slack Integration

Add Slack webhook:
```python
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

slack_alert = SlackWebhookOperator(
    task_id='slack_alert',
    http_conn_id='slack_webhook',
    message='Research pipeline completed!',
    dag=dag
)
```

### Custom Metrics

Log metrics:
```python
from airflow.metrics import Stats

Stats.incr('research_pipeline.sources_found', count=len(sources))
Stats.gauge('research_pipeline.quality_score', score)
```

## 🔧 Troubleshooting

### DAG not appearing

```bash
# Check DAG location
airflow dags list

# Parse DAG for errors
python 5_Pipeline/airflow_dag.py
```

### Task failing

```bash
# View logs
airflow tasks logs research_intelligence_pipeline <task_id> <execution_date>

# Test task locally
airflow tasks test research_intelligence_pipeline discovery_task 2025-01-01
```

### XCom size limit

If data is too large for XCom, use external storage:
```python
# Save to file
import pickle
with open(f'/tmp/data_{run_id}.pkl', 'wb') as f:
    pickle.dump(data, f)

# Pass file path via XCom
ti.xcom_push(key='data_path', value=file_path)
```

## 🚀 Advanced

### Parallel Execution

Enable parallelism in `airflow.cfg`:
```ini
[core]
parallelism = 32
dag_concurrency = 16
max_active_runs_per_dag = 16
```

### Custom Executors

Switch from SequentialExecutor:
- **LocalExecutor**: Multiple processes on one machine
- **CeleryExecutor**: Distributed workers
- **KubernetesExecutor**: Kubernetes pods

### Dynamic DAG Generation

Generate DAGs programmatically:
```python
for topic in ['AI', 'ML', 'DL']:
    dag_id = f'research_pipeline_{topic.lower()}'
    # Create DAG dynamically
```

## 📝 Best Practices

1. **Idempotency**: Tasks should produce same result when rerun
2. **Small XComs**: Keep XCom data < 1MB
3. **Error Handling**: Always add retry logic
4. **Logging**: Use structured logging
5. **Testing**: Test tasks locally before deploying
6. **Monitoring**: Set up alerts for failures

## 🔗 Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [XCom Guide](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html)

---

For Docker setup and more details, see the main project's `AIRFLOW_README.md`

