# 9_Deployment - Streamlit Web UI

This directory contains the Streamlit web application for the Research Intelligence Platform.

## 🚀 Quick Start

```bash
# From project root
streamlit run 9_Deployment/app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎨 Features

### Main Tabs

1. **🚀 New Research**
   - Enter research queries
   - Select research depth
   - Track progress in real-time
   - View agent execution stages

2. **📊 Results**
   - Executive summary
   - Key metrics dashboard
   - Interactive knowledge graph
   - Key concepts visualization
   - Consensus findings
   - Research gaps analysis
   - ML analysis (topics, clusters, predictions)
   - Top sources with citations
   - Download options (TXT, JSON, PNG)

3. **📚 History**
   - View past research sessions
   - Access previous reports and graphs

### Sidebar Configuration

- **API Key Status**: Shows if keys are configured (from .env or environment)
- **System Info**: View platform capabilities
- **About**: Version and framework information

**Note:** API keys are loaded automatically from `.env` file or environment variables. No manual entry needed!

## 📦 Dependencies

Required packages (from `requirements.txt`):
```
streamlit==1.32.0
Pillow==12.0.0
```

Plus all dependencies from the main pipeline.

## 🎯 Usage

1. **Ensure API Keys**: Make sure `.env` file exists with your GROQ_API_KEY
2. **Enter Query**: Type your research question
3. **Select Depth**: Choose quick/standard/deep
4. **Start Research**: Click the button and wait
5. **View Results**: Switch to Results tab
6. **Download**: Export reports and visualizations

The app will show API key status in the sidebar (✅ if configured, ❌ if missing).

## 🔧 Customization

### Modify UI Theme

Edit the custom CSS in `app.py`:
```python
st.markdown("""
<style>
    .main-header {
        /* Your custom styles */
    }
</style>
""", unsafe_allow_html=True)
```

### Add New Features

The app uses Streamlit session state for data persistence:
```python
st.session_state.research_result = result
```

### Change Layout

Modify the column structure:
```python
col1, col2, col3 = st.columns([2, 1, 1])
```

## 🌐 Deployment

### Streamlit Community Cloud (Free)

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repository
4. Add secrets (API keys) in dashboard
5. Deploy!

### Docker

```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "9_Deployment/app.py"]
```

### Other Platforms

- **Heroku**: Use Procfile with streamlit command
- **Railway**: Deploy directly from GitHub
- **AWS/GCP/Azure**: Use container services

## 🐛 Troubleshooting

### Port Already in Use

```bash
streamlit run 9_Deployment/app.py --server.port 8502
```

### Import Errors

Make sure you're running from the project root:
```bash
cd /path/to/datascienceproj
streamlit run 9_Deployment/app.py
```

### API Key Issues

**Solution:**
- Keys are automatically loaded from `.env` file or environment variables
- Check sidebar for API key status (✅ = configured, ❌ = missing)
- Create `.env` file: `rename env.example .env` (Windows) or `mv env.example .env` (Mac/Linux)
- Add your `GROQ_API_KEY=your_key_here` to the `.env` file
- Restart the app after creating/updating `.env` file

## 📝 Notes

- The app automatically saves results to session state
- Knowledge graphs are loaded from generated PNG files
- Reports are read from the `report_*.txt` files
- All file paths are relative to project root

---

For more information, see the main project README.

