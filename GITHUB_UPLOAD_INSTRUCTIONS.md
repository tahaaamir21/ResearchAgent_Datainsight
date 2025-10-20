# 📤 GitHub Upload Instructions

## Step-by-Step Guide to Upload Your Project

### Prerequisites
✅ You've created a GitHub repository  
✅ You have the repository URL  

---

## 🚀 Upload Commands

### Step 1: Initialize Git (if not already done)

```bash
cd C:\Users\PMLS\Desktop\datascienceproj
git init
```

### Step 2: Add All Files

```bash
git add .
```

This will add all files except those in `.gitignore` (API keys, large files, etc.)

### Step 3: Check What Will Be Uploaded

```bash
git status
```

**Make sure these are NOT listed:**
- ❌ `.env` file
- ❌ `chroma_db/` folder
- ❌ Large PNG files

**These SHOULD be listed:**
- ✅ All Python files
- ✅ README files
- ✅ requirements.txt
- ✅ Project folders (1_Documentation, 4_Src_Code, etc.)

### Step 4: Commit Your Changes

```bash
git commit -m "Initial commit: Multi-Agent Research Intelligence Platform"
```

### Step 5: Connect to GitHub Repository

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

**Example:**
```bash
git remote add origin https://github.com/johndoe/research-platform.git
```

### Step 6: Rename Branch to 'main' (if needed)

```bash
git branch -M main
```

### Step 7: Push to GitHub

```bash
git push -u origin main
```

You may be prompted for your GitHub credentials.

---

## 🔐 GitHub Authentication

### Option 1: Personal Access Token (Recommended)

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo` (full control of private repositories)
4. Copy the token
5. Use it as password when pushing

### Option 2: GitHub Desktop

1. Download GitHub Desktop: https://desktop.github.com/
2. Sign in with your account
3. Add existing repository
4. Commit and push via GUI

---

## ✅ Verify Upload

After pushing, check:

1. Go to your GitHub repository
2. Refresh the page
3. You should see:
   - All project folders
   - README.md displayed nicely
   - Files and folders structure
   
---

## 📝 Update README on GitHub

The main README is visible by default. To make it look better:

1. Copy content from `README_GITHUB.md`
2. Replace `README.md` content with it
3. Update your GitHub username in links
4. Commit and push:

```bash
copy README_GITHUB.md README.md
git add README.md
git commit -m "Update README with GitHub-specific content"
git push
```

---

## 🔄 Future Updates

Whenever you make changes:

```bash
# 1. Add changed files
git add .

# 2. Commit with message
git commit -m "Description of changes"

# 3. Push to GitHub
git push
```

---

## 🚨 Troubleshooting

### Error: "fatal: remote origin already exists"

```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

### Error: "Authentication failed"

Use a Personal Access Token instead of your password.

### Error: "rejected because the remote contains work"

```bash
git pull origin main --rebase
git push origin main
```

### Check if .env is Excluded

```bash
git check-ignore .env
```

Should output: `.env` (confirming it's ignored)

---

## 📌 Important Notes

### Files That SHOULD Be Uploaded
- ✅ All Python code
- ✅ README files
- ✅ requirements.txt
- ✅ env.example (template)
- ✅ .gitignore
- ✅ Launch scripts
- ✅ Documentation files

### Files That SHOULD NOT Be Uploaded
- ❌ .env (contains API keys!)
- ❌ chroma_db/ (large vector database)
- ❌ logs/ (Airflow logs)
- ❌ *.png, *.txt output files
- ❌ __pycache__/
- ❌ .streamlit/secrets.toml

**Our `.gitignore` already handles this!**

---

## 🎉 Once Uploaded

After successful upload:

1. ✅ Add repository description on GitHub
2. ✅ Add topics/tags: `machine-learning`, `ai`, `research`, `langchain`, `streamlit`
3. ✅ Enable GitHub Issues
4. ✅ Add a LICENSE file (MIT recommended)
5. ✅ Consider making it public to share with others
6. ✅ Add to your portfolio/resume!

---

## 🚀 Next: Deploy to Streamlit Cloud

See `DEPLOYMENT_GUIDE.md` for instructions on deploying your Streamlit app!

---

**Ready to upload? Run the commands in Step 1-7 above!** 🚀

