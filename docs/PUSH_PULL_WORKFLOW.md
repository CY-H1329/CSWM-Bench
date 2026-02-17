# Push / Pull Workflow

Workflow for syncing code between local machine and remote server (e.g., H100).

## 1. Push from remote server to GitHub

On the **remote server** (where experiments run):

```bash
cd ~/CY/Spatial_MAS  # or your project path
git add .
git status
git commit -m "Update: experiment results, config changes"
git push origin main
```

## 2. Pull on local machine

On your **local machine**:

```bash
cd ~/Desktop/Spatial_MAS  # or your local path
git pull origin main
```

## 3. Push from local to GitHub

On your **local machine** (after edits, documentation updates):

```bash
git add .
git status
git commit -m "Docs: organize for paper submission"
git push origin main
```

## 4. Pull on remote server

On the **remote server**:

```bash
cd ~/CY/Spatial_MAS
git pull origin main
```

## Recommended flow

1. **Remote**: Run experiments → commit results → push to GitHub
2. **Local**: Pull → review → edit docs → commit → push
3. **Remote**: Pull to get updated docs and config

## Notes

- Avoid committing large result files (e.g., `results/` with many samples). Add to `.gitignore` if needed.
- Use meaningful commit messages for reproducibility.
