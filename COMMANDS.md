# Project Commands Reference

## Run with Docker (recommended)
```bash
docker compose up -d          # start everything
docker compose down           # stop everything
docker compose build          # rebuild images after code changes
docker compose logs -f        # watch live logs
```

Open http://localhost:3000 in browser.

---

## Run locally (without Docker)
```bash
# Terminal 1 — API
~/.local/bin/uv run uvicorn api.inference:app --port 8000

# Terminal 2 — React UI
cd frontend && npm start
```

---

## Tests
```bash
~/.local/bin/uv run pytest tests/ -v
~/.local/bin/uv run pytest tests/ --cov=api --cov=cleaning --cov-report=term-missing
```

---

## Dependencies
```bash
~/.local/bin/uv sync           # install production deps
~/.local/bin/uv sync --dev     # install including pytest, httpx, ruff
```

---

## Git
```bash
git add <files>
git commit -m "your message"
git push origin main
git pull origin main --rebase  # sync with remote before pushing
```
