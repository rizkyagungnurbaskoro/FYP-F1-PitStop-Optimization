# Pitwall Web (Next.js)

## Run

```bash
cd pitwall_web
npm install
npm run dev
```

Open http://localhost:3000

## API
Set API base (optional):

```bash
set NEXT_PUBLIC_API_BASE=http://localhost:8000
```

## Notes
- UI-only fallback if API is down.
- Streamlit dashboard is untouched.

## Streamlit Embed (Demo page)
Run Streamlit separately, then the Demo page will embed it:

```bash
streamlit run dashboard/streamlit_app.py
```
