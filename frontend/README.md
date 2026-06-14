# Shastra AI Frontend

React TypeScript frontend for the Shastra AI application.

## Run

Start the API from the repository root:

```bash
uvicorn server:app --reload
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

Use `.env.local` if the API runs somewhere else:

```bash
VITE_API_BASE_URL=http://localhost:8000
```
