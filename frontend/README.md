# FinLearn Frontend

This folder contains the final React dashboard for FinLearn Tutor.

## What It Uses

- Vite + React + TypeScript
- Tailwind + shadcn/ui styling
- Real simulation data from the Python backend at `/api/simulation`

## Local Development

1. Start the Python API from the repository root:

```bash
python -m server.app
```

2. In this folder, install dependencies:

```bash
npm install
```

3. Copy the example env file if needed:

```bash
cp .env.example .env
```

4. Start the frontend:

```bash
npm run dev
```

The app uses the local Vite proxy by default, so calls to `/api` and `/reset` are forwarded to `http://127.0.0.1:7860`.

## Production Build

```bash
npm run build
```
