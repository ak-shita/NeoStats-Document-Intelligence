# NeoStats Document Intelligence

NeoStats is an AI-assisted document-intelligence application prepared for the NeoStats AI Engineer Intern assessment. It accepts a supported financial document, extracts structured data, runs deterministic validation checks, and persists the complete result for retrieval in the web UI and API.

> Live frontend URL: not deployed yet
>
> Live backend/API URL: not deployed yet
>
> Swagger UI: not deployed yet

## Architecture and processing flow

```text
HTML/CSS/JS frontend (Render static site)
        |
        v
FastAPI API (Railway) --> Railway MySQL
        |
        +--> file validation --> OCR.Space --> Gemini structured extraction
                                      |              |
                                      +--------------+
                                                     v
                                      Pydantic validation --> deterministic financial validation
```

The processing API follows this sequence:

1. Validate a PDF, JPEG, or PNG upload.
2. Extract page-level text with OCR.Space. Transient OCR failures are retried at most three times.
3. Send the OCR text to Gemini with a document-specific structured-output schema.
4. Validate the Gemini result with Pydantic models.
5. Run deterministic financial checks; the service never asks the LLM to perform validation arithmetic.
6. Store the complete structured result as JSON in MySQL and return it through the API/UI.

## Supported input

- Document types: `invoice`, `balance_sheet`, `profit_and_loss`, `cash_flow`.
- File types: PDF, JPEG/JPG, PNG.
- Maximum pages: 3.
- OCR images larger than the OCR.Space free-tier 1 MB ceiling are compressed only for the OCR upload. PDFs remain PDFs.

## Financial validation

The validation response contains checks with `name`, `formula`, `operands`, `calculated_value`, `reported_value`, `variance`, and `status`. A check can be `PASS`, `FAIL`, or `NOT_APPLICABLE` when the required source fields are absent. A financial `FAIL` is returned as validation information; it does not turn an otherwise completed processing request into a pipeline failure.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Liveness check; does not call OCR or Gemini. |
| `POST` | `/api/v1/documents/process` | Process a multipart `file` with `document_type`. |
| `GET` | `/api/v1/documents` | List persisted documents. |
| `GET` | `/api/v1/documents/{document_name}` | Retrieve one persisted processing result. |
| `GET` | `/docs` | Swagger UI. |

Document-result reads use `Cache-Control: no-store` so a reprocessed filename does not show a browser-cached historical validation result.

## Local setup

Prerequisites: Python 3.12+, a MySQL-compatible database for persistence, an OCR.Space key, and a Gemini API key.

```sh
python -m venv venv
# Windows: venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # macOS/Linux: cp .env.example .env
```

Set the following values in `.env` (never commit this file):

```dotenv
OCR_SPACE_API_KEY=
GEMINI_API_KEY=
DATABASE_URL=mysql+pymysql://username:password@localhost:3306/neostats_document_intelligence
# Needed when a separate static frontend calls the API, for example:
CORS_ORIGINS=https://your-frontend.onrender.com
```

Start the backend from the repository root:

```sh
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

For local same-origin use, open the frontend served by FastAPI at `http://localhost:8000`. If hosting the static frontend separately, set its API base URL in the Settings view or configure `frontend/js/runtime-config.js` with the backend URL.

## Tests

```sh
python -m pytest -q
```

The normal suite uses isolated SQLite and mocks OCR/Gemini. Real MySQL integration tests are opt-in:

```sh
RUN_MYSQL_INTEGRATION=1 python -m pytest -m mysql_integration -q
```

Do not run live OCR/Gemini processing as part of ordinary tests.

## Deployment

### Railway: FastAPI and MySQL

`railway.json` installs `requirements.txt`, starts `uvicorn` bound to `0.0.0.0:$PORT`, and checks `/api/v1/health`.

1. Create a Railway project from this repository and add a MySQL service.
2. Deploy the repository root as the FastAPI service.
3. Set `DATABASE_URL` to Railway's MySQL connection URL, plus `OCR_SPACE_API_KEY`, `GEMINI_API_KEY`, and `CORS_ORIGINS`.
4. Set `CORS_ORIGINS` to the eventual Render URL, with no trailing slash.
5. Generate a Railway public domain and verify `/api/v1/health`, `/api/v1/documents`, and `/docs`.

Tables are created safely on application startup when `DATABASE_URL` is configured. Existing data is never deleted automatically.

### Render: static frontend

`render.yaml` defines a Render Static Site rooted at `frontend/`. Configure its `API_BASE_URL` environment variable to the public Railway API base URL (for example, `https://your-api.up.railway.app`). The build script writes that public URL into `js/runtime-config.js`; it does not embed credentials.

After the Render URL exists, update Railway `CORS_ORIGINS` to that exact origin and redeploy the backend. The frontend can also be pointed at a different backend URL from its Settings screen without changing code.

## Sample outputs

Representative, non-secret fixtures are stored under [`sample_outputs/`](sample_outputs):

- [`validation_smoke/Consolidated_Balance_Sheet_2017.validation.json`](sample_outputs/validation_smoke/Consolidated_Balance_Sheet_2017.validation.json) demonstrates financial-validation JSON.
- [`extraction_smoke/Consolidated_Balance_Sheet_2017.extraction.json`](sample_outputs/extraction_smoke/Consolidated_Balance_Sheet_2017.extraction.json) demonstrates a structured extraction fixture.
- `ocr_smoke/` contains OCR text and metadata examples.

The source assessment dataset is intentionally ignored by Git to keep the public submission lightweight.

## Current limitations

- OCR.Space and Gemini are external services. Their free tiers can rate-limit, reject oversized files, or be temporarily unavailable.
- The app retries only transient OCR transport/provider failures; it does not retry invalid uploads or configuration errors.
- Gemini free-tier capacity can block a live processing run. Health and persisted-document APIs remain usable without invoking Gemini/OCR.
- Financial validation only evaluates relationships for values actually extracted from the source; unavailable operands produce `NOT_APPLICABLE` rather than invented values.

## AI/tool usage declaration

This assessment project uses OCR.Space for OCR and Google Gemini for structured extraction. The codebase and tests were developed with AI-assisted coding support. Financial validation, input validation, API contracts, persistence, and tests are explicit application code and do not rely on an LLM for arithmetic decisions.

## Future updates and redeployment

1. Make and test a change locally.
2. Commit and push it to the connected GitHub repository.
3. Railway redeploys the backend from the configured branch; verify `/api/v1/health` and `/docs`.
4. Render redeploys the static frontend from the configured branch; verify the browser uses the Railway URL and can call `/api/v1/health`.
5. If the Render domain changes, update Railway `CORS_ORIGINS` and redeploy the backend.
