# NeoStats Document Intelligence

NeoStats is an AI-assisted document-intelligence application prepared for the NeoStats AI Engineer Intern assessment. It accepts a supported financial document, extracts structured data, runs deterministic validation checks, and persists the complete result for retrieval in the web UI and API.

> Live frontend URL:
> https://neostats-document-intelligence-pwpu.onrender.com

> Live backend/API URL:
> https://neostats-document-intelligence-frontend-production.up.railway.app

> Swagger UI:
> https://neostats-document-intelligence-frontend-production.up.railway.app/docs

> Public GitHub Repository
> https://github.com/ak-shita/NeoStats-Document-Intelligence

## Deployment Status

- Frontend: Render
- Backend/API: Railway
- Database: Railway MySQL
- API health: Live
- Swagger: Live
- Public repository: GitHub

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

## Technology Stack

- Python: Core backend & processing -
  Strong ecosystem for AI, OCR, data processing and validation
- FastAPI: REST API -
  Lightweight, fast, typed and provides automatic OpenAPI/Swagger documentation
- OCR.Space: OCR - 
  External OCR service suitable for scanned documents and rapid implementation
- Google Gemini: AI extraction -
  Capable of semantic document understanding and structured extraction
- MySQL: Database -
  Relational storage suitable for document metadata and structured processing results
- HTML,CSS,JS: Frontend
- Render: Frontend Deployment
- Railway: Backend + MySQL deployment
- pytest: Testing

## Supported input

- Document types: `invoice`, `balance_sheet`, `profit_and_loss`, `cash_flow`.
- File types: PDF, JPEG/JPG, PNG.
- Maximum pages: 3.
- OCR images larger than the OCR.Space free-tier 1 MB ceiling are compressed only for the OCR upload. PDFs remain PDFs.

## Local setup

Prerequisites: Python 3.12+, a MySQL-compatible database for persistence, an OCR.Space key, and a Gemini API key.

### Clone the Repository

git clone https://github.com/ak-shita/NeoStats-Document-Intelligence.git
cd NeoStats-Document-Intelligence

### Create and Activate Virtual Environment

Windows:
python -m venv venv
venv\Scripts\activate

Linux/macOS:
python3 -m venv venv
source venv/bin/activate

### Install Dependencies

pip install -r requirements.txt

### Configure Environment Variables

Create:
backend/.env
using:
backend/.env.example

### Run the Backend

cd backend
uvicorn app.main:app --reload

backend will be available at: 
http://127.0.0.1:8000

Swagger/OpenAPI documentation will be available at:
http://127.0.0.1:8000/docs

### Run the Frontend

It can be served using a local static HTTP server.


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
# Railway's MYSQL_URL can be referenced directly; mysql:// and mysql+pymysql:// are supported.
DATABASE_URL=mysql+pymysql://username:password@localhost:3306/neostats_document_intelligence
# Needed when a separate static frontend calls the API, for example:
CORS_ORIGINS=https://your-frontend.onrender.com
```

## Environment Variables

```
OCR_SPACE_API_KEY=your_ocr_space_api_key
OCR_SPACE_ENDPOINT=https://api.ocr.space/parse/image
OCR_SPACE_MAX_UPLOAD_BYTES=1000000
OCR_SPACE_TIMEOUT_SECONDS=120
OCR_SPACE_ENGINE=2

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.6-flash
GEMINI_TIMEOUT_SECONDS=90

DATABASE_URL=mysql+pymysql://username:password@host:3306/database_name

CORS_ORIGINS=http://localhost:5500
```

## Tests

```sh
python -m pytest -q
```

The normal suite uses isolated SQLite and mocks OCR/Gemini. Real MySQL integration tests are opt-in:

```sh
RUN_MYSQL_INTEGRATION=1 python -m pytest -m mysql_integration -q
```

Do not run live OCR/Gemini processing as part of ordinary tests.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Liveness check; does not call OCR or Gemini. |
| `POST` | `/api/v1/documents/process` | Process a multipart `file` with `document_type`. |
| `GET` | `/api/v1/documents` | List persisted documents. |
| `GET` | `/api/v1/documents/{document_name}` | Retrieve one persisted processing result. |
| `GET` | `/docs` | Swagger UI. |

Document-result reads use `Cache-Control: no-store` so a reprocessed filename does not show a browser-cached historical validation result.

### API Examples

> Health Check:

```
curl -X GET \
  "https://neostats-document-intelligence-frontend-production.up.railway.app/api/v1/health"
```
> Process / Upload Document:

```
POST /api/v1/documents/process
```
Example:
```
curl -X POST \
  "https://neostats-document-intelligence-frontend-production.up.railway.app/api/v1/documents/process" \
  -F "file=@sample_invoice.jpg" \
  -F "document_type=invoice"
```

> Get Document by Name

```
GET /api/v1/documents/{document_name}
```

Example:
```
curl -X GET \
  "https://neostats-document-intelligence-frontend-production.up.railway.app/api/v1/documents/Consolidated%20Balance%20Sheet%202017.pdf"
```
> Dashboard / List Documents
```
GET /api/v1/documents
```
Example:
```
curl -X GET \
  "https://neostats-document-intelligence-frontend-production.up.railway.app/api/v1/documents"
```

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

## Confidence Calculation and Interpretation

The extraction schema supports confidence information for extracted fields where available.

Confidence represents the extraction model's confidence in identifying a value. It should not be interpreted as proof that the extracted value is factually correct.

Financial correctness is determined separately through deterministic validation rules.

The application therefore distinguishes between:
- Extraction confidence
- Processing status
- Financial validation status

## Financial validation

The validation response contains checks with `name`, `formula`, `operands`, `calculated_value`, `reported_value`, `variance`, and `status`. A check can be `PASS`, `FAIL`, or `NOT_APPLICABLE` when the required source fields are absent. A financial `FAIL` is returned as validation information; it does not turn an otherwise completed processing request into a pipeline failure.

## Database and Persistence

The application uses MySQL through SQLAlchemy.

Each processed document is persisted using the document name as a unique identifier.

Stored information includes:
- Document ID
- Document name
- Document type
- Processing status
- Overall confidence
- Structured extraction result
- Validation result
- Creation timestamp
- Last updated timestamp

The structured processing result is stored as JSON so that extracted fields, tables, evidence and validation results can be preserved without requiring a separate database column for every possible document field.

Repeated processing of the same document uses an update/upsert approach rather than creating uncontrolled duplicate records.

## Current limitations

- OCR.Space and Gemini are external services. Their free tiers can rate-limit, reject oversized files, or be temporarily unavailable.
- Gemini free-tier capacity can block a live processing run. Health and persisted-document APIs remain usable without invoking Gemini/OCR.
- LLM/API availability can affect extraction because semantic extraction depends on the Gemini API.
- OCR accuracy depends on document quality, resolution, scanning quality and layout.
- OCR can introduce character or number recognition errors, particularly in low-quality scans.
- The OCR provider's free-tier limits can restrict processing throughput.

## What I Would Change for a Production Deployment

- Use asynchronous background workers and a job queue for document processing.
- Add OCR/LLM fallback services and stronger retry/rate-limit handling.
- Add authentication, authorization, secure file storage and audit logging.
- Add monitoring, structured logging, metrics and alerting.
- Build a larger labeled dataset for field-level extraction accuracy and regression testing.
- Add human review for low-confidence or failed extractions.
- Add database migrations, stronger indexing and production-scale storage.

## AI/tool usage declaration

This assessment project uses OCR.Space for OCR and Google Gemini for structured extraction. The codebase and tests were developed with AI-assisted coding support. Financial validation, input validation, API contracts, persistence, and tests are explicit application code and do not rely on an LLM for arithmetic decisions.

## Future updates and redeployment

1. Make and test a change locally.
2. Commit and push it to the connected GitHub repository.
3. Railway redeploys the backend from the configured branch; verify `/api/v1/health` and `/docs`.
4. Render redeploys the static frontend from the configured branch; verify the browser uses the Railway URL and can call `/api/v1/health`.
5. If the Render domain changes, update Railway `CORS_ORIGINS` and redeploy the backend.
