import asyncio
from google import genai
from google.genai import types
from app.core.config import Settings
from pydantic import BaseModel, Field
from typing import Any

class Evidence(BaseModel):
    source_text: str | None = None
    page_number: int | None = None

class ExtractedField(BaseModel):
    field: str
    value: str | None = None
    confidence: float | None = None
    evidence: Evidence | None = None

class TinyInvoice(BaseModel):
    invoice_number: ExtractedField | None = None

async def try_model(client, model, label, config):
    try:
        r = await client.aio.models.generate_content(
            model=model,
            contents="OCR: Invoice no: 94404257. Extract invoice_number.",
            config=config,
        )
        print(label, "OK", (r.text or "")[:120].replace("\n"," "))
    except Exception as e:
        print(label, "FAIL", type(e).__name__, str(e)[:300])

async def main():
    s = Settings()
    client = genai.Client(api_key=s.gemini_api_key.strip())
    models = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]
    for m in models:
        await try_model(client, m, f"{m}:plain", types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
    for m in models:
        await try_model(client, m, f"{m}:tiny_schema", types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TinyInvoice,
            temperature=0.1,
        ))
    # also list a few models
    try:
        found = []
        async for m in client.aio.models.list():
            name = getattr(m, "name", str(m))
            if "flash" in name.lower() or "gemini" in name.lower():
                found.append(name)
            if len(found) >= 20:
                break
        print("models_sample", found)
    except Exception as e:
        print("list_fail", e)
    await client.aio.aclose()

asyncio.run(main())
