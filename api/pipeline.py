import os
import json
import base64
import mimetypes
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI


def bytes_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def parse_price_any(price_text: str) -> Optional[Dict[str, Any]]:
    if not price_text:
        return None
    patterns = [
        (r"(USD|US\s?\$|\$)\s*([0-9][0-9,]*(?:\.[0-9]{2})?)", "USD"),
        (r"(EUR|€)\s*([0-9][0-9,]*(?:\.[0-9]{2})?)", "EUR"),
        (r"(GBP|£)\s*([0-9][0-9,]*(?:\.[0-9]{2})?)", "GBP"),
        (r"(CAD|C\$)\s*([0-9][0-9,]*(?:\.[0-9]{2})?)", "CAD"),
        (r"(AUD|A\$)\s*([0-9][0-9,]*(?:\.[0-9]{2})?)", "AUD"),
        (r"(JPY|¥)\s*([0-9][0-9,]*)", "JPY"),
    ]
    for pat, cur in patterns:
        m = re.search(pat, price_text, flags=re.IGNORECASE)
        if m:
            amt = m.group(2).replace(",", "")
            try:
                return {"currency": cur, "amount": float(amt)}
            except ValueError:
                return {"currency": cur, "amount": amt}
    return None


IDENT_SCHEMA_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "name": "luxury_identification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "brand": {"type": "string"},
            "model": {"type": "string"},
            "category": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "attributes": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "primary_color": {"type": "string"},
                    "material": {"type": "string"},
                    "metal_finish": {"type": "string"},
                    "closure": {"type": "string"},
                    "notable_markings": {"type": "string"},
                },
                "required": ["primary_color", "material", "metal_finish", "closure", "notable_markings"],
            },
            "typical_price_range_usd": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"low": {"type": "number"}, "high": {"type": "number"}},
                "required": ["low", "high"],
            },
            "estimated_market_value_usd": {"type": "number"},
            "suggested_queries": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [
            "brand",
            "model",
            "category",
            "aliases",
            "confidence",
            "attributes",
            "typical_price_range_usd",
            "estimated_market_value_usd",
            "suggested_queries",
            "rationale",
        ],
    },
}

LISTINGS_SCHEMA_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "name": "similar_listings",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "queries_used": {"type": "array", "items": {"type": "string"}},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "source": {"type": "string"},
                        "price_text": {"type": "string"},
                        "date_text": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["title", "url", "source", "price_text", "date_text", "notes"],
                },
            },
        },
        "required": ["queries_used", "results"],
    },
}


class LuxuryPipeline:
    def __init__(self, api_key: Optional[str] = None, model_ident: str = "gpt-4.1-mini", model_search: str = "gpt-4.1-mini"):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY (set env var or enter in the app).")
        self.client = OpenAI(api_key=api_key)
        self.model_ident = model_ident
        self.model_search = model_search

    def identify(self, image_data_urls: List[str], description_text: str = "", classifier_hint: str = "") -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "You are a luxury goods identification assistant.\n"
                    "Use the images, the user description, and the classifier hint (may be wrong).\n"
                    "Return structured fields for brand/model/category, aliases, attributes, confidence, price range, market value,\n"
                    "suggested search queries, and a short rationale.\n"
                    "Rules: do not fabricate stamps/serials; if not visible, say 'not visible'."
                ),
            },
        ]
        if description_text.strip():
            content.append({"type": "input_text", "text": f"User description:\n{description_text.strip()}"})
        if classifier_hint.strip():
            content.append({"type": "input_text", "text": f"Classifier hint:\n{classifier_hint.strip()}"})

        for du in image_data_urls:
            content.append({"type": "input_image", "image_url": du})

        resp = self.client.responses.create(
            model=self.model_ident,
            input=[{"role": "user", "content": content}],
            text={"format": IDENT_SCHEMA_FORMAT},
        )
        return json.loads(resp.output_text)

    def search_listings(self, ident: Dict[str, Any], max_results: int = 10) -> Dict[str, Any]:
        brand = (ident.get("brand") or "").strip()
        model = (ident.get("model") or "").strip()
        category = (ident.get("category") or "").strip()
        aliases = ident.get("aliases") or []

        suggested = (ident.get("suggested_queries") or [])[:8]
        fallbacks = [f'{brand} "{model}" price', f'{brand} "{model}" {category} listing']
        if aliases:
            fallbacks.append(f'{brand} "{aliases[0]}" {category} listing')

        queries = [q for q in (suggested + fallbacks) if q][:10]

        prompt = (
            "You are a research agent.\n"
            "Use web search to find comparable listings/sales for the identified luxury product.\n"
            "Return up to 12 results (or fewer if low quality) with title, url, source, price_text, date_text if visible, notes.\n"
            "Rules:\n"
            "- Prefer official brand/retailer pages and major resale marketplaces.\n"
            "- Avoid generic blogs unless they include an actual listing with price.\n"
            "- Do not invent prices or dates; if missing, leave empty and explain in notes.\n"
            f"- Target up to {max_results} results.\n\n"
            f"Item:\nBrand: {brand}\nModel: {model}\nCategory: {category}\nAliases: {aliases}\n\n"
            f"Queries:\n{queries}"
        )

        try:
            resp = self.client.responses.create(
                model=self.model_search,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                tools=[{"type": "web_search"}],
                text={"format": LISTINGS_SCHEMA_FORMAT},
            )
            data = json.loads(resp.output_text)
            for r in data.get("results", []):
                r["parsed_price"] = parse_price_any(r.get("price_text", ""))
            return data
        except Exception as e:
            return {"queries_used": queries, "results": [], "error": f"web_search unavailable or failed: {type(e).__name__}: {e}"}

    def run(self, image_data_urls: List[str], description_text: str = "", classifier_hint: str = "", max_results: int = 10) -> Dict[str, Any]:
        ident = self.identify(image_data_urls, description_text=description_text, classifier_hint=classifier_hint)
        listings = self.search_listings(ident, max_results=max_results)
        return {"identification": ident, "listings": listings}
