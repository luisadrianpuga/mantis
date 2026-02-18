import httpx
from bs4 import BeautifulSoup


async def fetch(url: str) -> str:
    """
    Fetch page text (first ~2000 chars) from a URL.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            return text[:2000]
    except Exception as exc:  # pragma: no cover - defensive guardrail
        return f"HTTP fetch error: {exc}"
