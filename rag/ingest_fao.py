from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

URL = "https://www.fao.org/land-water/databases-and-software/crop-information/wheat/en/"

def extract_clean_text(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")

        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted sections
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()

    content = []

    # Extract headings
    for h in soup.find_all(["h1", "h2", "h3"]):
        text = h.get_text(strip=True)
        if text:
            content.append(f"\n## {text}\n")

    # Extract paragraphs
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text:
            content.append(text)

    # Extract tables (preserve structure)
    for table in soup.find_all("table"):
        content.append("\n--- TABLE START ---")
        for row in table.find_all("tr"):
            cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
            if cols:
                content.append(" | ".join(cols))
        content.append("--- TABLE END ---\n")

    return "\n".join(content)


def save_to_txt(output_path="fao_wheat_clean.txt"):
    text = extract_clean_text(URL)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Source: {URL}\n\n")
        f.write(text)

    print("✅ Saved clean FAO wheat data to:", output_path)


if __name__ == "__main__":
    save_to_txt()