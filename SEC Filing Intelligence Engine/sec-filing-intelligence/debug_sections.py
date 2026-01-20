from src.document_parser import SECFilingParser

parser = SECFilingParser("data/apple_10k_2025.html")
parser.load()

financial_text = parser.extract_section("financial performance")

print(f"Extracted {len(financial_text)} characters\n")
print("=" * 80)
print("FINANCIAL TEXT PREVIEW (first 3000 chars):")
print("=" * 80)
print(financial_text[:3000])
print("\n" + "=" * 80)
print("SEARCHING FOR KEY TERMS:")
print("=" * 80)

keywords = ["iPhone", "Mac", "iPad", "Net sales", "revenue", "201", "202"]
for kw in keywords:
    if kw in financial_text:
        idx = financial_text.find(kw)
        snippet = financial_text[max(0, idx-100):idx+200]
        print(f"\n'{kw}' found at position {idx}:")
        print(f"...{snippet}...")