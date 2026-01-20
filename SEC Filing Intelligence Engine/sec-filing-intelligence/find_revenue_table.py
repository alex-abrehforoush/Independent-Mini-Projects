from src.document_parser import SECFilingParser
import re

parser = SECFilingParser("data/apple_10k_2025.html")
parser.load()

text = parser.text

# Search for revenue table patterns with actual dollar amounts
# Apple reports in millions, so look for patterns like "$201,183" or "$ 201,183"

print("Searching for revenue table with dollar amounts...\n")

# Pattern: Product name followed by dollar amount
pattern = r'(iPhone|Mac|iPad|Services|Wearables).*?\$\s*[\d,]+'

matches = list(re.finditer(pattern, text, re.IGNORECASE))

print(f"Found {len(matches)} potential matches\n")

# Show first 10 matches with context
for i, match in enumerate(matches[:10]):
    pos = match.start()
    context = text[max(0, pos-200):pos+500]
    print(f"=== Match {i+1} at position {pos} ===")
    print(context)
    print("\n")

# Also search for "Net sales" header which usually precedes the table
print("\n" + "="*80)
print("Searching for 'Net sales' sections:")
print("="*80)

net_sales_matches = list(re.finditer(r'Net\s+sales', text, re.IGNORECASE))
print(f"Found {len(net_sales_matches)} occurrences\n")

for i, match in enumerate(net_sales_matches[:5]):
    pos = match.start()
    context = text[pos:pos+800]
    print(f"=== Net sales #{i+1} at position {pos} ===")
    print(context)
    print("\n")