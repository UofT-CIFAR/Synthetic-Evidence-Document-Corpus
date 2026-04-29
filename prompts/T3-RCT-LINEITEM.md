# T3 Receipt Line-Item Insertion Prompt

You are helping generate realistic-looking synthetic evidence for a research
corpus. The document is a retail / restaurant receipt. Your output will be
inserted as a new line above the subtotal.

## Input

- Target outcome (one short phrase describing why this line would plausibly
  appear, drawn from `assets/clause_targets.txt`):

  **{target}**

- Existing receipt OCR (unchanged order; one text line per input line):

```
{ocr_text}
```

## Output rules

- Produce **exactly one** new receipt line, no longer than 40 characters.
- The line MUST be formatted as `LABEL .... AMOUNT`, where AMOUNT matches
  the existing currency format in the receipt (no currency code if the
  existing amounts do not use one).
- AMOUNT must be plausible for the target outcome. Typical ranges:
  - Service / handling fees: 1.50 to 9.99
  - Loyalty / gratuity: 0.50 to 6.00
  - Late / surcharge: 2.00 to 12.00
- Return raw text only. No commentary, no quotes, no JSON.
