# T4 Receipt Fabrication Prompt

You are fabricating a single plausible retail / restaurant receipt for a
research corpus that trains detectors of AI-generated evidence. Output MUST
be valid JSON that matches the schema below. Do not include any commentary,
do not wrap the JSON in code fences.

## Identity

- Customer: {customer_name}, {customer_address}
- Merchant: {merchant_name}, {merchant_address}
- Merchant phone: {merchant_phone}

## Style anchors (OCR text from three real receipts, for tone only)

{anchor_block}

## Target

- Arithmetic sub-variant: **{sub_variant}**
  - If `consistent`, subtotal + tax must equal total (to 2 decimals).
  - If `inconsistent`, tax must be wrong by at least 5%, total must still
    match subtotal + the wrong tax.

## Output schema

```
{
  "merchant": string,
  "merchant_address": string,
  "date": "DD/MM/YYYY",
  "line_items": [{"label": string, "amount": number}],  // 3-7 items
  "subtotal": number,
  "tax": number,
  "total": number,
  "payment_method": string  // e.g. "CASH", "VISA ****1234"
}
```
