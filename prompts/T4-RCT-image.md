# T4 Receipt fabrication (image output)

You are generating **one full raster image** of a retail or restaurant **thermal-paper receipt** scan for a research corpus on synthetic document detection.

Use the attached reference receipt images **only** for overall visual style: paper tone, speckle, fold shadows, font roughness, and general layout density. Do **not** copy merchant names, logos, or line items from references verbatim. The new receipt must be **original** content.

## Identity (must appear legibly on the receipt)

- Customer name and address: **{customer_name}**, **{customer_address}**
- Merchant name, address, and phone: **{merchant_name}**, **{merchant_address}**, **{merchant_phone}**

## Anchor OCR (tone and item style only — do not transcribe verbatim)

{anchor_block}

## Arithmetic sub-variant: **{sub_variant}**

- If **consistent**: printed **subtotal**, **tax**, and **total** must satisfy subtotal + tax = total to two decimal places.
- If **inconsistent**: **tax** must be wrong by at least 5% relative to a plausible rate on the subtotal, while the printed **total** must still equal subtotal + the **wrong** tax (so the slip is internally “consistent” but tax is unrealistic).

## Output

Return **only** a single photorealistic receipt image (narrow portrait aspect, white/off-white paper, dark print). No JSON, no markdown, no surrounding explanation in the image itself.
