# Tier-1 email date edit — audit / manifest text

Procedure reference for **X-T1-DATE-EML** (spec §6):

1. Parse the `.eml` / RFC822 source.
2. Apply offset uniformly chosen from {±1 hour, ±6 hours, ±1 day, ±7 days, ±30 days} per batch item index mapping.
3. Render a deterministic screenshot PNG showing headers + body (training signal uses vision edit).
4. Full-frame adapter edit updates only header timing fields per policy:
   - **coherent_received** = `{coherent_received}` — when true, Received hop lines should remain plausible vs the new Date; when false, only Date changes.
5. Save raster PNG with corpus provenance markers.

## Precomputed edit (recorded on this manifest row)

- Bounding box (header Date line region): `{bbox}`
- Old visible Date string: `{old_date}`
- New Date string: `{new_date}`
- Coherent Received policy: `{coherent_received}`
