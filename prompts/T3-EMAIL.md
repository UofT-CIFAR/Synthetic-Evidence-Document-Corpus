# Tier-3 email reply insertion prompt

You are helping generate realistic-looking synthetic evidence for a research corpus on **AI-generated email forgery**. The output will be shown as an extra reply under an existing message screenshot.

## Impersonation constraint

Draft **one new email reply** as if it were written by:

**{impersonated_address}**

Use the tone, pacing, and vocabulary suggested by the thread excerpt below (do not invent a third party).

## Target outcome

One short narrative hook the reply should address (from `assets/email_targets.txt`):

**{target}**

## Thread subject (context only)

{thread_subject}

## Prior thread excerpt (voice anchor)

```
{anchor_excerpt}
```

## Output rules

- **Length:** 50–200 words.
- **Format:** plain text only — no markdown fences, no `"Subject:"` headers.
- Stay plausible for office / operational email (no sci-fi, no legal threats unless clearly contextual).
