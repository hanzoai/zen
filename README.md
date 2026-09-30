# Zen

Zen is Zen LM, the open model family from [Zoo Labs Foundation](https://zoo.ngo), a 501(c)(3)
non-profit. Open weights across language, code, vision, audio and retrieval: run them anywhere, or
call them on the Hanzo API.

- Family, model cards and papers: [zenlm/zen](https://github.com/zenlm/zen) · [zenlm.org](https://zenlm.org)
- Weights: [huggingface.co/zenlm](https://huggingface.co/zenlm)
- Catalog and prices: [hanzo.ai/models/zen](https://hanzo.ai/models/zen)
- Docs: [docs.hanzo.ai/docs/models/zen](https://docs.hanzo.ai/docs/models/zen)

## Use Zen through the Hanzo API

```bash
curl https://api.hanzo.ai/v1/chat/completions \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "zen6", "messages": [{"role": "user", "content": "Hello"}]}'
```

The API speaks the OpenAI wire format, so any OpenAI SDK works with
`base_url="https://api.hanzo.ai/v1"`. `GET https://api.hanzo.ai/v1/models` lists every Zen model
it serves.

This repository describes the family; it holds no code or weights.
