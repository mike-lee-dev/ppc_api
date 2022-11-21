import os
import openai
openai.organization = "org-S9fmNV4i82AvIn9SJwQ9tZOy"

APIKEY="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"

openai.api_key = os.getenv(APIKEY)

curl https://api.openai.com/v1/completions \
-H "Content-Type: application/json" \
-H "Authorization: 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" \
-d '{"model": "text-davinci-002", "prompt": "Say this is a test", "temperature": 0, "max_tokens": 6}'