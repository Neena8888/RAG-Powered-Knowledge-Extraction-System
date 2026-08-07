import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

class RAGAPIError(Exception):
    """Custom exception class for RAG API failures."""
    pass

def call_openrouter_llm_safe(
    prompt_text,
    system_instruction=None,
    model_name="google/gemini-2.5-flash",
    max_retries=3,
    timeout_sec=30,
):
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RAGAPIError("OPENROUTER_API_KEY is missing in .env configuration.")

    if len(prompt_text) > 15000:
        prompt_text = prompt_text[:15000] + "\n...[Context truncated due to length limits]"

    # Correct OpenRouter Endpoint URL
    endpoint_url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt_text})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 600,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                endpoint_url, headers=headers, json=payload, timeout=timeout_sec
            )

            if response.status_code == 429:
                wait_time = attempt * 2
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            response_json = response.json()

            if "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["message"]["content"].strip()
            else:
                raise RAGAPIError("Empty response structure received from API.")

        except requests.exceptions.Timeout:
            if attempt == max_retries:
                raise RAGAPIError(f"API request timed out after {timeout_sec}s.")
            time.sleep(2)

        except requests.exceptions.HTTPError as http_err:
            raise RAGAPIError(f"HTTP Error encountered: {http_err} - {response.text}")

        except requests.exceptions.RequestException as req_err:
            if attempt == max_retries:
                raise RAGAPIError(f"Network error occurred: {req_err}")
            time.sleep(2)

    raise RAGAPIError("Failed to fetch response after maximum retries.")