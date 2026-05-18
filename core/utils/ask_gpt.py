import os
import json
from threading import Lock
import json_repair
from openai import OpenAI
from core.utils.config_utils import load_key, load_timeout
from rich import print as rprint
from core.utils.decorator import except_handler

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'

# ------------
# OpenAI client cache (per api_key + base_url).
# Reuses the underlying httpx connection pool across calls, which avoids paying
# TCP + TLS handshake on every request. Handshake cost dominates wall-clock under
# high concurrency on high-RTT links (e.g. local machine -> overseas API),
# where re-building a client for each ask_gpt() invocation made 60 workers
# behave almost serially. The OpenAI sync client is thread-safe.
# ------------
_CLIENT_CACHE = {}
_CLIENT_LOCK = Lock()


def _get_client():
    api_key = load_key("api.key")
    base_url = load_key("api.base_url")
    if 'ark' in base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3"  # huoshan base url
    elif 'v1' not in base_url:
        base_url = base_url.strip('/') + '/v1'
    cache_key = (api_key, base_url)
    client = _CLIENT_CACHE.get(cache_key)
    if client is not None:
        return client
    with _CLIENT_LOCK:
        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            # Some API endpoints behind a CDN/WAF (e.g. Cloudflare) silently block
            # requests whose User-Agent starts with "OpenAI/Python ...". Override
            # it with a generic UA so the request behaves like a normal HTTP client.
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={"User-Agent": "python-requests/2.32.3"},
                timeout=load_timeout("llm", 300),
            )
            _CLIENT_CACHE[cache_key] = client
        return client


def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append({"model": model, "prompt": prompt, "resp_content": resp_content, "resp_type": resp_type, "resp": resp, "message": message})
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        return item["resp"]
        return False

# ------------
# ask gpt once
# ------------

@except_handler("GPT request failed", retry=5)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    if not load_key("api.key"):
        raise ValueError("API key is not set")
    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        rprint("use cache response")
        return cached

    model = load_key("api.model")
    client = _get_client()
    response_format = {"type": "json_object"} if resp_type == "json" and load_key("api.llm_support_json") else None

    messages = [{"role": "user", "content": prompt}]

    params = dict(
        model=model,
        messages=messages,
        response_format=response_format,
        timeout=load_timeout("llm", 300)
    )
    resp_raw = client.chat.completions.create(**params)

    # process and return full result
    resp_content = resp_raw.choices[0].message.content
    if resp_type == "json":
        resp = json_repair.loads(resp_content)
    else:
        resp = resp_content
    
    # check if the response format is valid
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
    return resp


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
