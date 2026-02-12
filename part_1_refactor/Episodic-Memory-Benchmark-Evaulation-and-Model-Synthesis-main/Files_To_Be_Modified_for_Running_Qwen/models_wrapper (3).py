import requests
import json

class ModelsWrapper:
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", config={}):
        assert model_name is not None, f"model_name is required, got: {model_name}"
        self.model_name = model_name

        # OpenAI / "o" families
        if ("gpt-4o" in model_name) or ("o1" in model_name) or ("o3" in model_name) or ("o4" in model_name) or ("gpt-" in model_name):
            from openai import OpenAI
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)

        # Deepseek (prefer OpenRouter for long contexts)
        elif ("deepseek" in model_name):
            from openai import OpenAI
            self.client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
            from epbench.src.models.misc import no_ssl_verification
            no_ssl_verification()
            # prefer OpenRouter for long contexts
            self.client = None
            self.key = config.OPENROUTER_API_KEY

        # Grok / XAI
        elif ("grok" in model_name):
            from openai import OpenAI
            self.client = OpenAI(api_key=config.XAI_API_KEY, base_url="https://api.x.ai/v1")
            from epbench.src.models.misc import no_ssl_verification
            no_ssl_verification()

        # Claude -> route through OpenRouter (no Anthropic library)
        elif "claude-" in model_name:
            from epbench.src.models.misc import no_ssl_verification
            no_ssl_verification()
            self.client = None
            self.key = config.OPENROUTER_API_KEY

        # Google Gemini
        elif "gemini" in model_name:
            from epbench.src.models.misc import no_ssl_verification
            no_ssl_verification()
            from google import genai
            self.client = genai.Client(api_key=config.GOOGLE_API_KEY)

        # Llama via OpenRouter
        elif ("llama" in model_name):
            from epbench.src.models.misc import no_ssl_verification
            no_ssl_verification()
            self.client = None
            self.key = config.OPENROUTER_API_KEY

        # Qwen via OpenRouter
        elif "qwen" in model_name:
            from epbench.src.models.misc import no_ssl_verification
            no_ssl_verification()
            self.client = None
            self.key = config.OPENROUTER_API_KEY

        else:
            raise ValueError("Wrapper for this model name has not been coded, see ModelsWrapper class")

    def generate(self,
                 user_prompt: str = "Who are you?",
                 system_prompt: str = "You are a content event generator assistant.",
                 full_outputs=False,
                 max_new_tokens: int = 256,
                 temperature: float = 1.0,
                 keep_reasoning=False):

        reasoning = None
        outputs = None

        # ---------- OpenAI-style families ----------
        if "gpt-4o" in self.model_name:
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            if not full_outputs:
                outputs = outputs.choices[0].message.content

        elif "gpt-5" in self.model_name:
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=max_new_tokens,
                temperature=temperature
            )
            if not full_outputs:
                outputs = outputs.choices[0].message.content

        elif "gpt-4.1" in self.model_name:
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=max_new_tokens,
                temperature=temperature
            )
            if not full_outputs:
                outputs = outputs.choices[0].message.content

        # ---------- OpenAI "o" families ----------
        elif "o1" in self.model_name:
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": user_prompt}]
            )
            if not full_outputs:
                outputs = outputs.choices[0].message.content

        elif ("o3" in self.model_name) or ("o4" in self.model_name):
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": user_prompt}]
            )
            if not full_outputs:
                outputs = outputs.choices[0].message.content

        # ---------- Claude via OpenRouter ----------
        elif "claude-" in self.model_name:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            payload = {
                "model": "anthropic/" + self.model_name,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }

            headers = {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json"
            }

            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            resp.raise_for_status()
            parsed = resp.json()

            if not full_outputs:
                try:
                    outputs = parsed["choices"][0]["message"]["content"]
                except Exception:
                    outputs = parsed["choices"][0].get("text", "")
                reasoning = parsed["choices"][0].get("message", {}).get("reasoning", None)
            else:
                outputs = parsed

        # ---------- Gemini ----------
        elif "gemini" in self.model_name:
            outputs = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt
            )
            if not full_outputs:
                outputs = outputs.text

        # ---------- Llama via OpenRouter ----------
        elif "llama" in self.model_name:
            resp = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                data=json.dumps({
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "provider": {"order": ["Fireworks"]}
                })
            )
            resp.raise_for_status()
            parsed_dict = resp.json()
            if not full_outputs:
                outputs = parsed_dict['choices'][0]['message']['content']
            else:
                outputs = parsed_dict

        # ---------- Qwen via OpenRouter (no base_url) ----------
        elif "qwen" in self.model_name:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }

            headers = {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            parsed = response.json()

            if not full_outputs:
                try:
                    outputs = parsed["choices"][0]["message"]["content"]
                except Exception:
                    outputs = parsed["choices"][0].get("text", "")
                reasoning = parsed["choices"][0].get("message", {}).get("reasoning", None)
            else:
                outputs = parsed

        # ---------- Deepseek fallback to OpenRouter ----------
        elif "deepseek" in self.model_name:
            from epbench.src.generation.generate_3_secondary_entities import count_tokens
            nb_tokens_user_prompt = count_tokens(user_prompt)
            using_deepseek_api = True
            if nb_tokens_user_prompt > 60000:
                using_deepseek_api = False

            if using_deepseek_api:
                outputs = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=False
                )
                if not full_outputs:
                    reasoning = outputs.choices[0].message.reasoning_content
                    outputs = outputs.choices[0].message.content
            else:
                if self.model_name == "deepseek-reasoner":
                    model_name_here = "deepseek-r1"
                else:
                    model_name_here = self.model_name
                resp = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                    data=json.dumps({
                        "model": "deepseek/" + model_name_here,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "provider": {"order": ["Fireworks"],
                                     "ignore": ["Avian", "Novita", "DeepInfra", "Featherless", "DeepSeek", "Kluster", "Nebius", "Together"]},
                        "include_reasoning": True
                    })
                )
                resp.raise_for_status()
                parsed = resp.json()
                if not full_outputs:
                    outputs = parsed['choices'][0]['message']['content']
                    reasoning = parsed['choices'][0]['message'].get('reasoning', None)
                else:
                    outputs = parsed

        # ---------- Grok ----------
        elif "grok" in self.model_name:
            outputs = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False
            )
            if not full_outputs:
                outputs = outputs.choices[0].message.content

        else:
            raise ValueError("there is no generate function for this model name")

        if keep_reasoning:
            return outputs, reasoning

        return outputs