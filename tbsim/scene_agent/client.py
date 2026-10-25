"""Bounded OpenAI, DeepSeek and Qwen calls. Credentials and provider error bodies never enter logs."""
import json
import base64
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


class BudgetExhausted(RuntimeError):
    pass


class QwenClient:
    def __init__(self, config, log_path=None):
        self.config = config
        self.log_path = Path(log_path) if log_path else None
        self.requests = self.tokens = 0
        self.http_requests = 0

    def _credential(self):
        env_name = self.config.get('api_key_env', 'DEEPSEEK_API_KEY')
        key = os.environ.get(env_name, '').strip()
        optional_file = self.config.get('api_key_file')
        if not key and optional_file:
            key = Path(optional_file).expanduser().read_text().strip()
        if not key:
            raise ValueError('Missing API credential: set ' + env_name)
        return key

    def _request_json(self, request, record):
        """Bounded retries for rate limits/overload; never retry account errors."""
        c=self.config
        use_responses = c.get('provider') == 'openai'
        if use_responses:
            from .openai_responses import request_body, chat_response
            body = request_body(json.loads(request.data), c)
            request = urllib.request.Request(c['base_url'].rstrip('/')+'/responses',
                data=json.dumps(body).encode(), headers=dict(request.header_items()), method='POST')
        retries=max(0,min(3,int(c.get('http_max_retries',0))))
        attempts=[]
        for index in range(retries+1):
            if self.http_requests>=c['max_requests']:
                record['error']='Qwen HTTP attempt budget exhausted'
                raise BudgetExhausted(record['error'])
            self.http_requests+=1
            started=time.monotonic()
            try:
                with urllib.request.urlopen(request,timeout=c['request_timeout_s']) as response:
                    data=json.load(response)
                attempts.append(dict(status=200,elapsed_s=round(time.monotonic()-started,3)))
                record['http_attempts']=attempts
                return chat_response(data) if use_responses else data
            except urllib.error.HTTPError as exc:
                attempts.append(dict(status=exc.code,elapsed_s=round(time.monotonic()-started,3)))
                record['http_attempts']=attempts
                if exc.code not in (429,500,503) or index==retries or self.http_requests>=c['max_requests']:
                    raise
                try:delay=float(exc.headers.get('Retry-After',''))
                except (TypeError,ValueError,AttributeError):delay=2**(index+1)+random.uniform(0,1)
                exc.close()
                time.sleep(max(0,min(30,delay)))

    def chat(self, messages, tools=None):
        """One native tool-calling turn; the caller owns the persistent history."""
        c=self.config
        if self.requests>=c['max_requests'] or self.tokens>=c['max_total_tokens']:
            raise BudgetExhausted('Qwen request/token budget exhausted')
        key=self._credential()
        body=dict(model=c['model'],messages=messages,temperature=c['temperature'],
                  max_tokens=c['response_max_tokens'])
        if c.get('provider')=='deepseek':
            body['thinking']={'type':'enabled' if c.get('enable_thinking',False) else 'disabled'}
            if c.get('enable_thinking') and c.get('reasoning_effort'):
                body['reasoning_effort']=c['reasoning_effort']
        else:
            body['enable_thinking']=c.get('enable_thinking',False)
        if tools:
            body['tools']=tools
            body['tool_choice']='auto'
        elif c.get('chat_json_output'):
            body['response_format']={'type':'json_object'}
        request=urllib.request.Request(c['base_url'].rstrip('/')+'/chat/completions',
            data=json.dumps(body,ensure_ascii=False).encode(),
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        self.requests+=1
        started=time.monotonic()
        # A full history contains image data URLs. Log its shape, never its payload.
        record=dict(call=self.requests,message_count=len(messages),
            image_count=sum(sum(part.get('type')=='image_url' for part in msg.get('content',[]))
                            for msg in messages if isinstance(msg.get('content'),list)))
        try:
            data=self._request_json(request,record)
            usage=data.get('usage') or {}
            self.tokens+=int(usage.get('total_tokens',0))
            # A truncated answer is still billed; retain usage without its text.
            record.update(usage=usage,request_id=data.get('id'),model=data.get('model'))
            choice=data['choices'][0]
            if choice.get('finish_reason')=='length':
                raise ValueError('Qwen response truncated')
            message=choice['message']
            if key in json.dumps(message,ensure_ascii=False):
                raise ValueError('Credential unexpectedly present in model response')
            record.update(usage=usage,request_id=data.get('id'),model=data.get('model'),
                response=dict(content=message.get('content'),
                    tool_names=[call.get('function',{}).get('name') for call in message.get('tool_calls',[])]))
            return message
        except urllib.error.HTTPError as exc:
            status=exc.code;exc.close()
            record['error']='Qwen HTTP '+str(status)
            raise RuntimeError(record['error']) from None
        except (urllib.error.URLError,TimeoutError,OSError) as exc:
            record['error']='Qwen transport error: '+type(exc).__name__
            raise RuntimeError(record['error']) from None
        except (ValueError,KeyError,IndexError,TypeError) as exc:
            record['error']='Qwen format error: '+str(exc)[:180]
            raise ValueError(record['error']) from None
        finally:
            record['elapsed_s']=round(time.monotonic()-started,3)
            if self.log_path:
                self.log_path.parent.mkdir(parents=True,exist_ok=True)
                with self.log_path.open('a') as f:
                    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')

    def complete(self, system, payload, reserve_tokens=0, image_path=None):
        c = self.config
        if self.requests >= c['max_requests'] or self.tokens >= c['max_total_tokens']:
            raise BudgetExhausted('Qwen request/token budget exhausted')
        encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False)
        # Conservative estimate for this JSON/Chinese workload, not provider tokenization.
        estimated=(len(system.encode())+len(encoded.encode()))//2+c['response_max_tokens']
        if reserve_tokens and self.tokens+estimated+reserve_tokens>c['max_total_tokens']:
            raise BudgetExhausted('Stop decisions early to preserve final semantic-review budget')
        if reserve_tokens and self.requests>=c['max_requests']-1:
            raise BudgetExhausted('Final semantic-review request reserved')
        key = self._credential()
        user_content=encoded
        if image_path is not None:
            image_path=Path(image_path)
            mime='image/png' if image_path.suffix.lower()=='.png' else 'image/jpeg'
            data_url='data:'+mime+';base64,'+base64.b64encode(image_path.read_bytes()).decode('ascii')
            user_content=[dict(type='text',text=encoded),
                          dict(type='image_url',image_url=dict(url=data_url))]
        body = dict(model=c['model'], messages=[dict(role='system', content=system),
                    dict(role='user', content=user_content)],
                    response_format={'type': 'json_object'}, temperature=c['temperature'],
                    max_tokens=c['response_max_tokens'])
        if c.get('provider')=='deepseek':
            body['thinking']={'type':'disabled'}
        else:
            body['enable_thinking']=False
        request = urllib.request.Request(c['base_url'].rstrip('/')+'/chat/completions',
                   data=json.dumps(body).encode(), headers={'Authorization': 'Bearer '+key,
                   'Content-Type': 'application/json'}, method='POST')
        self.requests += 1
        started = time.monotonic()
        # Keep the image file path in the audit log, never its large base64 body.
        logged_body=dict(body)
        logged_body['messages']=[dict(role='system',content=system),
            dict(role='user',content=encoded)]
        record = dict(call=self.requests, request=logged_body)
        if image_path is not None:record['image_path']=str(image_path)
        try:
            data=self._request_json(request,record)
            usage = data.get('usage') or {}
            self.tokens += int(usage.get('total_tokens', 0))
            record.update(usage=usage, request_id=data.get('id'), model=data.get('model'))
            choice = data['choices'][0]
            if choice.get('finish_reason') == 'length':
                raise ValueError('Qwen response truncated; reduce proposal length')
            raw = choice['message']['content']
            if key in raw:
                raise ValueError('Credential unexpectedly present in model response')
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError('Expected a JSON object')
            record['response'] = result
            return result
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            record['error'] = 'Qwen HTTP '+str(status)
            raise RuntimeError(record['error']) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            record['error'] = 'Qwen transport error: '+type(exc).__name__
            raise RuntimeError(record['error']) from None
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            record['error'] = 'Qwen format error: '+type(exc).__name__
            raise ValueError(record['error']) from None
        finally:
            record['elapsed_s'] = round(time.monotonic()-started, 3)
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open('a') as f:
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
