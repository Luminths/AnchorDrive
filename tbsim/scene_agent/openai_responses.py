"""Translate the pipeline's conversation format to the OpenAI Responses API."""
import copy


def request_body(chat, config):
    items = []
    for message in chat['messages']:
        role = message['role']
        if role == 'assistant' and message.get('_responses_output'):
            items.extend(copy.deepcopy(message['_responses_output']))
            continue
        if role == 'tool':
            items.append(dict(type='function_call_output', call_id=message['tool_call_id'], output=message['content']))
            continue
        content = message.get('content')
        if content:
            if isinstance(content, str):
                parts = [dict(type='output_text' if role == 'assistant' else 'input_text', text=content)]
            else:
                parts = []
                for part in content:
                    if part['type'] == 'text':
                        parts.append(dict(type='input_text', text=part['text']))
                    elif part['type'] == 'image_url':
                        parts.append(dict(type='input_image', image_url=part['image_url']['url']))
                    else:
                        raise ValueError('Unsupported OpenAI content type: '+part['type'])
            items.append(dict(role='developer' if role == 'system' else role, content=parts))
        for call in message.get('tool_calls', []):
            items.append(dict(type='function_call', call_id=call['id'], name=call['function']['name'], arguments=call['function']['arguments']))
    result = dict(model=chat['model'], input=items, store=False,
                  include=['reasoning.encrypted_content'], reasoning={'effort':config.get('reasoning_effort', 'low')},
                  max_output_tokens=chat['max_tokens'])
    if chat.get('tools'):
        result['tools'] = [dict(type='function', name=t['function']['name'],
            description=t['function'].get('description',''), parameters=t['function']['parameters'], strict=False)
            for t in chat['tools']]
        result['tool_choice'] = 'auto'
        result['parallel_tool_calls'] = False
    elif chat.get('response_format', {}).get('type') == 'json_object':
        result['text'] = {'format':{'type':'json_object'}}
    return result


def chat_response(response):
    if response.get('status') not in ('completed','incomplete'):
        raise ValueError('OpenAI response did not complete: '+str(response.get('status')))
    output = response.get('output', [])
    content = ''.join(p.get('text','') for item in output if item.get('type') == 'message'
                      for p in item.get('content', []) if p.get('type') == 'output_text')
    calls = [dict(id=item['call_id'],type='function',function=dict(name=item['name'],arguments=item['arguments']))
             for item in output if item.get('type') == 'function_call']
    message = dict(role='assistant',content=content,_responses_output=output)
    if calls: message['tool_calls'] = calls
    usage = response.get('usage') or {}
    inp, out = usage.get('input_tokens',0), usage.get('output_tokens',0)
    return dict(id=response.get('id'),model=response.get('model'),
        usage=dict(prompt_tokens=inp,completion_tokens=out,total_tokens=usage.get('total_tokens',inp+out),
            prompt_tokens_details=usage.get('input_tokens_details',{}),completion_tokens_details=usage.get('output_tokens_details',{})),
        choices=[dict(finish_reason='length' if response.get('status') == 'incomplete' else 'tool_calls' if calls else 'stop', message=message)])
