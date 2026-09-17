"""CPU V4.1 protocol regression: reference parity, output validation and truncation.
Usage: python3 tools/ds41_protocol_test.py <fixture executable> <model encoding directory>
The encoder is read only; no model weights or accelerator are loaded.
"""
import copy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
sys.dont_write_bytecode = True
binary, ref_dir = sys.argv[1:]
spec=importlib.util.spec_from_file_location('reference',Path(ref_dir)/'encoding.py')
ref=importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
p=subprocess.Popen([binary],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
count=0

def call(c):
    p.stdin.write(json.dumps(c,ensure_ascii=False)+'\n');p.stdin.flush()
    return json.loads(p.stdout.readline())

def parity(c):
    global count
    messages=copy.deepcopy(c['messages'])
    if c.get('tools'):
        if messages[0]['role']!='system': messages.insert(0,{'role':'system','content':''})
        messages[0]['tools']=c['tools']
    want=ref.encode_messages(messages,thinking_mode=c.get('thinking_mode','chat'),
        reasoning_effort=c.get('reasoning_effort'),drop_thinking=c.get('drop_thinking',True),
        add_default_bos_token=c.get('add_default_bos_token',True))
    got=call(c)
    assert not got['error'],got
    if got['prompt']!=want:
        i=next((i for i,(a,b) in enumerate(zip(got['prompt'],want)) if a!=b),min(len(got['prompt']),len(want)))
        raise AssertionError((count,i,got['prompt'][max(0,i-40):i+100],want[max(0,i-40):i+100]))
    count+=1

try:
    # Actual shipped text/tool case and plain conversation. Internal task/reminder
    # and vision fixtures are outside this text+tools inference port.
    for n in [1,2]:
        raw=json.loads((Path(ref_dir)/f'tests/test_input_{n}.json').read_text())
        case=raw if isinstance(raw,dict) else {'messages':raw}
        parity(case)
        expected=(Path(ref_dir)/f'tests/test_output_{n}.txt').read_text()
        assert call(case)['prompt']==expected, f'shipped golden {n}'
    tool={'type':'function','function':{'name':'lookup','description':'Look up 雪 "q", key: value',
        'parameters':{'type':'object','properties':{'query':{'type':'string'},'count':{'type':'number'}},'required':['query']}}}
    rng=random.Random(41)
    for i in range(120):
        args={'query':'雪 <｜DSML｜ calls> \\ "\nnext','count':rng.randint(-1000,1000),'nested':{'x':[True,None,2.5,'a,b: c']}}
        calls=[{'id':'a','type':'function','function':{'name':'lookup','arguments':json.dumps(args,ensure_ascii=False)}},
               {'id':'b','type':'function','function':{'name':'lookup','arguments':{}}}]
        if i%5==0: calls[0]['function']['arguments']=json.dumps(calls[0]['function']['arguments'])
        tools=[copy.deepcopy(tool)]
        if i%3==0:
            tools[0]['namespace']={'name':'web','description':'Search namespace'}
            for tc in calls: tc['namespace']='web'
        messages=[{'role':'user','content':'question 雪'}, {'role':'assistant','content':'summary','reasoning_content':'keep this reason','tool_calls':calls},
            {'role':'tool','tool_call_id':'b','content':'second'}, {'role':'user','content':'between'},
            {'role':'tool','tool_call_id':'a','content':'first'}, {'role':'user','content':'answer please'}]
        if i%2: messages.insert(0,{'role':'system','content':'helpful'})
        parity({'messages':messages,'tools':tools,'thinking_mode':'thinking' if i%2 else 'chat','drop_thinking':bool(i%3),
            'reasoning_effort':[1,50,75,100,'low','high','max'][i%7],'add_default_bos_token':bool(i%4)})
    for number in [1e-4,1e-5,1e-6,1e-7,1e15,1e16,1e20,1e21,1.23e-12,-0.0]:
        t=copy.deepcopy(tool); t['function']['parameters']['default']=number
        parity({'messages':[{'role':'user','content':'test'}],'tools':[t]})
    # Tool history without a current tool list, and merged plain-user turns.
    for thinking in ['chat','thinking']:
        parity({'messages':messages,'thinking_mode':thinking})
        parity({'messages':[{'role':'user','content':'one'},{'role':'user','content':'two'}],'thinking_mode':thinking})
    EOS='<｜end▁of▁sentence｜>'
    tc='\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n<｜DSML｜ parameter name="query" string="true">雪 \\ "hello"\n<｜DSML｜ calls></｜DSML｜ parameter>\n<｜DSML｜ parameter name="count" string="false">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    for thinking in [False,True]:
        text=('reason</think>' if thinking else '')+'summary'+tc+EOS
        got=call({'completion':text,'thinking':thinking})
        want=ref.parse_message_from_completion_text(text,thinking_mode='thinking' if thinking else 'chat')
        assert not got['error'],got
        for key in ['content','reasoning_content','tool_calls']:
            assert got[key]==want.get(key,[] if key=='tool_calls' else ''),(key,got,want)
        if got['tool_calls']:
            stream=[json.loads(s[6:]) for s in got['sse'].splitlines() if s.startswith('data: {')]
            actual=[c['function'] for frame in stream for c in frame['choices'][0]['delta'].get('tool_calls',[])]
            assert actual==[c['function'] for c in got['tool_calls']]
            assert all('<｜DSML｜' not in f['choices'][0]['delta'].get('content','') for f in stream)
            response=got['response']['choices'][0]
            assert response['finish_reason']=='tool_calls'
            history={'messages':[{'role':'user','content':'lookup'},response['message'],
                {'role':'tool','tool_call_id':response['message']['tool_calls'][0]['id'],'content':'value'}],
                'tools':[tool],'enable_thinking':thinking}
            through_http=call({'request':history})
            direct=call({**history,'thinking_mode':'thinking' if thinking else 'chat'})
            assert through_http==direct,(through_http,direct)
        count+=1
        for end in range(len(text)-len(EOS)):
            bad=call({'completion':text[:end],'thinking':thinking})
            assert bad['error'] and not bad['tool_calls'],(end,bad)
            count+=1
    for bad in [tc.replace('>2</','>invalid</'),tc+"tail",tc.replace('name="count"','name="query"'),tc.replace('</｜DSML｜ invoke>','')]:
        got=call({'completion':bad+EOS});assert got['error'] and not got['tool_calls'],got;count+=1
    empty='\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="ping">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'+EOS
    assert call({'completion':empty})['tool_calls'][0]['function']['arguments']=='{}';count+=1
    # Every possible byte/text fragment of the marker stays out of visible content.
    marker='<｜DSML｜'
    for i in range(1,len(marker)+1):
        assert call({'partial':'hello'+marker[:i]})['content']=='hello';count+=1
    for msg in [{'role':'latest_reminder','content':'unsupported'},{'role':'user','content':[{'type':'image_url','image_url':{'url':'x'}}]}]:
        got=call({'messages':[msg]});assert got['error'] and not got['prompt'];count+=1
    print(f'PASS: {count} protocol checks; shipped golden files 1 and 2 match byte for byte. Internal-task/reminder and vision files 3–5 remain unsupported.')
finally:
    p.stdin.close();p.wait(timeout=10)
