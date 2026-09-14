"""Actual installed PyO3 gateway regression in a network-none, GPU-free container.

Only localhost fake workers are used. No model credential or real inference.
"""
import argparse
import hashlib
import http.client
import http.server
import importlib.util
import json
import pathlib
import socket
import subprocess
import threading
import time

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--expect-buffered', action='store_true')
a = p.parse_args()
out = pathlib.Path(a.output)
out.mkdir(parents=True, exist_ok=False)
events = {}
fixtures = {
    'timing': {'p_delay': 3, 'chunks': 12, 'gap': .05},
    'late-failure': {'p_delay': .4, 'p_status': 500, 'chunks': 100, 'gap': .05},
    'cancel': {'p_delay': 1, 'chunks': 100, 'gap': .05},
    'nonstream': {'p_delay': .5, 'chunks': 1, 'gap': 0},
}


class Worker(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def respond(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()

    def do_GET(self):
        if self.path in ['/server_info', '/get_server_info']:
            self.respond(200, {'model_path': 'mock-model', 'served_model_name': 'mock-model',
                'dp_size': 1, 'tp_size': 1, 'context_length': 32768, 'version': '0.3.0',
                'internal_states': [{'waiting_queue_size': 0, 'running_queue_size': 0}]})
        elif self.path in ['/get_model_info', '/model_info']:
            self.respond(200, {'model_path': 'mock-model', 'tokenizer_path': 'mock-tokenizer',
                              'is_generation': True})
        elif self.path == '/v1/models':
            self.respond(200, {'object': 'list', 'data': [{'id': 'mock-model', 'object': 'model'}]})
        else:
            self.respond(200, {'status': 'ok', 'running_requests': 0, 'waiting_requests': 0})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        body = json.loads(raw or b'{}')
        case = body.get('text')
        if case not in fixtures:
            self.respond(200, {'success': True})
            return
        cfg = fixtures[case]
        role = self.server.role
        record = events.setdefault(case, {}).setdefault(role, {'chunks': 0})
        record['accepted'] = time.monotonic()
        try:
            if role == 'prefill':
                time.sleep(cfg['p_delay'])
                self.respond(cfg.get('p_status', 200), {'text': '', 'meta_info': {
                    'prompt_tokens': 4, 'completion_tokens': 1}})
                record['body_sent'] = time.monotonic()
            elif not body.get('stream'):
                self.respond(200, {'text': 'fixture complete', 'meta_info': {
                    'prompt_tokens': 4, 'completion_tokens': 1}})
                record['body_sent'] = time.monotonic()
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.flush()
                record['headers_sent'] = time.monotonic()
                for i in range(cfg['chunks']):
                    self.wfile.write(('data: ' + json.dumps({'text': 'x' * (i + 1),
                        'meta_info': {'prompt_tokens': 4, 'completion_tokens': i + 1}}) + '\n\n').encode())
                    self.wfile.flush()
                    record['chunks'] += 1
                    time.sleep(cfg['gap'])
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
                record['body_sent'] = time.monotonic()
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            record['disconnected'] = time.monotonic()
        finally:
            record['finished'] = time.monotonic()


servers = []
for port, role in [(30201, 'prefill'), (30202, 'decode')]:
    server = http.server.ThreadingHTTPServer(('127.0.0.1', port), Worker)
    server.daemon_threads = True
    server.role = role
    threading.Thread(target=server.serve_forever, daemon=True).start()
    servers.append(server)
command = ['python3', '-m', 'sglang_router.launch_router', '--host', '127.0.0.1',
    '--port', '30200', '--pd-disaggregation', '--prefill', 'http://127.0.0.1:30201',
    '--decode', 'http://127.0.0.1:30202', '--policy', 'round_robin', '--disable-retries',
    '--worker-startup-timeout-secs', '30', '--worker-startup-check-interval', '1',
    '--request-timeout-secs', '15']
binary = pathlib.Path(importlib.util.find_spec('sglang_router.sglang_router_rs').origin)
result = {'command': command, 'extension_path': str(binary),
    'extension_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'test_boundary': 'network-none container localhost fake workers; no GPU/model/TOKEN',
    'expect_buffered': a.expect_buffered, 'cases': [], 'completed': False}


def save():
    result['worker_events'] = events
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


def run_case(name):
    c = http.client.HTTPConnection('127.0.0.1', 30200, timeout=10)
    t = time.monotonic()
    c.request('POST', '/generate', body=json.dumps({'text': name, 'stream': name != 'nonstream'}),
              headers={'Content-Type': 'application/json'})
    response = c.getresponse()
    row = {'name': name, 'http': response.status, 'headers_s': time.monotonic() - t,
           'data_times_s': [], 'payloads': []}
    if name == 'nonstream' or response.status != 200:
        row['body'] = response.read().decode()
    else:
        while line := response.readline():
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip().decode()
            row['data_times_s'].append(time.monotonic() - t)
            row['payloads'].append(payload)
            if name == 'cancel' and len(row['payloads']) >= 2:
                break
            if payload == '[DONE]':
                break
    row['client_elapsed_s'] = time.monotonic() - t
    response.close()
    c.close()
    # Allow the pending prefill body drain / upstream disconnect to finish.
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        state = events.get(name, {})
        if all('finished' in state.get(role, {}) for role in ['prefill', 'decode']):
            break
        time.sleep(.02)
    state = events.get(name, {})
    if name == 'timing':
        first = row['data_times_s'][0] if row['data_times_s'] else float('inf')
        row['buffering_reproduced'] = row['headers_s'] > 2.5 and first > 2.5
        row['streaming_pass'] = (row['http'] == 200 and row['headers_s'] < 1.5 and
            first < 1.5 and row['data_times_s'][-1] - first > .4 and
            row['payloads'][-1] == '[DONE]' and 'body_sent' in state.get('prefill', {}))
        row['pass'] = row['buffering_reproduced'] if a.expect_buffered else row['streaming_pass']
    elif name == 'late-failure':
        error = any(x != '[DONE]' and json.loads(x).get('error', {}).get('type') == 'prefill_failed'
                    for x in row['payloads'])
        row['pass'] = (row['http'] == 200 and error and '[DONE]' not in row['payloads'] and
            state.get('decode', {}).get('chunks', 100) < 100)
    elif name == 'cancel':
        row['pass'] = ('disconnected' in state.get('decode', {}) and
            state['decode']['chunks'] < 100 and 'body_sent' in state.get('prefill', {}))
    else:
        row['pass'] = (row['http'] == 200 and json.loads(row['body'])['text'] == 'fixture complete'
                       and row['headers_s'] >= .45)
    result['cases'].append(row)
    save()
    print(json.dumps(row), flush=True)


gateway = None
try:
    with (out / 'gateway.log').open('w') as log:
        gateway = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if gateway.poll() is not None:
                raise RuntimeError('Gateway exited during startup')
            try:
                conn = http.client.HTTPConnection('127.0.0.1', 30200, timeout=1)
                conn.request('GET', '/v1/models')
                response = conn.getresponse()
                raw = response.read()
                conn.close()
                if response.status == 200 and b'mock-model' in raw:
                    break
            except OSError:
                pass
            time.sleep(.25)
        else:
            raise RuntimeError('Gateway readiness timed out')
        for name in (['timing'] if a.expect_buffered else fixtures):
            run_case(name)
        result['completed'] = True
        result['pass'] = all(row['pass'] for row in result['cases'])
except Exception as exc:
    result['failure_type'] = type(exc).__name__
    result['failure'] = str(exc)
finally:
    if gateway is not None and gateway.poll() is None:
        gateway.terminate()
        try:
            gateway.wait(timeout=20)
        except subprocess.TimeoutExpired:
            gateway.kill()
            gateway.wait()
    for server in servers:
        server.shutdown()
    save()
print(json.dumps({key: result.get(key) for key in ['completed', 'pass', 'failure']}), flush=True)
raise SystemExit(0 if result.get('pass') else 1)
