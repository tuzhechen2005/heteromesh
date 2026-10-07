"""Real TLS handshake: a mismatched DER pin must send no HTTP authorization."""
import hashlib
import http.server
import json
import os
from pathlib import Path
import ssl
import subprocess
import tempfile
import threading


def main():
    binary = Path(__file__).resolve().parents[1] / '.build/debug/apple-node'
    with tempfile.TemporaryDirectory() as tmp:
        cert, key = Path(tmp)/'cert.pem', Path(tmp)/'key.pem'
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1',
                        '-keyout',str(key),'-out',str(cert),'-subj','/CN=localhost'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        digest = hashlib.sha256(ssl.PEM_cert_to_DER_cert(cert.read_text())).hexdigest()
        seen = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                seen.append(self.headers.get('Authorization'))
                data = json.dumps({'protocol_version':1,'status':'ok'}).encode()
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        server = http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert,key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            address = 'https://127.0.0.1:' + str(server.server_address[1])
            env = dict(os.environ,HETEROMESH_TEST_TOKEN='synthetic-test-secret')
            wrong = subprocess.run([str(binary),'probe',address,'0'*64],env=env,capture_output=True,timeout=20)
            assert wrong.returncode != 0, 'mismatched certificate accepted'
            assert seen == [], 'Authorization/HTTP sent before pin validation'
            right = subprocess.run([str(binary),'probe',address,digest],env=env,capture_output=True,timeout=20)
            assert right.returncode == 0, right.stderr.decode()+right.stdout.decode()
            assert seen == ['Bearer synthetic-test-secret'], seen
            print('PASS: real TLS pin mismatch sends no HTTP; confirmed pin permits request')
        finally:
            server.shutdown();server.server_close();thread.join()

if __name__ == '__main__':
    main()
