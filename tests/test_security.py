import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from heteromesh.security import SecurityStore, SecurityError, create_identity

class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.now=100
        self.store=SecurityStore(Path(self.tmp.name)/'db',clock=lambda:self.now)
    def tearDown(self): self.store.close(); self.tmp.cleanup()
    def test_pair_once_and_revoke(self):
        token=self.store.create_pairing()['token']
        node=self.store.register(token,{'backend':'cpu'})
        self.assertEqual(self.store.authenticate(node['token']),node['node_id'])
        with self.assertRaises(SecurityError): self.store.register(token,{})
        self.store.revoke(node['node_id'])
        with self.assertRaises(SecurityError): self.store.authenticate(node['token'])
    def test_expired_token(self):
        token=self.store.create_pairing()['token']; self.now+=301
        with self.assertRaises(SecurityError): self.store.register(token,{})
    def test_database_does_not_store_secrets(self):
        pairing=self.store.create_pairing()['token']
        node=self.store.register(pairing,{})
        dump=''.join(self.store.db.iterdump())
        self.assertNotIn(pairing,dump); self.assertNotIn(node['token'],dump)
    def test_identity_persists_and_permissions(self):
        root=Path(self.tmp.name)/'identity'
        first=create_identity(root); second=create_identity(root)
        self.assertEqual(first,second)
        self.assertEqual(len(first['fingerprint']),64)
        if os.name != 'nt': self.assertEqual(root.joinpath('key.pem').stat().st_mode & 0o777,0o600)
    def test_heartbeat_updates_only_authenticated_node(self):
        node=self.store.register(self.store.create_pairing()['token'],{'backend':'cpu'})
        self.now+=2; self.store.heartbeat(node['node_id'],{'backend':'cpu'})
        self.assertEqual(self.store.nodes()[0]['last_seen'],self.now)

class ClientPinTests(unittest.TestCase):
    def test_pin_checked_before_authorization_sent(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import ssl, threading
        from heteromesh.client import PinnedClient, PinError
        with tempfile.TemporaryDirectory() as tmp:
            identity=create_identity(Path(tmp)/'tls')
            observed=[]
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    observed.append(self.headers.get('Authorization'))
                    data=b'{"protocol_version":1}'
                    self.send_response(200); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
                def log_message(self,*args): pass
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
            context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); context.load_cert_chain(identity['cert'],identity['key'])
            server.socket=context.wrap_socket(server.socket,server_side=True)
            thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
            try:
                bad=PinnedClient('127.0.0.1',server.server_port,'0'*64,token='secret')
                with self.assertRaises(PinError): bad.request('GET','/v1/health')
                self.assertEqual(observed,[])
                good=PinnedClient('127.0.0.1',server.server_port,identity['fingerprint'],token='secret')
                self.assertEqual(good.request('GET','/v1/health')['protocol_version'],1)
                self.assertEqual(observed,['Bearer secret'])
            finally: server.shutdown(); server.server_close(); thread.join()
