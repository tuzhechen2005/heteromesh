"""Local TLS identity and one-time device pairing; bearer secrets are hashed at rest."""
from __future__ import annotations
import hashlib
import json
import os
import secrets
import sqlite3
import ssl
import subprocess
import threading
import time
from pathlib import Path

class SecurityError(ValueError):
    pass

def _hash(token):
    if not isinstance(token,str) or len(token)>256: raise SecurityError('UNAUTHORIZED')
    return hashlib.sha256(token.encode()).hexdigest()

def create_identity(directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    cert,key,admin=(directory/name for name in ('cert.pem','key.pem','admin.token'))
    if cert.exists()!=key.exists(): raise SecurityError('INCOMPLETE_IDENTITY')
    if not cert.exists():
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-sha256','-days','365','-subj','/CN=HeteroMesh Local Coordinator','-keyout',str(key),'-out',str(cert)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    key.chmod(0o600)
    if not admin.exists():
        fd=os.open(admin,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f: f.write(secrets.token_hex(32))
    admin.chmod(0o600)
    der=ssl.PEM_cert_to_DER_cert(cert.read_text())
    return {'cert':str(cert),'key':str(key),'admin_token':admin.read_text().strip(),'fingerprint':hashlib.sha256(der).hexdigest()}

class SecurityStore:
    def __init__(self,path,*,clock=time.time):
        self.clock=clock; self.lock=threading.RLock()
        self.db=sqlite3.connect(str(path),check_same_thread=False,isolation_level=None)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS pairing(token_hash TEXT PRIMARY KEY,expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY,token_hash TEXT UNIQUE NOT NULL,capabilities TEXT NOT NULL,last_seen REAL NOT NULL,revoked INTEGER NOT NULL DEFAULT 0);''')
    def close(self): self.db.close()
    def create_pairing(self):
        token=secrets.token_hex(32); expires=int(self.clock()+300)
        with self.lock:
            self.db.execute('INSERT INTO pairing VALUES(?,?)',(_hash(token),expires))
        return {'token':token,'expires':expires}
    def register(self,token,capabilities):
        encoded=json.dumps(capabilities,allow_nan=False)
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                pair=self.db.execute('SELECT * FROM pairing WHERE token_hash=?',(_hash(token),)).fetchone()
                if pair is None or pair['expires']<=self.clock(): raise SecurityError('UNAUTHORIZED')
                self.db.execute('DELETE FROM pairing WHERE token_hash=?',(_hash(token),))
                node_id=secrets.token_hex(16); node_token=secrets.token_hex(32)
                self.db.execute('INSERT INTO nodes VALUES(?,?,?,?,0)',(node_id,_hash(node_token),encoded,self.clock()))
                self.db.execute('COMMIT')
                return {'node_id':node_id,'token':node_token}
            except BaseException:
                self.db.execute('ROLLBACK'); raise
    def authenticate(self,token):
        with self.lock:
            node=self.db.execute('SELECT id FROM nodes WHERE token_hash=? AND revoked=0',(_hash(token),)).fetchone()
            if node is None: raise SecurityError('UNAUTHORIZED')
            return node['id']
    def revoke(self,node_id):
        with self.lock: self.db.execute('UPDATE nodes SET revoked=1 WHERE id=?',(node_id,))
    def heartbeat(self,node_id,capabilities):
        with self.lock:
            changed=self.db.execute('UPDATE nodes SET last_seen=?,capabilities=? WHERE id=? AND revoked=0',(self.clock(),json.dumps(capabilities,allow_nan=False),node_id)).rowcount
            if not changed: raise SecurityError('UNAUTHORIZED')
    def nodes(self):
        with self.lock:
            return [{'node_id':r['id'],'capabilities':json.loads(r['capabilities']),'last_seen':int(r['last_seen']),'revoked':bool(r['revoked'])} for r in self.db.execute('SELECT * FROM nodes')]
