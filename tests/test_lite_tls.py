"""Exercise AnyIO's TLS hostname verification on loopback only."""
import socket
import ssl
import subprocess
import tempfile
import threading
from pathlib import Path

import anyio


def check_hostname(cert, key, hostname, expected):
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert, key)
    client_context = ssl.create_default_context(cafile=cert)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen(1)
    port = sock.getsockname()[1]

    def serve():
        try:
            client, _ = sock.accept()
            with server_context.wrap_socket(client, server_side=True) as conn:
                if conn.recv(4) == b'ping':
                    conn.sendall(b'pong')
                    conn.unwrap().close()
        except ssl.SSLError:
            pass  # A rejected certificate causes a TLS alert on the server.
        finally:
            sock.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    async def connect():
        try:
            stream = await anyio.connect_tcp('127.0.0.1', port, tls=True,
                                             tls_hostname=hostname, ssl_context=client_context)
            async with stream:
                await stream.send(b'ping')
                assert await stream.receive(4) == b'pong'
            return 'accepted'
        except ssl.SSLCertVerificationError:
            return 'rejected'

    actual = anyio.run(connect)
    thread.join(timeout=5)
    assert not thread.is_alive(), 'loopback TLS server did not finish'
    assert actual == expected, (hostname, actual, expected)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        cert = str(Path(tmp) / 'cert.pem')
        key = str(Path(tmp) / 'key.pem')
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                        '-keyout', key, '-out', cert, '-days', '1', '-subj', '/CN=fass.test',
                        '-addext', 'subjectAltName=DNS:fass.test'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        check_hostname(cert, key, 'fass.test', 'accepted')
        check_hostname(cert, key, 'faß.test', 'rejected')
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                        '-keyout', key, '-out', cert, '-days', '1', '-subj', '/CN=xn--fa-hia.test',
                        '-addext', 'subjectAltName=DNS:xn--fa-hia.test'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        check_hostname(cert, key, 'faß.test', 'accepted')
    print('loopback TLS accepts matching ASCII and IDNA names and rejects the IDNA mismatch')


if __name__ == '__main__':
    main()
