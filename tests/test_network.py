"""Network diagnostics must not expose exception strings or proxy credentials."""
import errno
import json
import socket
import ssl
import unittest
import urllib.error

from bridge import ProtocolError, network_error_details


class NetworkDetailsTests(unittest.TestCase):
    def test_network_categories_keep_underlying_reason(self):
        cases = [
            (socket.gaierror(socket.EAI_AGAIN, 'private'), 'dns_temporary'),
            (socket.gaierror(socket.EAI_NONAME, 'private'), 'dns_resolution'),
            (ConnectionRefusedError(errno.ECONNREFUSED, 'private'), 'connection_refused'),
            (ConnectionResetError(errno.ECONNRESET, 'private'), 'connection_reset'),
            (TimeoutError('private'), 'timeout'),
            (ssl.SSLEOFError(8, 'private'), 'tls_eof'),
            (ssl.SSLCertVerificationError(1, 'private'), 'tls_certificate_verification'),
            (ssl.SSLError(1, 'private'), 'tls_error'),
        ]
        for error, expected in cases:
            with self.subTest(expected=expected):
                details = network_error_details(urllib.error.URLError(error))
                self.assertEqual(details['category'], expected)
                self.assertEqual(details['reason_type'], type(error).__name__)
                self.assertNotIn('private', json.dumps(details))

    def test_proxy_and_unknown_reason_strings_are_not_logged(self):
        secret = 'https://user:private-password@proxy.example/?token=private-token'
        for reason, category in [
            ('Tunnel connection failed: 502 ' + secret, 'proxy_tunnel_failure'),
            (secret, 'network_error'),
        ]:
            details = network_error_details(urllib.error.URLError(reason))
            self.assertEqual(details['category'], category)
            self.assertNotIn(secret, json.dumps(details))
            self.assertNotIn('private', json.dumps(details))

    def test_http_and_protocol_errors_are_not_network_connect_errors(self):
        error = urllib.error.HTTPError('https://example.invalid', 503, 'private', {}, None)
        try:
            self.assertIsNone(network_error_details(error))
        finally:
            error.close()
        self.assertIsNone(network_error_details(ProtocolError('invalid request')))


if __name__ == '__main__':
    unittest.main()
