#!/usr/bin/python

import socketserver
from http.server import SimpleHTTPRequestHandler


def serve():
    PORT = 8082
    Handler = SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", PORT), Handler)
    httpd.serve_forever()
    print("serving at port", PORT)
