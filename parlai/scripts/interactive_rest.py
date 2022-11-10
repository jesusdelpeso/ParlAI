"""
Talk with a model using a REST API.

## Examples

```shell
parlai interactive_rest --model-file "zoo:tutorial_transformer_generator/model"
```

TODO: Add example messages
"""

from copy import copy, deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from typing import Dict, Any
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging

import json
import time

from parlai.agents.local_human.local_human import LocalHumanAgent

HOST_NAME = 'localhost'
PORT = 8080

SHARED: Dict[Any, Any] = {}


class MyHandler(BaseHTTPRequestHandler):
    """
    Handle HTTP requests.
    """

    def _interactive_running(self, opt, conversation_id, reply_text):
        # Manage conversation
        active_conversations = SHARED["active_conversations"]
        if conversation_id not in active_conversations:
            active_conversations[conversation_id] = self._clone_agent(SHARED["agent"])
            active_conversations[conversation_id].reset()
        reply = {'episode_done': False, 'text': reply_text}
        active_conversations[conversation_id].observe(reply)
        model_res = active_conversations[conversation_id].act()
#        SHARED['agent'].observe(reply)
#        model_res = SHARED['agent'].act()
        return model_res

    def _clone_agent(self, agent):
        res = copy(agent)
        res.history = deepcopy(agent.history)
        return res

    def do_HEAD(self):
        """
        Handle HEAD requests.
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        """
        Handle POST request, especially replying to a chat message.
        """
        if self.path == '/interact':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            decoded_body = json.loads(body.decode('utf-8'))
            conversation_id = decoded_body.get("conversation_id", None)
            text = decoded_body["text"]
            print(f"*** Text: {text}")  # Debug
            model_response = self._interactive_running(
                SHARED.get('opt'), conversation_id, text
            )
            print(f"*** RESP: {model_response}")  # Debug
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            json_str = json.dumps(model_response.json_safe_payload())
            self.wfile.write(bytes(json_str, 'utf-8'))
        elif self.path == '/reset':
            # TODO Save conversation to logs
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            decoded_body = json.loads(body.decode('utf-8'))
            conversation_id = decoded_body.get("conversation_id", None)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            SHARED["active_conversations"][conversation_id].reset()
            del SHARED["active_conversations"][conversation_id]
#            SHARED['agent'].reset()
            self.wfile.write(bytes("{}", 'utf-8'))
        else:
            self.send_response(500)
            self.wfile.write(bytes(str({'status': 500}), 'utf-8'))


def setup_interweb_args(shared):
    """
    Build and parse CLI opts.
    """
    parser = setup_args()
    parser.description = 'Interactive chat with a model in a web browser'
    parser.add_argument('--port', type=int, default=PORT, help='Port to listen on.')
    parser.add_argument(
        '--host',
        default=HOST_NAME,
        type=str,
        help='Host from which allow requests, use 0.0.0.0 to allow all IPs',
    )
    return parser


def shutdown():
    global SHARED
    if 'server' in SHARED:
        SHARED['server'].shutdown()
    SHARED.clear()


def wait():
    global SHARED
    while not SHARED.get('ready'):
        time.sleep(0.01)


def interactive_web(opt):
    global SHARED

    human_agent = LocalHumanAgent(opt)

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    agent.opt.log()
    SHARED['opt'] = agent.opt
    SHARED['agent'] = agent
    SHARED['world'] = create_task(opt, [human_agent, SHARED['agent']])
    SHARED['active_conversations'] = {}

    MyHandler.protocol_version = 'HTTP/1.0'
    httpd = HTTPServer((opt['host'], opt['port']), MyHandler)
    SHARED['server'] = httpd
    logging.info('http://{}:{}/'.format(opt['host'], opt['port']))

    try:
        SHARED['ready'] = True
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()


@register_script('interactive_web', aliases=['iweb'], hidden=True)
class InteractiveWeb(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_interweb_args(SHARED)

    def run(self):
        return interactive_web(self.opt)


if __name__ == '__main__':
    InteractiveWeb.main()
