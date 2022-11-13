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
from parlai.utils.misc import Timer
import parlai.utils.logging as logging
from parlai.utils.world_logging import WorldLogger

import json
import os
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
            # Create new agent, world and world logger.
            _agent_clone = self._clone_agent(SHARED["agent"])
            active_conversations[conversation_id] = {
                "agent": _agent_clone,
                "world": self._clone_world(SHARED["world"], _agent_clone)
            }
            active_conversations[conversation_id]["agent"].reset()
            world_logger = WorldLogger(SHARED["opt"])
            active_conversations[conversation_id]["world_logger"] = world_logger

        reply = {'episode_done': False, 'text': reply_text}
        active_conversations[conversation_id]["agent"].observe(reply)
        model_res = active_conversations[conversation_id]["agent"].act()

        _world = active_conversations[conversation_id]["world"]
        _world.acts[0] = reply
        _world.acts[1] = model_res
        active_conversations[conversation_id]["world_logger"].log(_world)

#        SHARED['agent'].observe(reply)
#        model_res = SHARED['agent'].act()
        return model_res

    def _clone_agent(self, agent):
        res = copy(agent)
        res.history = deepcopy(agent.history)
        return res

    def _clone_world(self, world, agent_clone):
        res = copy(world)
        res.agents[0] = deepcopy(world.agents[0])
        res.agents[1] = agent_clone
        res.acts = deepcopy(world.acts)
        res.time = Timer()
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
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            decoded_body = json.loads(body.decode('utf-8'))
            conversation_id = decoded_body.get("conversation_id", None)

            # Save conversation to logs
            if SHARED["opt"]["outdir"]:
                out_file = self._get_out_file(SHARED["opt"]["outdir"], conversation_id)
                save_format = SHARED["opt"]["save_format"]
                logging.info("Saving conversation to {} in {} format.".format(out_file, save_format))
                self._save_conversation(conversation_id, out_file, save_format)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            SHARED["active_conversations"][conversation_id]["agent"].reset()
            del SHARED["active_conversations"][conversation_id]
#            SHARED['agent'].reset()
            self.wfile.write(bytes("{}", 'utf-8'))
        else:
            self.send_response(500)
            self.wfile.write(bytes(str({'status': 500}), 'utf-8'))

    def _get_out_file(self, out_dir, conversation_id):
        return os.path.join(out_dir, f"{conversation_id}_{time.time_ns()}")

    def _save_conversation(self, conversation_id, out_file, save_format):
        world_logger = SHARED["active_conversations"][conversation_id]["world_logger"]
        world_logger.reset_world()
        world_logger.write(out_file, SHARED["active_conversations"][conversation_id]["world"], save_format)


def setup_inter_rest_args(shared):
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
    parser.add_argument(
        '--outdir',
        type=str,
        default='',
        help='Saves a jsonl file containing all of the task examples and '
        'model replies in the specified directory. Save one file per episode. '
        'Each episode is saved when the reset operation is received. '
        'Set to the empty string to not save at all',
    )
    parser.add_argument(
        '--save-format',
        type=str,
        default='conversations',
        choices=['conversations', 'parlai'],
        help='Format to save logs in. conversations is a jsonl format, parlai is a text format.',
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
        return setup_inter_rest_args(SHARED)

    def run(self):
        return interactive_web(self.opt)


if __name__ == '__main__':
    InteractiveWeb.main()
