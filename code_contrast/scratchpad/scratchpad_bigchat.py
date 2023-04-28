# {'temperature': 0.3, 'top_p': 1.0, 'top_n': 0, 'max_tokens': 500, 'created': 1682669743.180327, 'stop_tokens': [], 'id': 'uchat-Gv1CSwCGXEM1-CSZJlmLgIVTu', 'object': 'chat_completion_req',
# 'account': 'oleg@smallcloud.tech',
# 'model': 'bigcode/15b',
# 'messages': [{'role': 'user', 'content': "import pygame\nimport random\n\n\nclass Game:\n    def __init__(self):\n        pygame.init()\n        self.screen = pygame.display.set_mode((800, 600))\n        self.clock = pygame.time.Clock()\n        self.running = True\n        self.circles = []\n        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]\n        self.color_index = 0\n        self.score = 0\n        self.font = pygame.font.SysFont('Arial', 24)\n        self.text = self.font.render('Score: ' + str(self.score), True, (255, 255, 255))\n        self.text_rect = self.text.get_rect()\n        self.text_rect.center = (400, 300)\n\n    def run(self):\n        while self.running:\n            self.clock.tick(60)\n            self.events()\n            self.updat\n            self.draw()\n\n    def update(self):\n        for circle in self.circles:\n            circle.update()\n\n    def draw(self):\n        self.screen.fill((0, 0, 0))\n        for circle in self.circles:\n            circle.draw(self.screen)\n        self.screen.blit(self.text, self.text_rect)\n        pygame.display.flip()\n\n    def events(self):\n        for event in pygame.event.get():\n            if event.type == pygame.QUIT:\n                self.running = False\n\n    def init_circles(self):\n        self.circles = []\n        for i in range(10):\n            self.circles.append(Circle(self, self.colors[self.color_index]))\n            self.color_index = (self.color_index + 1) % len(self.colors)\n        self.color_index = 0\n\n\nclass Circle:\n    def __init__(self, game, color):\n        self.game = game\n        self.color = color\n        self.radius = random.randrange(10, 30)\n        self.x = random.randrange(self.radius, self.game.screen.get_width() - self.radius)\n        self.y = random.randrange(self.radius, self.game.screen.get_height() - self.radius)\n        self.x_speed = random.randrange(-2, 2)\n        self.y_speed = random.randrange(-2, 2)\n\n    def update(self):\n        self.x += self.x_speed"}, {'role': 'assistant', 'content': "Thanks for context, what's your question?"}, {'role': 'user', 'content': 'aaa'}, {'role': 'assistant', 'content': ''}, {'role': 'user', 'content': 'aaa'}],
# 'stream': True}


from typing import List, Optional, Dict

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.scratchpad.messages import base_msg


class ScratchpadBigChat:
    def __init__(
            self,
            enc: SMCEncoding,
            temperature: int,
            top_p: float,
            top_n: int,
            max_tokens: int,
            created: float,
            stop_tokens: List[Optional[str]],
            id: str,
            object: str,
            account: str,
            model: str,
            messages: List[Dict[str, str]],
            stream: bool = True,
            **unused
    ):
        self.enc = enc
        self.messages = messages
        for k, v in unused.items():
            print("ScratchpadBase: unused parameter '%s' = '%s'" % (k, v))

        self._completion = []

    def prompt_infill(self, T: int):
        prompt: List[int] = [
            *self.enc.encode(base_msg),
        ]
        print(f'tokens_cnt: {len(prompt)}')
        self._completion.clear()
        return prompt

    def completion(self, final: bool):
        result = {}
        completion_text = self.enc.decode(self._completion)
        return completion_text

