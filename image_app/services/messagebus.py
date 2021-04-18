#!/usr/bin/env python3

import collections
import logging

from typing import Deque, List

from . import commands, events, handlers
from .handlers import Message

logger = logging.getLogger(__name__)


class MessageBus(object):

    def handle(self, message: commands.Command) -> List[object]:
        """Handle commands and events.
        This will return the final result."""
        self.queue = collections.deque([message])
        while self.queue:
            message = self.queue.popleft()
            if isinstance(message, commands.Command):
                result = self.handle_command(message)
            elif isinstance(message, events.Event):
                result = self.handle_events(message)
            else:  # pragma: no cover
                raise ValueError('Invalid message type')
        return result

    def handle_events(self, event: events.Event):
        handler = handlers.EVENT_HANDLERS[type(event)]
        result = handler(event, self.queue)
        return result

    def handle_command(self, cmd: commands.Command):
        handler = handlers.COMMAND_HANDLERS[type(cmd)]
        result = handler(cmd, self.queue)
        return result


messagebus = MessageBus()
