#!/usr/bin/env python3

import logging

from typing import List

from . import commands, handlers

logger = logging.getLogger(__name__)


class MessageBus(object):

    def handle(self, message: commands.Command) -> List[object]:
        """Handle commands and events.
        This will return the final result."""
        if isinstance(message, commands.Command):
            result = self.handle_command(message)
        else:
            raise ValueError('Invalid message type')
        return result

    def handle_command(self, cmd: commands.Command):
        handler = handlers.COMMAND_HANDLERS[type(cmd)]
        result = handler(cmd)
        return result


messagebus = MessageBus()
