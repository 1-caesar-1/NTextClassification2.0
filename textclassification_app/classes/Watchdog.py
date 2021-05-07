from threading import Timer, Lock


class Watchdog(Exception):
    """
    Create a watchdog that can see to it that an activity is taking place.
        If the watchdog does not receive a signal, it will activate the Handler.
        Call reset() whenever you want to give a sign of life.
        Call stop() at the end of using the watchdog.

        watchdog = Watchdog(x)\n
        try:
            # do something that might take too long
            watchdog.reset()
        except Watchdog:
            # handle watchdog error
        watchdog.stop()
    """

    def __init__(self, timeout, user_handler=None, handler_params=None):
        """
        :param timeout: The number of seconds before entering panic mode
        :param user_handler: Optional, A function that operates in panic mode
        """
        self.handler_params = handler_params or list()
        self.timeout = timeout
        self.handler = user_handler or self.default_handler
        self.lock = Lock()
        self.timer = Timer(self.timeout, self.handler, self.handler_params)

    def reset(self):
        with self.lock:
            self.timer.cancel()
            self.timer = Timer(self.timeout, self.handler, self.handler_params)
            self.timer.start()

    def stop(self):
        with self.lock:
            self.timer.cancel()

    def start(self):
        with self.lock:
            self.timer.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def default_handler(self):
        print("No life signal was received for {} seconds!".format(self.timeout))
        raise self
