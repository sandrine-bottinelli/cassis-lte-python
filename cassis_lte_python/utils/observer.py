import abc
import pprint

"""
This module is used to implement an Observer design pattern
"""


class Event:

    """
    Class representing an event
    """

    # Used for debug
    def __str__(self):
        pp = pprint.PrettyPrinter(indent=4)
        return pp.pformat(self.__dict__)


class Observer:

    """
    Class representing an observer
    """

    @abc.abstractmethod
    def handle_event(self, event, debug=False):
        """
        Abstract method to handle the given event.

        :param event: Event transmitted by an Observable
        :param debug: If True, will print the content of the event
        """
        if debug:
            print(event)
        pass


class Observable:

    """
    Class representing an observable
    """

    def __init__(self):
        self._observers = set()

    @abc.abstractmethod
    def fire_state(self):
        """
        Abstract method to fire a state of an observable
        """
        pass

    def subscribe(self, observer):
        """
        Adds an observer to the list of subscribers

        :param observer: The observer to add
        """
        self._observers.add(observer)

    def unsubscribe(self, observer):
        """
        Removes an observer from the list of subscribers

        :param observer: The observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def unsubscribe_all(self):
        """
        Removes all observers from the list of subscribers
        """
        self._observers.clear()

    def fire(self, **att):
        """
        Creates an event and send it to the subscribes observers.
        Any number of attributes can be attached to the event using a list of keyed arguments
        A source attribute is automatically added to the event, with its value being the sender

        Users should use at least the 'type' keys to identify the event.

        :param att: The list of keyed argument
        """
        e = Event()
        e.source = self
        debug = False
        for k, v in att.items():
            if k == 'debug':
                debug = v
            setattr(e, k, v)
        for obs in self._observers:
            obs.handle_event(e, debug)


class ProcessModel(Observable):

    """
    Special observable used to represent long-running processes
    """

    def __init__(self, name=""):
        """
        Constructor

        :param name: Name of the process
        """
        super(ProcessModel, self).__init__()
        self._name = name

    def fire_state(self):
        pass

    def process_start(self, min_steps=0, max_steps=0):
        """
        Sends an event notifying the start of the process

        :param min_steps: The starting number of steps (usually 0), uses key 'min_steps'
        :param max_steps: The maximum number of steps, uses key 'max_steps'
        """
        self.fire(type='process_start', name=self._name, min_steps=min_steps, max_steps=max_steps)

    def step_start(self, description=""):
        """
        Sends an event notifying the end of a sub-process

        :param description: A description of the sub-process, uses key 'description'
        """
        self.fire(type='step_start', description=description)

    def step_end(self):
        """
        Sends an event notifying the end of the last sub-process
        """
        self.fire(type='step_end', increment=1)

    def process_end(self, result=None, **kwargs):
        """
        Sends an event notifying the end of the process

        :param result: The result of the process, uses key 'value'
        """
        self.fire(type='process_end', value=result, **kwargs)
