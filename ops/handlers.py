from datetime import datetime
import logging
import os

# TODO: Move to sensible location

# Modified from: https://stackoverflow.com/a/33492520/1883424
class DatestampFileHandler(logging.FileHandler):

    # Pass the file name and header string to the constructor.
    def __init__(self, filename, header=None, mode='a', encoding=None, delay=0):
        if not filename:
            raise ValueError("Must provide a filename!")

        # We want our filename to have a timestamp, so:
        parts = filename.split(".")
        # Insert the timestamp just before the extension
        parts.insert(-1, datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        filename = ".".join(parts)

        # Call the parent __init__
        super(DatestampFileHandler, self).__init__(filename, mode, encoding, delay)


class MermaidHandler(DatestampFileHandler):

    # Pass the file name and header string to the constructor.
    def __init__(self, header=None, *args, **kwargs):
        # Call the parent __init__
        super(MermaidHandler, self).__init__(*args, **kwargs)

        # Store the header information.
        if not header:
            header = 'sequenceDiagram'

        # Write the header if delay is False and a file stream was created.
        if not delay and self.stream is not None:
            self.stream.write('{}\n'.format(header))
