"This module is for configuring our logging for this application"

import os

# RUNTLS_LOG_DIR = os.getenv('TELESPY_LOG_DIR', '.')
RUNTLS_LOG_DIR = "/home/scratch/pmargani/LASSI/runTLSlogs" #os.getenv('TELESPY_LOG_DIR', '.')

config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s',
        },
        'simple': {
            'format': '%(levelname)s %(message)s',
        },
        'super_simple': {
            'format': '%(message)s',
        },
        'mermaid_formatter': {
            # We want the time at the end here; that way it is shown as the label
            # in the mermaid sequence diagram
            'format': '%(message)s: %(asctime)s',
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            # Log to console only in debug mode
            # 'filters': ['filters.OnlyInDebugFilter']
        },
        'teleSpyFile': {
            'level': 'DEBUG',
            'formatter': 'verbose',
            'class': 'handlers.DatestampFileHandler',
            # Pull the log path from the environment. If this isn't set, an error
            # will be thrown. To log to the current directory, set this to .
            # 'filename': os.path.join(SPARROW_LOG_DIR, 'gfm_logs', 'gfm.log'),
            'filename': os.path.join(RUNTLS_LOG_DIR, 'teleSpy.log')
        },
    },
    # No loggers currently specified
    # 'loggers':
    'root': {
        'handlers': ['console', 'teleSpyFile'],
        # Make sure we don't miss anything...
        'level': 'DEBUG',
    },
}
