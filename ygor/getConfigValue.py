# Copyright (C) 2011 Associated Universities, Inc. Washington DC, USA.
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
# 
# Correspondence concerning GBT software should be addressed as follows:
#       GBT Operations
#       National Radio Astronomy Observatory
#       P. O. Box 2
#       Green Bank, WV 24944-0002 USA

import os

def getConfigValue(defaultValue, name, configFile = "system.conf"):
    # Get installation definition
    ygorTelescope = os.environ["YGOR_TELESCOPE"]
    # Open the configuration file
    if configFile[0] == '/':
        filename = configFile
    else:
        filename = ygorTelescope + "/etc/config/" + configFile
    config_file = open(filename, 'r')

    # Get the configuration values
    keywords = { "YGOR_TELESCOPE" : ygorTelescope }
    for line in config_file.readlines():
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
           continue
        tokens = line.split(':=')
        if len(tokens) == 2:
            # Remove quotes
            parse = tokens[1].split('"')
            if len(parse) == 3:
                value = parse[1]
            else:
                value = tokens[1]
            keywords[tokens[0].strip()] = value.strip()
        else:
            print("Bad syntax in file %s: %s" % (filename, line))

    # Get the value from the environment, the file, or the default
    # if os.environ.has_key(name):
    if name in os.environ:
        return os.environ[name]
    # if keywords.has_key(name):
    if name in keywords:
        return keywords[name]
    if defaultValue:
        return defaultValue
    raise KeyError("No defined or default value for %s" % name)

