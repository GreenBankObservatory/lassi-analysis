from collections import namedtuple

Actuator = namedtuple('Actuator', ['hoop', 'rib', 'x', 'y', 'rho', 'theta', 'phi'])

class AsZernikeFile:

    """
    This class is responsible for parsing the contents of the
    etc/config/AsZernike.conf file, which contains info on actuator
    positions and angles to the bore sight
    """

    def __init__(self, filename):


        self.filename = filename
        self.actuatorList = []
        self.actuators = {}

    def parse(self):
        "Read the file and figure out it's contents"

        # re-initialize
        self.actuatorList = []
        self.actuators = {}

        with open(self.filename, 'r') as f:
            ls = f.readlines()

        # avoid the initial comments
        ls = ls[7:]

        
        # each line should look like this:
        # actuator[rib][hoop] x y rho theta phi rho_y theta_y phi_y
        # act[45][-70] 103.2276438124404 12.67476091855193 1.01666 1.8228 1.32331 1.01666 1.8228 1.32331
        for l in ls:
            ps = l.split(' ')
            actHoopRib = ps[0]
            hoop, rib = self.parseHoopRib(actHoopRib)
            x = float(ps[1])
            y = float(ps[2])
            rho = float(ps[3])
            theta = float(ps[4])
            phi = float(ps[5])
            # the *_y terms are all redundant
            act = Actuator(hoop=hoop,
                           rib=rib,
                           x=x,
                           y=y,
                           rho=rho,
                           theta=theta,
                           phi=phi)
            self.actuatorList.append(act)
            self.actuators[(hoop, rib)] = act

    def parseHoopRib(self, actHoopRib):
        " 'act[1][-3]'' -> (1, -3)"

        # look for the x value first
        i1 = actHoopRib.find('[')
        if i1 == -1:
            print("could not find first [", actHoopRib)
            return None

        i2 = actHoopRib.find(']')
        if i2 == -1:
            print("could not find first ]", actHoopRib)
            return None

        x = int(actHoopRib[i1+1:i2])
        
        # y value follows
        y = int(actHoopRib[i2+2:-1])

        return x, y


if __name__ == '__main__':
    asz = AsZernikeFile("/home/gbt/etc/config/AsZernike.conf")
    asz.parse()
    #for act in asz.actuatorList:
    #    print act
    ks = sorted(asz.actuators.keys())
    for k in ks:
        print(k, asz.actuators[k].phi)
    print("Found %d actuators specified" % len(asz.actuatorList))
