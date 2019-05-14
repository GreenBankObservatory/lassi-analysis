import os

from astropy.io import fits

from ProjectScanLog import ProjectScanLog

def main(proj):
    print "Examining scan numbers for: ", proj

    p = ProjectScanLog(proj)
    p.open()

    ant = 'Antenna'
    go = 'GO'
    scan = 'SCAN'

    badScans = []
    print "Reporting on scan numbers from ScanLog.fits,"
    print "Antenna and GO FITS files"

    print "%5s %5s %5s %30s" % ('SCAN', 'ANT', 'GO', "FILENAME")
    for scanNumber, v in p.scans.items():
        antf = v[ant]
        gof = v[go]
        assert gof == antf
        antp = os.path.join(proj, ant, antf)
        gop = os.path.join(proj, go, gof)
        antHs = fits.open(antp)
        goHs = fits.open(gop)
        antScan = int(antHs[0].header[scan])
        goScan = int(goHs[0].header[scan])
        print "%5d %5s %5s %30s" % (scanNumber, antScan, goScan, gof)
        if scanNumber != antScan or scanNumber != goScan:
            badScans.append((scanNumber, antScan, goScan, gof))

    print ""
    print "Scans with possible problems: "
    print "%5s %5s %5s %30s" % ('SCAN', 'ANT', 'GO', 'FILENAME')
    for scanNumber, antScan, goScan, filename in badScans:
        print "%5d %5s %5s %30s" % (scanNumber, antScan, goScan, filename)

if __name__ == '__main__':
    proj = "/home/gbtdata/TGBT19A_504_02"
    main(proj)
