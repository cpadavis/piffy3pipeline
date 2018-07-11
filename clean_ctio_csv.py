# wtf is with the csv aaron gave?
from __future__ import print_function

path = '/nfs/slac/g/ki/ki06/roodman/CtioDB/db-part1.csv'
out_path = '/nfs/slac/g/ki/ki19/des/cpd/piff_test/CtioDB_db-part1.csv'
Nbad = 0
Ntot = 0
with open(path) as f:
    with open(out_path, 'w') as fo:
        for i, line in enumerate(f):
            entries = line.split(',')
            length = len(entries)
            if length != 161:
                print(i, length, entries[0])
                Nbad += 1
            else:
                fo.write(line)
            Ntot += 1
print(Nbad, Ntot)
