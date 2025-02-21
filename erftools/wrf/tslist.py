name_len = 25

class TSList(object):
    """Process WRF tslist input file"""

    def __init__(self,fpath):
        with open(fpath,'r') as f:
            f.readline()
            header = f.readline().upper()
            if (' I ' in header) and (' J ' in header):
                self.have_ij = True
            else:
                assert (' LAT ' in header) and (' LON ' in header)
                self.have_ij = False
            f.readline()
            towers = f.readlines()
        self.ntowers = len(towers)
        self.name = []
        self.prefix = []
        self.i, self.j = [],[]
        self.lat, self.lon = [],[]
        for line in towers:
            name = line[:name_len]
            defs = line[name_len:].split()
            self.name.append(name.strip())
            self.prefix.append(defs[0])
            if self.have_ij:
                self.i.append(int(defs[1])-1)
                self.j.append(int(defs[2])-1)
            else:
                self.lat.append(float(defs[1]))
                self.lon.append(float(defs[2]))

    def get_ijk_lists(self,nz):
        """Get indices for ERF line sampling"""
        lo,hi = [],[]
        for i,j in zip(self.i,self.j):
            lo.extend([i,j,0])
            hi.extend([i,j,nz-1])
        return lo, hi
