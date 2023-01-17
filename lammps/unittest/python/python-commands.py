
import sys,os,unittest
from lammps import lammps, LMP_VAR_ATOM, LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR

has_manybody=False
try:
    machine=None
    if 'LAMMPS_MACHINE_NAME' in os.environ:
        machine=os.environ['LAMMPS_MACHINE_NAME']
    lmp=lammps(name=machine)
    has_manybody = lmp.has_style("pair","sw")
    lmp.close()
except:
    pass

class PythonCommand(unittest.TestCase):

    def setUp(self):
        machine=None
        if 'LAMMPS_MACHINE_NAME' in os.environ:
            machine=os.environ['LAMMPS_MACHINE_NAME']
        self.lmp=lammps(name=machine,
                        cmdargs=['-nocite',
                                 '-log','none',
                                 '-echo','screen',
                                 '-var','zpos','1.5',
                                 '-var','x','2'])
        # create demo input strings and files
        # a few commands to set up a box with a single atom
        self.demo_input="""
region       box block 0 $x 0 2 0 2
create_box 1 box
create_atoms 1 single 1.0 1.0 ${zpos}
"""
        # another command to add an atom and use a continuation line
        self.cont_input="""
create_atoms 1 single &
            0.2 0.1 0.1
"""
        self.demo_file='in.test'
        with open(self.demo_file,'w') as f:
            f.write(self.demo_input)
        self.cont_file='in.cont'
        with open(self.cont_file,'w') as f:
            f.write(self.cont_input)

    # clean up temporary files
    def tearDown(self):
        if os.path.exists(self.demo_file):
            os.remove(self.demo_file)
        if os.path.exists(self.cont_file):
            os.remove(self.cont_file)

    ##############################
    def testFile(self):
        """Test reading commands from a file"""
        natoms = self.lmp.get_natoms()
        self.assertEqual(natoms,0)
        self.lmp.file(self.demo_file)
        natoms = self.lmp.get_natoms()
        self.assertEqual(natoms,1)
        self.lmp.file(self.cont_file)
        natoms = self.lmp.get_natoms()
        self.assertEqual(natoms,2)

    def testNoFile(self):
        """Test (not) reading commands from no file"""
        self.lmp.file(None)
        natoms = self.lmp.get_natoms()
        self.assertEqual(natoms,0)

    def testCommand(self):
        """Test executing individual commands"""
        natoms = self.lmp.get_natoms()
        self.assertEqual(natoms,0)
        cmds = self.demo_input.splitlines()
        for cmd in cmds:
            self.lmp.command(cmd)
        natoms = self.lmp.get_natoms()
        self.assertEqual(natoms,1)

    def testCommandsList(self):
        """Test executing commands from list of strings"""
        natoms = self.lmp.get_natoms()
        self.assertEqual(natoms,0)
        cmds = self.demo_input.splitlines()+self.cont_input.splitlines()
        self.lmp.commands_list(cmds)
        natoms = self.lmp.get_natoms()
        self.assertEqual(natoms,2)

    def testCommandsString(self):
        """Test executing block of commands from string"""
        natoms = self.lmp.get_natoms()
        self.assertEqual(natoms,0)
        self.lmp.commands_string(self.demo_input+self.cont_input)
        natoms = self.lmp.get_natoms()
        self.assertEqual(natoms,2)

    def testNeighborListSimple(self):
        self.lmp.commands_string("""
        units lj
        atom_style atomic
        atom_modify map array
        boundary f f f
        region box block 0 2 0 2 0 2
        create_box 1 box""")

        x = [ 1.0, 1.0, 1.0,  1.0, 1.0, 1.5 ]
        types = [1, 1]

        self.assertEqual(self.lmp.create_atoms(2, id=None, type=types, x=x), 2)
        nlocal = self.lmp.extract_global("nlocal")
        self.assertEqual(nlocal, 2)

        self.lmp.commands_string("""
        mass 1 1.0
        velocity all create 3.0 87287
        pair_style lj/cut 2.5
        pair_coeff 1 1 1.0 1.0 2.5
        neighbor 0.1 bin
        neigh_modify every 20 delay 0 check no
        run 0 post no""")

        idx = self.lmp.find_pair_neighlist("lj/cut")
        self.assertNotEqual(idx, -1)
        self.assertEqual(self.lmp.find_pair_neighlist("morse"), -1)
        nlist = self.lmp.get_neighlist(idx)
        self.assertEqual(len(nlist), 2)
        atom_i, numneigh_i, neighbors_i = nlist[0]
        atom_j, numneigh_j, _ = nlist[1]

        self.assertEqual(atom_i, 0)
        self.assertEqual(atom_j, 1)

        self.assertEqual(numneigh_i, 1)
        self.assertEqual(numneigh_j, 0)

        self.assertEqual(1, neighbors_i[0])

    def testNeighborListHalf(self):
        self.lmp.commands_string("""
        boundary f f f
        units real
        region box block -5 5 -5 5 -5 5
        create_box 1 box
        mass 1 1.0
        pair_style lj/cut 4.0
        pair_coeff 1 1 0.2 2.0
        """)
        x = [ 0.0,  0.0,  0.0,  -1.1,  0.0,  0.0,  1.0,  0.0,  0.0,
              0.0, -1.1,  0.0,   0.0,  1.0,  0.0,  0.0,  0.0, -1.1,
              0.0,  0.0,  1.0 ]
        tags = [1, 2, 3, 4, 5, 6, 7]
        types = [1, 1, 1, 1, 1, 1, 1]

        self.assertEqual(self.lmp.create_atoms(7, id=tags, type=types, x=x), 7)
        nlocal = self.lmp.extract_global("nlocal")
        self.assertEqual(nlocal, 7)

        self.lmp.command("run 0 post no")

        self.assertEqual(self.lmp.find_pair_neighlist("lj/cut"),0)
        nlist = self.lmp.get_neighlist(0)
        self.assertEqual(nlist.size, 7)
        for i in range(0,nlist.size):
            idx, num, neighs = nlist.get(i)
            self.assertEqual(idx,i)
            self.assertEqual(num,nlocal-1-i)

        # look up neighbor list by atom index
        num, neighs = nlist.find(2)
        self.assertEqual(num,4)
        self.assertIsNotNone(neighs,None)
        # this one will fail
        num, neighs = nlist.find(10)
        self.assertEqual(num,-1)
        self.assertIsNone(neighs,None)

    @unittest.skipIf(not has_manybody,"Full neighbor list test for manybody potential")
    def testNeighborListFull(self):
        self.lmp.commands_string("""
        boundary f f f
        units metal
        region box block -5 5 -5 5 -5 5
        create_box 1 box
        mass 1 1.0
        pair_style sw
        pair_coeff * * Si.sw Si
        """)
        x = [ 0.0,  0.0,  0.0,  -1.1,  0.0,  0.0,  1.0,  0.0,  0.0,
              0.0, -1.1,  0.0,   0.0,  1.0,  0.0,  0.0,  0.0, -1.1,
              0.0,  0.0,  1.0 ]
        tags = [1, 2, 3, 4, 5, 6, 7]
        types = [1, 1, 1, 1, 1, 1, 1]

        self.assertEqual(self.lmp.create_atoms(7, id=tags, type=types, x=x), 7)
        nlocal = self.lmp.extract_global("nlocal")
        self.assertEqual(nlocal, 7)

        self.lmp.command("run 0 post no")

        self.assertEqual(self.lmp.find_pair_neighlist("sw"),0)
        nlist = self.lmp.get_neighlist(0)
        self.assertEqual(nlist.size, 7)
        for i in range(0,nlist.size):
            idx, num, neighs = nlist.get(i)
            self.assertEqual(idx,i)
            self.assertEqual(num,nlocal-1)

    @unittest.skipIf(not has_manybody,"Hybrid neighbor list test for manybody potential")
    def testNeighborListHybrid(self):
        self.lmp.commands_string("""
        boundary f f f
        units metal
        region box block -5 5 -5 5 -5 5
        create_box 2 box
        mass * 1.0
        pair_style hybrid/overlay morse 4.0 lj/cut 4.0 lj/cut 4.0 sw
        pair_coeff * * sw Si.sw Si NULL
        pair_coeff 1 2 morse 0.2 2.0 2.0
        pair_coeff 2 2 lj/cut 1 0.1 2.0
        pair_coeff * * lj/cut 2 0.01 2.0
        """)
        x = [ 0.0,  0.0,  0.0,  -1.1,  0.0,  0.0,  1.0,  0.0,  0.0,
              0.0, -1.1,  0.0,   0.0,  1.0,  0.0,  0.0,  0.0, -1.1,
              0.0,  0.0,  1.0 ]
        tags = [1, 2, 3, 4, 5, 6, 7]
        types = [1, 1, 1, 1, 2, 2, 2]

        self.assertEqual(self.lmp.create_atoms(7, id=tags, type=types, x=x), 7)
        nlocal = self.lmp.extract_global("nlocal")
        self.assertEqual(nlocal, 7)

        self.lmp.command("run 0 post no")

        # valid and invalid lookups
        self.assertNotEqual(self.lmp.find_pair_neighlist("sw"),-1)
        self.assertNotEqual(self.lmp.find_pair_neighlist("morse"),-1)
        self.assertNotEqual(self.lmp.find_pair_neighlist("lj/cut",nsub=1),-1)
        self.assertNotEqual(self.lmp.find_pair_neighlist("lj/cut",nsub=2),-1)
        self.assertEqual(self.lmp.find_pair_neighlist("lj/cut"),-1)
        self.assertEqual(self.lmp.find_pair_neighlist("hybrid/overlay"),-1)
        self.assertNotEqual(self.lmp.get_neighlist(4).size,0)
        self.assertEqual(self.lmp.get_neighlist(5).size,-1)

        # full neighbor list for 4 type 1 atoms
        # all have 3 type 1 atom neighbors
        nlist = self.lmp.get_neighlist(self.lmp.find_pair_neighlist("sw"))
        self.assertEqual(nlist.size, 4)
        for i in range(0,nlist.size):
            idx, num, neighs = nlist.get(i)
            self.assertEqual(idx,i)
            self.assertEqual(num,3)

        # half neighbor list for all pairs between type 1 and type 2
        # 4 type 1 atoms with 3 type 2 neighbors and 3 type 2 atoms without neighbors
        nlist = self.lmp.get_neighlist(self.lmp.find_pair_neighlist("morse"))
        self.assertEqual(nlist.size, 7)
        for i in range(0,nlist.size):
            idx, num, neighs = nlist.get(i)
            if (i < 4): self.assertEqual(num,3)
            else: self.assertEqual(num,0)

        # half neighbor list between type 2 atoms only
        # 3 pairs with 2, 1, 0 neighbors
        nlist = self.lmp.get_neighlist(self.lmp.find_pair_neighlist("lj/cut",nsub=1))
        self.assertEqual(nlist.size, 3)
        for i in range(0,nlist.size):
            idx, num, neighs = nlist.get(i)
            self.assertEqual(num,2-i)

        # half neighbor list between all pairs. same as simple lj/cut case
        nlist = self.lmp.get_neighlist(self.lmp.find_pair_neighlist("lj/cut",nsub=2))
        self.assertEqual(nlist.size, 7)
        for i in range(0,nlist.size):
            idx, num, neighs = nlist.get(i)
            self.assertEqual(num,nlocal-1-i)

    def testNeighborListCompute(self):
        self.lmp.commands_string("""
        boundary f f f
        units real
        region box block -5 5 -5 5 -5 5
        create_box 1 box
        mass 1 1.0
        pair_style lj/cut 4.0
        pair_coeff 1 1 0.2 2.0
        compute dist all pair/local dist
        fix dist all ave/histo 1 1 1 0.0 3.0 4 c_dist mode vector
        thermo_style custom f_dist[*]
        """)
        x = [ 0.0,  0.0,  0.0,  -1.1,  0.0,  0.0,  1.0,  0.0,  0.0,
              0.0, -1.1,  0.0,   0.0,  1.0,  0.0,  0.0,  0.0, -1.1,
              0.0,  0.0,  1.0 ]
        tags = [1, 2, 3, 4, 5, 6, 7]
        types = [1, 1, 1, 1, 1, 1, 1]

        self.assertEqual(self.lmp.create_atoms(7, id=tags, type=types, x=x), 7)
        nlocal = self.lmp.extract_global("nlocal")
        self.assertEqual(nlocal, 7)

        self.lmp.command("run 0 post no")
        # check compute data from histogram summary
        nhisto = self.lmp.extract_fix("dist",LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR,nrow=0)
        nskip = self.lmp.extract_fix("dist",LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR,nrow=1)
        minval = self.lmp.extract_fix("dist",LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR,nrow=2)
        maxval = self.lmp.extract_fix("dist",LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR,nrow=3)
        # 21 pair distances counted, none skipped, smallest 1.0, largest 2.1
        self.assertEqual(nhisto,21)
        self.assertEqual(nskip,0)
        self.assertEqual(minval,1.0)
        self.assertEqual(maxval,2.1)

        self.assertNotEqual(self.lmp.find_pair_neighlist("lj/cut"),-1)
        self.assertNotEqual(self.lmp.find_compute_neighlist("dist"),-1)
        self.assertEqual(self.lmp.find_compute_neighlist("xxx"),-1)
        self.assertEqual(self.lmp.find_fix_neighlist("dist"),-1)

        # the compute has a half neighbor list
        nlist = self.lmp.get_neighlist(self.lmp.find_compute_neighlist("dist"))
        self.assertEqual(nlist.size, 7)
        for i in range(0,nlist.size):
            idx, num, neighs = nlist.get(i)
            self.assertEqual(idx,i)
            self.assertEqual(num,nlocal-1-i)

    def test_extract_box_non_periodic(self):
        self.lmp.command("boundary f f f")
        self.lmp.command("region box block 0 2 0 2 0 2")
        self.lmp.command("create_box 1 box")

        boxlo, boxhi, xy, yz, xz, periodicity, box_change = self.lmp.extract_box()

        self.assertEqual(boxlo, [0.0, 0.0, 0.0])
        self.assertEqual(boxhi, [2.0, 2.0, 2.0])
        self.assertEqual(xy, 0.0)
        self.assertEqual(yz, 0.0)
        self.assertEqual(xz, 0.0)
        self.assertEqual(periodicity, [0, 0, 0])
        self.assertEqual(box_change, 0)

    def test_extract_box_periodic(self):
        self.lmp.command("boundary p p p")
        self.lmp.command("region box block 0 2 0 2 0 2")
        self.lmp.command("create_box 1 box")

        boxlo, boxhi, xy, yz, xz, periodicity, box_change = self.lmp.extract_box()

        self.assertEqual(boxlo, [0.0, 0.0, 0.0])
        self.assertEqual(boxhi, [2.0, 2.0, 2.0])
        self.assertEqual(xy, 0.0)
        self.assertEqual(yz, 0.0)
        self.assertEqual(xz, 0.0)
        self.assertEqual(periodicity, [1, 1, 1])
        self.assertEqual(box_change, 0)

    def test_extract_box_triclinic(self):
        self.lmp.command("boundary p p p")
        self.lmp.command("region box block 0 2 0 2 0 2")
        self.lmp.command("create_box 1 box")
        self.lmp.command("change_box all triclinic")
        self.lmp.command("change_box all xy final 0.1 yz final 0.2 xz final 0.3")

        boxlo, boxhi, xy, yz, xz, periodicity, box_change = self.lmp.extract_box()

        self.assertEqual(boxlo, [0.0, 0.0, 0.0])
        self.assertEqual(boxhi, [2.0, 2.0, 2.0])
        self.assertEqual(xy, 0.1)
        self.assertEqual(yz, 0.2)
        self.assertEqual(xz, 0.3)
        self.assertEqual(periodicity, [1, 1, 1])
        self.assertEqual(box_change, 0)

    def test_reset_box(self):
        self.lmp.command("boundary p p p")
        self.lmp.command("region box block 0 2 0 2 0 2")
        self.lmp.command("create_box 1 box")
        self.lmp.command("change_box all triclinic")
        self.lmp.command("change_box all xy final 0.1 yz final 0.2 xz final 0.3")
        self.lmp.reset_box([0,0,0], [1,1,1], 0, 0, 0)

        boxlo, boxhi, xy, yz, xz, periodicity, box_change = self.lmp.extract_box()

        self.assertEqual(boxlo, [0.0, 0.0, 0.0])
        self.assertEqual(boxhi, [1.0, 1.0, 1.0])
        self.assertEqual(xy, 0)
        self.assertEqual(yz, 0)
        self.assertEqual(xz, 0)
        self.assertEqual(periodicity, [1, 1, 1])
        self.assertEqual(box_change, 0)

    def test_extract_variable_equalstyle(self):
        self.lmp.command("variable a equal 100")
        a = self.lmp.extract_variable("a")
        self.assertEqual(a, 100)

        self.lmp.command("variable a equal 3.14")
        a = self.lmp.extract_variable("a")
        self.assertEqual(a, 3.14)

    def test_extract_variable_atomstyle(self):
        self.lmp.command("units lj")
        self.lmp.command("atom_style atomic")
        self.lmp.command("atom_modify map array")
        self.lmp.command("boundary f f f")
        self.lmp.command("region box block 0 2 0 2 0 2")
        self.lmp.command("create_box 1 box")

        x = [
          1.0, 1.0, 1.0,
          1.0, 1.0, 1.5
        ]

        types = [1, 1]

        self.assertEqual(self.lmp.create_atoms(2, id=None, type=types, x=x), 2)
        self.lmp.command("variable a atom x*x+y*y+z*z")
        a = self.lmp.extract_variable("a", "all", LMP_VAR_ATOM)
        self.assertEqual(a[0], x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        self.assertEqual(a[1], x[3]*x[3]+x[4]*x[4]+x[5]*x[5])

    def test_get_thermo(self):
        self.lmp.command("units lj")
        self.lmp.command("atom_style atomic")
        self.lmp.command("atom_modify map array")
        self.lmp.command("boundary f f f")
        self.lmp.command("region box block 0 2 0 2 0 2")
        self.lmp.command("create_box 1 box")

        x = [
          1.0, 1.0, 1.0,
          1.0, 1.0, 1.5
        ]

        types = [1, 1]
        self.lmp.create_atoms(2, id=None, type=types, x=x)

        state = {
            "step": 0,
            "dt" : 0.005,
            "time" : 0.0,
            "atoms" : 2.0,
            "vol" : 8.0,
            "lx" : 2.0,
            "ly" : 2.0,
            "lz" : 2.0,
            "xlo" : 0,
            "xhi" : 2.0,
            "ylo" : 0,
            "yhi" : 2.0,
            "zlo" : 0,
            "zhi" : 2.0
        }

        for key, value in state.items():
            result = self.lmp.get_thermo(key)
            self.assertEqual(value, result, key)

    def test_extract_global(self):
        self.lmp.command("region box block -1 1 -2 2 -3 3")
        self.lmp.command("create_box 1 box")
        self.assertEqual(self.lmp.extract_global("units"), "lj")
        self.assertEqual(self.lmp.extract_global("ntimestep"), 0)
        self.assertEqual(self.lmp.extract_global("dt"), 0.005)

        self.assertEqual(self.lmp.extract_global("boxxlo"), -1.0)
        self.assertEqual(self.lmp.extract_global("boxxhi"), 1.0)
        self.assertEqual(self.lmp.extract_global("boxylo"), -2.0)
        self.assertEqual(self.lmp.extract_global("boxyhi"), 2.0)
        self.assertEqual(self.lmp.extract_global("boxzlo"), -3.0)
        self.assertEqual(self.lmp.extract_global("boxzhi"), 3.0)
        self.assertEqual(self.lmp.extract_global("boxlo"), [-1.0, -2.0, -3.0])
        self.assertEqual(self.lmp.extract_global("boxhi"), [1.0, 2.0, 3.0])
        self.assertEqual(self.lmp.extract_global("sublo"), [-1.0, -2.0, -3.0])
        self.assertEqual(self.lmp.extract_global("subhi"), [1.0, 2.0, 3.0])
        self.assertEqual(self.lmp.extract_global("periodicity"), [1,1,1])
        self.assertEqual(self.lmp.extract_global("triclinic"), 0)
        self.assertEqual(self.lmp.extract_global("sublo_lambda"), None)
        self.assertEqual(self.lmp.extract_global("subhi_lambda"), None)
        self.assertEqual(self.lmp.extract_global("respa_levels"), None)
        self.assertEqual(self.lmp.extract_global("respa_dt"), None)

        # set and initialize r-RESPA
        self.lmp.command("run_style respa 3 5 2 pair 2 kspace 3")
        self.lmp.command("mass * 1.0")
        self.lmp.command("run 1 post no")
        self.assertEqual(self.lmp.extract_global("ntimestep"), 1)
        self.assertEqual(self.lmp.extract_global("respa_levels"), 3)
        self.assertEqual(self.lmp.extract_global("respa_dt"), [0.0005, 0.0025, 0.005])

        # checks only for triclinic boxes
        self.lmp.command("change_box all triclinic")
        self.assertEqual(self.lmp.extract_global("triclinic"), 1)
        self.assertEqual(self.lmp.extract_global("sublo_lambda"), [0.0, 0.0, 0.0])
        self.assertEqual(self.lmp.extract_global("subhi_lambda"), [1.0, 1.0, 1.0])

##############################
if __name__ == "__main__":
    unittest.main()
