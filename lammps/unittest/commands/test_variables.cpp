/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "lammps.h"

#include "atom.h"
#include "domain.h"
#include "group.h"
#include "info.h"
#include "input.h"
#include "math_const.h"
#include "region.h"
#include "variable.h"

#include "../testing/core.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstring>
#include <vector>

// whether to print verbose output (i.e. not capturing LAMMPS screen output).
bool verbose = false;

using LAMMPS_NS::MathConst::MY_PI;
using LAMMPS_NS::utils::split_words;

namespace LAMMPS_NS {
using ::testing::ExitedWithCode;
using ::testing::MatchesRegex;
using ::testing::StrEq;

class VariableTest : public LAMMPSTest {
protected:
    Group *group;
    Domain *domain;
    Variable *variable;

    void SetUp() override
    {
        testbinary = "VariableTest";
        args       = {"-log", "none", "-echo", "screen", "-nocite", "-v", "num", "1"};
        LAMMPSTest::SetUp();
        group    = lmp->group;
        domain   = lmp->domain;
        variable = lmp->input->variable;
    }

    void TearDown() override
    {
        LAMMPSTest::TearDown();
        unlink("test_variable.file");
        unlink("test_variable.atomfile");
    }

    void atomic_system()
    {
        BEGIN_HIDE_OUTPUT();
        command("units real");
        command("lattice sc 1.0 origin 0.125 0.125 0.125");
        command("region box block -2 2 -2 2 -2 2");
        command("create_box 8 box");
        command("create_atoms 1 box");
        command("mass * 1.0");
        command("region left block -2.0 -1.0 INF INF INF INF");
        command("region right block 0.5  2.0 INF INF INF INF");
        command("region top block INF INF -2.0 -1.0 INF INF");
        command("set region left type 2");
        command("set region right type 3");
        END_HIDE_OUTPUT();
    }

    void molecular_system()
    {
        BEGIN_HIDE_OUTPUT();
        command("fix props all property/atom mol rmass q");
        END_HIDE_OUTPUT();
        atomic_system();
        BEGIN_HIDE_OUTPUT();
        command("variable molid atom floor(id/4)+1");
        command("variable charge atom 2.0*sin(PI/32*id)");
        command("set atom * mol v_molid");
        command("set atom * charge v_charge");
        command("set type 1 mass 0.5");
        command("set type 2*4 mass 2.0");
        END_HIDE_OUTPUT();
    }

    void file_vars()
    {
        FILE *fp = fopen("test_variable.file", "w");
        fputs("# test file for file style variable\n\n\none\n  two  \n\n"
              "three  # with comment\nfour   ! with non-comment\n"
              "# comments only\n	five\n#END\n",
              fp);
        fclose(fp);

        fp = fopen("test_variable.atomfile", "w");
        fputs("# test file for atomfile style variable\n\n"
              "4  # four lines\n4 0.5   #with comment\n"
              "2 -0.5         \n3 1.5\n1 -1.5\n\n"
              "2\n10 1.0 # test\n13 1.0\n\n######\n"
              "4\n1 4.0 # test\n2 3.0\n3 2.0\n4 1.0\n#END\n",
              fp);
        fclose(fp);
    }
};

TEST_F(VariableTest, CreateDelete)
{
    file_vars();
    ASSERT_EQ(variable->nvar, 1);
    BEGIN_HIDE_OUTPUT();
    command("shell putenv TEST_VARIABLE=simpletest2");
    command("shell putenv TEST_VARIABLE2=simpletest OTHER_VARIABLE=2");
    command("variable one    index     1 2 3 4");
    command("variable two    equal     1");
    command("variable two    equal     2");
    command("variable three  string    four");
    command("variable three  string    three");
    command("variable four1  loop      4");
    command("variable four2  loop      2 4");
    command("variable five1  loop      100 pad");
    command("variable five2  loop      10 200 pad");
    command("variable six    world     one");
    command("variable seven  format    two \"%5.2f\"");
    command("variable eight  getenv    TEST_VARIABLE2");
    command("variable eight  getenv    XXX");
    command("variable nine   file      test_variable.file");
    command("variable ten    internal  1.0");
    command("variable ten    internal  10.0");
    command("variable ten1   universe  1 2 3 4");
    command("variable ten2   uloop     4");
    command("variable ten3   uloop     4 pad");
    command("variable dummy  index     0");
    command("variable file   equal     is_file(MYFILE)");
    END_HIDE_OUTPUT();
    ASSERT_EQ(variable->nvar, 18);
    BEGIN_HIDE_OUTPUT();
    command("variable dummy  delete");
    END_HIDE_OUTPUT();
    ASSERT_EQ(variable->nvar, 17);
    ASSERT_THAT(variable->retrieve("three"), StrEq("three"));
    variable->set_string("three", "four");
    ASSERT_THAT(variable->retrieve("three"), StrEq("four"));
    ASSERT_THAT(variable->retrieve("four2"), StrEq("2"));
    ASSERT_THAT(variable->retrieve("five1"), StrEq("001"));
    ASSERT_THAT(variable->retrieve("seven"), StrEq(" 2.00"));
    ASSERT_THAT(variable->retrieve("ten"), StrEq("1"));
    ASSERT_THAT(variable->retrieve("eight"), StrEq(""));
    variable->internal_set(variable->find("ten"), 2.5);
    ASSERT_THAT(variable->retrieve("ten"), StrEq("2.5"));
    ASSERT_THAT(variable->retrieve("file"), StrEq("0"));
    FILE *fp = fopen("MYFILE", "w");
    fputs(" ", fp);
    fclose(fp);
    ASSERT_THAT(variable->retrieve("file"), StrEq("1"));
    unlink("MYFILE");
    ASSERT_THAT(variable->retrieve("file"), StrEq("0"));

    BEGIN_HIDE_OUTPUT();
    command("variable seven delete");
    command("variable seven getenv TEST_VARIABLE");
    command("variable eight getenv OTHER_VARIABLE");
    END_HIDE_OUTPUT();
    ASSERT_THAT(variable->retrieve("seven"), StrEq("simpletest2"));
    ASSERT_THAT(variable->retrieve("eight"), StrEq("2"));

    ASSERT_EQ(variable->equalstyle(variable->find("one")), 0);
    ASSERT_EQ(variable->equalstyle(variable->find("two")), 1);
    ASSERT_EQ(variable->equalstyle(variable->find("ten")), 1);

    ASSERT_EQ(variable->internalstyle(variable->find("two")), 0);
    ASSERT_EQ(variable->internalstyle(variable->find("ten")), 1);

    TEST_FAILURE(".*ERROR: Illegal variable command.*", command("variable"););
    TEST_FAILURE(".*ERROR: Illegal variable command.*", command("variable dummy index"););
    TEST_FAILURE(".*ERROR: Illegal variable command.*", command("variable dummy delete xxx"););
    TEST_FAILURE(".*ERROR: Illegal variable command.*", command("variable dummy loop -1"););
    TEST_FAILURE(".*ERROR: Illegal variable command.*", command("variable dummy loop 10 1"););
    TEST_FAILURE(".*ERROR: Illegal variable command.*", command("variable dummy xxxx"););
    TEST_FAILURE(".*ERROR: Cannot redefine variable as a different style.*",
                 command("variable two string xxx"););
    TEST_FAILURE(".*ERROR: Cannot redefine variable as a different style.*",
                 command("variable two getenv xxx"););
    TEST_FAILURE(".*ERROR: Cannot redefine variable as a different style.*",
                 command("variable one equal 2"););
    TEST_FAILURE(".*ERROR: Cannot redefine variable as a different style.*",
                 command("variable one internal 2"););
    TEST_FAILURE(".*ERROR: Cannot use atomfile-style variable unless an atom map exists.*",
                 command("variable eleven    atomfile  test_variable.atomfile"););
    TEST_FAILURE(".*ERROR on proc 0: Cannot open file variable file test_variable.xxx.*",
                 command("variable nine1  file      test_variable.xxx"););
    TEST_FAILURE(".*ERROR: World variable count doesn't match # of partitions.*",
                 command("variable ten10 world xxx xxx"););
    TEST_FAILURE(".*ERROR: All universe/uloop variables must have same # of values.*",
                 command("variable ten4   uloop     2"););
    TEST_FAILURE(".*ERROR: Incorrect conversion in format string.*",
                 command("variable ten11  format    two \"%08f\""););
    TEST_FAILURE(".*ERROR: Variable name 'ten@12' must have only alphanumeric characters or.*",
                 command("variable ten@12  index    one two three"););
    TEST_FAILURE(".*ERROR: Variable evaluation before simulation box is defined.*",
                 variable->compute_equal("c_thermo_press"););
    TEST_FAILURE(".*ERROR: Invalid variable reference v_unknown in variable formula.*",
                 variable->compute_equal("v_unknown"););
}

TEST_F(VariableTest, AtomicSystem)
{
    HIDE_OUTPUT([&] {
        command("atom_modify map array");
    });
    atomic_system();
    file_vars();

    BEGIN_HIDE_OUTPUT();
    command("variable  one  index     1 2 3 4");
    command("variable  id   atom      type");
    command("variable  id   atom      id");
    command("variable  ten  atomfile  test_variable.atomfile");

    command("compute  press all pressure NULL pair");
    command("compute  rg    all gyration");
    command("compute  vacf  all vacf");
    command("fix  press all ave/time 1 1 1 c_press mode vector");
    command("fix  rg    all ave/time 1 1 1 c_rg mode vector");
    command("fix  vacf  all ave/time 1 1 1 c_vacf mode vector");

    command("variable  press  vector  f_press");
    command("variable  rg     vector  f_rg");
    command("variable  vacf   vector  f_vacf");
    command("variable  press  vector  f_press+0.0");
    command("variable  self   vector  v_self+f_press");
    command("variable  circle vector  f_press+v_circle");
    command("variable  sum    vector  v_press+v_rg");
    command("variable  sum2   vector  v_vacf+v_rg");
    command("variable  pmax   equal   max(v_press)");
    command("variable  psum   equal   sum(v_press)");
    command("variable  rgmax  equal   max(v_rg)");
    command("variable  rgsum  equal   sum(v_rg)");
    command("variable  loop   equal   v_loop+1");
    command("run 0 post no");
    END_HIDE_OUTPUT();

    ASSERT_EQ(variable->atomstyle(variable->find("one")), 0);
    ASSERT_EQ(variable->atomstyle(variable->find("id")), 1);
    ASSERT_EQ(variable->atomstyle(variable->find("ten")), 1);

    ASSERT_EQ(variable->vectorstyle(variable->find("one")), 0);
    ASSERT_EQ(variable->vectorstyle(variable->find("press")), 1);

    ASSERT_DOUBLE_EQ(variable->compute_equal("v_pmax"), 0.0);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_psum"), 0.0);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_rgmax"), 1.25);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_rgsum"), 3.75);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_sum[1]"), 1.25);

    TEST_FAILURE(".*ERROR: Cannot redefine variable as a different style.*",
                 command("variable one atom x"););
    TEST_FAILURE(".*ERROR: Cannot redefine variable as a different style.*",
                 command("variable one vector f_press"););
    TEST_FAILURE(".*ERROR on proc 0: Cannot open file variable file test_variable.xxx.*",
                 command("variable ten1   atomfile  test_variable.xxx"););
    TEST_FAILURE(".*ERROR: Variable loop: has a circular dependency.*",
                 variable->compute_equal("v_loop"););
    TEST_FAILURE(".*Variable self: Vector-style variable in equal-style variable formula.*",
                 variable->compute_equal("v_self"););
    TEST_FAILURE(".*ERROR: Variable sum2: Inconsistent lengths in vector-style variable.*",
                 variable->compute_equal("max(v_sum2)"););
}

TEST_F(VariableTest, Expressions)
{
    atomic_system();
    BEGIN_HIDE_OUTPUT();
    command("variable one    index     1");
    command("variable two    equal     2");
    command("variable three  equal     v_one+v_two");
    command("variable four   equal     PI");
    command("variable five   equal     version");
    command("variable six    equal     XXX");
    command("variable seven  equal     -v_one");
    command("variable eight  equal     v_three-0.5");
    command("variable nine   equal     v_two*(v_one+v_three)");
    command("variable ten    equal     (1.0/v_two)^2");
    command("variable eleven equal     v_three%2");
    command("variable twelve equal     1==2");
    command("variable ten3   equal     1!=v_two");
    command("variable ten4   equal     1<2");
    command("variable ten5   equal     2>1");
    command("variable ten6   equal     (1<=v_one)&&(v_ten>=0.2)");
    command("variable ten7   equal     !(1<v_two)");
    command("variable ten8   equal     1|^0");
    command("variable ten9   equal     v_one-v_ten9");
    command("variable ten10  internal  100.0");
    command("variable ten11  equal     (1!=1)+(2<1)+(2<=1)+(1>2)+(1>=2)+(1&&0)+(0||0)+(1|^1)+10^0");
    command("variable ten12  equal     yes+no+on+off+true+false");
    command("variable err1   equal     v_one/v_ten7");
    command("variable err2   equal     v_one%v_ten7");
    command("variable err3   equal     v_ten7^-v_one");
    variable->set("dummy  index     1 2");
    END_HIDE_OUTPUT();

    int ivar = variable->find("one");
    ASSERT_FALSE(variable->equalstyle(ivar));
    ivar = variable->find("two");
    ASSERT_TRUE(variable->equalstyle(ivar));
    ASSERT_DOUBLE_EQ(variable->compute_equal(ivar), 2.0);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_three"), 3.0);
    ASSERT_FLOAT_EQ(variable->compute_equal("v_four"), MY_PI);
    ASSERT_GE(variable->compute_equal("v_five"), 20210310);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_seven"), -1);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_eight"), 2.5);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_nine"), 8);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_ten"), 0.25);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_eleven"), 1);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_twelve"), 0);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_ten3"), 1);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_ten4"), 1);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_ten5"), 1);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_ten6"), 1);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_ten7"), 0);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_ten8"), 1);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_ten10"), 100);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_ten11"), 1);
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_ten12"), 3);

    TEST_FAILURE(".*ERROR: Variable six: Invalid thermo keyword 'XXX' in variable formula.*",
                 command("print \"${six}\""););
    TEST_FAILURE(".*ERROR: Variable ten9: has a circular dependency.*",
                 command("print \"${ten9}\""););
    TEST_FAILURE(".*ERROR on proc 0: Variable err1: Divide by 0 in variable formula.*",
                 command("print \"${err1}\""););
    TEST_FAILURE(".*ERROR on proc 0: Variable err2: Modulo 0 in variable formula.*",
                 command("print \"${err2}\""););
    TEST_FAILURE(".*ERROR on proc 0: Variable err3: Invalid power expression in variable formula.*",
                 command("print \"${err3}\""););
}

TEST_F(VariableTest, Functions)
{
    atomic_system();
    file_vars();

    BEGIN_HIDE_OUTPUT();
    command("variable seed   index     643532");
    command("variable one    index     1");
    command("variable two    equal     random(1,2,v_seed)");
    command("variable three  equal     atan2(v_one,1)");
    command("variable four   equal     atan2()");
    command("variable five   equal     sqrt(v_one+v_one)");
    command("variable six    equal     exp(ln(0.1))");
    command("variable seven  equal     abs(log(1.0/100.0))");
    command("variable eight  equal     0.5*PI");
    command("variable nine   equal     round(sin(v_eight)+cos(v_eight))");
    command("variable ten    equal     floor(1.85)+ceil(1.85)");
    command("variable ten1   equal     tan(v_eight/2.0)");
    command("variable ten2   equal     asin(-1.0)+acos(0.0)");
    command("variable ten3   equal     floor(100*random(0.2,0.8,v_seed)+1)");
    END_HIDE_OUTPUT();

    ASSERT_GT(variable->compute_equal(variable->find("two")), 0.99);
    ASSERT_LT(variable->compute_equal(variable->find("two")), 2.01);
    ASSERT_DOUBLE_EQ(variable->compute_equal(variable->find("three")), 0.25 * MY_PI);
    ASSERT_DOUBLE_EQ(variable->compute_equal(variable->find("five")), sqrt(2.0));
    ASSERT_DOUBLE_EQ(variable->compute_equal(variable->find("six")), 0.1);
    ASSERT_DOUBLE_EQ(variable->compute_equal(variable->find("seven")), 2);
    ASSERT_DOUBLE_EQ(variable->compute_equal(variable->find("nine")), 1);
    ASSERT_DOUBLE_EQ(variable->compute_equal(variable->find("ten")), 3);
    ASSERT_FLOAT_EQ(variable->compute_equal(variable->find("ten1")), 1);
    ASSERT_GT(variable->compute_equal(variable->find("ten3")), 19);
    ASSERT_LT(variable->compute_equal(variable->find("ten3")), 81);

    TEST_FAILURE(".*ERROR: Variable four: Invalid syntax in variable formula.*",
                 command("print \"${four}\""););
}

TEST_F(VariableTest, IfCommand)
{
    BEGIN_HIDE_OUTPUT();
    command("variable one index 1");
    END_HIDE_OUTPUT();

    BEGIN_CAPTURE_OUTPUT();
    command("if 1>0 then 'print \"bingo!\"'");
    auto text = END_CAPTURE_OUTPUT();
    ASSERT_THAT(text, MatchesRegex(".*bingo!.*"));

    BEGIN_CAPTURE_OUTPUT();
    command("if 1>2 then 'print \"bingo!\"' else 'print \"nope?\"'");
    text = END_CAPTURE_OUTPUT();
    ASSERT_THAT(text, MatchesRegex(".*nope\?.*"));

    BEGIN_CAPTURE_OUTPUT();
    command("if (1<=0) then 'print \"bingo!\"' else 'print \"nope?\"'");
    text = END_CAPTURE_OUTPUT();
    ASSERT_THAT(text, MatchesRegex(".*nope\?.*"));

    BEGIN_CAPTURE_OUTPUT();
    command("if (-1.0e-1<0.0E+0)|^(1<0) then 'print \"bingo!\"'");
    text = END_CAPTURE_OUTPUT();
    ASSERT_THAT(text, MatchesRegex(".*bingo!.*"));

    BEGIN_CAPTURE_OUTPUT();
    command("if (${one}==1.0)&&(2>=1) then 'print \"bingo!\"'");
    text = END_CAPTURE_OUTPUT();
    ASSERT_THAT(text, MatchesRegex(".*bingo!.*"));

    BEGIN_CAPTURE_OUTPUT();
    command("if !((${one}!=1.0)||(2|^1)) then 'print \"missed\"' else 'print \"bingo!\"'");
    text = END_CAPTURE_OUTPUT();
    ASSERT_THAT(text, MatchesRegex(".*bingo!.*"));

    BEGIN_CAPTURE_OUTPUT();
    command("if (1>=2)&&(0&&1) then 'print \"missed\"' else 'print \"bingo!\"'");
    text = END_CAPTURE_OUTPUT();
    ASSERT_THAT(text, MatchesRegex(".*bingo!.*"));

    BEGIN_CAPTURE_OUTPUT();
    command("if !1 then 'print \"missed\"' else 'print \"bingo!\"'");
    text = END_CAPTURE_OUTPUT();
    ASSERT_THAT(text, MatchesRegex(".*bingo!.*"));

    BEGIN_CAPTURE_OUTPUT();
    command("if !(a==b) then 'print \"bingo!\"'");
    text = END_CAPTURE_OUTPUT();
    ASSERT_THAT(text, MatchesRegex(".*bingo!.*"));

    BEGIN_CAPTURE_OUTPUT();
    command("if x==x|^1==0 then 'print \"bingo!\"'");
    text = END_CAPTURE_OUTPUT();
    ASSERT_THAT(text, MatchesRegex(".*bingo!.*"));

    BEGIN_CAPTURE_OUTPUT();
    command("if x!=x|^a!=b then 'print \"bingo!\"'");
    text = END_CAPTURE_OUTPUT();

    ASSERT_THAT(text, MatchesRegex(".*bingo!.*"));

    TEST_FAILURE(".*ERROR: Invalid Boolean syntax in if command.*",
                 command("if () then 'print \"bingo!\"'"););
    TEST_FAILURE(".*ERROR: Invalid Boolean syntax in if command.*",
                 command("if \"1 1\" then 'print \"bingo!\"'"););
    TEST_FAILURE(".*ERROR: Invalid Boolean syntax in if command.*",
                 command("if 1a then 'print \"bingo!\"'"););
    TEST_FAILURE(".*ERROR: Invalid Boolean syntax in if command.*",
                 command("if 1=<2 then 'print \"bingo!\"'"););
    TEST_FAILURE(".*ERROR: Invalid Boolean syntax in if command.*",
                 command("if 1!=a then 'print \"bingo!\"'"););
    TEST_FAILURE(".*ERROR: Invalid Boolean syntax in if command.*",
                 command("if 1&<2 then 'print \"bingo!\"'"););
    TEST_FAILURE(".*ERROR: Invalid Boolean syntax in if command.*",
                 command("if 1|<2 then 'print \"bingo!\"'"););
    TEST_FAILURE(".*ERROR: Invalid Boolean syntax in if command.*",
                 command("if (1)( then 'print \"bingo!\"'"););
    TEST_FAILURE(".*ERROR: Invalid Boolean syntax in if command.*",
                 command("if (1)1 then 'print \"bingo!\"'"););
    TEST_FAILURE(".*ERROR: Invalid Boolean syntax in if command.*",
                 command("if (v_one==1.0)&&(2>=1) then 'print \"bingo!\"'"););
}

TEST_F(VariableTest, NextCommand)
{
    file_vars();

    BEGIN_HIDE_OUTPUT();
    command("variable one    index  1 2");
    command("variable two    equal  2");
    command("variable three  file   test_variable.file");
    command("variable four   loop   2 4");
    command("variable five   index  1 2");
    END_HIDE_OUTPUT();
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_one"), 1);
    ASSERT_THAT(variable->retrieve("three"), StrEq("one"));
    BEGIN_HIDE_OUTPUT();
    command("next one");
    command("next three");
    END_HIDE_OUTPUT();
    ASSERT_DOUBLE_EQ(variable->compute_equal("v_one"), 2);
    ASSERT_THAT(variable->retrieve("three"), StrEq("two"));
    ASSERT_GE(variable->find("one"), 0);
    BEGIN_HIDE_OUTPUT();
    command("next one");
    command("next three");
    END_HIDE_OUTPUT();
    // index style variable is deleted if no more next element
    ASSERT_EQ(variable->find("one"), -1);
    ASSERT_GE(variable->find("three"), 0);
    BEGIN_HIDE_OUTPUT();
    command("next three");
    command("next three");
    command("next three");
    END_HIDE_OUTPUT();
    // file style variable is deleted if no more next element
    ASSERT_EQ(variable->find("three"), -1);

    TEST_FAILURE(".*ERROR: Illegal next command.*", command("next"););
    TEST_FAILURE(".*ERROR: Invalid variable 'xxx' in next command.*", command("next xxx"););
    TEST_FAILURE(".*ERROR: Invalid variable style with next command.*", command("next two"););
    TEST_FAILURE(".*ERROR: All variables in next command must have same style.*",
                 command("next five four"););
}
} // namespace LAMMPS_NS

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleMock(&argc, argv);

    if (Info::get_mpi_vendor() == "Open MPI" && !LAMMPS_NS::Info::has_exceptions())
        std::cout << "Warning: using OpenMPI without exceptions. "
                     "Death tests will be skipped\n";

    // handle arguments passed via environment variable
    if (const char *var = getenv("TEST_ARGS")) {
        std::vector<std::string> env = split_words(var);
        for (auto arg : env) {
            if (arg == "-v") {
                verbose = true;
            }
        }
    }

    if ((argc > 1) && (strcmp(argv[1], "-v") == 0)) verbose = true;

    int rv = RUN_ALL_TESTS();
    MPI_Finalize();
    return rv;
}
