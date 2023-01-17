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

#include "../testing/core.h"
#include "../testing/utils.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "info.h"
#include "input.h"
#include "lammps.h"
#include "update.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdio>
#include <mpi.h>
#include <string>

using namespace LAMMPS_NS;

using testing::MatchesRegex;
using testing::StrEq;

using utils::read_lines_from_file;
using utils::sfgets;
using utils::sfread;
using utils::split_words;

// whether to print verbose output (i.e. not capturing LAMMPS screen output).
bool verbose = false;

class FileOperationsTest : public LAMMPSTest {
protected:
    void SetUp() override
    {
        testbinary = "FileOperationsTest";
        LAMMPSTest::SetUp();
        ASSERT_NE(lmp, nullptr);

        std::ofstream out("safe_file_read_test.txt", std::ios_base::out | std::ios_base::binary);
        ASSERT_TRUE(out.good());
        out << "one line\ntwo_lines\n\nno newline";
        out.close();

        out.open("file_with_long_lines_test.txt", std::ios_base::out | std::ios_base::binary);
        ASSERT_TRUE(out.good());
        out << "zero ##########################################################"
               "##################################################################"
               "##################################################################"
               "############################################################\n";
        out << "one line\ntwo_lines\n\n";
        for (int i = 0; i < 100; ++i)
            out << "one two ";
        out << "\nthree\nfour five #";
        for (int i = 0; i < 1000; ++i)
            out << '#';
        out.close();
    }

    void TearDown() override
    {
        LAMMPSTest::TearDown();
        remove("safe_file_read_test.txt");
        remove("file_with_long_lines_test.txt");
    }
};

static constexpr int MAX_BUF_SIZE = 128;

TEST_F(FileOperationsTest, safe_fgets)
{
    char buf[MAX_BUF_SIZE];

    FILE *fp = fopen("safe_file_read_test.txt", "rb");
    ASSERT_NE(fp, nullptr);

    memset(buf, 0, MAX_BUF_SIZE);
    utils::sfgets(FLERR, buf, MAX_BUF_SIZE, fp, "safe_file_read_test.txt", lmp->error);
    ASSERT_THAT(buf, StrEq("one line\n"));

    memset(buf, 0, MAX_BUF_SIZE);
    utils::sfgets(FLERR, buf, MAX_BUF_SIZE, fp, "safe_file_read_test.txt", lmp->error);
    ASSERT_THAT(buf, StrEq("two_lines\n"));

    memset(buf, 0, MAX_BUF_SIZE);
    utils::sfgets(FLERR, buf, MAX_BUF_SIZE, fp, "safe_file_read_test.txt", lmp->error);
    ASSERT_THAT(buf, StrEq("\n"));

    memset(buf, 0, MAX_BUF_SIZE);
    utils::sfgets(FLERR, buf, 4, fp, "safe_file_read_test.txt", lmp->error);
    ASSERT_THAT(buf, StrEq("no "));

    memset(buf, 0, MAX_BUF_SIZE);
    utils::sfgets(FLERR, buf, MAX_BUF_SIZE, fp, "safe_file_read_test.txt", lmp->error);
    ASSERT_THAT(buf, StrEq("newline"));

    memset(buf, 0, MAX_BUF_SIZE);
    TEST_FAILURE(
        ".*ERROR on proc 0: Unexpected end of file while "
        "reading file 'safe_file_read_test.txt'.*",
        utils::sfgets(FLERR, buf, MAX_BUF_SIZE, fp, "safe_file_read_test.txt", lmp->error););
    fclose(fp);
}

TEST_F(FileOperationsTest, fgets_trunc)
{
    char buf[MAX_BUF_SIZE];
    char *ptr;

    FILE *fp = fopen("safe_file_read_test.txt", "rb");
    ASSERT_NE(fp, nullptr);

    // read line shorter than buffer
    memset(buf, 0, MAX_BUF_SIZE);
    ptr = utils::fgets_trunc(buf, MAX_BUF_SIZE, fp);
    ASSERT_THAT(buf, StrEq("one line\n"));
    ASSERT_NE(ptr, nullptr);

    // read line of exactly the buffer length
    memset(buf, 0, MAX_BUF_SIZE);
    ptr = utils::fgets_trunc(buf, sizeof("two_lines\n"), fp);
    ASSERT_THAT(buf, StrEq("two_lines\n"));
    ASSERT_NE(ptr, nullptr);

    memset(buf, 0, MAX_BUF_SIZE);
    ptr = utils::fgets_trunc(buf, MAX_BUF_SIZE, fp);
    ASSERT_THAT(buf, StrEq("\n"));
    ASSERT_NE(ptr, nullptr);

    memset(buf, 0, MAX_BUF_SIZE);
    ptr = utils::fgets_trunc(buf, 4, fp);
    ASSERT_THAT(buf, StrEq("no\n"));
    ASSERT_NE(ptr, nullptr);

    ptr = utils::fgets_trunc(buf, MAX_BUF_SIZE, fp);
    ASSERT_EQ(ptr, nullptr);
    fclose(fp);

    fp = fopen("file_with_long_lines_test.txt", "r");
    ASSERT_NE(fp, nullptr);

    memset(buf, 0, MAX_BUF_SIZE);
    ptr = utils::fgets_trunc(buf, MAX_BUF_SIZE, fp);
    ASSERT_NE(ptr, nullptr);
    ASSERT_THAT(buf, StrEq("zero ##########################################################"
                           "###############################################################\n"));

    ptr = utils::fgets_trunc(buf, MAX_BUF_SIZE, fp);
    ASSERT_THAT(buf, StrEq("one line\n"));
    ASSERT_NE(ptr, nullptr);

    ptr = utils::fgets_trunc(buf, MAX_BUF_SIZE, fp);
    ASSERT_THAT(buf, StrEq("two_lines\n"));
    ASSERT_NE(ptr, nullptr);

    ptr = utils::fgets_trunc(buf, MAX_BUF_SIZE, fp);
    ASSERT_THAT(buf, StrEq("\n"));
    ASSERT_NE(ptr, nullptr);

    ptr = utils::fgets_trunc(buf, MAX_BUF_SIZE, fp);
    ASSERT_NE(ptr, nullptr);
    ASSERT_THAT(buf, StrEq("one two one two one two one two one two one two one two one two "
                           "one two one two one two one two one two one two one two one tw\n"));

    fclose(fp);
}

TEST_F(FileOperationsTest, safe_fread)
{
    char buf[MAX_BUF_SIZE];

    FILE *fp = fopen("safe_file_read_test.txt", "rb");
    ASSERT_NE(fp, nullptr);

    memset(buf, 0, MAX_BUF_SIZE);
    utils::sfread(FLERR, buf, 1, 9, fp, "safe_file_read_test.txt", lmp->error);
    ASSERT_THAT(buf, StrEq("one line\n"));

    memset(buf, 0, MAX_BUF_SIZE);
    utils::sfread(FLERR, buf, 1, 10, fp, "safe_file_read_test.txt", nullptr);
    ASSERT_THAT(buf, StrEq("two_lines\n"));

    TEST_FAILURE(".*ERROR on proc 0: Unexpected end of file while "
                 "reading file 'safe_file_read_test.txt'.*",
                 utils::sfread(FLERR, buf, 1, 100, fp, "safe_file_read_test.txt", lmp->error););

    // short read but no error triggered due to passing a NULL pointer
    memset(buf, 0, MAX_BUF_SIZE);
    clearerr(fp);
    rewind(fp);
    utils::sfread(FLERR, buf, 1, 100, fp, "safe_file_read_test.txt", nullptr);
    ASSERT_THAT(buf, StrEq("one line\ntwo_lines\n\nno newline"));
    fclose(fp);
}

TEST_F(FileOperationsTest, read_lines_from_file)
{
    char *buf      = new char[MAX_BUF_SIZE];
    FILE *fp       = nullptr;
    MPI_Comm world = MPI_COMM_WORLD;
    int me, rv;
    memset(buf, 0, MAX_BUF_SIZE);
    MPI_Comm_rank(world, &me);

    rv = utils::read_lines_from_file(nullptr, 1, MAX_BUF_SIZE, buf, me, world);
    ASSERT_EQ(rv, 1);

    if (me == 0) {
        fp = fopen("safe_file_read_test.txt", "r");
        ASSERT_NE(fp, nullptr);
    } else
        ASSERT_EQ(fp, nullptr);

    rv = utils::read_lines_from_file(fp, 2, MAX_BUF_SIZE / 2, buf, me, world);
    ASSERT_EQ(rv, 0);
    ASSERT_THAT(buf, StrEq("one line\ntwo_lines\n"));

    rv = utils::read_lines_from_file(fp, 2, MAX_BUF_SIZE / 2, buf, me, world);
    ASSERT_EQ(rv, 0);
    ASSERT_THAT(buf, StrEq("\nno newline\n"));

    rv = utils::read_lines_from_file(fp, 2, MAX_BUF_SIZE / 2, buf, me, world);
    ASSERT_EQ(rv, 1);
    delete[] buf;
}

TEST_F(FileOperationsTest, logmesg)
{
    char buf[64];
    BEGIN_HIDE_OUTPUT();
    command("echo none");
    END_HIDE_OUTPUT();
    BEGIN_CAPTURE_OUTPUT();
    utils::logmesg(lmp, "one\n");
    command("log test_logmesg.log");
    utils::logmesg(lmp, "two\n");
    utils::logmesg(lmp, "three={}\n", 3);
    utils::logmesg(lmp, "four {}\n");
    utils::logmesg(lmp, "five\n", 5);
    command("log none");
    std::string out = END_CAPTURE_OUTPUT();
    memset(buf, 0, 64);
    FILE *fp = fopen("test_logmesg.log", "r");
    fread(buf, 1, 64, fp);
    fclose(fp);
    ASSERT_THAT(out, StrEq("one\ntwo\nthree=3\nargument not found\nfive\n"));
    ASSERT_THAT(buf, StrEq("two\nthree=3\nargument not found\nfive\n"));
    remove("test_logmesg.log");
}

TEST_F(FileOperationsTest, error_message_warn)
{
    char buf[64];
    BEGIN_HIDE_OUTPUT();
    command("echo none");
    command("log test_error_warn.log");
    END_HIDE_OUTPUT();
    BEGIN_CAPTURE_OUTPUT();
    lmp->error->message("testme.cpp", 10, "message me");
    lmp->error->warning("testme.cpp", 100, "warn me");
    command("log none");
    std::string out = END_CAPTURE_OUTPUT();
    memset(buf, 0, 64);
    FILE *fp = fopen("test_error_warn.log", "r");
    fread(buf, 1, 64, fp);
    fclose(fp);
    auto msg = StrEq("message me (testme.cpp:10)\n"
                     "WARNING: warn me (testme.cpp:100)\n");
    ASSERT_THAT(out, msg);
    ASSERT_THAT(buf, msg);
    remove("test_error_warn.log");
}

TEST_F(FileOperationsTest, error_all_one)
{
    BEGIN_HIDE_OUTPUT();
    command("echo none");
    command("log none");
    END_HIDE_OUTPUT();
    TEST_FAILURE(".*ERROR: exit \\(testme.cpp:10\\).*", lmp->error->all("testme.cpp", 10, "exit"););
    TEST_FAILURE(".*ERROR: exit too \\(testme.cpp:10\\).*",
                 lmp->error->all("testme.cpp", 10, "exit {}", "too"););
    TEST_FAILURE(".*ERROR: argument not found \\(testme.cpp:10\\).*",
                 lmp->error->all("testme.cpp", 10, "exit {} {}", "too"););
    TEST_FAILURE(".*ERROR on proc 0: exit \\(testme.cpp:10\\).*",
                 lmp->error->one("testme.cpp", 10, "exit"););
    TEST_FAILURE(".*ERROR on proc 0: exit too \\(testme.cpp:10\\).*",
                 lmp->error->one("testme.cpp", 10, "exit {}", "too"););
    TEST_FAILURE(".*ERROR on proc 0: argument not found \\(testme.cpp:10\\).*",
                 lmp->error->one("testme.cpp", 10, "exit {} {}", "too"););
}

TEST_F(FileOperationsTest, write_restart)
{
    BEGIN_HIDE_OUTPUT();
    command("echo none");
    END_HIDE_OUTPUT();
    TEST_FAILURE(".*ERROR: Write_restart command before simulation box is defined.*",
                 command("write_restart test.restart"););

    BEGIN_HIDE_OUTPUT();
    command("region box block -2 2 -2 2 -2 2");
    command("create_box 1 box");
    command("create_atoms 1 single 0.0 0.0 0.0");
    command("mass 1 1.0");
    command("reset_timestep 333");
    command("comm_modify cutoff 0.2");
    command("write_restart noinit.restart noinit");
    command("run 0 post no");
    command("write_restart test.restart");
    command("write_restart step*.restart");
    command("write_restart multi-%.restart");
    command("write_restart multi2-%.restart fileper 2");
    command("write_restart multi3-%.restart nfile 1");
    if (info->has_package("MPIIO")) command("write_restart test.restart.mpiio");
    END_HIDE_OUTPUT();

    ASSERT_FILE_EXISTS("noinit.restart");
    ASSERT_FILE_EXISTS("test.restart");
    ASSERT_FILE_EXISTS("step333.restart");
    ASSERT_FILE_EXISTS("multi-base.restart");
    ASSERT_FILE_EXISTS("multi-0.restart");
    ASSERT_FILE_EXISTS("multi2-base.restart");
    ASSERT_FILE_EXISTS("multi2-0.restart");
    ASSERT_FILE_EXISTS("multi3-base.restart");
    ASSERT_FILE_EXISTS("multi3-0.restart");
    if (info->has_package("MPIIO")) ASSERT_FILE_EXISTS("test.restart.mpiio");

    if (!info->has_package("MPIIO")) {
        TEST_FAILURE(".*ERROR: Writing to MPI-IO filename when MPIIO package is not inst.*",
                     command("write_restart test.restart.mpiio"););
    } else {
        TEST_FAILURE(".*ERROR: Restart file MPI-IO output not allowed with % in filename.*",
                     command("write_restart test.restart-%.mpiio"););
    }

    TEST_FAILURE(".*ERROR: Illegal write_restart command.*", command("write_restart"););
    TEST_FAILURE(".*ERROR: Illegal write_restart command.*",
                 command("write_restart test.restart xxxx"););
    TEST_FAILURE(".*ERROR on proc 0: Cannot open restart file some_crazy_dir/test.restart:"
                 " No such file or directory.*",
                 command("write_restart some_crazy_dir/test.restart"););
    BEGIN_HIDE_OUTPUT();
    command("clear");
    END_HIDE_OUTPUT();
    ASSERT_EQ(lmp->atom->natoms, 0);
    ASSERT_EQ(lmp->update->ntimestep, 0);
    ASSERT_EQ(lmp->domain->triclinic, 0);

    TEST_FAILURE(
        ".*ERROR on proc 0: Cannot open restart file noexist.restart: No such file or directory.*",
        command("read_restart noexist.restart"););

    BEGIN_HIDE_OUTPUT();
    command("read_restart step333.restart");
    command("change_box all triclinic");
    command("write_restart triclinic.restart");
    END_HIDE_OUTPUT();
    ASSERT_EQ(lmp->atom->natoms, 1);
    ASSERT_EQ(lmp->update->ntimestep, 333);
    ASSERT_EQ(lmp->domain->triclinic, 1);
    BEGIN_HIDE_OUTPUT();
    command("clear");
    END_HIDE_OUTPUT();
    ASSERT_EQ(lmp->atom->natoms, 0);
    ASSERT_EQ(lmp->update->ntimestep, 0);
    ASSERT_EQ(lmp->domain->triclinic, 0);
    BEGIN_HIDE_OUTPUT();
    command("read_restart triclinic.restart");
    END_HIDE_OUTPUT();
    ASSERT_EQ(lmp->atom->natoms, 1);
    ASSERT_EQ(lmp->update->ntimestep, 333);
    ASSERT_EQ(lmp->domain->triclinic, 1);

    // clean up
    delete_file("noinit.restart");
    delete_file("test.restart");
    delete_file("step333.restart");
    delete_file("multi-base.restart");
    delete_file("multi-0.restart");
    delete_file("multi2-base.restart");
    delete_file("multi2-0.restart");
    delete_file("multi3-base.restart");
    delete_file("multi3-0.restart");
    delete_file("triclinic.restart");
    if (info->has_package("MPIIO")) delete_file("test.restart.mpiio");
}

TEST_F(FileOperationsTest, write_data)
{
    BEGIN_HIDE_OUTPUT();
    command("echo none");
    END_HIDE_OUTPUT();
    TEST_FAILURE(".*ERROR: Write_data command before simulation box is defined.*",
                 command("write_data test.data"););

    BEGIN_HIDE_OUTPUT();
    command("region box block -2 2 -2 2 -2 2");
    command("create_box 2 box");
    command("create_atoms 1 single 0.5 0.0 0.0");
    command("pair_style zero 1.0");
    command("pair_coeff * *");
    command("mass * 1.0");
    command("reset_timestep 333");
    command("write_data noinit.data noinit");
    command("write_data nocoeff.data nocoeff");
    command("run 0 post no");
    command("write_data test.data");
    command("write_data step*.data pair ij");
    command("fix q all property/atom q");
    command("set type 1 charge -0.5");
    command("write_data charge.data");
    command("write_data nofix.data nofix");
    END_HIDE_OUTPUT();

    ASSERT_FILE_EXISTS("noinit.data");
    ASSERT_EQ(count_lines("noinit.data"), 26);
    ASSERT_FILE_EXISTS("test.data");
    ASSERT_EQ(count_lines("test.data"), 26);
    ASSERT_FILE_EXISTS("step333.data");
    ASSERT_EQ(count_lines("step333.data"), 27);
    ASSERT_FILE_EXISTS("nocoeff.data");
    ASSERT_EQ(count_lines("nocoeff.data"), 21);
    ASSERT_FILE_EXISTS("nofix.data");
    ASSERT_EQ(count_lines("nofix.data"), 26);
    ASSERT_FILE_EXISTS("charge.data");
    ASSERT_EQ(count_lines("charge.data"), 30);

    TEST_FAILURE(".*ERROR: Illegal write_data command.*", command("write_data"););
    TEST_FAILURE(".*ERROR: Illegal write_data command.*", command("write_data test.data xxxx"););
    TEST_FAILURE(".*ERROR: Illegal write_data command.*", command("write_data test.data pair xx"););
    TEST_FAILURE(".*ERROR on proc 0: Cannot open data file some_crazy_dir/test.data:"
                 " No such file or directory.*",
                 command("write_data some_crazy_dir/test.data"););

    BEGIN_HIDE_OUTPUT();
    command("clear");
    END_HIDE_OUTPUT();
    ASSERT_EQ(lmp->domain->box_exist, 0);
    ASSERT_EQ(lmp->atom->natoms, 0);
    ASSERT_EQ(lmp->update->ntimestep, 0);
    ASSERT_EQ(lmp->domain->triclinic, 0);

    TEST_FAILURE(".*ERROR: Cannot open file noexist.data: No such file or directory.*",
                 command("read_data noexist.data"););

    BEGIN_HIDE_OUTPUT();
    command("pair_style zero 1.0");
    command("read_data step333.data");
    command("change_box all triclinic");
    command("write_data triclinic.data");
    END_HIDE_OUTPUT();
    ASSERT_EQ(lmp->atom->natoms, 1);
    ASSERT_EQ(lmp->update->ntimestep, 0);
    ASSERT_EQ(lmp->domain->triclinic, 1);
    BEGIN_HIDE_OUTPUT();
    command("clear");
    END_HIDE_OUTPUT();
    ASSERT_EQ(lmp->atom->natoms, 0);
    ASSERT_EQ(lmp->domain->triclinic, 0);
    BEGIN_HIDE_OUTPUT();
    command("pair_style zero 1.0");
    command("read_data triclinic.data");
    END_HIDE_OUTPUT();
    ASSERT_EQ(lmp->atom->natoms, 1);
    ASSERT_EQ(lmp->domain->triclinic, 1);

    // clean up
    delete_file("charge.data");
    delete_file("nocoeff.data");
    delete_file("noinit.data");
    delete_file("nofix.data");
    delete_file("test.data");
    delete_file("step333.data");
    delete_file("triclinic.data");
}

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
