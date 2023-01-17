// clang-format off
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

#include "input.h"

#include "accelerator_kokkos.h"
#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "comm_brick.h"
#include "comm_tiled.h"
#include "command.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "min.h"
#include "modify.h"
#include "neighbor.h"
#include "output.h"
#include "pair.h"
#include "special.h"
#include "style_command.h"      // IWYU pragma: keep
#include "thermo.h"
#include "timer.h"
#include "universe.h"
#include "update.h"
#include "variable.h"

#include <cstring>
#include <cerrno>
#include <cctype>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _WIN32
#include <direct.h>
#endif

using namespace LAMMPS_NS;

#define DELTALINE 256
#define DELTA 4

/* ---------------------------------------------------------------------- */

/** \class LAMMPS_NS::Input
 *  \brief Class for processing commands and input files
 *
\verbatim embed:rst

The Input class contains methods for reading, pre-processing and
parsing LAMMPS commands and input files and will dispatch commands
to the respective class instances or contains the code to execute
the commands directly.  It also contains the instance of the
Variable class which performs computations and text substitutions.

\endverbatim */

/** Input class constructor
 *
\verbatim embed:rst

This sets up the input processing, processes the *-var* and *-echo*
command line flags, holds the factory of commands and creates and
initializes an instance of the Variable class.

To execute a command, a specific class instance, derived from
:cpp:class:`Command`, is created, then its ``command()`` member
function executed, and finally the class instance is deleted.

\endverbatim
 *
 * \param  lmp   pointer to the base LAMMPS class
 * \param  argc  number of entries in *argv*
 * \param  argv  argument vector  */

Input::Input(LAMMPS *lmp, int argc, char **argv) : Pointers(lmp)
{
  MPI_Comm_rank(world,&me);

  maxline = maxcopy = maxwork = 0;
  line = copy = work = nullptr;
  narg = maxarg = 0;
  arg = nullptr;

  echo_screen = 0;
  echo_log = 1;

  label_active = 0;
  labelstr = nullptr;
  jump_skip = 0;
  utf8_warn = true;

  if (me == 0) {
    nfile = 1;
    maxfile = 16;
    infiles = new FILE *[maxfile];
    infiles[0] = infile;
  } else infiles = nullptr;

  variable = new Variable(lmp);

  // fill map with commands listed in style_command.h

  command_map = new CommandCreatorMap();

#define COMMAND_CLASS
#define CommandStyle(key,Class) \
  (*command_map)[#key] = &command_creator<Class>;
#include "style_command.h"      // IWYU pragma: keep
#undef CommandStyle
#undef COMMAND_CLASS

  // process command-line args
  // check for args "-var" and "-echo"
  // caller has already checked that sufficient arguments exist

  int iarg = 1;
  while (iarg < argc) {
    if (strcmp(argv[iarg],"-var") == 0 || strcmp(argv[iarg],"-v") == 0) {
      int jarg = iarg+3;
      while (jarg < argc && argv[jarg][0] != '-') jarg++;
      variable->set(argv[iarg+1],jarg-iarg-2,&argv[iarg+2]);
      iarg = jarg;
    } else if (strcmp(argv[iarg],"-echo") == 0 ||
               strcmp(argv[iarg],"-e") == 0) {
      narg = 1;
      char **tmp = arg;        // trick echo() into using argv instead of arg
      arg = &argv[iarg+1];
      echo();
      arg = tmp;
      iarg += 2;
     } else iarg++;
  }
}

/* ---------------------------------------------------------------------- */

Input::~Input()
{
  // don't free command and arg strings
  // they just point to other allocated memory

  memory->sfree(line);
  memory->sfree(copy);
  memory->sfree(work);
  if (labelstr) delete[] labelstr;
  memory->sfree(arg);
  delete[] infiles;
  delete variable;

  delete command_map;
}

/** Process all input from the ``FILE *`` pointer *infile*
 *
\verbatim embed:rst

This will read lines from *infile*, parse and execute them until the end
of the file is reached.  The *infile* pointer will usually point to
``stdin`` or the input file given with the ``-in`` command line flag.

\endverbatim */

void Input::file()
{
  int m,n;

  while (1) {

    // read a line from input script
    // n = length of line including str terminator, 0 if end of file
    // if line ends in continuation char '&', concatenate next line

    if (me == 0) {

      m = 0;
      while (1) {

        if (infile == nullptr) {
          n = 0;
          break;
        }

        if (maxline-m < 2) reallocate(line,maxline,0);

        // end of file reached, so break
        // n == 0 if nothing read, else n = line with str terminator

        if (fgets(&line[m],maxline-m,infile) == nullptr) {
          if (m) n = strlen(line) + 1;
          else n = 0;
          break;
        }

        // continue if last char read was not a newline
        // could happen if line is very long

        m = strlen(line);
        if (line[m-1] != '\n') continue;

        // continue reading if final printable char is & char
        // or if odd number of triple quotes
        // else break with n = line with str terminator

        m--;
        while (m >= 0 && isspace(line[m])) m--;
        if (m < 0 || line[m] != '&') {
          if (numtriple(line) % 2) {
            m += 2;
            continue;
          }
          line[m+1] = '\0';
          n = m+2;
          break;
        }
      }
    }

    // bcast the line
    // if n = 0, end-of-file
    // error if label_active is set, since label wasn't encountered
    // if original input file, code is done
    // else go back to previous input file

    MPI_Bcast(&n,1,MPI_INT,0,world);
    if (n == 0) {
      if (label_active) error->all(FLERR,"Label wasn't found in input script");
      break;
    }

    if (n > maxline) reallocate(line,maxline,n);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // echo the command unless scanning for label

    if (me == 0 && label_active == 0) {
      if (echo_screen && screen) fprintf(screen,"%s\n",line);
      if (echo_log && logfile) fprintf(logfile,"%s\n",line);
    }

    // parse the line
    // if no command, skip to next line in input script

    parse();
    if (command == nullptr) continue;

    // if scanning for label, skip command unless it's a label command

    if (label_active && strcmp(command,"label") != 0) continue;

    // execute the command

    if (execute_command() && line)
      error->all(FLERR,"Unknown command: {}",line);
  }
}

/** Process all input from the file *filename*
 *
\verbatim embed:rst

This function opens the file at the path *filename*, put the current
file pointer stored in *infile* on a stack and instead assign *infile*
with the newly opened file pointer.  Then it will call the
:cpp:func:`Input::file() <LAMMPS_NS::Input::file()>` function to read,
parse and execute the contents of that file.  When the end of the file
is reached, it is closed and the previous file pointer from the infile
file pointer stack restored to *infile*.

\endverbatim
 *
 * \param  filename  name of file with LAMMPS commands */

void Input::file(const char *filename)
{
  // error if another nested file still open, should not be possible
  // open new filename and set infile, infiles[0], nfile
  // call to file() will close filename and decrement nfile

  if (me == 0) {
    if (nfile == maxfile)
      error->one(FLERR,"Too many nested levels of input scripts");

    infile = fopen(filename,"r");
    if (infile == nullptr)
      error->one(FLERR,"Cannot open input script {}: {}",
                                   filename, utils::getsyserror());

    infiles[nfile++] = infile;
  }

  // process contents of file

  file();

  if (me == 0) {
    fclose(infile);
    nfile--;
    infile = infiles[nfile-1];
  }
}

/** Process a single command from a string in *single*
 *
\verbatim embed:rst

This function takes the text in *single*, makes a copy, parses that,
executes the command and returns the name of the command (without the
arguments).  If there was no command in *single* it will return
``nullptr``.

\endverbatim
 *
 * \param  single  string with LAMMPS command
 * \return         string with name of the parsed command w/o arguments */

char *Input::one(const std::string &single)
{
  int n = single.size() + 1;
  if (n > maxline) reallocate(line,maxline,n);
  strcpy(line,single.c_str());

  // echo the command unless scanning for label

  if (me == 0 && label_active == 0) {
    if (echo_screen && screen) fprintf(screen,"%s\n",line);
    if (echo_log && logfile) fprintf(logfile,"%s\n",line);
  }

  // parse the line
  // if no command, just return a null pointer

  parse();
  if (command == nullptr) return nullptr;

  // if scanning for label, skip command unless it's a label command

  if (label_active && strcmp(command,"label") != 0) return nullptr;

  // execute the command and return its name

  if (execute_command())
    error->all(FLERR,"Unknown command: {}",line);

  return command;
}

/* ----------------------------------------------------------------------
   Send text to active echo file pointers
------------------------------------------------------------------------- */
void Input::write_echo(const std::string &txt)
{
  if (me == 0) {
    if (echo_screen && screen) fputs(txt.c_str(),screen);
    if (echo_log && logfile) fputs(txt.c_str(),logfile);
  }
}

/* ----------------------------------------------------------------------
   parse copy of command line by inserting string terminators
   strip comment = all chars from # on
   replace all $ via variable substitution except within quotes
   command = first word
   narg = # of args
   arg[] = individual args
   treat text between single/double/triple quotes as one arg via nextword()
------------------------------------------------------------------------- */

void Input::parse()
{
  // duplicate line into copy string to break into words

  int n = strlen(line) + 1;
  if (n > maxcopy) reallocate(copy,maxcopy,n);
  strcpy(copy,line);

  // strip any # comment by replacing it with 0
  // do not strip from a # inside single/double/triple quotes
  // quoteflag = 1,2,3 when encounter first single/double,triple quote
  // quoteflag = 0 when encounter matching single/double,triple quote

  int quoteflag = 0;
  char *ptr = copy;
  while (*ptr) {
    if (*ptr == '#' && !quoteflag) {
      *ptr = '\0';
      break;
    }
    if (quoteflag == 0) {
      if (strstr(ptr,"\"\"\"") == ptr) {
        quoteflag = 3;
        ptr += 2;
      }
      else if (*ptr == '"') quoteflag = 2;
      else if (*ptr == '\'') quoteflag = 1;
    } else {
      if (quoteflag == 3 && strstr(ptr,"\"\"\"") == ptr) {
        quoteflag = 0;
        ptr += 2;
      }
      else if (quoteflag == 2 && *ptr == '"') quoteflag = 0;
      else if (quoteflag == 1 && *ptr == '\'') quoteflag = 0;
    }
    ptr++;
  }

  if (utils::has_utf8(copy)) {
    std::string buf = utils::utf8_subst(copy);
    strcpy(copy,buf.c_str());
    if (utf8_warn && (comm->me == 0))
      error->warning(FLERR,"Detected non-ASCII characters in input. "
                     "Will try to continue by replacing with ASCII "
                     "equivalents where known.");
    utf8_warn = false;
  }

  // perform $ variable substitution (print changes)
  // except if searching for a label since earlier variable may not be defined

  if (!label_active) substitute(copy,work,maxcopy,maxwork,1);

  // command = 1st arg in copy string

  char *next;
  command = nextword(copy,&next);
  if (command == nullptr) return;

  // point arg[] at each subsequent arg in copy string
  // nextword() inserts string terminators into copy string to delimit args
  // nextword() treats text between single/double/triple quotes as one arg

  narg = 0;
  ptr = next;
  while (ptr) {
    if (narg == maxarg) {
      maxarg += DELTA;
      arg = (char **) memory->srealloc(arg,maxarg*sizeof(char *),"input:arg");
    }
    arg[narg] = nextword(ptr,&next);
    if (!arg[narg]) break;
    narg++;
    ptr = next;
  }
}

/* ----------------------------------------------------------------------
   find next word in str
   insert 0 at end of word
   ignore leading whitespace
   treat text between single/double/triple quotes as one arg
   matching quote must be followed by whitespace char if not end of string
   strip quotes from returned word
   return ptr to start of word or null pointer if no word in string
   also return next = ptr after word
------------------------------------------------------------------------- */

char *Input::nextword(char *str, char **next)
{
  char *start,*stop;

  // start = first non-whitespace char

  start = &str[strspn(str," \t\n\v\f\r")];
  if (*start == '\0') return nullptr;

  // if start is single/double/triple quote:
  //   start = first char beyond quote
  //   stop = first char of matching quote
  //   next = first char beyond matching quote
  //   next must be null char or whitespace
  // if start is not single/double/triple quote:
  //   stop = first whitespace char after start
  //   next = char after stop, or stop itself if stop is null char

  if (strstr(start,"\"\"\"") == start) {
    stop = strstr(&start[3],"\"\"\"");
    if (!stop) error->all(FLERR,"Unbalanced quotes in input line");
    start += 3;
    *next = stop+3;
    if (**next && !isspace(**next))
      error->all(FLERR,"Input line quote not followed by white-space");
  } else if (*start == '"' || *start == '\'') {
    stop = strchr(&start[1],*start);
    if (!stop) error->all(FLERR,"Unbalanced quotes in input line");
    start++;
    *next = stop+1;
    if (**next && !isspace(**next))
      error->all(FLERR,"Input line quote not followed by white-space");
  } else {
    stop = &start[strcspn(start," \t\n\v\f\r")];
    if (*stop == '\0') *next = stop;
    else *next = stop+1;
  }

  // set stop to null char to terminate word

  *stop = '\0';
  return start;
}

/* ----------------------------------------------------------------------
   substitute for $ variables in str using work str2 and return it
   reallocate str/str2 to hold expanded version if necessary & reset max/max2
   print updated string if flag is set and not searching for label
   label_active will be 0 if called from external class
------------------------------------------------------------------------- */

void Input::substitute(char *&str, char *&str2, int &max, int &max2, int flag)
{
  // use str2 as scratch space to expand str, then copy back to str
  // reallocate str and str2 as necessary
  // do not replace $ inside single/double/triple quotes
  // var = pts at variable name, ended by null char
  //   if $ is followed by '{', trailing '}' becomes null char
  //   else $x becomes x followed by null char
  // beyond = points to text following variable

  int i,n,paren_count;
  char immediate[256];
  char *var,*value,*beyond;
  int quoteflag = 0;
  char *ptr = str;

  n = strlen(str) + 1;
  if (n > max2) reallocate(str2,max2,n);
  *str2 = '\0';
  char *ptr2 = str2;

  while (*ptr) {

    // variable substitution

    if (*ptr == '$' && !quoteflag) {

      // value = ptr to expanded variable
      // variable name between curly braces, e.g. ${a}

      if (*(ptr+1) == '{') {
        var = ptr+2;
        i = 0;

        while (var[i] != '\0' && var[i] != '}') i++;

        if (var[i] == '\0') error->one(FLERR,"Invalid variable name");
        var[i] = '\0';
        beyond = ptr + strlen(var) + 3;
        value = variable->retrieve(var);

      // immediate variable between parenthesis, e.g. $(1/3) or $(1/3:%.6g)

      } else if (*(ptr+1) == '(') {
        var = ptr+2;
        paren_count = 0;
        i = 0;

        while (var[i] != '\0' && !(var[i] == ')' && paren_count == 0)) {
          switch (var[i]) {
          case '(': paren_count++; break;
          case ')': paren_count--; break;
          default: ;
          }
          i++;
        }

        if (var[i] == '\0') error->one(FLERR,"Invalid immediate variable");
        var[i] = '\0';
        beyond = ptr + strlen(var) + 3;

        // check if an inline format specifier was appended with a colon

        char fmtstr[64] = "%.20g";
        char *fmtflag;
        if ((fmtflag=strrchr(var, ':')) && (fmtflag[1]=='%')) {
          strncpy(fmtstr,&fmtflag[1],sizeof(fmtstr)-1);
          *fmtflag='\0';
        }

        // quick check for proper format string

        if (!utils::strmatch(fmtstr,"%[0-9 ]*\\.[0-9]+[efgEFG]"))
          error->all(FLERR,"Incorrect conversion in format string");

        snprintf(immediate,256,fmtstr,variable->compute_equal(var));
        value = immediate;

      // single character variable name, e.g. $a

      } else {
        var = ptr;
        var[0] = var[1];
        var[1] = '\0';
        beyond = ptr + 2;
        value = variable->retrieve(var);
      }

      if (value == nullptr)
        error->one(FLERR,"Substitution for illegal variable {}",var);

      // check if storage in str2 needs to be expanded
      // re-initialize ptr and ptr2 to the point beyond the variable.

      n = strlen(str2) + strlen(value) + strlen(beyond) + 1;
      if (n > max2) reallocate(str2,max2,n);
      strcat(str2,value);
      ptr2 = str2 + strlen(str2);
      ptr = beyond;

      // output substitution progress if requested

      if (flag && me == 0 && label_active == 0) {
        if (echo_screen && screen) fprintf(screen,"%s%s\n",str2,beyond);
        if (echo_log && logfile) fprintf(logfile,"%s%s\n",str2,beyond);
      }

      continue;
    }

    // quoteflag = 1,2,3 when encounter first single/double,triple quote
    // quoteflag = 0 when encounter matching single/double,triple quote
    // copy 2 extra triple quote chars into str2

    if (quoteflag == 0) {
      if (strstr(ptr,"\"\"\"") == ptr) {
        quoteflag = 3;
        *ptr2++ = *ptr++;
        *ptr2++ = *ptr++;
      }
      else if (*ptr == '"') quoteflag = 2;
      else if (*ptr == '\'') quoteflag = 1;
    } else {
      if (quoteflag == 3 && strstr(ptr,"\"\"\"") == ptr) {
        quoteflag = 0;
        *ptr2++ = *ptr++;
        *ptr2++ = *ptr++;
      }
      else if (quoteflag == 2 && *ptr == '"') quoteflag = 0;
      else if (quoteflag == 1 && *ptr == '\'') quoteflag = 0;
    }

    // copy current character into str2

    *ptr2++ = *ptr++;
    *ptr2 = '\0';
  }

  // set length of input str to length of work str2
  // copy work string back to input str

  if (max2 > max) reallocate(str,max,max2);
  strcpy(str,str2);
}

/* ----------------------------------------------------------------------
   return number of triple quotes in line
------------------------------------------------------------------------- */

int Input::numtriple(char *line)
{
  int count = 0;
  char *ptr = line;
  while ((ptr = strstr(ptr,"\"\"\""))) {
    ptr += 3;
    count++;
  }
  return count;
}

/* ----------------------------------------------------------------------
   rellocate a string
   if n > 0: set max >= n in increments of DELTALINE
   if n = 0: just increment max by DELTALINE
------------------------------------------------------------------------- */

void Input::reallocate(char *&str, int &max, int n)
{
  if (n) {
    while (n > max) max += DELTALINE;
  } else max += DELTALINE;

  str = (char *) memory->srealloc(str,max*sizeof(char),"input:str");
}

/* ----------------------------------------------------------------------
   process a single parsed command
   return 0 if successful, -1 if did not recognize command
------------------------------------------------------------------------- */

int Input::execute_command()
{
  int flag = 1;

  if (!strcmp(command,"clear")) clear();
  else if (!strcmp(command,"echo")) echo();
  else if (!strcmp(command,"if")) ifthenelse();
  else if (!strcmp(command,"include")) include();
  else if (!strcmp(command,"jump")) jump();
  else if (!strcmp(command,"label")) label();
  else if (!strcmp(command,"log")) log();
  else if (!strcmp(command,"next")) next_command();
  else if (!strcmp(command,"partition")) partition();
  else if (!strcmp(command,"print")) print();
  else if (!strcmp(command,"python")) python();
  else if (!strcmp(command,"quit")) quit();
  else if (!strcmp(command,"shell")) shell();
  else if (!strcmp(command,"variable")) variable_command();

  else if (!strcmp(command,"angle_coeff")) angle_coeff();
  else if (!strcmp(command,"angle_style")) angle_style();
  else if (!strcmp(command,"atom_modify")) atom_modify();
  else if (!strcmp(command,"atom_style")) atom_style();
  else if (!strcmp(command,"bond_coeff")) bond_coeff();
  else if (!strcmp(command,"bond_style")) bond_style();
  else if (!strcmp(command,"bond_write")) bond_write();
  else if (!strcmp(command,"boundary")) boundary();
  else if (!strcmp(command,"box")) box();
  else if (!strcmp(command,"comm_modify")) comm_modify();
  else if (!strcmp(command,"comm_style")) comm_style();
  else if (!strcmp(command,"compute")) compute();
  else if (!strcmp(command,"compute_modify")) compute_modify();
  else if (!strcmp(command,"dielectric")) dielectric();
  else if (!strcmp(command,"dihedral_coeff")) dihedral_coeff();
  else if (!strcmp(command,"dihedral_style")) dihedral_style();
  else if (!strcmp(command,"dimension")) dimension();
  else if (!strcmp(command,"dump")) dump();
  else if (!strcmp(command,"dump_modify")) dump_modify();
  else if (!strcmp(command,"fix")) fix();
  else if (!strcmp(command,"fix_modify")) fix_modify();
  else if (!strcmp(command,"group")) group_command();
  else if (!strcmp(command,"improper_coeff")) improper_coeff();
  else if (!strcmp(command,"improper_style")) improper_style();
  else if (!strcmp(command,"kspace_modify")) kspace_modify();
  else if (!strcmp(command,"kspace_style")) kspace_style();
  else if (!strcmp(command,"lattice")) lattice();
  else if (!strcmp(command,"mass")) mass();
  else if (!strcmp(command,"min_modify")) min_modify();
  else if (!strcmp(command,"min_style")) min_style();
  else if (!strcmp(command,"molecule")) molecule();
  else if (!strcmp(command,"neigh_modify")) neigh_modify();
  else if (!strcmp(command,"neighbor")) neighbor_command();
  else if (!strcmp(command,"newton")) newton();
  else if (!strcmp(command,"package")) package();
  else if (!strcmp(command,"pair_coeff")) pair_coeff();
  else if (!strcmp(command,"pair_modify")) pair_modify();
  else if (!strcmp(command,"pair_style")) pair_style();
  else if (!strcmp(command,"pair_write")) pair_write();
  else if (!strcmp(command,"processors")) processors();
  else if (!strcmp(command,"region")) region();
  else if (!strcmp(command,"reset_timestep")) reset_timestep();
  else if (!strcmp(command,"restart")) restart();
  else if (!strcmp(command,"run_style")) run_style();
  else if (!strcmp(command,"special_bonds")) special_bonds();
  else if (!strcmp(command,"suffix")) suffix();
  else if (!strcmp(command,"thermo")) thermo();
  else if (!strcmp(command,"thermo_modify")) thermo_modify();
  else if (!strcmp(command,"thermo_style")) thermo_style();
  else if (!strcmp(command,"timestep")) timestep();
  else if (!strcmp(command,"timer")) timer_command();
  else if (!strcmp(command,"uncompute")) uncompute();
  else if (!strcmp(command,"undump")) undump();
  else if (!strcmp(command,"unfix")) unfix();
  else if (!strcmp(command,"units")) units();

  else flag = 0;

  // return if command was listed above

  if (flag) return 0;

  // invoke commands added via style_command.h

  if (command_map->find(command) != command_map->end()) {
    CommandCreator &command_creator = (*command_map)[command];
    Command *cmd = command_creator(lmp);
    cmd->command(narg,arg);
    delete cmd;
    return 0;
  }

  // unrecognized command

  return -1;
}

/* ----------------------------------------------------------------------
   one instance per command in style_command.h
------------------------------------------------------------------------- */

template <typename T>
Command *Input::command_creator(LAMMPS *lmp)
{
  return new T(lmp);
}

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void Input::clear()
{
  if (narg > 0) error->all(FLERR,"Illegal clear command");
  lmp->destroy();
  lmp->create();
  lmp->post_create();
}

/* ---------------------------------------------------------------------- */

void Input::echo()
{
  if (narg != 1) error->all(FLERR,"Illegal echo command");

  if (strcmp(arg[0],"none") == 0) {
    echo_screen = 0;
    echo_log = 0;
  } else if (strcmp(arg[0],"screen") == 0) {
    echo_screen = 1;
    echo_log = 0;
  } else if (strcmp(arg[0],"log") == 0) {
    echo_screen = 0;
    echo_log = 1;
  } else if (strcmp(arg[0],"both") == 0) {
    echo_screen = 1;
    echo_log = 1;
  } else error->all(FLERR,"Illegal echo command");
}

/* ---------------------------------------------------------------------- */

void Input::ifthenelse()
{
  if (narg < 3) error->all(FLERR,"Illegal if command");

  // substitute for variables in Boolean expression for "if"
  // in case expression was enclosed in quotes
  // must substitute on copy of arg else will step on subsequent args

  int n = strlen(arg[0]) + 1;
  if (n > maxline) reallocate(line,maxline,n);
  strcpy(line,arg[0]);
  substitute(line,work,maxline,maxwork,0);

  // evaluate Boolean expression for "if"

  double btest = variable->evaluate_boolean(line);

  // bound "then" commands

  if (strcmp(arg[1],"then") != 0) error->all(FLERR,"Illegal if command");

  int first = 2;
  int iarg = first;
  while (iarg < narg &&
         (strcmp(arg[iarg],"elif") != 0 && strcmp(arg[iarg],"else") != 0))
    iarg++;
  int last = iarg-1;

  // execute "then" commands
  // make copies of all arg string commands
  // required because re-parsing a command via one() will wipe out args

  if (btest != 0.0) {
    int ncommands = last-first + 1;
    if (ncommands <= 0) error->all(FLERR,"Illegal if command");

    char **commands = new char*[ncommands];
    ncommands = 0;
    for (int i = first; i <= last; i++) {
      n = strlen(arg[i]) + 1;
      if (n == 1) error->all(FLERR,"Illegal if command");
      commands[ncommands] = new char[n];
      strcpy(commands[ncommands],arg[i]);
      ncommands++;
    }

    for (int i = 0; i < ncommands; i++) {
      one(commands[i]);
      delete[] commands[i];
    }
    delete[] commands;

    return;
  }

  // done if no "elif" or "else"

  if (iarg == narg) return;

  // check "elif" or "else" until find commands to execute
  // substitute for variables and evaluate Boolean expression for "elif"
  // must substitute on copy of arg else will step on subsequent args
  // bound and execute "elif" or "else" commands

  while (iarg != narg) {
    if (iarg+2 > narg) error->all(FLERR,"Illegal if command");
    if (strcmp(arg[iarg],"elif") == 0) {
      n = strlen(arg[iarg+1]) + 1;
      if (n > maxline) reallocate(line,maxline,n);
      strcpy(line,arg[iarg+1]);
      substitute(line,work,maxline,maxwork,0);
      btest = variable->evaluate_boolean(line);
      first = iarg+2;
    } else {
      btest = 1.0;
      first = iarg+1;
    }

    iarg = first;
    while (iarg < narg &&
           (strcmp(arg[iarg],"elif") != 0 && strcmp(arg[iarg],"else") != 0))
      iarg++;
    last = iarg-1;

    if (btest == 0.0) continue;

    int ncommands = last-first + 1;
    if (ncommands <= 0) error->all(FLERR,"Illegal if command");

    char **commands = new char*[ncommands];
    ncommands = 0;
    for (int i = first; i <= last; i++) {
      n = strlen(arg[i]) + 1;
      if (n == 1) error->all(FLERR,"Illegal if command");
      commands[ncommands] = new char[n];
      strcpy(commands[ncommands],arg[i]);
      ncommands++;
    }

    // execute the list of commands

    for (int i = 0; i < ncommands; i++) {
      one(commands[i]);
      delete[] commands[i];
    }
    delete[] commands;

    return;
  }
}

/* ---------------------------------------------------------------------- */

void Input::include()
{
  if (narg != 1) error->all(FLERR,"Illegal include command");

  if (me == 0) {
    if (nfile == maxfile)
      error->one(FLERR,"Too many nested levels of input scripts");

    infile = fopen(arg[0],"r");
    if (infile == nullptr)
      error->one(FLERR,"Cannot open input script {}: {}",
                                   arg[0], utils::getsyserror());

    infiles[nfile++] = infile;
  }

  // process contents of file

  file();

  if (me == 0) {
    fclose(infile);
    nfile--;
    infile = infiles[nfile-1];
  }
}

/* ---------------------------------------------------------------------- */

void Input::jump()
{
  if (narg < 1 || narg > 2) error->all(FLERR,"Illegal jump command");

  if (jump_skip) {
    jump_skip = 0;
    return;
  }

  if (me == 0) {
    if (strcmp(arg[0],"SELF") == 0) rewind(infile);
    else {
      if (infile && infile != stdin) fclose(infile);
      infile = fopen(arg[0],"r");
      if (infile == nullptr)
        error->one(FLERR,"Cannot open input script {}: {}",
                                     arg[0], utils::getsyserror());

      infiles[nfile-1] = infile;
    }
  }

  if (narg == 2) {
    label_active = 1;
    if (labelstr) delete[] labelstr;
    labelstr = utils::strdup(arg[1]);
  }
}

/* ---------------------------------------------------------------------- */

void Input::label()
{
  if (narg != 1) error->all(FLERR,"Illegal label command");
  if (label_active && strcmp(labelstr,arg[0]) == 0) label_active = 0;
}

/* ---------------------------------------------------------------------- */

void Input::log()
{
  if ((narg < 1) || (narg > 2)) error->all(FLERR,"Illegal log command");

  int appendflag = 0;
  if (narg == 2) {
    if (strcmp(arg[1],"append") == 0) appendflag = 1;
    else error->all(FLERR,"Illegal log command");
  }

  if (me == 0) {
    if (logfile) fclose(logfile);
    if (strcmp(arg[0],"none") == 0) logfile = nullptr;
    else {
      if (appendflag) logfile = fopen(arg[0],"a");
      else logfile = fopen(arg[0],"w");

      if (logfile == nullptr)
        error->one(FLERR,"Cannot open logfile {}: {}",
                                     arg[0], utils::getsyserror());

    }
    if (universe->nworlds == 1) universe->ulogfile = logfile;
  }
}

/* ---------------------------------------------------------------------- */

void Input::next_command()
{
  if (variable->next(narg,arg)) jump_skip = 1;
}

/* ---------------------------------------------------------------------- */

void Input::partition()
{
  if (narg < 3) error->all(FLERR,"Illegal partition command");

  int yesflag = 0;
  if (strcmp(arg[0],"yes") == 0) yesflag = 1;
  else if (strcmp(arg[0],"no") == 0) yesflag = 0;
  else error->all(FLERR,"Illegal partition command");

  int ilo,ihi;
  utils::bounds(FLERR,arg[1],1,universe->nworlds,ilo,ihi,error);

  // new command starts at the 3rd argument,
  // which must not be another partition command

  if (strcmp(arg[2],"partition") == 0)
    error->all(FLERR,"Illegal partition command");

  char *cmd = strstr(line,arg[2]);

  // execute the remaining command line on requested partitions

  if (yesflag) {
    if (universe->iworld+1 >= ilo && universe->iworld+1 <= ihi) one(cmd);
  } else {
    if (universe->iworld+1 < ilo || universe->iworld+1 > ihi) one(cmd);
  }
}

/* ---------------------------------------------------------------------- */

void Input::print()
{
  if (narg < 1) error->all(FLERR,"Illegal print command");

  // copy 1st arg back into line (copy is being used)
  // check maxline since arg[0] could have been expanded by variables
  // substitute for $ variables (no printing) and print arg

  int n = strlen(arg[0]) + 1;
  if (n > maxline) reallocate(line,maxline,n);
  strcpy(line,arg[0]);
  substitute(line,work,maxline,maxwork,0);

  // parse optional args

  FILE *fp = nullptr;
  int screenflag = 1;
  int universeflag = 0;

  int iarg = 1;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"file") == 0 || strcmp(arg[iarg],"append") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal print command");
      if (me == 0) {
        if (fp != nullptr) fclose(fp);
        if (strcmp(arg[iarg],"file") == 0) fp = fopen(arg[iarg+1],"w");
        else fp = fopen(arg[iarg+1],"a");
        if (fp == nullptr)
          error->one(FLERR,"Cannot open print file {}: {}",
                                       arg[iarg+1], utils::getsyserror());
      }
      iarg += 2;
    } else if (strcmp(arg[iarg],"screen") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal print command");
      if (strcmp(arg[iarg+1],"yes") == 0) screenflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) screenflag = 0;
      else error->all(FLERR,"Illegal print command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"universe") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal print command");
      if (strcmp(arg[iarg+1],"yes") == 0) universeflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) universeflag = 0;
      else error->all(FLERR,"Illegal print command");
      iarg += 2;
    } else error->all(FLERR,"Illegal print command");
  }

  if (me == 0) {
    if (screenflag && screen) fprintf(screen,"%s\n",line);
    if (screenflag && logfile) fprintf(logfile,"%s\n",line);
    if (fp) {
      fprintf(fp,"%s\n",line);
      fclose(fp);
    }
  }
  if (universeflag && (universe->me == 0)) {
    if (universe->uscreen)  fprintf(universe->uscreen, "%s\n",line);
    if (universe->ulogfile) fprintf(universe->ulogfile,"%s\n",line);
  }
}

/* ---------------------------------------------------------------------- */

void Input::python()
{
  variable->python_command(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::quit()
{
  if (narg == 0) error->done(0); // 1 would be fully backwards compatible
  if (narg == 1) error->done(utils::inumeric(FLERR,arg[0],false,lmp));
  error->all(FLERR,"Illegal quit command");
}

/* ---------------------------------------------------------------------- */

char *shell_failed_message(const char* cmd, int errnum)
{
  std::string errmsg = fmt::format("Shell command '{}' failed with error '{}'",
                                   cmd, strerror(errnum));
  char *msg = new char[errmsg.size()+1];
  strcpy(msg, errmsg.c_str());
  return msg;
}

void Input::shell()
{
  int rv,err;

  if (narg < 1) error->all(FLERR,"Illegal shell command");

  if (strcmp(arg[0],"cd") == 0) {
    if (narg != 2) error->all(FLERR,"Illegal shell cd command");
    rv = (chdir(arg[1]) < 0) ? errno : 0;
    MPI_Reduce(&rv,&err,1,MPI_INT,MPI_MAX,0,world);
    if (me == 0 && err != 0) {
      char *message = shell_failed_message("cd",err);
      error->warning(FLERR,message);
      delete[] message;
    }

  } else if (strcmp(arg[0],"mkdir") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal shell mkdir command");
    if (me == 0)
      for (int i = 1; i < narg; i++) {
#if defined(_WIN32)
        rv = _mkdir(arg[i]);
#else
        rv = mkdir(arg[i], S_IRWXU | S_IRGRP | S_IXGRP);
#endif
        if (rv < 0) {
          char *message = shell_failed_message("mkdir",errno);
          error->warning(FLERR,message);
          delete[] message;
        }
      }

  } else if (strcmp(arg[0],"mv") == 0) {
    if (narg != 3) error->all(FLERR,"Illegal shell mv command");
    rv = (rename(arg[1],arg[2]) < 0) ? errno : 0;
    MPI_Reduce(&rv,&err,1,MPI_INT,MPI_MAX,0,world);
    if (me == 0 && err != 0) {
      char *message = shell_failed_message("mv",err);
      error->warning(FLERR,message);
      delete[] message;
    }

  } else if (strcmp(arg[0],"rm") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal shell rm command");
    if (me == 0)
      for (int i = 1; i < narg; i++) {
        if (unlink(arg[i]) < 0) {
          char *message = shell_failed_message("rm",errno);
          error->warning(FLERR,message);
          delete[] message;
        }
      }

  } else if (strcmp(arg[0],"rmdir") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal shell rmdir command");
    if (me == 0)
      for (int i = 1; i < narg; i++) {
        if (rmdir(arg[i]) < 0) {
          char *message = shell_failed_message("rmdir",errno);
          error->warning(FLERR,message);
          delete[] message;
        }
      }

  } else if (strcmp(arg[0],"putenv") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal shell putenv command");
    for (int i = 1; i < narg; i++) {
      rv = 0;
#ifdef _WIN32
      if (arg[i]) rv = _putenv(utils::strdup(arg[i]));
#else
      if (arg[i]) {
        std::string vardef(arg[i]);
        auto found = vardef.find_first_of('=');
        if (found == std::string::npos) {
          rv = setenv(vardef.c_str(),"",1);
        } else {
          rv = setenv(vardef.substr(0,found).c_str(),
                      vardef.substr(found+1).c_str(),1);
        }
      }
#endif
      rv = (rv < 0) ? errno : 0;
      MPI_Reduce(&rv,&err,1,MPI_INT,MPI_MAX,0,world);
      if (me == 0 && err != 0) {
        char *message = shell_failed_message("putenv",err);
        error->warning(FLERR,message);
        delete[] message;
      }
    }

  // use work string to concat args back into one string separated by spaces
  // invoke string in shell via system()

  } else {
    int n = 0;
    for (int i = 0; i < narg; i++) n += strlen(arg[i]) + 1;
    if (n > maxwork) reallocate(work,maxwork,n);

    strcpy(work,arg[0]);
    for (int i = 1; i < narg; i++) {
      strcat(work," ");
      strcat(work,arg[i]);
    }

    if (me == 0)
      if (system(work) != 0)
        error->warning(FLERR,"Shell command returned with non-zero status");
  }
}

/* ---------------------------------------------------------------------- */

void Input::variable_command()
{
  variable->set(narg,arg);
}

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   one function for each LAMMPS-specific input script command
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void Input::angle_coeff()
{
  if (domain->box_exist == 0)
    error->all(FLERR,"Angle_coeff command before simulation box is defined");
  if (force->angle == nullptr)
    error->all(FLERR,"Angle_coeff command before angle_style is defined");
  if (atom->avec->angles_allow == 0)
    error->all(FLERR,"Angle_coeff command when no angles allowed");
  force->angle->coeff(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::angle_style()
{
  if (narg < 1) error->all(FLERR,"Illegal angle_style command");
  if (atom->avec->angles_allow == 0)
    error->all(FLERR,"Angle_style command when no angles allowed");
  force->create_angle(arg[0],1);
  if (force->angle) force->angle->settings(narg-1,&arg[1]);
}

/* ---------------------------------------------------------------------- */

void Input::atom_modify()
{
  atom->modify_params(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::atom_style()
{
  if (narg < 1) error->all(FLERR,"Illegal atom_style command");
  if (domain->box_exist)
    error->all(FLERR,"Atom_style command after simulation box is defined");
  atom->create_avec(arg[0],narg-1,&arg[1],1);
}

/* ---------------------------------------------------------------------- */

void Input::bond_coeff()
{
  if (domain->box_exist == 0)
    error->all(FLERR,"Bond_coeff command before simulation box is defined");
  if (force->bond == nullptr)
    error->all(FLERR,"Bond_coeff command before bond_style is defined");
  if (atom->avec->bonds_allow == 0)
    error->all(FLERR,"Bond_coeff command when no bonds allowed");
  force->bond->coeff(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::bond_style()
{
  if (narg < 1) error->all(FLERR,"Illegal bond_style command");
  if (atom->avec->bonds_allow == 0)
    error->all(FLERR,"Bond_style command when no bonds allowed");
  force->create_bond(arg[0],1);
  if (force->bond) force->bond->settings(narg-1,&arg[1]);
}

/* ---------------------------------------------------------------------- */

void Input::bond_write()
{
  if (atom->avec->bonds_allow == 0)
    error->all(FLERR,"Bond_write command when no bonds allowed");
  if (force->bond == nullptr)
    error->all(FLERR,"Bond_write command before bond_style is defined");
  else force->bond->write_file(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::boundary()
{
  if (domain->box_exist)
    error->all(FLERR,"Boundary command after simulation box is defined");
  domain->set_boundary(narg,arg,0);
}

/* ---------------------------------------------------------------------- */

void Input::box()
{
  if (domain->box_exist)
    error->all(FLERR,"Box command after simulation box is defined");
  domain->set_box(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::comm_modify()
{
  comm->modify_params(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::comm_style()
{
  if (narg < 1) error->all(FLERR,"Illegal comm_style command");
  if (strcmp(arg[0],"brick") == 0) {
    if (comm->style == 0) return;
    Comm *oldcomm = comm;
    comm = new CommBrick(lmp,oldcomm);
    delete oldcomm;
  } else if (strcmp(arg[0],"tiled") == 0) {
    if (comm->style == 1) return;
    Comm *oldcomm = comm;

    if (lmp->kokkos) comm = new CommTiledKokkos(lmp,oldcomm);
    else comm = new CommTiled(lmp,oldcomm);

    delete oldcomm;
  } else error->all(FLERR,"Illegal comm_style command");
}

/* ---------------------------------------------------------------------- */

void Input::compute()
{
  modify->add_compute(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::compute_modify()
{
  modify->modify_compute(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::dielectric()
{
  if (narg != 1) error->all(FLERR,"Illegal dielectric command");
  force->dielectric = utils::numeric(FLERR,arg[0],false,lmp);
}

/* ---------------------------------------------------------------------- */

void Input::dihedral_coeff()
{
  if (domain->box_exist == 0)
    error->all(FLERR,"Dihedral_coeff command before simulation box is defined");
  if (force->dihedral == nullptr)
    error->all(FLERR,"Dihedral_coeff command before dihedral_style is defined");
  if (atom->avec->dihedrals_allow == 0)
    error->all(FLERR,"Dihedral_coeff command when no dihedrals allowed");
  force->dihedral->coeff(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::dihedral_style()
{
  if (narg < 1) error->all(FLERR,"Illegal dihedral_style command");
  if (atom->avec->dihedrals_allow == 0)
    error->all(FLERR,"Dihedral_style command when no dihedrals allowed");
  force->create_dihedral(arg[0],1);
  if (force->dihedral) force->dihedral->settings(narg-1,&arg[1]);
}

/* ---------------------------------------------------------------------- */

void Input::dimension()
{
  if (narg != 1) error->all(FLERR,"Illegal dimension command");
  if (domain->box_exist)
    error->all(FLERR,"Dimension command after simulation box is defined");
  domain->dimension = utils::inumeric(FLERR,arg[0],false,lmp);
  if (domain->dimension != 2 && domain->dimension != 3)
    error->all(FLERR,"Illegal dimension command");

  // must reset default extra_dof of all computes
  // since some were created before dimension command is encountered

  for (int i = 0; i < modify->ncompute; i++)
    modify->compute[i]->reset_extra_dof();
}

/* ---------------------------------------------------------------------- */

void Input::dump()
{
  output->add_dump(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::dump_modify()
{
  output->modify_dump(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::fix()
{
  modify->add_fix(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::fix_modify()
{
  modify->modify_fix(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::group_command()
{
  group->assign(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::improper_coeff()
{
  if (domain->box_exist == 0)
    error->all(FLERR,"Improper_coeff command before simulation box is defined");
  if (force->improper == nullptr)
    error->all(FLERR,"Improper_coeff command before improper_style is defined");
  if (atom->avec->impropers_allow == 0)
    error->all(FLERR,"Improper_coeff command when no impropers allowed");
  force->improper->coeff(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::improper_style()
{
  if (narg < 1) error->all(FLERR,"Illegal improper_style command");
  if (atom->avec->impropers_allow == 0)
    error->all(FLERR,"Improper_style command when no impropers allowed");
  force->create_improper(arg[0],1);
  if (force->improper) force->improper->settings(narg-1,&arg[1]);
}

/* ---------------------------------------------------------------------- */

void Input::kspace_modify()
{
  if (force->kspace == nullptr)
    error->all(FLERR,"KSpace style has not yet been set");
  force->kspace->modify_params(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::kspace_style()
{
  force->create_kspace(arg[0],1);
  if (force->kspace) force->kspace->settings(narg-1,&arg[1]);
}

/* ---------------------------------------------------------------------- */

void Input::lattice()
{
  domain->set_lattice(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::mass()
{
  if (narg != 2) error->all(FLERR,"Illegal mass command");
  if (domain->box_exist == 0)
    error->all(FLERR,"Mass command before simulation box is defined");
  atom->set_mass(FLERR,narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::min_modify()
{
  update->minimize->modify_params(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::min_style()
{
  if (domain->box_exist == 0)
    error->all(FLERR,"Min_style command before simulation box is defined");
  update->create_minimize(narg,arg,1);
}

/* ---------------------------------------------------------------------- */

void Input::molecule()
{
  atom->add_molecule(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::neigh_modify()
{
  neighbor->modify_params(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::neighbor_command()
{
  neighbor->set(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::newton()
{
  int newton_pair=1,newton_bond=1;

  if (narg == 1) {
    if (strcmp(arg[0],"off") == 0) newton_pair = newton_bond = 0;
    else if (strcmp(arg[0],"on") == 0) newton_pair = newton_bond = 1;
    else error->all(FLERR,"Illegal newton command");
  } else if (narg == 2) {
    if (strcmp(arg[0],"off") == 0) newton_pair = 0;
    else if (strcmp(arg[0],"on") == 0) newton_pair= 1;
    else error->all(FLERR,"Illegal newton command");
    if (strcmp(arg[1],"off") == 0) newton_bond = 0;
    else if (strcmp(arg[1],"on") == 0) newton_bond = 1;
    else error->all(FLERR,"Illegal newton command");
  } else error->all(FLERR,"Illegal newton command");

  force->newton_pair = newton_pair;

  if (domain->box_exist && (newton_bond != force->newton_bond))
    error->all(FLERR,"Newton bond change after simulation box is defined");
  force->newton_bond = newton_bond;

  if (newton_pair || newton_bond) force->newton = 1;
  else force->newton = 0;
}

/* ---------------------------------------------------------------------- */

void Input::package()
{
  if (domain->box_exist)
    error->all(FLERR,"Package command after simulation box is defined");
  if (narg < 1) error->all(FLERR,"Illegal package command");

  // same checks for packages existing as in LAMMPS::post_create()
  // since can be invoked here by package command in input script

  if (strcmp(arg[0],"gpu") == 0) {
    if (!modify->check_package("GPU"))
      error->all(FLERR,"Package gpu command without GPU package installed");

    std::string fixcmd = "package_gpu all GPU";
    for (int i = 1; i < narg; i++) fixcmd += std::string(" ") + arg[i];
    modify->add_fix(fixcmd);

  } else if (strcmp(arg[0],"kokkos") == 0) {
    if (lmp->kokkos == nullptr || lmp->kokkos->kokkos_exists == 0)
      error->all(FLERR,
                 "Package kokkos command without KOKKOS package enabled");
    lmp->kokkos->accelerator(narg-1,&arg[1]);

  } else if (strcmp(arg[0],"omp") == 0) {
    if (!modify->check_package("OMP"))
      error->all(FLERR,
                 "Package omp command without OPENMP package installed");

    std::string fixcmd = "package_omp all OMP";
    for (int i = 1; i < narg; i++) fixcmd += std::string(" ") + arg[i];
    modify->add_fix(fixcmd);

 } else if (strcmp(arg[0],"intel") == 0) {
    if (!modify->check_package("INTEL"))
      error->all(FLERR,
                 "Package intel command without INTEL package installed");

    std::string fixcmd = "package_intel all INTEL";
    for (int i = 1; i < narg; i++) fixcmd += std::string(" ") + arg[i];
    modify->add_fix(fixcmd);

  } else error->all(FLERR,"Illegal package command");
}

/* ---------------------------------------------------------------------- */

void Input::pair_coeff()
{
  if (domain->box_exist == 0)
    error->all(FLERR,"Pair_coeff command before simulation box is defined");
  if (force->pair == nullptr)
    error->all(FLERR,"Pair_coeff command before pair_style is defined");
  if ((narg < 2) || (force->pair->one_coeff && ((strcmp(arg[0],"*") != 0)
                                               || (strcmp(arg[1],"*") != 0))))
    error->all(FLERR,"Incorrect args for pair coefficients");
  force->pair->coeff(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::pair_modify()
{
  if (force->pair == nullptr)
    error->all(FLERR,"Pair_modify command before pair_style is defined");
  force->pair->modify_params(narg,arg);
}

/* ----------------------------------------------------------------------
   if old pair style exists and new style is same, just change settings
   else create new pair class
------------------------------------------------------------------------- */

void Input::pair_style()
{
  if (narg < 1) error->all(FLERR,"Illegal pair_style command");
  if (force->pair) {
    std::string style = arg[0];
    int match = 0;
    if (style == force->pair_style) match = 1;
    if (!match && lmp->suffix_enable) {
      if (lmp->suffixp)
        if (style + "/" + lmp->suffixp == force->pair_style) match = 1;

      if (lmp->suffix && !lmp->suffixp)
        if (style + "/" + lmp->suffix == force->pair_style) match = 1;

      if (lmp->suffix2)
        if (style + "/" + lmp->suffix2 == force->pair_style) match = 1;
    }
    if (match) {
      force->pair->settings(narg-1,&arg[1]);
      return;
    }
  }

  force->create_pair(arg[0],1);
  if (force->pair) force->pair->settings(narg-1,&arg[1]);
}

/* ---------------------------------------------------------------------- */

void Input::pair_write()
{
  if (force->pair == nullptr)
    error->all(FLERR,"Pair_write command before pair_style is defined");
  force->pair->write_file(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::processors()
{
  if (domain->box_exist)
    error->all(FLERR,"Processors command after simulation box is defined");
  comm->set_processors(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::region()
{
  domain->add_region(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::reset_timestep()
{
  update->reset_timestep(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::restart()
{
  output->create_restart(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::run_style()
{
  if (domain->box_exist == 0)
    error->all(FLERR,"Run_style command before simulation box is defined");
  update->create_integrate(narg,arg,1);
}

/* ---------------------------------------------------------------------- */

void Input::special_bonds()
{
  // store 1-3,1-4 and dihedral/extra flag values before change
  // change in 1-2 coeffs will not change the special list

  double lj2 = force->special_lj[2];
  double lj3 = force->special_lj[3];
  double coul2 = force->special_coul[2];
  double coul3 = force->special_coul[3];
  int angle = force->special_angle;
  int dihedral = force->special_dihedral;

  force->set_special(narg,arg);

  // if simulation box defined and saved values changed, redo special list

  if (domain->box_exist && atom->molecular == Atom::MOLECULAR) {
    if (lj2 != force->special_lj[2] || lj3 != force->special_lj[3] ||
        coul2 != force->special_coul[2] || coul3 != force->special_coul[3] ||
        angle != force->special_angle ||
        dihedral != force->special_dihedral) {
      Special special(lmp);
      special.build();
    }
  }
}

/* ---------------------------------------------------------------------- */

void Input::suffix()
{
  if (narg < 1) error->all(FLERR,"Illegal suffix command");

  if (strcmp(arg[0],"off") == 0) lmp->suffix_enable = 0;
  else if (strcmp(arg[0],"on") == 0) {
    if (!lmp->suffix)
      error->all(FLERR,"May only enable suffixes after defining one");
    lmp->suffix_enable = 1;
  } else {
    lmp->suffix_enable = 1;

    delete[] lmp->suffix;
    delete[] lmp->suffix2;
    lmp->suffix = lmp->suffix2 = nullptr;

    if (strcmp(arg[0],"hybrid") == 0) {
      if (narg != 3) error->all(FLERR,"Illegal suffix command");
      lmp->suffix = utils::strdup(arg[1]);
      lmp->suffix2 = utils::strdup(arg[2]);
    } else {
      if (narg != 1) error->all(FLERR,"Illegal suffix command");
      lmp->suffix = utils::strdup(arg[0]);
    }
  }
}

/* ---------------------------------------------------------------------- */

void Input::thermo()
{
  output->set_thermo(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::thermo_modify()
{
  output->thermo->modify_params(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::thermo_style()
{
  output->create_thermo(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::timer_command()
{
  timer->modify_params(narg,arg);
}

/* ---------------------------------------------------------------------- */

void Input::timestep()
{
  if (narg != 1) error->all(FLERR,"Illegal timestep command");
  update->dt = utils::numeric(FLERR,arg[0],false,lmp);
  update->dt_default = 0;
}

/* ---------------------------------------------------------------------- */

void Input::uncompute()
{
  if (narg != 1) error->all(FLERR,"Illegal uncompute command");
  modify->delete_compute(arg[0]);
}

/* ---------------------------------------------------------------------- */

void Input::undump()
{
  if (narg != 1) error->all(FLERR,"Illegal undump command");
  output->delete_dump(arg[0]);
}

/* ---------------------------------------------------------------------- */

void Input::unfix()
{
  if (narg != 1) error->all(FLERR,"Illegal unfix command");
  modify->delete_fix(arg[0]);
}

/* ---------------------------------------------------------------------- */

void Input::units()
{
  if (narg != 1) error->all(FLERR,"Illegal units command");
  if (domain->box_exist)
    error->all(FLERR,"Units command after simulation box is defined");
  update->set_units(arg[0]);
}
