/*
*   This routine reads the data from a .frc forcefield file and stores it in
*   dynamically allocated memory. This allows for fast searches of the
*   file.
*
*/

#include "msi2lmp.h"
#include "Forcefield.h"

#include <stdlib.h>
#include <string.h>

struct FrcFieldItem ff_atomtypes, equivalence, ff_vdw, ff_bond, ff_morse, ff_ang, ff_tor, ff_oop,
  ff_bonbon, ff_bonang, ff_angtor, ff_angangtor, ff_endbontor, ff_midbontor, ff_angang, ff_bonbon13;


void ClearFrcData(void)
{
  ClearFrcItem(&ff_atomtypes);
  ClearFrcItem(&equivalence);
  ClearFrcItem(&ff_vdw);
  ClearFrcItem(&ff_bond);
  if (forcefield & FF_TYPE_CLASS1) {  /* Morse bond terms for class I */
      ClearFrcItem(&ff_morse);
  }
  ClearFrcItem(&ff_ang);
  ClearFrcItem(&ff_tor);
  ClearFrcItem(&ff_oop);

  if (forcefield & FF_TYPE_CLASS2) {  /* Cross terms for class II */
    ClearFrcItem(&ff_bonbon);
    ClearFrcItem(&ff_bonang);
    ClearFrcItem(&ff_angtor);
    ClearFrcItem(&ff_angangtor);
    ClearFrcItem(&ff_endbontor);
    ClearFrcItem(&ff_midbontor);
    ClearFrcItem(&ff_bonbon13);
    ClearFrcItem(&ff_angang);
  }
}

void ReadFrcFile(void)
{
  const char *val;

  /* Open Forcefield File */
  if ( (FrcF = fopen(FrcFileName,"r")) == NULL ) {
    fprintf(stderr,"Cannot open %s\n", FrcFileName);
    exit(72);
  }
  InitializeItems(); /* sets keywords, number of members and number of
                        parameters for each structure */
  /* allocate memory to and search and fill each structure */

  val = SearchAndCheck("type");

  if (forcefield & (FF_TYPE_CLASS1|FF_TYPE_OPLSAA)) {
    if (strcmp(val,"A-B") == 0) {
      ljtypeflag = 0;
    } else {
      fprintf(stderr," Inconsistent data in force field file.\n"
              " Expected: 'A-B' type non-bonded data, found: %s\n",val);
      exit(73);
    }
  } else if (forcefield & FF_TYPE_CLASS2) {
    if (strcmp(val,"r-eps") == 0) {
      ljtypeflag = 1;
    } else {
      fprintf(stderr,"Inconsistent parameter file.\n"
              " Expected: 'r-eps' type non-bonded data, found: %s\n",val);
      exit(74);
    }
  }
  free((void *)val);

  SearchAndFill(&ff_atomtypes);
  SearchAndFill(&equivalence);
  SearchAndFill(&ff_vdw);
  SearchAndFill(&ff_bond);
  if (forcefield & FF_TYPE_CLASS1) {  /* Morse bond terms for class I */
    SearchAndFill(&ff_morse);
  }
  SearchAndFill(&ff_ang);
  SearchAndFill(&ff_tor);
  SearchAndFill(&ff_oop);

  if (forcefield & FF_TYPE_CLASS2) {  /* Cross terms for class II */
    SearchAndFill(&ff_bonbon);
    SearchAndFill(&ff_bonang);
    SearchAndFill(&ff_angtor);
    SearchAndFill(&ff_angangtor);
    SearchAndFill(&ff_endbontor);
    SearchAndFill(&ff_midbontor);
    SearchAndFill(&ff_bonbon13);
    SearchAndFill(&ff_angang);
  }
  if (pflag > 1) {

    fprintf(stderr,"\n Item %s has %d entries\n",
            ff_atomtypes.keyword,ff_atomtypes.entries);
    fprintf(stderr," Item %s has %d entries\n",
            equivalence.keyword,equivalence.entries);
    fprintf(stderr," Item %s has %d entries\n",
            ff_vdw.keyword,ff_vdw.entries);
    fprintf(stderr," Item %s has %d entries\n",
            ff_bond.keyword,ff_bond.entries);
    if (forcefield & FF_TYPE_CLASS1)
      fprintf(stderr," Item %s has %d entries\n",
              ff_morse.keyword,ff_morse.entries);
    fprintf(stderr," Item %s has %d entries\n",
            ff_ang.keyword,ff_ang.entries);
    if (forcefield & FF_TYPE_CLASS2) {
      fprintf(stderr," Item %s has %d entries\n",
              ff_bonbon.keyword,ff_bonbon.entries);
      fprintf(stderr," Item %s has %d entries\n",
              ff_bonang.keyword,ff_bonang.entries);
    }
    fprintf(stderr," Item %s has %d entries\n",
            ff_tor.keyword,ff_tor.entries);
    if (forcefield & FF_TYPE_CLASS2) {
      fprintf(stderr," Item %s has %d entries\n",
              ff_angtor.keyword,ff_angtor.entries);
      fprintf(stderr," Item %s has %d entries\n",
              ff_angangtor.keyword,ff_angangtor.entries);
      fprintf(stderr," Item %s has %d entries\n",
              ff_endbontor.keyword,ff_endbontor.entries);
      fprintf(stderr," Item %s has %d entries\n",
              ff_midbontor.keyword,ff_midbontor.entries);
      fprintf(stderr," Item %s has %d entries\n",
              ff_bonbon13.keyword,ff_bonbon13.entries);
    }
    fprintf(stderr," Item %s has %d entries\n",
            ff_oop.keyword,ff_oop.entries);
    if (forcefield & FF_TYPE_CLASS2) {
      fprintf(stderr," Item %s has %d entries\n",
              ff_angang.keyword,ff_angang.entries);
    }
    fprintf(stderr,"\n");
  }
  fclose(FrcF);
}

