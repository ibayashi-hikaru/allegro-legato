# Install/unInstall package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update

mode=$1

# enforce using portable C locale
LC_ALL=C
export LC_ALL

# arg1 = file, arg2 = file it depends on

action () {
  if (test $mode = 0) then
    rm -f ../$1
  elif (! cmp -s $1 ../$1) then
    if (test -z "$2" || test -e ../$2) then
      cp $1 ..
      if (test $mode = 2) then
        echo "  updating src/$1"
      fi
    fi
  elif (test -n "$2") then
    if (test ! -e ../$2) then
      rm -f ../$1
    fi
  fi
}

# all package files with no dependencies

for file in *.cpp *.h; do
  test -f ${file} && action $file
done

# edit 2 Makefile.package files to include/exclude package info

if (test $1 = 1) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*qmmm[^ \t]* //g' ../Makefile.package
    sed -i -e 's|^PKG_INC =[ \t]*|&-I..\/..\/lib\/qmmm |' ../Makefile.package
    sed -i -e 's|^PKG_PATH =[ \t]*|&-L..\/..\/lib\/qmmm$(LIBOBJDIR) |' ../Makefile.package
    sed -i -e 's|^PKG_LIB =[ \t]*|&-lqmmm |' ../Makefile.package
    sed -i -e 's|^PKG_SYSINC =[ \t]*|&$(qmmm_SYSINC) |' ../Makefile.package
    sed -i -e 's|^PKG_SYSLIB =[ \t]*|&$(qmmm_SYSLIB) |' ../Makefile.package
    sed -i -e 's|^PKG_SYSPATH =[ \t]*|&$(qmmm_SYSPATH) |' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/^include.*qmmm.*$/d' ../Makefile.package.settings
    # multiline form needed for BSD sed on Macs
    sed -i -e '4 i \
include ..\/..\/lib\/qmmm\/Makefile.lammps
' ../Makefile.package.settings

  fi

elif (test $1 = 0) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]*qmmm[^ \t]* //g' ../Makefile.package
  fi

  if (test -e ../Makefile.package.settings) then
    sed -i -e '/^include.*qmmm.*$/d' ../Makefile.package.settings
  fi

fi
