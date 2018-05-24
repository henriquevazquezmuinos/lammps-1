/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   Electronic stopping power fix authors: K.Avchachov and T.Metspalu
   Created in Feb 2015. More information: konstantin.avchachov@helsinki.fi
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(elstop,FixElstop)

#else

#ifndef LMP_FIX_ELSTOP_H
#define LMP_FIX_ELSTOP_H

#include "fix.h"
#include "neighbor.h"
#include "compute.h"


namespace LAMMPS_NS {

class FixElstop : public Fix {
 public:
  FixElstop(class LAMMPS *, int, char **);
  ~FixElstop();
  int setmask();
  void init();
  void post_force(int);
  void init_list(int, class NeighList *);
  void read_table(char *);
  double compute_scalar();

 private:
  double Ecut;	// E_stop cutoff energy
  //double ftm2v;  // conversion of ft/m to velocity
  //double dt; // timestep
  double SeLoss, SeLoss_all; // Electronic energy loss
  int eflag;
  int table_entries;
  double **elstop_ranges; 
  int respaflag;
  int regionflag;
  int iregion;
  char *idregion;
  char *file_name;
  char *groupname; 
  class NeighList *list;
  class Compute *c_ke;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

W: Dump dcd/xtc timestamp may be wrong with fix dt/reset

If the fix changes the timestep, the dump dcd file will not
reflect the change.

*/