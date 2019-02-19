/* ----------------------------------------------------------------------
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
   Electronic stopping power contributing authors: K.Avchachov and T.Metspalu
   Updated: 18th of May 2015. Information: konstantin.avchachov@helsinki.fi
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "fix_elstop.h"
#include "mpi.h"
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "region.h"
#include "group.h"
#include "force.h"
#include "pair.h"
#include "fix.h"
#include "compute.h"
#include "modify.h"
#include "memory.h"
#include "comm.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG 1.0e20
#define MAXLINE 1024
#define MAXTABLEN 300
#define MINNEIGHB 5

/* ---------------------------------------------------------------------- */

FixElstop::FixElstop(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal fix elstop command");

  // set time_depend, else elapsed time accumulation can be messed up
  time_depend = 1;

  scalar_flag = 1;  // Has compute_scalar
  global_freq = 1;  // SeLoss computed every step
  extscalar = 0;    // SeLoss is intensive???
  nevery = 1;       // Run fix every step

  regionflag = 0;
  iregion = -1;

  int iarg = 0;

  // args: 0 = fix ID, 1 = group ID,  2 = "elstop"
  //       3 = Ecut,   4 = file path
  // optional rest: "region" region name

  // reading in the numerical parameters from fix ID group-ID elstop N Ecut:
  Ecut = force->numeric(FLERR, arg[3]);
  if (Ecut <= 0.0) error->all(FLERR,"Illegal fix elstop command: cutoff energy cannot be 0 or negative!");

  int n = strlen(arg[4]) + 1;
  file_name = new char[n];
  strcpy(file_name, arg[4]);

  iarg = 5;
  // reading in the regions
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix elstop command: region name missing!");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"region ID for fix elstop does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      regionflag = 1;
      iarg += 2;
    }
    else iarg++;
  }
  //fprintf(screen, "Debugging >> %s %s %i\n", file_name, arg[4], narg);
}

/* ---------------------------------------------------------------------- */

FixElstop::~FixElstop()
{
  list=NULL;
  memory->destroy(elstop_ranges);
  modify->delete_compute("ke_atom");
}

/* ---------------------------------------------------------------------- */

int FixElstop::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixElstop::init()
{
  eflag = 0;
  SeLoss=0.0;
  const int nt = atom->ntypes + 1;
  // set rRESPA (Reversible reference system propagation algorithm) flag
  respaflag = 0;
  if (strstr(update->integrate_style,"respa")) respaflag = 1;

  int ikeatom = modify->find_compute("ke_atom");
  if (ikeatom < 0 ){
    char **newarg = new char*[3];
    newarg[0] = (char *) "ke_atom";
    newarg[1] = group->names[igroup];
    newarg[2] = (char *) "ke/atom";
    modify->add_compute(3,newarg);
    ikeatom = modify->find_compute("ke_atom");
    delete [] newarg;
  }

  c_ke = modify->compute[ikeatom];


  memory->create(elstop_ranges,nt, MAXTABLEN, "elstop:tabs");
  memset(&elstop_ranges[0][0],0,nt*MAXTABLEN*sizeof(double));

  if (comm->me == 0){
    read_table(file_name);
  }

  MPI_Bcast(&table_entries, 1 , MPI_INT, 0, world);
  MPI_Bcast(&elstop_ranges[0][0],nt*MAXTABLEN,MPI_DOUBLE,0,world);

  // need an occasional full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

}

/* ---------------------------------------------------------------------- */

void FixElstop::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixElstop::post_force(int vflag)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;

  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;	// number of atoms in the group
  int *numneigh;
  int i;
  double ENERGY;

  eflag = 0;
  c_ke->invoked_peratom = 1;
  c_ke->compute_peratom();
  double *ke = c_ke->vector_atom;
  double dt = update->dt;
  neighbor->build_one(list);
  numneigh = list->numneigh;

  for (int i = 0; i < nlocal; ++i){


    if (regionflag) {
      if (domain->regions[iregion]->match(x[i][0],x[i][1],x[i][2]) != 1) continue;  // Start stopping when the group has entered a predefined region
    }

    if (numneigh[i]> MINNEIGHB ) {  // Avoiding dimers, trimers and even tetramers in case of really high energies
      if (mask[i] & groupbit) {
        int iup=0, idown=1, ihalf=0;
        int itype = type[i];
        ENERGY=ke[i];
        fprintf(screen, "HERE %d %d %g %g %g %d\n", i, type[i], ke[i], ENERGY, elstop_ranges[0][table_entries-1], table_entries);

        // for (int n = 0; n < 3; n++)
        //fprintf(screen, "Debugging >> speed in %i-direction: %4.5f | force: %4.5f | energy: %4.5f | timestep length: %2.7f \n ", n, v[i][n], f[i][n], ENERGY, dt);

        /* ---------------------------------------------------------------------- */

        if (ENERGY >= Ecut && ENERGY >= elstop_ranges[0][0] && ENERGY <= elstop_ranges[0][table_entries-1] ) {
          double Se, Se_lo, Se_hi, E_lo, E_hi;
          iup=table_entries-1; idown=0;
          while (true) {
            ihalf=idown+(iup-idown)/2;
            if (ihalf==idown) break;
            if (elstop_ranges[0][ihalf] < ENERGY) idown=ihalf;
            else iup=ihalf;
          }
          Se_lo = elstop_ranges[itype][idown];
          Se_hi = elstop_ranges[itype][iup];
          E_lo = elstop_ranges[0][idown];
          E_hi = elstop_ranges[0][iup];

          /* Get elstop with a simple linear interpolation */
          Se=(Se_hi-Se_lo)/(E_hi-E_lo)*(ENERGY-E_lo)+Se_lo;

          double v2 = 0.0;
          for (int n=0 ; n<3 ; ++n ) {
            v2+=v[i][n]*v[i][n];
          }

          //fprintf(screen, "Debugging >> %lf %lf %lf %i %i\n", Se, ENERGY, dt, idown,  iup);
          for (int n=0 ; n<3 ; ++n ) {
            f[i][n] = f[i][n]-v[i][n]/sqrt(v2)*Se;
          }
          SeLoss+=Se*sqrt(v2)*dt; //very rough approx
        }

        /* ---------------------------------------------------------------------- */

      }
    }
  }
}

void FixElstop::read_table(char *file)
{
  char line[MAXLINE];

  fprintf(screen, "Reading file %s\n", file);

  FILE *fp = force->open_potential(file);
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open stopping range table %s",file);
    error->one(FLERR,str);
  }

  const int nt = atom->ntypes + 1;

  // Why skip the first line???
  //if (fgets(line,MAXLINE,fp) == NULL)
  //    error->one(FLERR,"Did not find any data in table file");

  int l = 0;
  while (l < MAXTABLEN) {
    if (fgets(line, MAXLINE, fp) == NULL) break; // no match, skip section
    if (line[0] == '#') continue; // comment
    if (strspn(line," \t\n\r") == strlen(line)) continue;  // blank line

    // fprintf(screen, "line %d: %s\n", l, line);
    char *pch = strtok (line," \t\n\r");
    for (int i = 0; i < nt; i++){
      fprintf(screen, "line %d word %d: '%s' = %g\n", l, i, pch, atof(pch));
      elstop_ranges[i][l] = atof(pch);
      pch = strtok (NULL, " \t\n\r");
    }
    l++;
  }
  table_entries=l;

  if (table_entries == 0) error->one(FLERR,"Did not find any data in table file");

  if (fgets(line,MAXLINE,fp) != NULL){
    fprintf(screen, "Warning: Only %d entries have been read from the elstop table\n"
        "Please increase MAXTABLEN=%d value and recompile\n",
        MAXTABLEN, table_entries);
  }

  fclose(fp);
}


double FixElstop::compute_scalar()
{
  // only sum across procs one time

  if (eflag == 0) {
    MPI_Allreduce(&SeLoss,&SeLoss_all,1,MPI_DOUBLE,MPI_SUM,world);
    eflag = 1;
  }
  return SeLoss_all;
}
