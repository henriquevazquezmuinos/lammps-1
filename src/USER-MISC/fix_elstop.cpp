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

#define DEFAULT_MAXLINES 300
#define MAXLINE 1024
#define MINNEIGHB 5

/* ---------------------------------------------------------------------- */

FixElstop::FixElstop(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  scalar_flag = 1;  // Has compute_scalar
  global_freq = 1;  // SeLoss computed every step
  extscalar = 0;    // SeLoss compute_scalar is intensive
  nevery = 1;       // Run fix every step

  // args: 0 = fix ID, 1 = group ID,  2 = "elstop"
  //       3 = Ecut,   4 = file path
  // optional rest: "region" <region name>
  //                "maxlines" <max number of lines in table>

  if (narg < 5) error->all(FLERR, "Illegal fix elstop command: too few arguments");

  Ecut = force->numeric(FLERR, arg[3]);
  if (Ecut <= 0.0) error->all(FLERR, "Illegal fix elstop command: cutoff energy cannot be 0 or negative");

  int n = strlen(arg[4]) + 1;
  file_name = new char[n];
  strcpy(file_name, arg[4]);

  int iarg = 5;
  iregion = -1;
  maxlines = 0;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "region") == 0) {
      if (iregion >= 0) error->all(FLERR, "Illegal fix elstop command: region given twice");
      if (iarg+2 > narg) error->all(FLERR, "Illegal fix elstop command: region name missing");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion < 0)
        error->all(FLERR, "region ID for fix elstop does not exist");
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "maxlines") == 0) {
      if (maxlines > 0) error->all(FLERR, "Illegal fix elstop command: maxlines given twice");
      if (iarg+2 > narg) error->all(FLERR, "Illegal fix elstop command: maxlines value missing");
      maxlines = force->numeric(FLERR, arg[iarg+1]);
      if (maxlines <= 0) error->all(FLERR, "Illegal fix elstop command: maxlines <= 0");
      iarg += 2;
    }
    else error->all(FLERR, "Illegal fix elstop command: unknown argument");
  }

  if (maxlines == 0) maxlines = DEFAULT_MAXLINES;
}

/* ---------------------------------------------------------------------- */

FixElstop::~FixElstop()
{
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
  SeLoss_sync_flag = 0;
  SeLoss = 0.0;

  int ikeatom = modify->find_compute("ke_atom");
  if (ikeatom < 0) {
    char *newarg[3];
    newarg[0] = (char *) "ke_atom";
    newarg[1] = group->names[igroup];
    newarg[2] = (char *) "ke/atom";
    modify->add_compute(3, newarg);
    ikeatom = modify->find_compute("ke_atom");
  }

  c_ke = modify->compute[ikeatom];


  const int ncol = atom->ntypes + 1;
  memory->create(elstop_ranges, ncol, maxlines, "elstop:tabs");
  memset(&elstop_ranges[0][0], 0, ncol*maxlines*sizeof(double));

  if (comm->me == 0)
    read_table(file_name);

  MPI_Bcast(&table_entries, 1 , MPI_INT, 0, world);
  MPI_Bcast(&elstop_ranges[0][0], ncol*maxlines, MPI_DOUBLE, 0, world);


  // need an occasional full neighbor list
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;
}

/* ---------------------------------------------------------------------- */

void FixElstop::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixElstop::post_force(int /*vflag*/)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;

  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *numneigh;

  SeLoss_sync_flag = 0;
  c_ke->invoked_peratom = 1; //????
  c_ke->compute_peratom();
  double *ke = c_ke->vector_atom;
  double dt = update->dt;
  neighbor->build_one(list);
  numneigh = list->numneigh;

  for (int i = 0; i < nlocal; ++i) {

    // Do fast checks first, only then the region check

    // Avoiding dimers, trimers and even tetramers in case of really high energies
    if (numneigh[i] <= MINNEIGHB) continue;

    if (!(mask[i] & groupbit)) continue;

    double energy = ke[i];
    if (energy < Ecut) continue;
    if (energy < elstop_ranges[0][0]) continue;
    if (energy > elstop_ranges[0][table_entries - 1]) continue; // ???

    if (iregion >= 0) {
      // Only apply in the given region
      if (domain->regions[iregion]->match(x[i][0], x[i][1], x[i][2]) != 1)
        continue;
    }

    // Binary search to find correct energy range
    int iup = table_entries - 1;
    int idown = 0;
    while (true) {
      int ihalf = idown + (iup - idown) / 2;
      if (ihalf == idown) break;
      if (elstop_ranges[0][ihalf] < energy) idown = ihalf;
      else iup = ihalf;
    }

    int itype = type[i];
    double Se_lo = elstop_ranges[itype][idown];
    double Se_hi = elstop_ranges[itype][iup];
    double E_lo = elstop_ranges[0][idown];
    double E_hi = elstop_ranges[0][iup];

    // Get elstop with a simple linear interpolation
    double Se = (Se_hi - Se_lo) / (E_hi - E_lo) * (energy - E_lo) + Se_lo;

    double v2 = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
    double vabs = sqrt(v2);
    double factor = -Se / vabs;

    f[i][0] += v[i][0] * factor;
    f[i][1] += v[i][1] * factor;
    f[i][2] += v[i][2] * factor;

    SeLoss += Se * vabs * dt; // very rough approx
  }
}

/* ---------------------------------------------------------------------- */

void FixElstop::read_table(const char *file)
{
  char line[MAXLINE];

  fprintf(screen, "Reading elstop table %s\n", file);

  FILE *fp = force->open_potential(file);
  if (fp == NULL) {
    char str[128];
    snprintf(str, 128, "Cannot open stopping range table %s", file);
    error->one(FLERR, str);
  }

  const int ncol = atom->ntypes + 1;

  int l = 0;
  while (l < maxlines) {
    if (fgets(line, MAXLINE, fp) == NULL) break; // end of file
    if (line[0] == '#') continue; // comment

    char *pch = strtok(line, " \t\n\r");
    if (pch == NULL) continue; // blank line

    for (int i = 0; i < ncol; i++){
      if (pch == NULL) error->one(FLERR, "fix elstop: Invalid table line");
      elstop_ranges[i][l] = atof(pch);
      pch = strtok(NULL, " \t\n\r");
    }
    if (pch != NULL) error->one(FLERR, "fix elstop: Invalid table line");
    l++;
  }
  table_entries = l;

  if (table_entries == 0) error->one(FLERR, "Did not find any data in elstop table file");

  if (fgets(line, MAXLINE, fp) != NULL)
    error->one(FLERR, "fix elstop: Table too long. Increase maxlines.");

  fclose(fp);
}


double FixElstop::compute_scalar()
{
  // only sum across procs when changed since last call

  if (SeLoss_sync_flag == 0) {
    MPI_Allreduce(&SeLoss, &SeLoss_all, 1, MPI_DOUBLE, MPI_SUM, world);
    SeLoss_sync_flag = 1;
  }
  return SeLoss_all;
}
