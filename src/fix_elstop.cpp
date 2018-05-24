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

#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "fix_elstop.h"
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
#include <iostream> // For cout capabilities (good for debugging)
#include <string> // For cout string capabilities (good for debugging)

using namespace std;
using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG 1.0e20
#define MAXLINE 1024
#define MAXTABLEN 300
#define MINNEIGHB 5

/* ---------------------------------------------------------------------- */

template<typename T>
T signum(T n)
{
if (n < 0) return -1;
if (n > 0) return 1;
return 0;
}

/* ---------------------------------------------------------------------- */

FixElstop::FixElstop(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal fix elstop command");

  // set time_depend, else elapsed time accumulation can be messed up

  time_depend = 1; //?????
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 0;
  extvector = 0;

  regionflag = 0; 
  iregion = -1;

  int iarg = 0;

  // reading in the numerical parameters from fix ID group-ID elstop N Ecut:
  nevery = 1;	// Do this every step
  groupname=group->names[igroup];
  Ecut = force->numeric(FLERR,arg[3]);
  
  int n = strlen(arg[4]) + 1;
  file_name = new char[n];
  strcpy(file_name,arg[4]);
  
  if (Ecut <= 0.0) error->all(FLERR,"Illegal fix elstop command: cutoff energy cannot be 0 or negative!");
 
  iarg = 5;
  // reading in the regions
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix elstop command: region name missing!");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"REGION (not group) ID for fix elstop does not exist");
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
		newarg[1] = groupname;
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
	
		  //fprintf(screen, "Debugging >> %lf %lf %lf %i %i\n", Se, ENERGY, dt, idown,  iup);
		  double v2 = 0.0;
          for (int n=0 ; n<3 ; ++n ) {
            f[i][n] = f[i][n]-(signum(v[i][n]))*Se;
			v2+=v[i][n]*v[i][n];
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

  if (fgets(line,MAXLINE,fp) == NULL)
      error->one(FLERR,"Did not find any data in table file");

  int l=0;
  while (l < MAXTABLEN) {
    if (fgets(line,MAXLINE,fp) == NULL) break;                         // no match, skip section
    if (line == NULL) break;     
    if (strspn(line," \t\n\r") == strlen(line)) continue;  // blank line
    if (line[0] == '#') continue;                          // comment
    
    //fprintf(screen, "line %d: %s\n", l, line);
	char * pch;
    pch = strtok (line," \t\n\r");
    for (int i = 0; i < nt; i++){
        elstop_ranges[i][l]=atof(pch);
        pch = strtok (NULL, " \t\n\r");
    }
    l++;
  }
  table_entries=l;

  if (fgets(line,MAXLINE,fp) != NULL){
    fprintf(screen, "Warning: Only 300 entries have been read from the elstop table\n Please increase MAXTABLEN=%i value and recompile\n", table_entries);
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


/* ---------------------------------------------------------------------- */