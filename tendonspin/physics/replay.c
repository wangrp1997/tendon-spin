/* Exact native command playback, with original per-step physical gates.
 * This function evolves the supplied data. It never resets or adopts forecasts.
 */
#include <math.h>
#include <string.h>
#include <mujoco/mujoco.h>

int replay_version(void) { return mjVERSION_HEADER; }

int replay(const mjModel* m, mjData* d, int steps, const double* controls,
           const int* owner, int object_geom, int object_qadr,
           int floor_geom, int support_eq, const double* center, const double* axis,
           double* q, double* v, double* motor, double* sensors,
           double* metrics, int* flags, int stop_on_failure) {
  for (int t = 0; t < steps; ++t) {
    memcpy(d->ctrl, controls+(size_t)t*m->nu, m->nu*sizeof(double));
    mj_step(m, d);
    memcpy(q+(size_t)t*m->nq, d->qpos, m->nq*sizeof(double));
    memcpy(v+(size_t)t*m->nv, d->qvel, m->nv*sizeof(double));
    memcpy(motor+(size_t)t*m->nu, d->actuator_force, m->nu*sizeof(double));
    if (sensors) memcpy(sensors+(size_t)t*m->nsensordata, d->sensordata, m->nsensordata*sizeof(double));
    double f[6][3]={{0}}, normal[6]={0};
    double outside=0, floor=0, obj_pen=0, self_pen=0, self_force=0;
    for (int k=0; k<d->ncon; ++k) {
      const mjContact* c=d->contact+k;
      double cf[6]; mj_contactForce(m,d,k,cf);
      if (owner[c->geom[0]]>=0 && owner[c->geom[1]]>=0) {
        self_force+=cf[0]; self_pen=fmin(self_pen,c->dist);
      }
      if (c->geom[0]!=object_geom && c->geom[1]!=object_geom) continue;
      int other=c->geom[1]==object_geom?c->geom[0]:c->geom[1];
      double sign=c->geom[1]==object_geom?1.:-1., world[3];
      for (int j=0; j<3; ++j)
        world[j]=sign*(c->frame[j]*cf[0]+c->frame[3+j]*cf[1]+c->frame[6+j]*cf[2]);
      obj_pen=fmin(obj_pen,c->dist);
      if (owner[other]>=0) {
        int a=owner[other];
        for (int j=0; j<3; ++j) f[a][j]+=world[j];
        normal[a]+=cf[0];
      } else {
        double n=sqrt(world[0]*world[0]+world[1]*world[1]+world[2]*world[2]);
        outside+=n;
        if (other==floor_geom) floor+=n;
      }
    }
    double delta[3], R[9], drift=0, maximum=0;
    for (int j=0; j<3; ++j) { delta[j]=d->qpos[object_qadr+j]-center[j]; drift+=delta[j]*delta[j]; }
    drift=sqrt(drift);
    mju_quat2Mat(R,d->qpos+object_qadr+3);
    double dot=R[2]*axis[0]+R[5]*axis[1]+R[8]*axis[2];
    double tilt=acos(fmin(1.,fmax(-1.,dot)))*180./mjPI;
    int loaded=0;
    for (int a=0; a<6; ++a) {
      maximum=fmax(maximum,sqrt(f[a][0]*f[a][0]+f[a][1]*f[a][1]+f[a][2]*f[a][2]));
      if (a<5 && normal[a]>1e-6) loaded++;
    }
    int external=d->eq_active[support_eq]!=0, warning=0, nonfinite=0;
    for (int j=0; j<m->nbody*6; ++j) external|=d->xfrc_applied[j]!=0;
    for (int j=0; j<m->nv; ++j) external|=d->qfrc_applied[j]!=0;
    for (int j=0; j<mjNWARNING; ++j) warning|=d->warning[j].number!=0;
    for (int j=0; j<m->nq; ++j) nonfinite|=!isfinite(d->qpos[j]);
    for (int j=0; j<m->nv; ++j) nonfinite|=!isfinite(d->qvel[j]);
    int code=(drift>.005) | ((tilt>15)<<1) | ((maximum>12)<<2) | ((loaded<2)<<3)
      | ((obj_pen<-.0015)<<4) | ((self_pen<-.0015)<<5) | ((outside>.05)<<6)
      | ((floor>.05)<<7) | (external<<8) | (warning<<9) | (nonfinite<<10);
    double* z=metrics+(size_t)t*14;
    z[0]=drift*1000; z[1]=tilt;
    for (int a=0; a<6; ++a) z[2+a]=normal[a];
    z[8]=maximum; z[9]=self_force; z[10]=-obj_pen*1000;
    z[11]=-self_pen*1000; z[12]=outside; z[13]=floor;
    flags[t]=code;
    if (code && stop_on_failure) return t+1;
  }
  return steps;
}
