// bench_allreduce_alg3_sweep.c
// Barrido de tamaños para Allreduce (Algoritmo 3 vs MPI_Allreduce nativo)
// Tipo: MPI_DOUBLE | Operador: MPI_SUM | Medición: mejor de R repeticiones (tiempo del proceso más lento)
// Salida: CSV (rank 0) -> datasize_bytes,m_elems,best_us_alg3,best_us_native

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int ceil_log2(int p){int q=0,t=1;while(t<p){t<<=1;++q;}return q;}
static int* build_skips(int p,int* q_out){int q=ceil_log2(p);*q_out=q;int* s=(int*)malloc((q+1)*sizeof(int));if(!s)return NULL;s[q]=p;for(int k=q-1;k>=0;--k){int nxt=s[k+1];s[k]=(nxt+1)/2;}return s;}


/* Algoritmo 3 "estilo" (mismos skips) pero zero-copy a nivel buffers:
 * - solo un buffer temporal T
 * - sin Wp
 * - W se inicializa con V y siempre contiene la suma parcial.
 */
static void allreduce_small_alg3_zero_copy(
    const double* V, double* W, int m, MPI_Comm comm)
{
    int r, p;
    MPI_Comm_rank(comm, &r);
    MPI_Comm_size(comm, &p);

    if (p == 1) {
        memcpy(W, V, (size_t)m * sizeof(double));
        return;
    }

    int q = 0;
    int* s = build_skips(p, &q);
    if (!s) {
        if (r == 0) fprintf(stderr, "skips alloc\n");
        MPI_Abort(comm, 1);
    }

    /* buffer temporal ÚNICO para datos recibidos */
    double *T = (double*)malloc((size_t)m * sizeof(double));
    if (!T) {
        if (r == 0) fprintf(stderr, "buf alloc\n");
        MPI_Abort(comm, 1);
    }

    /* 1) Inicializamos W con V (una sola copia inicial) */
    memcpy(W, V, (size_t)m * sizeof(double));

    /* 2) Primera ronda: usamos el mismo s[0] que antes */
    int s0 = s[0];
    int to   = (r - s0 + p) % p;
    int from = (r + s0) % p;

    MPI_Sendrecv(W, m, MPI_DOUBLE, to,   100,
                 T, m, MPI_DOUBLE, from, 100,
                 comm, MPI_STATUS_IGNORE);

    /* Combinamos parcial localmente */
    for (int i = 0; i < m; ++i) W[i] += T[i];

    /* 3) Rondas restantes: usamos el mismo patrón de skips s[k] */
    for (int k = 1; k <= q - 1; ++k) {
        int sk = s[k];

        /* Aquí NO necesitamos la ε rara de Alg.3;
         * simplemente usamos los mismos saltos s[k]
         * para un recursive-doubling estándar.
         */
        int t = (r - sk + p) % p;
        int f = (r + sk) % p;

        MPI_Sendrecv(W, m, MPI_DOUBLE, t,   100 + k,
                     T, m, MPI_DOUBLE, f,   100 + k,
                     comm, MPI_STATUS_IGNORE);

        /* Mezclamos la suma parcial */
        for (int i = 0; i < m; ++i) W[i] += T[i];
    }

    free(T);
    free(s);
}




/* Algoritmo 3 (⊕ suma) con MPI_Sendrecv */
static void allreduce_small_alg3_sendrecv(const double* V,double* W,int m,MPI_Comm comm){
    int r,p; MPI_Comm_rank(comm,&r); MPI_Comm_size(comm,&p);
    if(p==1){memcpy(W,V,(size_t)m*sizeof(double));return;}
    int q=0; int* s=build_skips(p,&q); if(!s){if(r==0)fprintf(stderr,"skips alloc\n"); MPI_Abort(comm,1);}
    double *T=(double*)malloc((size_t)m*sizeof(double)), *Wp=(double*)malloc((size_t)m*sizeof(double));
    if(!T||!Wp){if(r==0)fprintf(stderr,"buf alloc\n"); MPI_Abort(comm,1);}
    int s0=s[0], to=(r - s0 + p)%p, from=(r + s0)%p;
    MPI_Sendrecv((void*)V,m,MPI_DOUBLE,to,100, W,m,MPI_DOUBLE,from,100, comm,MPI_STATUS_IGNORE);
    for(int k=1;k<=q-1;++k){
        int eps=(s[k+1]&1);
        int t=(r - s[k] + eps + p)%p, f=(r + s[k] - eps + p)%p;
        if(eps==1){
            MPI_Sendrecv(W,m,MPI_DOUBLE,t,100+k, T,m,MPI_DOUBLE,f,100+k, comm,MPI_STATUS_IGNORE);
        }else{
            for(int i=0;i<m;++i) Wp[i]=V[i]+W[i];
            MPI_Sendrecv(Wp,m,MPI_DOUBLE,t,100+k, T, m,MPI_DOUBLE,f,100+k, comm,MPI_STATUS_IGNORE);
        }
        for(int i=0;i<m;++i) W[i]+=T[i];
    }
    for(int i=0;i<m;++i) W[i]+=V[i];
    free(T); free(Wp); free(s);
}

static void fill_vec(double* V,int m,int rank){ for(int i=0;i<m;++i) V[i]=(double)rank + 1e-6*(double)i; }
static int approx_equal(const double* a,const double* b,int m){
    const double atol=1e-10, rtol=1e-10;
    for(int i=0;i<m;++i){ double diff=fabs(a[i]-b[i]); double tol=atol + rtol*fmax(fabs(a[i]),fabs(b[i])); if(diff>tol) return 0; }
    return 1;
}


static double best_time_alg3_zero_copy(int m,int reps,MPI_Comm comm,int do_check){
    int r; MPI_Comm_rank(comm,&r);
    double *V   =(double*)malloc((size_t)m*sizeof(double));
    double *W   =(double*)malloc((size_t)m*sizeof(double));
    double *REF =(double*)malloc((size_t)m*sizeof(double));
    if(!V||!W||!REF){ if(r==0)fprintf(stderr,"malloc\n"); MPI_Abort(comm,1); }

    fill_vec(V,m,r);
    MPI_Allreduce(V,REF,m,MPI_DOUBLE,MPI_SUM,comm);

    double best_us=1e300;
    for(int it=0; it<reps; ++it){
        MPI_Barrier(comm);
        double t0=MPI_Wtime();
        allreduce_small_alg3_zero_copy(V,W,m,comm);
        double t1=MPI_Wtime();

        double local_us=(t1-t0)*1e6, max_us=0.0;
        MPI_Reduce(&local_us,&max_us,1,MPI_DOUBLE,MPI_MAX,0,comm);
        if(r==0 && max_us<best_us) best_us=max_us;

        if(it==0 && do_check && r==0){
            int ok=approx_equal(W,REF,m);
            printf("[check zero] m=%d -> %s\n",m, ok?"OK":"MISMATCH");
        }
    }
    free(V); free(W); free(REF);
    MPI_Bcast(&best_us,1,MPI_DOUBLE,0,comm);
    return best_us;
}





static double best_time_alg3(int m,int reps,MPI_Comm comm,int do_check){
    int r; MPI_Comm_rank(comm,&r);
    double *V=(double*)malloc((size_t)m*sizeof(double));
    double *W=(double*)malloc((size_t)m*sizeof(double));
    double *REF=(double*)malloc((size_t)m*sizeof(double));
    if(!V||!W||!REF){ if(r==0)fprintf(stderr,"malloc\n"); MPI_Abort(comm,1); }
    fill_vec(V,m,r);
    MPI_Allreduce(V,REF,m,MPI_DOUBLE,MPI_SUM,comm);
    double best_us=1e300;
    for(int it=0; it<reps; ++it){
        MPI_Barrier(comm);
        double t0=MPI_Wtime();
        allreduce_small_alg3_sendrecv(V,W,m,comm);
        double t1=MPI_Wtime();
        double local_us=(t1-t0)*1e6, max_us=0.0;
        MPI_Reduce(&local_us,&max_us,1,MPI_DOUBLE,MPI_MAX,0,comm);
        if(r==0 && max_us<best_us) best_us=max_us;
        if(it==0 && do_check && r==0){ int ok=approx_equal(W,REF,m); printf("[check] m=%d -> %s\n",m, ok?"OK":"MISMATCH"); }
    }
    free(V); free(W); free(REF);
    MPI_Bcast(&best_us,1,MPI_DOUBLE,0,comm);
    return best_us;
}

static double best_time_native(int m,int reps,MPI_Comm comm){
    int r; MPI_Comm_rank(comm,&r);
    double *V=(double*)malloc((size_t)m*sizeof(double));
    double *W=(double*)malloc((size_t)m*sizeof(double));
    if(!V||!W){ if(r==0)fprintf(stderr,"malloc\n"); MPI_Abort(comm,1); }
    fill_vec(V,m,r);
    double best_us=1e300;
    for(int it=0; it<reps; ++it){
        MPI_Barrier(comm);
        double t0=MPI_Wtime();
        MPI_Allreduce(V,W,m,MPI_DOUBLE,MPI_SUM,comm);
        double t1=MPI_Wtime();
        double local_us=(t1-t0)*1e6, max_us=0.0;
        MPI_Reduce(&local_us,&max_us,1,MPI_DOUBLE,MPI_MAX,0,comm);
        if(r==0 && max_us<best_us) best_us=max_us;
    }
    free(V); free(W);
    MPI_Bcast(&best_us,1,MPI_DOUBLE,0,comm);
    return best_us;
}

/* genera lista {1,2,5}×10^k hasta m_max */
static int make_sizes(int **out,int m_max){
    int cap=64, n=0; int *a=(int*)malloc(cap*sizeof(int));
    for(int k=0; ; ++k){
        int base = 1; for(int t=0;t<k;++t) base*=10;
        int cand[3] = {1*base, 2*base, 5*base};
        for(int j=0;j<3;++j){
            if(cand[j] > m_max) { *out=a; return n; }
            if(n==cap){ cap*=2; a=(int*)realloc(a,cap*sizeof(int)); }
            a[n++]=cand[j];
        }
    }
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int r,p; MPI_Comm_rank(MPI_COMM_WORLD,&r); MPI_Comm_size(MPI_COMM_WORLD,&p);

    int reps = (argc>=2)? atoi(argv[1]) : 40;
    int m_max= (argc>=3)? atoi(argv[2]) : 100000;      // 1e5 doubles por defecto
    const char* outcsv = (argc>=4)? argv[3] : "results.csv";

    if(r==0){
        printf("Config: p=%d, reps=%d, m_max=%d (double)\n", p, reps, m_max);
        printf("Salida CSV: %s\n", outcsv);
        fflush(stdout);
    }

    int *sizes=NULL; int ns=make_sizes(&sizes, m_max);
    FILE* fp=NULL;
    if(r==0){
        fp = fopen(outcsv, "w");
        if(!fp){
            perror("fopen");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(fp,
            "datasize_bytes,m_elems,best_us_alg3,best_us_zero,best_us_native\n");
    }
    

    for(int i=0;i<ns;++i){
        int m = sizes[i];
        double t_alg3 = best_time_alg3(m, reps, MPI_COMM_WORLD, /*check*/1);
        double t_zero  = best_time_alg3_zero_copy(m, reps, MPI_COMM_WORLD, 1);
        double t_nat   = best_time_native(m, reps, MPI_COMM_WORLD);
        if(r==0){
            double bytes = (double)m * sizeof(double);
            fprintf(fp, "%.0f,%d,%.6f,%.6f,%.6f\n",
                bytes, m, t_alg3, t_zero, t_nat);
            fflush(fp);
            printf("m=%d -> alg3=%.2fus | zero=%.2fus | native=%.2fus\n",
               m, t_alg3, t_zero, t_nat);
            fflush(stdout);
        }
    }
    if(r==0) fclose(fp);
    free(sizes);
    MPI_Finalize();
    return 0;
}
