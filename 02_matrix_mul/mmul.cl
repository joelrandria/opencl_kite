/**
 * Calcul d'une case de la matrice résultat p/ work item.
 */
__kernel void mmul_cij_gmem(const int m1_rows, const int m1_cols, __global const float* g_m1,
			    const int m2_rows, const int m2_cols, __global const float* g_m2,
			    __global float* g_r)
{
  int i;

  int rr; // ligne de la case à calculer
  int rc; // colonne de la case à calculer

  rr = get_global_id(0);
  rc = get_global_id(1);

  g_r[rr * m2_cols + rc] = 0;
  for (i = 0; i < m1_cols; ++i)
    g_r[rr * m2_cols + rc] += g_m1[rr * m1_cols + i] * g_m2[i * m2_cols + rc];
}

/**
 * Calcul d'une ligne de la matrice résultat p/ work item.
 */
__kernel void mmul_ci_gmem(const int m1_rows, const int m1_cols, __global const float* g_m1,
			   const int m2_rows, const int m2_cols, __global const float* g_m2,
			   __global float* g_r)
{
  int i;
  int j;

  int rr; // ligne à calculer

  rr = get_global_id(0);

  for (j = 0; j < m2_cols; ++j)
  {
    g_r[rr * m2_cols + j] = 0;
    for (i = 0; i < m1_cols; ++i)
      g_r[rr * m2_cols + j] += g_m1[rr * m1_cols + i] * g_m2[i * m2_cols + j];
  }
}

/**
 * Calcul d'une ligne de la matrice résultat p/ work item:
 *
 * - Copie privée de la ligne de m1
 */
__kernel void mmul_ci_pmemr_gmemc(const int m1_rows, const int m1_cols, __global const float* g_m1,
				  const int m2_rows, const int m2_cols, __global const float* g_m2,
				  __global float* g_r)
{
  int i;
  int j;

  int rr; // ligne à calculer

  float p_m1r[1024];

  rr = get_global_id(0);

  // Copie privée de la ligne
  for (i = 0; i < m1_cols; ++i)
    p_m1r[i] = g_m1[rr * m1_cols + i];

  for (j = 0; j < m2_cols; ++j)
  {
    g_r[rr * m2_cols + j] = 0;
    for (i = 0; i < m1_cols; ++i)
      g_r[rr * m2_cols + j] += p_m1r[i] * g_m2[i * m2_cols + j];
  }
}

/**
 * Calcul d'une ligne de la matrice résultat p/ work item:
 *
 * - Copie privée de la ligne de m1
 * - Copie locale de la colonne de m2
 */
__kernel void mmul_ci_pmemr_lmemc(const int m1_rows, const int m1_cols, __global const float* g_m1,
				  const int m2_rows, const int m2_cols, __global const float* g_m2,
				  __global float* g_r,
				  __local float* l_c)
{
  int i;
  int j;

  int rr;	// ligne à calculer

  int iloc;	// ID local du work item
  int nloc;	// taille du work group

  float p_m1r[1024];

  rr = get_global_id(0);

  iloc = get_local_id(0);
  nloc = get_local_size(0);

  // Copie privée de la ligne
  for (i = 0; i < m1_cols; ++i)
    p_m1r[i] = g_m1[rr * m1_cols + i];

  for (j = 0; j < m2_cols; ++j)
  {
    // Copie locale de la colonne courante, partagée par tous les work items du work group dans l'itération courante
    for (i = iloc; i < m2_cols; i += nloc)
      l_c[i] = g_m2[i * m2_cols + j];

    barrier(CLK_LOCAL_MEM_FENCE); // Synchronisation des work items du work group

    g_r[rr * m2_cols + j] = 0;
    for (i = 0; i < m1_cols; ++i)
      g_r[rr * m2_cols + j] += p_m1r[i] * l_c[i];
  }
}
