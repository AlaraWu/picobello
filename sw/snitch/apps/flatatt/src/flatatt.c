#include "blas.h"
#include "dnn.h"
#include "data.h"
#include "snrt.h"
#include "exp_baseline.h"
#include <stdalign.h>
#include <stdint.h>

#define HW_MCAST
#ifdef HW_MCAST
#define BCAST_MASK_ALL 0x003C0000
#define BCAST_MASK_COL 0x000C0000
#define BCAST_MASK_ROW 0x00300000
#endif

typedef struct {
    uint32_t L;
    uint32_t S;
    uint32_t d;
    uint32_t B_r;
    uint32_t B_c;
    void *Q;
    void *K;
    void *V;
    void *O;
    precision_t dtype;
    uint32_t baseline;
    gemm_fp_t gemm_implementation;
} flatattention_singlehead_layer_t;

// Duplicate to a column of l2 tiles
// pos: 0 for left col, 1 for right col
static inline void duplicate_l2_tiles_col(void *src, uint32_t size, uint32_t prec, int pos){
    if (snrt_is_dm_core()){
        for (int i = 0; i < PB_CLUSTER_PER_COL; i++)
        {
            uintptr_t dst = pb_l2_tile_address(pos * PB_CLUSTER_PER_COL + i) + pb_l2_tile_offset((uintptr_t)src);
            if (dst != (uintptr_t)src){
                snrt_dma_start_1d((void *)dst, src, size * prec);
            }
        }
        snrt_dma_wait_all();
    }
}

// Allocate QKVO in L2 to better map kernel in NoC system
static inline void matrix_remapping_on_memtiles(mha_layer_t *args) {
    uint32_t prec = args->dtype;

    uint32_t size_q = (size_t)args->L * (size_t)args->d;
    uint32_t size_k = (size_t)args->S * (size_t)args->d;
    uint32_t size_v = (size_t)args->S * (size_t)args->d;
    // uint32_t size_o = (size_t)args->L * (size_t)args->d;

    // Q, O mapped to the left
    // K, V mapped to the right
    for (int i = 0; i < args->num_heads; i++) {
        duplicate_l2_tiles_col(((float **)args->Q)[i], size_q, prec, 0);
        duplicate_l2_tiles_col(((float **)args->K)[i], size_k, prec, 1);
        duplicate_l2_tiles_col(((float **)args->V)[i], size_v, prec, 1);
        // duplicate_l2_tiles_col(((float **)args->O)[i], size_o, prec, 0);
    }
    
}

typedef float (*row_op_t)(float acc, float val);

static inline void row_reduce(float *in, float *out, int cols, row_op_t op, float init_val) {
    *out = in[0];
    for (int j = 1; j < cols; j++) {
        *out = op(*out, in[j]);
    }
}

static inline void row_broadcast_from_left(void *dst, uint32_t size_bytes) {
    // Here we assume each cluster has such allocated dst buffer
    // Column num is fixed to 4
    if (snrt_is_dm_core()) {
        uint32_t cluster_id = snrt_cluster_idx();
#if defined(HW_MCAST)
        if (pb_cluster_col() == 0){
            snrt_dma_start_1d_mcast(snrt_remote_l1_ptr(dst, cluster_id, cluster_id + 4), // dst
                                    dst,            // src
                                    size_bytes,     // size
                                    BCAST_MASK_ROW  // mcast
                                );
            snrt_dma_wait_all();
        }
#elif defined(SW_MCAST)
        if (pb_cluster_col() == 0){
            snrt_dma_start_1d(snrt_remote_l1_ptr(dst, cluster_id, cluster_id + 2 * PB_CLUSTER_PER_COL), // dst
                                    dst,             // src
                                    size_bytes       // size
                                );
            snrt_dma_wait_all();
        }
        snrt_inter_cluster_barrier();
        if (pb_cluster_col() % 2 == 0){
            snrt_dma_start_1d(snrt_remote_l1_ptr(dst, cluster_id, cluster_id + PB_CLUSTER_PER_COL), // dst
                                    dst,             // src
                                    size_bytes       // size
                                );
            snrt_dma_wait_all();
        }
#else
        if (pb_cluster_col() == 0){
            for (int i = 1; i < PB_CLUSTER_PER_ROW; i++) {
                snrt_dma_start_1d(snrt_remote_l1_ptr(dst, cluster_id, cluster_id + i * PB_CLUSTER_PER_COL), // dst
                                    dst,             // src
                                    size_bytes       // size
                                );
            }
            snrt_dma_wait_all();
        }
#endif
        snrt_mcycle();
    }
}

// TODO(chenwu): can refer to runtime/src/sync.c: function snrt_global_reduction_dma for generalization
static inline void inter_reduce(float *dst, float *tmp, int row_num, size_t size_bytes,
                                  row_op_t op) {
    // Step 1: Inter-cluster reduction in a tree fashion, saving results in cluster 0

    // 1->0, 3->2
    snrt_comm_t comm;
    snrt_comm_create(snrt_cluster_num(), &comm);
    // Distribute rows evenly to the cores in a cluster.
    // Last core handles remainder rows.
    uint32_t rows_per_core = row_num / snrt_cluster_compute_core_num();
    uint32_t start_row = rows_per_core * snrt_cluster_core_idx();
    char is_last_compute_core =
        snrt_cluster_core_idx() == (snrt_cluster_compute_core_num() - 1);
    uint32_t end_row =
        is_last_compute_core ? row_num : start_row + rows_per_core;

    snrt_mcycle();

    if (pb_cluster_col() % 2 == 0 && snrt_is_dm_core()) {
        snrt_dma_start_1d(tmp, snrt_remote_l1_ptr(dst, snrt_cluster_idx(), snrt_cluster_idx() + 4), size_bytes);
        snrt_dma_wait_all();
    }

    snrt_global_barrier(comm);
    // snrt_cluster_hw_barrier();
    snrt_mcycle();

    if (pb_cluster_col() % 2 == 0 && snrt_is_compute_core()) {
        // parallelize over compute cores
        for (int row_idx = start_row; row_idx < end_row; row_idx++) {
            dst[row_idx] = op(dst[row_idx], tmp[row_idx]);
        }
    }
    // Synchronize compute and DM cores for next tree level
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();
    snrt_mcycle();


    // 2->0
    if (pb_cluster_col() == 0 && snrt_is_dm_core()) {
        snrt_dma_start_1d(tmp, snrt_remote_l1_ptr(dst, snrt_cluster_idx(), snrt_cluster_idx() + 8), size_bytes);
        snrt_dma_wait_all();
    }

    snrt_global_barrier(comm);
    // snrt_cluster_hw_barrier();
    snrt_mcycle();

    if (pb_cluster_col() == 0 && snrt_is_compute_core()) {
        // parallelize over compute cores
        for (int row_idx = start_row; row_idx < end_row; row_idx++) {
            dst[row_idx] = op(dst[row_idx], tmp[row_idx]);
        }
    }
    // // Synchronize compute and DM cores for next tree level
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();
    // snrt_global_barrier(comm);
    snrt_mcycle();


    // Step 2: Broadcast
    row_broadcast_from_left(dst, size_bytes);
    snrt_global_barrier(comm);
}

// row-wise max operation
static inline float max_op(float acc, float val) {
    return acc > val ? acc : val;
}

// sum operation
static inline float sum_op(float acc, float val) {
    return acc + val;
}

static inline void flatatt_singlehead_fp32(flatattention_singlehead_layer_t layer){
    // alias layer parameters
    uint32_t dtype = layer.dtype;
    uint32_t L = layer.L;
    uint32_t S = layer.S;
    uint32_t d = layer.d;
    uint32_t B_r = layer.B_r;
    uint32_t B_c = layer.B_c;
    uint32_t baseline = layer.baseline;
    gemm_fp_t gemm_implementation = layer.gemm_implementation;
    float *Q_l2 = (float *)layer.Q;
    float *K_l2 = (float *)layer.K;
    float *V_l2 = (float *)layer.V;
    float *O_l2 = (float *)layer.O;

    // gemm specific parameters
    sc_st_gemm_args_t gemm_args;
    gemm_args.prec = dtype;
    if (!baseline) {
        gemm_args.setup_ssr = 1;
    } else {
        gemm_args.setup_ssr = 0;
    }
    gemm_args.partition_banks = 0;
    gemm_args.transa = 0;
    gemm_args.m = B_r;
    gemm_args.alpha = 1;

    // alias system parameters
    uint32_t compute_id = snrt_cluster_core_idx();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t num_cores = snrt_cluster_compute_core_num();
    uint32_t num_clusters = snrt_cluster_num();
    // Distribute rows evenly to the cores in a cluster.
    // Last core handles remainder rows.
    uint32_t rows_per_core, start_row, end_row;
    if (B_r < num_cores) {
        rows_per_core = (compute_id < B_r)? 1 : 0;
        start_row = (compute_id < B_r)? compute_id: 0;
        end_row = (compute_id < B_r)? compute_id + 1 : 0;
    }
    else {
        rows_per_core = B_r / num_cores;
        start_row = rows_per_core * compute_id;
        char is_last_compute_core =
            compute_id == (num_cores - 1);
        end_row =
            is_last_compute_core ? B_r : start_row + rows_per_core;
    }

    snrt_comm_t comm;
    snrt_comm_create(num_clusters, &comm);

    // compute the tiling parameters
    uint32_t T_r = L / B_r;  // number of row blocks
    uint32_t T_c = S / B_c;  // number of column blocks

    // compute the size of the matrices
    uint32_t q_fa_elems = B_r * d;
    uint32_t k_fa_elems = B_c * d;
    uint32_t v_fa_elems = B_c * d;
    uint32_t s_fa_elems = B_r * B_c;
    uint32_t p_fa_elems = B_r * B_c;
    uint32_t o_fa_elems = B_r * d;
    uint32_t q_fa_size = q_fa_elems * sizeof(float);
    uint32_t k_fa_size = k_fa_elems * sizeof(float);
    uint32_t v_fa_size = v_fa_elems * sizeof(float);
    uint32_t s_fa_size = s_fa_elems * sizeof(float);
    uint32_t m_size = B_r * sizeof(float);
    uint32_t p_fa_size = p_fa_elems * sizeof(float);
    uint32_t l_size = B_r * sizeof(float);
    uint32_t o_fa_size = o_fa_elems * sizeof(float);
    // allocate memory in TCDM
    float *Q_fa =
        (float *)snrt_l1_alloc_cluster_local(q_fa_size, alignof(float));
    float *K_fa =
        (float *)snrt_l1_alloc_cluster_local(k_fa_size, alignof(float));
    float *V_fa =
        (float *)snrt_l1_alloc_cluster_local(v_fa_size, alignof(float));
    float *S_fa =
        (float *)snrt_l1_alloc_cluster_local(s_fa_size, alignof(float));
    float *P_fa =
        (float *)snrt_l1_alloc_cluster_local(p_fa_size, alignof(float));
    float *O_fa =
        (float *)snrt_l1_alloc_cluster_local(o_fa_size, alignof(float));
    float *m =
        (float *)snrt_l1_alloc_cluster_local(m_size, alignof(float));
    float *l =
        (float *)snrt_l1_alloc_cluster_local(l_size, alignof(float));
    uint64_t *T_local = 
        (uint64_t *)snrt_l1_alloc_cluster_local(64 * sizeof(uint64_t), alignof(uint64_t));

    if (snrt_is_dm_core() && cluster_id == 0) {
        snrt_dma_start_1d_mcast(T_local,               // dst
                          T,                     // src
                          64 * sizeof(uint64_t), // size
                          BCAST_MASK_ALL         // mcast
                          );
        snrt_dma_wait_all();
    }

    float *m_tmp, *l_tmp, *O_tmp;
    m_tmp = (float *) snrt_l1_alloc_cluster_local(m_size, alignof(float));
    l_tmp = (float *) snrt_l1_alloc_cluster_local(l_size, alignof(float));
    O_tmp = (float *) snrt_l1_alloc_cluster_local(o_fa_size, alignof(float));

    if (snrt_is_compute_core()) {
        for (int row_idx = start_row; row_idx < end_row; row_idx++) {
            m[row_idx] = -INFINITY;
            l[row_idx] = 0.0f;
        }
    }

    // 1. Load QKV
    if (snrt_is_dm_core()) {
        snrt_mcycle();
#if defined(HW_MCAST)
        // clusters at the most left load, and broadcast (row-wise) Q
        if(pb_cluster_col() == 0){
            snrt_dma_load_1d_tile_mcast(Q_fa,           // dst
                                        Q_l2,           // src
                                        cluster_id,     // tile_idx
                                        q_fa_elems,     // tile_size
                                        dtype,          // prec
                                        BCAST_MASK_ROW  // mcast
                                    );
        }
        // clusters at the diagonal load, and broadcast (column-wise) K and V
        if (pb_cluster_row() == pb_cluster_col()){
            // K_fa: B_c x d, broadcast column-wise
            snrt_dma_load_1d_tile_mcast(K_fa,             // dst
                                        K_l2,             // src
                                        pb_cluster_col(), // tile_idx
                                        k_fa_elems,       // tile_size
                                        dtype,            // prec
                                        BCAST_MASK_COL    // mcast
                                    );
            // V_fa: B_c x d, broadcast column-wise
            snrt_dma_load_1d_tile_mcast(V_fa,             // dst
                                        V_l2,             // src
                                        pb_cluster_col(), // tile_idx
                                        v_fa_elems,       // tile_size
                                        dtype,            // prec
                                        BCAST_MASK_COL    // mcast
                                    );
        }
// #elif defined(SW_MCAST)
#else
        // Fetch Q from left. Cluster in the same row get the same Q (B_r x d)
        snrt_dma_load_1d_tile(Q_fa,           // dst
                            Q_l2,             // src
                            pb_cluster_row(), // tile_idx
                            q_fa_elems,       // tile_size
                            sizeof(float)             // prec
                            );
        // Fetch K, V from right. Cluster in the same column get the same K (B_c x d), V (B_c x d)
        snrt_dma_load_1d_tile(K_fa,           // dst
                            K_l2,             // src
                            pb_cluster_col(), // tile_idx
                            k_fa_elems,       // tile_size
                            sizeof(float)             // prec
                            );
        snrt_dma_load_1d_tile(V_fa,           // dst
                            V_l2,             // src
                            pb_cluster_col(), // tile_idx
                            v_fa_elems,       // tile_size
                            sizeof(float)             // prec
                            );
#endif
        snrt_dma_wait_all();
        snrt_mcycle();
    }

    snrt_global_barrier(comm);

    // 2. Compute S_fa = Q_fa * K_fa^T
    if (snrt_is_compute_core()) {
        snrt_mcycle();
        gemm_args.transb = 1;
        gemm_args.n = B_c;
        gemm_args.k = d;
        gemm_args.a = Q_fa;
        gemm_args.lda = d;
        gemm_args.b = K_fa;
        gemm_args.ldb = d;
        gemm_args.beta = 0;
        gemm_args.c = S_fa;
        gemm_args.ldc = B_c;
        sc_st_gemm(gemm_implementation, &gemm_args);

        snrt_fpu_fence();
        snrt_cluster_hw_barrier();
        snrt_mcycle();

        // 3. inner rowmax
        for (int row_idx = start_row; row_idx < end_row; row_idx++) {
            row_reduce(S_fa + row_idx * B_c, m + row_idx, B_c, max_op, -INFINITY);
        }
        snrt_mcycle();
    }
    else {
        snrt_cluster_hw_barrier();
    }

    snrt_global_barrier(comm);

    // 4. inter-cluster rowmax and broadcast
    inter_reduce(m, m_tmp, B_r, m_size, max_op);

    // 5. Compute P = exp(S - m) and inner rowsum on clusters
    if (snrt_is_compute_core()) {
        snrt_mcycle();
        for (int row_idx = start_row; row_idx < end_row; row_idx++) {
            float tmp_sum = 0.0f;
            // float tmp;
            float tmp[4];
            for (int j = 0; j < B_c ; j+=4) {
                for (int k = 0; k < 4; k++) {
                    tmp[k] = S_fa[row_idx * B_c + j + k] - m[row_idx];
                }
                vexpf_baseline(tmp, P_fa + row_idx * B_c + j, T_local);
                // for (int k = 0; k < 4; k++) {
                //     P_fa[row_idx * B_c + j + k] = expf(tmp[k]);
                // }
                for (int k = 0; k < 4; k++) {
                    tmp_sum += P_fa[row_idx * B_c + j + k];
                }
            }
            // for (int j = 0; j < B_c; j++) {
            //     tmp = S_fa[row_idx * B_c + j] - m[row_idx];
            //     P_fa[row_idx * B_c + j] = expf(S_fa[row_idx * B_c + j] - m[row_idx]);
            //     tmp_sum += P_fa[row_idx * B_c + j];
            // }
            l[row_idx] = tmp_sum;

            // row_reduce(P_fa + row_idx * B_c, l + row_idx, B_c, sum_op, 0.0f);
        }
        snrt_mcycle();
    }
    snrt_global_barrier(comm);

    // if (snrt_is_dm_core()) {
    //     snrt_dma_store_2d_tile(O_l2, P_fa, pb_cluster_row(), pb_cluster_col(), B_r, B_c, d, sizeof(float));
    //     // snrt_dma_store_1d_tile(O_l2, l, cluster_id, B_r, sizeof(float));
    //     snrt_dma_wait_all();
    // }
    // return;

    // 6. inter-cluster rowsum and broadcast
    inter_reduce(l, l_tmp, B_r, l_size, sum_op);


    // // 7. Row-wise normalization of P
    // if (snrt_is_compute_core()) {
    //     // for (int i = 0; i < B_r; i++) {
    //     //     for (int j = 0; j < B_c; j++) {
    //     //         P_fa[i * B_c + j] /= l[i];
    //     //     }
    //     // }
    //     // TODO(chenwu): non-divisible case?
    //     uint32_t items_per_core = p_fa_elems / num_cores;
    //     uint32_t core_offset = snrt_cluster_core_idx() * items_per_core;
    //     for (int i = 0; i < items_per_core; i++) {
    //         P_fa[core_offset + i] /= l[(core_offset + i) / B_c];
    //     }
    // }
    // snrt_cluster_hw_barrier();

    // if (snrt_is_dm_core()) {
    //     snrt_dma_store_1d_tile(O_l2, l, cluster_id, B_r, sizeof(float));
    //     snrt_dma_wait_all();
    // }
    // return;

    // 8. Compute O_fa = P_fa * V_fa
    if (snrt_is_compute_core()) {
        snrt_mcycle();
        gemm_args.transb = 0;
        gemm_args.n = d;
        gemm_args.k = B_c;
        gemm_args.a = P_fa;
        gemm_args.lda = B_c;
        gemm_args.b = V_fa;
        gemm_args.ldb = d;
        gemm_args.beta = 0;
        gemm_args.c = O_fa;
        gemm_args.ldc = d;
        sc_st_gemm(gemm_implementation, &gemm_args);
        snrt_mcycle();
    }
    snrt_global_barrier(comm);

    // 9. Compute O and write back
    // 1->0, 3->2
    snrt_mcycle();
    if (pb_cluster_col() % 2 == 0 && snrt_is_dm_core()) {
        snrt_dma_start_1d(O_tmp, snrt_remote_l1_ptr(O_fa, cluster_id, cluster_id + 4), o_fa_size);
        snrt_dma_wait_all();
    }
    snrt_global_barrier(comm);

    snrt_mcycle();

    if ((pb_cluster_col()) % 2 == 0 && snrt_is_compute_core()) {
        uint32_t items_per_core = o_fa_elems / num_cores;
        uint32_t core_offset = snrt_cluster_core_idx() * items_per_core;
        for (int i = 0; i < items_per_core; i++) {
            O_fa[core_offset + i] += O_tmp[core_offset + i];
        }
    }
    // Synchronize compute and DM cores for next tree level
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();
    snrt_mcycle();


    // 2->0
    if (pb_cluster_col() == 0 && snrt_is_dm_core()) {
        snrt_dma_start_1d(O_tmp, snrt_remote_l1_ptr(O_fa, cluster_id, cluster_id + 8), o_fa_size);
        snrt_dma_wait_all();
    }
    snrt_global_barrier(comm);
    snrt_mcycle();

    if (pb_cluster_col() == 0) {
        if (snrt_is_compute_core()) {
            // uint32_t items_per_core = o_fa_elems / num_cores;
            // uint32_t core_offset = snrt_cluster_core_idx() * items_per_core;
            // for (int i = 0; i < items_per_core; i++) {
            //     O_fa[core_offset + i] += O_tmp[core_offset + i];
            //     O_fa[core_offset + i] /= l[compute_id];
            // }

            for (int row_idx = start_row; row_idx < end_row; row_idx++) {
                for (int col_idx = 0; col_idx < d; col_idx++) {
                    O_fa[row_idx * d + col_idx] = (O_fa[row_idx * d + col_idx] + O_tmp[row_idx * d + col_idx]) / l[row_idx];
                }
            }
            // snrt_cluster_hw_barrier();
        } 
        // else {
        //     snrt_cluster_hw_barrier();
        // }
    }
    snrt_cluster_hw_barrier();
    snrt_mcycle();


    // Write back O
    if(pb_cluster_col() == 0 && snrt_is_dm_core()){
        uintptr_t o_dst = pb_l2_tile_address(0) + pb_l2_tile_offset((uintptr_t)layer.O) + o_fa_size * pb_cluster_row();
        snrt_dma_start_1d((void *)o_dst, O_fa, o_fa_size);
        snrt_dma_wait_all();
    }
    snrt_global_barrier(comm);
    snrt_mcycle();
}

static inline void flatatt_picobello(mha_layer_t layer) {
    // alias system parameters
    flatattention_singlehead_layer_t flat_args;
    flat_args.L = layer.L;
    flat_args.S = layer.S;
    flat_args.d = layer.d;
    flat_args.B_r = layer.B_r;
    flat_args.B_c = layer.B_c;
    flat_args.dtype = layer.dtype;
    flat_args.baseline = layer.baseline;
    flat_args.gemm_implementation = layer.gemm_implementation;
    uint32_t compute_id = snrt_cluster_core_idx();
    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t num_cores = snrt_cluster_compute_core_num();
    uint32_t num_clusters = snrt_cluster_num();

    // Duplicate Q,K,V in proper memory tiles
    if (cluster_id == 0) {
        matrix_remapping_on_memtiles(&layer);
    }
    snrt_global_barrier();
    // flatattention_singlehead_layer_t flat_args;
    // flat_args.L = layer.L;
    // flat_args.S = layer.S;
    // flat_args.d = layer.d;
    // flat_args.B_r = layer.B_r;
    // flat_args.B_c = layer.B_c;
    // flat_args.dtype = layer.dtype;
    // flat_args.baseline = layer.baseline;
    // flat_args.gemm_implementation = layer.gemm_implementation;
    for (int i = 0; i < layer.num_heads; i++) {
        // Prepare arguments for single-head FlatAttention
        void * Q_tile, * K_tile, * V_tile, * O_tile;
        Q_tile = (void *)(pb_l2_tile_address(pb_cluster_row()) + pb_l2_tile_offset((uintptr_t) ((float **)layer.Q)[i]));
        K_tile = (void *)(pb_l2_tile_address(pb_cluster_col() + PB_CLUSTER_PER_COL) + pb_l2_tile_offset((uintptr_t) ((float **)layer.K)[i]));
        V_tile = (void *)(pb_l2_tile_address(pb_cluster_col() + PB_CLUSTER_PER_COL) + pb_l2_tile_offset((uintptr_t) ((float **)layer.V)[i]));
        
        // flat_args.Q = ((float **)layer.Q)[i];
        // flat_args.K = ((float **)layer.K)[i];
        // flat_args.V = ((float **)layer.V)[i];
        flat_args.Q = Q_tile;
        flat_args.K = K_tile;
        flat_args.V = V_tile;
        flat_args.O = ((float **)layer.head_outputs)[i];
        // flat_args.O = layer.O;

        snrt_global_barrier();

        // Call single-head FlatAttention
        flatatt_singlehead_fp32(flat_args);
    }

    snrt_fpu_fence();
    snrt_global_barrier();

    // Prepare arguments for the FusedConcatLinear layer
    fused_concat_linear_layer_t fcl_args;
    fcl_args.num_inputs = layer.num_heads;
    fcl_args.input_shape[0] = layer.L;
    fcl_args.input_shape[1] = layer.d;
    fcl_args.output_shape[0] = layer.L;
    fcl_args.output_shape[1] = layer.d;
    fcl_args.inputs = layer.head_outputs;
    fcl_args.weights = layer.W;
    fcl_args.concat_output = nullptr;
    fcl_args.linear_output = layer.O;
    fcl_args.dtype = layer.dtype;
    fcl_args.gemm_implementation = layer.gemm_implementation;
    snrt_mcycle();

    // Call the FusedConcatLinear layer
    fused_concat_linear_layer(fcl_args);
    snrt_mcycle();
    
}

int main () {
    flatatt_picobello(layer);
    return 0;
}