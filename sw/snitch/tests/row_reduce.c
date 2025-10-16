// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Chen Wu <chenwu@iis.ee.ethz.ch>

// This application tests the row reduction
// 1. initialize
// 2. compute and intra reduce
// 3. inter reduce in a tree fashion
// 4. broadcast back (supports HW, SW and no multicast)
// 5. verify

#include <stdint.h>
#include "pb_addrmap.h"
#include "snrt.h"

#define SW_MCAST

#define INITIALIZER 0xAAAAAAAA

#ifndef LENGTH
#define LENGTH 8
#endif

#ifndef HEIGHT
#define HEIGHT 4
#endif

#ifndef N_CLUSTERS_TO_USE
#define N_CLUSTERS_TO_USE snrt_cluster_num()
#endif

#define BCAST_MASK_ROW 0x00300000

typedef int (*row_op_t)(int acc, int val);

static inline void row_reduce(int *in, int *out, int cols, row_op_t op, int init_val) {
    *out = init_val;
    for (int j = 0; j < cols; j++) {
        *out = op(*out, in[j]);
    }
}

static inline void intra_row_reduce(int *in, int *out, int rows, int cols,
                              row_op_t op, int init_val) {
    if (snrt_is_compute_core()) {
        // parallelize over compute cores
        uint32_t items_per_core = rows / snrt_cluster_compute_core_num();
        uint32_t remainder = rows % snrt_cluster_compute_core_num();
        for (int i = 0; i < snrt_cluster_compute_core_num(); i++) {
            if (snrt_cluster_core_idx() == i)
            {
                // Each core processes items_per_core rows
                for (int j = 0; j < items_per_core; j++)
                {
                    uint32_t abs_i = i + snrt_cluster_compute_core_num() * j;
                    out[abs_i] = init_val;
                    for (int c = 0; c < cols; c++) {
                        out[abs_i] = op(out[abs_i], in[abs_i * cols + c]);
                    }
                }
                // Handle remainder rows
                if (i < remainder)
                {
                    uint32_t abs_i = i + snrt_cluster_compute_core_num() * items_per_core;
                    out[abs_i] = init_val;
                    for (int c = 0; c < cols; c++) {
                        out[abs_i] = op(out[abs_i], in[abs_i * cols + c]);
                    }
                }
            }
        }
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
    }
}

// TODO(chenwu): can refer to runtime/src/sync.c: function snrt_global_reduction_dma for generalization
static inline void inter_reduce(int *dst, int *tmp, int row_num, size_t size_bytes,
                                  row_op_t op) {
    // Step 1: Inter-cluster reduction in a tree fashion, saving results in cluster 0

    // 1->0, 3->2
    snrt_comm_t comm;
    snrt_comm_create(snrt_cluster_num(), &comm);
    // Distribute rows evenly to the cores in a cluster.
    // Last core handles remainder rows.
    uint32_t rows_per_core = HEIGHT / snrt_cluster_compute_core_num();
    uint32_t start_row = rows_per_core * snrt_cluster_core_idx();
    char is_last_compute_core =
        snrt_cluster_core_idx() == (snrt_cluster_compute_core_num() - 1);
    uint32_t end_row =
        is_last_compute_core ? HEIGHT : start_row + rows_per_core;

    if (pb_cluster_col() % 2 == 0 && snrt_is_dm_core()) {
        snrt_dma_start_1d(tmp, snrt_remote_l1_ptr(dst, snrt_cluster_idx(), snrt_cluster_idx() + 4), size_bytes);
        snrt_dma_wait_all();
    }

    snrt_global_barrier(comm);
    // snrt_cluster_hw_barrier();

    if (pb_cluster_col() % 2 == 0 && snrt_is_compute_core()) {
        // parallelize over compute cores
        for (int row_idx = start_row; row_idx < end_row; row_idx++) {
            dst[row_idx] = op(dst[row_idx], tmp[row_idx]);
        }
    }
    // Synchronize compute and DM cores for next tree level
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();


    // 2->0
    if (pb_cluster_col() == 0 && snrt_is_dm_core()) {
        snrt_dma_start_1d(tmp, snrt_remote_l1_ptr(dst, snrt_cluster_idx(), snrt_cluster_idx() + 8), size_bytes);
        snrt_dma_wait_all();
    }

    snrt_global_barrier(comm);
    // snrt_cluster_hw_barrier();

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


    // Step 2: Broadcast
    row_broadcast_from_left(dst, size_bytes);
    snrt_global_barrier(comm);
}

// row-wise max operation
static inline int max_op(int acc, int val) {
    return acc > val ? acc : val;
}

// sum operation
static inline int sum_op(int acc, int val) {
    return acc + val;
}

int main() {
    snrt_interrupt_enable(IRQ_M_CLUSTER);

    uint32_t s_fa_size = HEIGHT * LENGTH * sizeof(int);
    uint32_t m_size = HEIGHT * sizeof(int);
    int *S_fa =
        (int *)snrt_l1_alloc_cluster_local(s_fa_size, alignof(int));
    int *golden_m =
        (int *)snrt_l1_alloc_cluster_local(m_size, alignof(int));
    int *m =
        (int *)snrt_l1_alloc_cluster_local(m_size, alignof(int));
    int *m_tmp;
    if (pb_cluster_col() % 2 == 0) {
        m_tmp = (int *) snrt_l1_alloc_cluster_local(m_size, alignof(int));
    }

    // Distribute rows evenly to the cores in a cluster.
    // Last core handles remainder rows.
    uint32_t rows_per_core = HEIGHT / snrt_cluster_compute_core_num();
    uint32_t start_row = rows_per_core * snrt_cluster_core_idx();
    char is_last_compute_core =
        snrt_cluster_core_idx() == (snrt_cluster_compute_core_num() - 1);
    uint32_t end_row =
        is_last_compute_core ? HEIGHT : start_row + rows_per_core;

    // Step 1: Initialize
    if (snrt_is_dm_core()) {
        for (uint32_t i = 0; i < HEIGHT; i++) {
            for (uint32_t j = 0; j < LENGTH; j++) {
                S_fa[i * LENGTH + j] = j + snrt_cluster_idx() * 10;
            }
            // golden_m[i] = LENGTH * (LENGTH - 1) / 2 + 10 * LENGTH * snrt_cluster_idx(); // this is for intra reduce only
            golden_m[i] = 4 * LENGTH * (LENGTH - 1) / 2 + 10 * LENGTH * (4 * pb_cluster_row() + 24);
            m[i] = 0;
        }
    }
    snrt_global_barrier();

    if (snrt_is_compute_core()) {
        for (int row_idx = start_row; row_idx < end_row; row_idx++) {
            row_reduce(&S_fa[row_idx * LENGTH], &m[row_idx], LENGTH, sum_op, 0);
        }
    }
    // intra_row_reduce(S_fa, m, HEIGHT, LENGTH, sum_op, 0);
    snrt_global_barrier();
    inter_reduce(m, m_tmp, HEIGHT, m_size, sum_op);

    if (snrt_is_dm_core() && (snrt_cluster_idx() < N_CLUSTERS_TO_USE)) {
        uint32_t n_errs = HEIGHT;
        for (uint32_t i = 0; i < HEIGHT; i++) {
            if (m[i]==golden_m[i]) n_errs--;
        }
        return n_errs;
    } 
    else
        return 0;

}