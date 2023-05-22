// Copyright 2023 Bytedance Ltd. and/or its affiliates.
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#pragma once
#include "common.h"
#include "reduce.h"
#include <cub/cub.cuh>
#define DIV_UP(x, y) ((x) + (y) - 1) / (y)

namespace bytetransformer {
template <typename T>
__global__ void softmax_kernel_warp(T *qk_buf, const T *atten_bias, const T *atten_mask,
                                    const int batch_size, const int head_num, const int seq_len) {
  int word_id = blockIdx.x;
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x;
  int head_id = threadIdx.y;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * seq_len;

  extern __shared__ float shmem[];
  float *s_row_qk = (float *)shmem + head_id * seq_len;

  float max_v = -1e20f;
  for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize) {
    float qk = (float)qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk += (float)atten_bias[((head_id * seq_len + seq_id) * seq_len) + col_id];
    float mask_val =
        (1.0f - (float)atten_mask[((batch_id * seq_len + seq_id) * seq_len) + col_id]) * -10000.0f;
    float tmp = qk + mask_val;
    s_row_qk[col_id] = tmp;
    max_v = tmp > max_v ? tmp : max_v;
  }
  max_v = warpReduceMax<float>(max_v);

  float exp_sum = 0.0f;
  for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize) {
    float qk = __expf(s_row_qk[col_id] - max_v);
    s_row_qk[col_id] = qk;
    exp_sum += qk;
  }
  exp_sum = warpReduceSum<float>(exp_sum);

  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize)
    qk_buf[qk_offset + col_id] = (T)(s_row_qk[col_id] * exp_sum);
}

template <typename T>
__global__ void softmax_kernel_warp_half2(half2 *qk_buf, const half2 *atten_bias,
                                          const half2 *atten_mask, const int batch_size,
                                          const int head_num, const int seq_len) {
  int word_id = blockIdx.x;
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x;
  int head_id = threadIdx.y;
  int half2_seq_len = seq_len / 2;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

  extern __shared__ float shmem[];
  float *s_qk_buf = (float *)shmem + head_id * seq_len;

  float max_val = -1e20f;
  for (int col_id = warp_tid; col_id < half2_seq_len; col_id += warpSize) {
    half2 qk = qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk = __hadd2(qk, atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id]);
    half2 mask_val = atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id];
    float mask_val_x = (1.0f - (float)mask_val.x) * -10000.0f,
          mask_val_y = (1.0f - (float)mask_val.y) * -10000.0f;
    float tmp_x = (float)qk.x + mask_val_x, tmp_y = (float)qk.y + mask_val_y;
    s_qk_buf[col_id * 2] = tmp_x, s_qk_buf[col_id * 2 + 1] = tmp_y;
    max_val = fmax(max_val, fmax(tmp_x, tmp_y));
  }
  max_val = warpReduceMax(max_val);

  float exp_sum = 0.0f;
  for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize) {
    float qk = __expf(s_qk_buf[col_id] - max_val);
    s_qk_buf[col_id] = qk;
    exp_sum += qk;
  }
  exp_sum = warpReduceSum(exp_sum);

  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  for (int col_id = warp_tid; col_id < half2_seq_len; col_id += warpSize)
    qk_buf[qk_offset + col_id] = __halves2half2((half)(s_qk_buf[col_id * 2] * exp_sum),
                                                (half)(s_qk_buf[col_id * 2 + 1] * exp_sum));
}

/*
dim3 grid(batch_size * seq_len, head_num), block(32, min(16, DIV_UP(seq_len, 64)));
const int shmem_size = DIV_UP(seq_len, 64) * (64 + 8) * sizeof(float);

gridDim.x = batch_size * seq_len;
gridDim.y = head_num;
blockDim.x = warpSize;
blockDim.y = seq_len / SPLIT / warpSize;
*/
template <typename T>
__global__ void softmax_kernel_block_half2(half2 *qk_buf, const half2 *atten_bias,
                                          const half2 *atten_mask, const int batch_size,
                                          const int head_num, const int seq_len) {
  int word_id = blockIdx.x;
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x % warpSize;
  int warp_idx = threadIdx.x / warpSize;
  int warpNum = blockDim.x / 32;
  // int head_id = threadIdx.y;
  int head_id = blockIdx.y;
  int half2_seq_len = seq_len / 2;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

  // one block for one row and one head
  // size 32 for float numbers in a warp, 
  // skew 8 to avoid bank conflicts
  // extern __shared__ float rshmem[][64 + 8];
  // float *s_qk_buf = (float *)shmem + head_id * seq_len;

  // change to register 
  // no communications needed except for reduction
  // seq len <= 10 * 16 * 32 * 2 = 10240
  float rshmem[10 * 2];
  // size equals to warpNum
  // extern __shared__ float holder[][2];
  // using cub, one row for a thread block
  __shared__ typename cub::BlockReduce<float, 512>::TempStorage temp_storage;
  __shared__ float s_sum, s_max;

  float max_val = -1e20f;
  int i = 0;
#pragma unroll
  for (int col_id = warp_idx * warpSize + warp_tid; col_id < half2_seq_len; col_id += warpSize * warpNum){
    half2 qk = qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk = __hadd2(qk, atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id]);
    half2 mask_val = atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id];
    float mask_val_x = (1.0f - (float)mask_val.x) * -10000.0f,
          mask_val_y = (1.0f - (float)mask_val.y) * -10000.0f;
    float tmp_x = (float)qk.x + mask_val_x, tmp_y = (float)qk.y + mask_val_y;
    // s_qk_buf[col_id * 2] = tmp_x, s_qk_buf[col_id * 2 + 1] = tmp_y;
    rshmem[i * 2] = tmp_x, rshmem[i * 2 + 1] = tmp_y;
    max_val = fmax(max_val, fmax(tmp_x, tmp_y));
    i += 1;
  }
  // TODO: replace the hand written block 
  // reduce with cub::BlockReduce
  // max_val = warpReduceMax(max_val);
  // using cub
  float cub_max_val = cub::BlockReduce<float, 512>(temp_storage).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0){s_max = cub_max_val;}
  __syncthreads();
  max_val = s_max;
  // if (warp_tid == 0){
  // holder[warp_idx][0] = max_val;
  // }
  // __syncthreads();
  // hand written block reduce
  // #pragma unroll 
  // for (int w = 0; w < warpNum; w++){
    // TODO: binary partition
    // max_val = fmax(max_val, holder[w][0]);
  // }

  float exp_sum = 0.0f;
  i = 0;
  for (int col_id = warp_idx * warpSize + warp_tid; col_id < half2_seq_len; col_id += warpSize * warpNum){
    // TODO: float2 store
    // the first one in half2
    float qk = __expf(rshmem[i * 2] - max_val);
    rshmem[i * 2] = qk;
    exp_sum += qk;
    // the second one in half2
    qk = __expf(rshmem[i * 2 + 1] - max_val);
    rshmem[i * 2 + 1] = qk;
    exp_sum += qk;
    i += 1;
  }
  // exp_sum = warpReduceSum(exp_sum);
  // if (warp_tid == 0){
  // holder[warp_idx][1] = exp_sum;
  // }
  // __syncthreads();
  // hand written block reduce
  // must be cleared first
  // exp_sum = 0.0f;
  // #pragma unroll 
  // for (int w = 0; w < warpNum; w++){
    // TODO: binary partition
    // exp_sum += holder[w][1];
  // }
  // using cub
  float cub_exp_sum = cub::BlockReduce<float, 512>(temp_storage).Sum(exp_sum);
  if (threadIdx.x == 0){s_sum = cub_exp_sum;}
  __syncthreads();
  exp_sum = s_sum;
  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  i = 0;
  for (int col_id = warp_idx * warpSize + warp_tid; col_id < half2_seq_len; col_id += warpSize * warpNum){
    qk_buf[qk_offset + col_id] = __halves2half2((half)(rshmem[i * 2] * exp_sum),
                                                (half)(rshmem[i * 2 + 1] * exp_sum));
    i += 1;
  }
}



/*template <typename T>
__global__ void softmax_kernel_warp_half2(float4 *qk_buf, const float4 *atten_bias,
                                          const float4 *atten_mask, const int batch_size,
                                          const int head_num, const int seq_len) {
  int word_id = blockIdx.x;
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x;
  int head_id = threadIdx.y;
  int float2_seq_len = seq_len / 8;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * float2_seq_len;

  extern __shared__ float shmem[];
  float *s_qk_buf = (float *)shmem + head_id * seq_len;

  float max_val = -1e20f;
  for (int col_id = warp_tid; col_id < float2_seq_len; col_id += warpSize) {
    // half2 qk = qk_buf[qk_offset + col_id];
    // *((float4*)&qk[0]) = *((float4*)(&qk_buf[qk_offset + col_id]));
    half qk[8];
    *((float4*)&qk[0]) = __ldg(&qk_buf[qk_offset + col_id]);
    if (atten_bias){
      // qk = __hadd2(qk, atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id]);
      half bias[8];
      *((float4*)&bias[0]) = __ldg(&atten_bias[((head_id * seq_len + seq_id) * float2_seq_len) + col_id]);
#pragma unroll
      for (int c = 0; c < 8; c += 2){
      *((half2*)(&qk[c])) = 
          __hadd2(*((half2*)(&qk[c])), *((half2*)(&bias[c])));
      }
    }
    // half2 mask_val = atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id];
    half mask_val[8];
    *((float4*)&mask_val[0]) = __ldg(&atten_mask[((batch_id * seq_len + seq_id) * float2_seq_len) + col_id]);
    // float mask_val_x = (1.0f - (float)mask_val.x) * -10000.0f,
    //       mask_val_y = (1.0f - (float)mask_val.y) * -10000.0f;
    // float tmp_x = (float)qk.x + mask_val_x, tmp_y = (float)qk.y + mask_val_y;
    // s_qk_buf[col_id * 2] = tmp_x, s_qk_buf[col_id * 2 + 1] = tmp_y;
    // max_val = fmax(max_val, fmax(tmp_x, tmp_y));
#pragma unroll
    for (int c = 0; c < 8; c++){
      s_qk_buf[col_id * 8 + c] = (float)qk[c] + (1.0f - (float)mask_val[c]) * -10000.0f;
      max_val = fmax(max_val, s_qk_buf[col_id * 8 + c]);
    }
  }
  max_val = warpReduceMax(max_val);

  float exp_sum = 0.0f;
  for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize) {
    float qk = __expf(s_qk_buf[col_id] - max_val);
    s_qk_buf[col_id] = qk;
    exp_sum += qk;
  }
  exp_sum = warpReduceSum(exp_sum);

  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  half to_store[8];
  for (int col_id = warp_tid; col_id < float2_seq_len; col_id += warpSize){
    // qk_buf[qk_offset + col_id] = __halves2half2((half)(s_qk_buf[col_id * 2] * exp_sum),
    //                                             (half)(s_qk_buf[col_id * 2 + 1] * exp_sum));
#pragma unroll
    for (int c = 0; c < 8; c++){
      to_store[c] = (half)(s_qk_buf[col_id * 4 + c] * exp_sum);
    }
    *((float4*)(&qk_buf[qk_offset + col_id])) = 
        *((float4*)(&to_store[0]));
  }
}*/


template <typename T>
__global__ void softmax_kernel_warp_et(T *qk_buf, const T *atten_bias, const T *atten_mask,
                                       const int batch_size, const int head_num, const int seq_len,
                                       int *batch_idx, int *word_idx) {
  int word_id = __ldg(&word_idx[blockIdx.x]);
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x;
  int head_id = threadIdx.y;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * seq_len;

  extern __shared__ float shmem[];
  float *s_row_qk = (float *)shmem + head_id * seq_len;

  float max_v = -1e20f;
  for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize) {
    float qk = (float)qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk += (float)atten_bias[((head_id * seq_len + seq_id) * seq_len) + col_id];
    float mask_val =
        (1.0f - (float)atten_mask[((batch_id * seq_len + seq_id) * seq_len) + col_id]) * -10000.0f;
    float tmp = qk + mask_val;
    s_row_qk[col_id] = tmp;
    max_v = tmp > max_v ? tmp : max_v;
  }
  max_v = warpReduceMax(max_v);

  float exp_sum = 0.0f;
  for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize) {
    float qk = __expf(s_row_qk[col_id] - max_v);
    s_row_qk[col_id] = qk;
    exp_sum += qk;
  }
  exp_sum = warpReduceSum(exp_sum);

  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize)
    qk_buf[qk_offset + col_id] = (T)(s_row_qk[col_id] * exp_sum);
}

template <typename T>
__global__ void softmax_kernel_warp_half2_et(half2 *qk_buf, const half2 *atten_bias,
                                             const half2 *atten_mask, const int batch_size,
                                             const int head_num, const int seq_len, int *batch_idx,
                                             int *word_idx) {
  int word_id = __ldg(&word_idx[blockIdx.x]);
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x;
  int head_id = threadIdx.y;
  int half2_seq_len = seq_len / 2;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

  extern __shared__ float shmem[];
  float *s_qk_buf = (float *)shmem + head_id * seq_len;

  float max_val = -1e20f;
  for (int col_id = warp_tid; col_id < half2_seq_len; col_id += warpSize) {
    half2 qk = qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk = __hadd2(qk, atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id]);
    half2 mask_val = atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id];
    float mask_val_x = (1.0f - (float)mask_val.x) * -10000.0f,
          mask_val_y = (1.0f - (float)mask_val.y) * -10000.0f;
    float tmp_x = (float)qk.x + mask_val_x, tmp_y = (float)qk.y + mask_val_y;
    s_qk_buf[col_id * 2] = tmp_x, s_qk_buf[col_id * 2 + 1] = tmp_y;
    max_val = fmax(max_val, fmax(tmp_x, tmp_y));
  }
  max_val = warpReduceMax(max_val);

  float exp_sum = 0.0f;
  for (int col_id = warp_tid; col_id < seq_len; col_id += warpSize) {
    float qk = __expf(s_qk_buf[col_id] - max_val);
    s_qk_buf[col_id] = qk;
    exp_sum += qk;
  }
  exp_sum = warpReduceSum(exp_sum);

  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  for (int col_id = warp_tid; col_id < half2_seq_len; col_id += warpSize)
    qk_buf[qk_offset + col_id] = __halves2half2((half)(s_qk_buf[col_id * 2] * exp_sum),
                                                (half)(s_qk_buf[col_id * 2 + 1] * exp_sum));
}

template <typename T, const int count, const bool need_padding>
__global__ void softmax_kernel_warp_half2_register(half2 *qk_buf, const half2 *atten_bias,
                                                   const half2 *atten_mask, const int batch_size,
                                                   const int head_num, const int seq_len) {
  int word_id = blockIdx.x;
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x;
  int head_id = threadIdx.y;
  int half2_seq_len = seq_len / 2;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

  float s_qk_buf[count];
  // [hk:] initialization to deal with short seq len < 64
  // but it seems the change has no impact on the end-to-end result
  // if (need_padding){
  // #pragma unroll
  //   for (int c = 0; c < count; c++){
  //     s_qk_buf[c] = -10000.0f;
  //   }
  // }
  if (need_padding)
    s_qk_buf[count - 2] = -10000.0f, s_qk_buf[count - 1] = -10000.0f;

  float max_val = -1e20f;
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + warpSize * i;
    if (need_padding && col_id >= half2_seq_len)
      break;

    half2 qk = qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk =
          __hadd2(qk, __ldg(&atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id]));
    half2 mask_val = __ldg(&atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id]);
    float mask_val_x = (1.0f - (float)mask_val.x) * -10000.0f,
          mask_val_y = (1.0f - (float)mask_val.y) * -10000.0f;
    s_qk_buf[i * 2] = (float)qk.x + mask_val_x, s_qk_buf[i * 2 + 1] = (float)qk.y + mask_val_y;
  }

  for (int i = 0; i < count; i++)
    max_val = fmax(max_val, s_qk_buf[i]);
  max_val = warpReduceMax(max_val);

  float exp_sum = 0.0f;
  for (int i = 0; i < count; i++) {
    s_qk_buf[i] = __expf(s_qk_buf[i] - max_val);
    exp_sum += s_qk_buf[i];
  }
  exp_sum = warpReduceSum(exp_sum);

  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + warpSize * i;
    if (need_padding && col_id >= half2_seq_len)
      return;
    qk_buf[qk_offset + col_id] =
        __halves2half2((half)(s_qk_buf[i * 2] * exp_sum), (half)(s_qk_buf[i * 2 + 1] * exp_sum));
  }
}


/*template <typename T, const int count, const bool need_padding>
__global__ void softmax_kernel_warp_half2_register(half2 *qk_buf, const half2 *atten_bias,
                                                   const half2 *atten_mask, const int batch_size,
                                                   const int head_num, const int seq_len) {
  int word_id = blockIdx.x;
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x;
  int head_id = threadIdx.y;
  int half2_seq_len = seq_len / 2;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

  float s_qk_buf[count];
  if (need_padding)
    s_qk_buf[count - 2] = -10000.0f, s_qk_buf[count - 1] = -10000.0f;

  float max_val = -1e20f;
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + warpSize * i;
    if (need_padding && col_id >= half2_seq_len)
      break;

    half2 qk = qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk =
          __hadd2(qk, __ldg(&atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id]));
    half2 mask_val = __ldg(&atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id]);
    float mask_val_x = (1.0f - (float)mask_val.x) * -10000.0f,
          mask_val_y = (1.0f - (float)mask_val.y) * -10000.0f;
    s_qk_buf[i * 2] = (float)qk.x + mask_val_x, s_qk_buf[i * 2 + 1] = (float)qk.y + mask_val_y;
    max_val = fmax(s_qk_buf[i * 2], max_val);
    max_val = fmax(s_qk_buf[i * 2 + 1], max_val);
  }

  // for (int i = 0; i < count; i++)
  //   max_val = fmax(max_val, s_qk_buf[i]);
  max_val = warpReduceMax(max_val);

  float exp_sum = 0.0f;
  for (int i = 0; i < count; i++) {
    s_qk_buf[i] = __expf(s_qk_buf[i] - max_val);
    exp_sum += s_qk_buf[i];
  }
  exp_sum = warpReduceSum(exp_sum);

  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + warpSize * i;
    if (need_padding && col_id >= half2_seq_len)
      return;
    qk_buf[qk_offset + col_id] =
        __halves2half2((half)(s_qk_buf[i * 2] * exp_sum), (half)(s_qk_buf[i * 2 + 1] * exp_sum));
  }
}*/


/*template <typename T, const int count, const bool need_padding>
__global__ void softmax_kernel_warp_half2_register(half *qk_buf, const half *atten_bias,
                                                   const half *atten_mask, const int batch_size,
                                                   const int head_num, const int seq_len) {
  int word_id = blockIdx.x;
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x;
  int head_id = threadIdx.y;
  // int float2_seq_len = seq_len / 4;
  // in the unit of half
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * seq_len;

  // 8 * 4 for sequence length = 1024
  float s_qk_buf[count];
  if (need_padding){
    s_qk_buf[count - 4] = -10000.0f, s_qk_buf[count - 3] = -10000.0f;
    s_qk_buf[count - 2] = -10000.0f, s_qk_buf[count - 1] = -10000.0f;
  }

  float max_val = -1e20f;
  for (int i = 0; i < count / 4; i++) {
    // stride 4 for float2
    int col_id = (warp_tid + warpSize * i) * 4;
    if (need_padding && col_id >= seq_len)
      break;

    // half2 qk = qk_buf[qk_offset + col_id];
    // float2 qk = *((float2*)(&qk_buf[qk_offset + col_id]));
    half qk[4];
    *((float2*)&qk[0]) = *((float2*)(&qk_buf[qk_offset + col_id]));
    if (atten_bias){
      half bias[4];
      *((float2*)&bias[0]) = *((float2*)(&atten_bias[((head_id * seq_len + seq_id) * seq_len) + col_id]));
      *((half2*)(&qk[0])) = 
          __hadd2(*((half2*)(&qk[0])), *((half2*)(&bias[0])));
      *((half2*)(&qk[2])) = 
          __hadd2(*((half2*)(&qk[2])), *((half2*)(&bias[2])));
    }
    // half2 mask_val = __ldg(&atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id]);
    half mask_val[4];
    *((float2*)&mask_val[0]) = *((float2*)(&atten_mask[((batch_id * seq_len + seq_id) * seq_len) + col_id]));
    float mask_val_0 = (1.0f - (float)mask_val[0]) * -10000.0f,
          mask_val_1 = (1.0f - (float)mask_val[1]) * -10000.0f,
          mask_val_2 = (1.0f - (float)mask_val[2]) * -10000.0f,
          mask_val_3 = (1.0f - (float)mask_val[3]) * -10000.0f;
    // s_qk_buf[i * 2] = (float)qk.x + mask_val_x, s_qk_buf[i * 2 + 1] = (float)qk.y + mask_val_y;
    s_qk_buf[i * 4] = (float)qk[0] + mask_val_0;
    s_qk_buf[i * 4 + 1] = (float)qk[1] + mask_val_1;
    s_qk_buf[i * 4 + 2] = (float)qk[2] + mask_val_2;
    s_qk_buf[i * 4 + 3] = (float)qk[3] + mask_val_3;
  }

  for (int i = 0; i < count; i++)
    max_val = fmax(max_val, s_qk_buf[i]);
  max_val = warpReduceMax(max_val);

  float exp_sum = 0.0f;
  for (int i = 0; i < count; i++) {
    s_qk_buf[i] = __expf(s_qk_buf[i] - max_val);
    exp_sum += s_qk_buf[i];
  }
  exp_sum = warpReduceSum(exp_sum);

  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  half to_store[4];
  for (int i = 0; i < count / 4; i++) {
    int col_id = (warp_tid + warpSize * i) * 4;
    if (need_padding && col_id >= seq_len)
      return;
    // qk_buf[qk_offset + col_id] =
    //     __halves2half2((half)(s_qk_buf[i * 2] * exp_sum), (half)(s_qk_buf[i * 2 + 1] * exp_sum));
#pragma unroll
    for (int c = 0; c < 4; c++){
      to_store[c] = (half)(s_qk_buf[i * 4 + c] * exp_sum);
    }
    *((float2*)(&qk_buf[qk_offset + col_id])) = 
        *((float2*)(&to_store[0]));
  }
}*/


/*template <typename T, const int count, const bool need_padding, const int cl_unit>
__global__ void softmax_kernel_warp_half2_register(float4 *qk_buf, const float4 *atten_bias,
                                                   const float4 *atten_mask, const int batch_size,
                                                   const int head_num, const int seq_len) {
  int word_id = blockIdx.x;
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x;
  int head_id = threadIdx.y;
  int float4_seq_len = seq_len / cl_unit;
  // in the unit of half
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * float4_seq_len;

  // 4 * 8 for sequence length = 1024, in the unit of float4
  float s_qk_buf[count];
  if (need_padding){
#pragma unroll
    for (int c = 0; c < cl_unit; c++){
      s_qk_buf[count - 1 - c] = -10000.0f;
    }
  }

  float max_val = -1e20f;
  for (int i = 0; i < count / cl_unit; i++) {
    // stride 4 for float2
    int col_id = (warp_tid + warpSize * i);
    if (need_padding && col_id >= float4_seq_len)
      break;

    // half2 qk = qk_buf[qk_offset + col_id];
    // float2 qk = *((float2*)(&qk_buf[qk_offset + col_id]));
    half qk[cl_unit];
    *((float4*)&qk[0]) = *((float4*)(&qk_buf[qk_offset + col_id]));
    if (atten_bias){
      half bias[cl_unit];
      *((float4*)&bias[0]) = *((float4*)(&atten_bias[((head_id * seq_len + seq_id) * float4_seq_len) + col_id]));
#pragma unroll
      for (int c = 0; c < cl_unit; c += 2){
      *((half2*)(&qk[c])) = 
          __hadd2(*((half2*)(&qk[c])), *((half2*)(&bias[c])));
      }
    }
    // half2 mask_val = __ldg(&atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id]);
    half mask_val[cl_unit];
    *((float4*)&mask_val[0]) = *((float4*)(&atten_mask[((batch_id * seq_len + seq_id) * float4_seq_len) + col_id]));
    // float mask_val_float[cl_unit];
#pragma unroll
    for (int c = 0; c < cl_unit; c++){
      s_qk_buf[i * cl_unit + c] = (float)qk[c] + (1.0f - (float)mask_val[c]) * -10000.0f;
      max_val = fmax(max_val, s_qk_buf[i * cl_unit + c]);
    }
  }

  // for (int i = 0; i < count; i++)
  //   max_val = fmax(max_val, s_qk_buf[i]);
  max_val = warpReduceMax(max_val);

  float exp_sum = 0.0f;
  for (int i = 0; i < count; i++) {
    s_qk_buf[i] = __expf(s_qk_buf[i] - max_val);
    exp_sum += s_qk_buf[i];
  }
  exp_sum = warpReduceSum(exp_sum);

  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  half to_store[cl_unit];
  for (int i = 0; i < count / cl_unit; i++) {
    int col_id = (warp_tid + warpSize * i);
    if (need_padding && col_id >= float4_seq_len)
      return;
    // qk_buf[qk_offset + col_id] =
    //     __halves2half2((half)(s_qk_buf[i * 2] * exp_sum), (half)(s_qk_buf[i * 2 + 1] * exp_sum));
#pragma unroll
    for (int c = 0; c < cl_unit; c++){
      to_store[c] = (half)(s_qk_buf[i * cl_unit + c] * exp_sum);
    }
    *((float4*)(&qk_buf[qk_offset + col_id])) = 
        *((float4*)(&to_store[0]));
  }
}*/


template <typename T, const int count, const bool need_padding>
__global__ void softmax_kernel_warp_half2_register_et(half2 *qk_buf, const half2 *atten_bias,
                                                      const half2 *atten_mask,
                                                      const int batch_size, const int head_num,
                                                      const int seq_len, int *batch_idx,
                                                      int *word_idx) {
  int word_id = __ldg(&word_idx[blockIdx.x]);
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = threadIdx.x;
  int head_id = threadIdx.y;
  int half2_seq_len = seq_len / 2;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

  float s_qk_buf[count];
  if (need_padding)
    s_qk_buf[count - 2] = -10000.0f, s_qk_buf[count - 1] = -10000.0f;

  float max_val = -1e20f;
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + warpSize * i;
    if (need_padding && col_id >= half2_seq_len)
      break;

    half2 qk = qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk =
          __hadd2(qk, __ldg(&atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id]));
    half2 mask_val = __ldg(&atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id]);
    float mask_val_x = (1.0f - (float)mask_val.x) * -10000.0f,
          mask_val_y = (1.0f - (float)mask_val.y) * -10000.0f;
    s_qk_buf[i * 2] = (float)qk.x + mask_val_x, s_qk_buf[i * 2 + 1] = (float)qk.y + mask_val_y;
  }

  for (int i = 0; i < count; i++)
    max_val = fmax(max_val, s_qk_buf[i]);
  max_val = warpReduceMax(max_val);

  float exp_sum = 0.0f;
  for (int i = 0; i < count; i++) {
    s_qk_buf[i] = __expf(s_qk_buf[i] - max_val);
    exp_sum += s_qk_buf[i];
  }
  exp_sum = warpReduceSum(exp_sum);

  exp_sum = __fdividef(1.0f, exp_sum + 1e-6f);
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + warpSize * i;
    if (need_padding && col_id >= half2_seq_len)
      return;
    qk_buf[qk_offset + col_id] =
        __halves2half2((half)(s_qk_buf[i * 2] * exp_sum), (half)(s_qk_buf[i * 2 + 1] * exp_sum));
  }
}

#define SOFTMAX_HALF2_REG(REG_COUNT)                                                            \
  if (seq_len % 64 == 0)                                                                        \
    softmax_kernel_warp_half2_register<half2, REG_COUNT, false>                                 \
        <<<grid, block, 0, stream>>>((half2 *)qk_buf, (half2 *)atten_bias, (half2 *)atten_mask, \
                                     batch_size, head_num, seq_len);                            \
  else                                                                                          \
    softmax_kernel_warp_half2_register<half2, REG_COUNT, true><<<grid, block, 0, stream>>>(     \
        (half2 *)qk_buf, (half2 *)atten_bias, (half2 *)atten_mask, batch_size, head_num, seq_len)

/*#define SOFTMAX_HALF2_REG(REG_COUNT)                                                            \
  if (seq_len % 64 == 0)                                                                        \
    softmax_kernel_warp_half2_register<half2, REG_COUNT, false, 8><<<grid, block, 0, stream>>>( \
        (float4 *)qk_buf, (float4 *)atten_bias, (float4 *)atten_mask, batch_size, head_num, seq_len); \
  else                                                                                          \
    softmax_kernel_warp_half2_register<half2, REG_COUNT, true, 8><<<grid, block, 0, stream>>>(  \
        (float4 *)qk_buf, (float4 *)atten_bias, (float4 *)atten_mask, batch_size, head_num, seq_len)*/

#define SOFTMAX_HALF2_REG_RM(REG_COUNT)                                                         \
  if (seq_len % 64 == 0)                                                                        \
    softmax_kernel_warp_half2_register_et<half2, REG_COUNT, false>                              \
        <<<grid, block, 0, stream>>>((half2 *)qk_buf, (half2 *)atten_bias, (half2 *)atten_mask, \
                                     batch_size, head_num, seq_len, batch_idx, word_idx);       \
  else                                                                                          \
    softmax_kernel_warp_half2_register_et<half2, REG_COUNT, true>                               \
        <<<grid, block, 0, stream>>>((half2 *)qk_buf, (half2 *)atten_bias, (half2 *)atten_mask, \
                                     batch_size, head_num, seq_len, batch_idx, word_idx)

template <OperationType OpType, typename T>
void softmax_kernelLauncher(T *qk_buf, const T *atten_bias, const T *atten_mask,
                            const int batch_size, const int seq_len, const int head_num,
                            cudaStream_t stream) {
  dim3 grid(batch_size * seq_len), block(32, head_num);

  const int shmem_size = head_num * seq_len * sizeof(float);

  // if (shmem_size > 64 * 1024)
  //  printf("Not Enough Shared Memory for Softmax\n");

  if ((seq_len & 0x1) == 0 && OpType == OperationType::HALF) {
    if (seq_len <= 1024) {
    // if (seq_len <= 960) {
      switch ((seq_len + 63) / 64) {
        case 1:
          SOFTMAX_HALF2_REG(1 * 2);
          break;
        case 2:
          SOFTMAX_HALF2_REG(2 * 2);
          break;
        case 3:
          SOFTMAX_HALF2_REG(3 * 2);
          break;
        case 4:
          SOFTMAX_HALF2_REG(4 * 2);
          break;
        case 5:
          SOFTMAX_HALF2_REG(5 * 2);
          break;
        case 6:
          SOFTMAX_HALF2_REG(6 * 2);
          break;
        case 7:
          SOFTMAX_HALF2_REG(7 * 2);
          break;
        case 8:
          SOFTMAX_HALF2_REG(8 * 2);
          break;
        case 9:
          SOFTMAX_HALF2_REG(9 * 2);
          break;
        case 10:
          SOFTMAX_HALF2_REG(10 * 2);
          break;
        case 11:
          SOFTMAX_HALF2_REG(11 * 2);
          break;
        case 12:
          SOFTMAX_HALF2_REG(12 * 2);
          break;
        case 13:
          SOFTMAX_HALF2_REG(13 * 2);
          break;
        case 14:
          SOFTMAX_HALF2_REG(14 * 2);
          break;
        case 15:
          SOFTMAX_HALF2_REG(15 * 2);
          break;
        case 16:
          SOFTMAX_HALF2_REG(16 * 2);
          // printf("new kernel!");
          break;
      }
    } else {
      // if (shmem_size > 48 * 1024)
      //   cudaFuncSetAttribute(softmax_kernel_warp_half2<half2>,
      //                         cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);

      // dim3 grid_new(batch_size * seq_len, head_num), block_new(32 * min(8, DIV_UP(seq_len, 64)));
      dim3 grid_new(batch_size * seq_len, head_num), block_new(32 * 16);
      // const int shmem_size_new = DIV_UP(seq_len, 64) * (64 + 8) * sizeof(float);
      const int shmem_size_new = 2 * (16 + 8) * sizeof(float);

      softmax_kernel_block_half2<half2><<<grid_new, block_new, shmem_size_new, stream>>>(
          (half2 *)qk_buf, (half2 *)atten_bias, (half2 *)atten_mask, batch_size, head_num,
          seq_len);
      /*softmax_kernel_warp_half2<half2><<<grid, block, shmem_size, stream>>>(
          (half2 *)qk_buf, (half2 *)atten_bias, (half2 *)atten_mask, batch_size, head_num,
          seq_len);*/
      /*softmax_kernel_warp_half2<half2><<<grid, block, shmem_size, stream>>>(
          (float4 *)qk_buf, (float4 *)atten_bias, (float4 *)atten_mask, batch_size, head_num,
          seq_len);*/
    }
  } else {
    if (shmem_size > 48 * 1024)
      cudaFuncSetAttribute(softmax_kernel_warp<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           64 * 1024);
    softmax_kernel_warp<T><<<grid, block, shmem_size, stream>>>(qk_buf, atten_bias, atten_mask,
                                                                batch_size, head_num, seq_len);
  }
}

template <OperationType OpType, typename T>
void softmax_et_kernelLauncher(T *qk_buf, const T *atten_bias, const T *atten_mask,
                               const int batch_size, const int seq_len, const int head_num,
                               cudaStream_t stream, int *batch_idx, int *word_idx,
                               int valid_word_num) {
  dim3 grid(valid_word_num), block(32, head_num);

  const int shmem_size = head_num * seq_len * sizeof(float);
  if (shmem_size > 64 * 1024)
    printf("Not Enough Shared Memory for Softmax\n");

  if ((seq_len & 0x1) == 0 && OpType == OperationType::HALF) {
    if (seq_len <= 1024) {
      switch ((seq_len + 63) / 64) {
        case 1:
          SOFTMAX_HALF2_REG_RM(1 * 2);
          break;
        case 2:
          SOFTMAX_HALF2_REG_RM(2 * 2);
          break;
        case 3:
          SOFTMAX_HALF2_REG_RM(3 * 2);
          break;
        case 4:
          SOFTMAX_HALF2_REG_RM(4 * 2);
          break;
        case 5:
          SOFTMAX_HALF2_REG_RM(5 * 2);
          break;
        case 6:
          SOFTMAX_HALF2_REG_RM(6 * 2);
          break;
        case 7:
          SOFTMAX_HALF2_REG_RM(7 * 2);
          break;
        case 8:
          SOFTMAX_HALF2_REG_RM(8 * 2);
          break;
        case 9:
          SOFTMAX_HALF2_REG_RM(9 * 2);
          break;
        case 10:
          SOFTMAX_HALF2_REG_RM(10 * 2);
          break;
        case 11:
          SOFTMAX_HALF2_REG_RM(11 * 2);
          break;
        case 12:
          SOFTMAX_HALF2_REG_RM(12 * 2);
          break;
        case 13:
          SOFTMAX_HALF2_REG_RM(13 * 2);
          break;
        case 14:
          SOFTMAX_HALF2_REG_RM(14 * 2);
          break;
        case 15:
          SOFTMAX_HALF2_REG_RM(15 * 2);
          break;
        case 16:
          SOFTMAX_HALF2_REG_RM(16 * 2);
          break;
      }
    } else {
      if (shmem_size > 48 * 1024)
        cudaFuncSetAttribute(softmax_kernel_warp_half2_et<half2>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
      softmax_kernel_warp_half2_et<half2><<<grid, block, shmem_size, stream>>>(
          (half2 *)qk_buf, (half2 *)atten_bias, (half2 *)atten_mask, batch_size, head_num, seq_len,
          batch_idx, word_idx);
    }
  } else {
    if (shmem_size > 48 * 1024)
      cudaFuncSetAttribute(softmax_kernel_warp_et<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           64 * 1024);
    softmax_kernel_warp_et<T><<<grid, block, shmem_size, stream>>>(
        qk_buf, atten_bias, atten_mask, batch_size, head_num, seq_len, batch_idx, word_idx);
  }
}
}  // namespace bytetransformer
