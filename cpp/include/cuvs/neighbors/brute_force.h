/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance_types.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup bruteforce_c_index Bruteforce index
 * @{
 */
/**
 * @brief Struct to hold address of cuvs::neighbors::brute_force::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} bruteForceIndex;

typedef bruteForceIndex* cuvsBruteForceIndex_t;

/**
 * @brief Allocate BRUTEFORCE index
 *
 * @param[in] index cuvsBruteForceIndex_t to allocate
 * @return cuvsError_t
 */
cuvsError_t bruteForceIndexCreate(cuvsBruteForceIndex_t* index);

/**
 * @brief De-allocate BRUTEFORCE index
 *
 * @param[in] index cuvsBruteForceIndex_t to de-allocate
 */
cuvsError_t bruteForceIndexDestroy(cuvsBruteForceIndex_t index);
/**
 * @}
 */

/**
 * @defgroup bruteforce_c_index_build Bruteforce index build
 * @{
 */
/**
 * @brief Build a BRUTEFORCE index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        3. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/brute_force.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create BRUTEFORCE index
 * cuvsBruteForceIndex_t index;
 * cuvsError_t index_create_status = bruteForceIndexCreate(&index);
 *
 * // Build the BRUTEFORCE Index
 * cuvsError_t build_status = bruteForceBuild(res, &dataset_tensor, L2Expanded, 0.f, index);
 *
 * // de-allocate `params`, `index` and `res`
 * cuvsError_t index_destroy_status = bruteForceIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[in] metric metric
 * @param[in] metric_arg metric_arg
 * @param[out] index cuvsBruteForceIndex_t Newly built BRUTEFORCE index
 * @return cuvsError_t
 */
cuvsError_t bruteForceBuild(cuvsResources_t res,
                            DLManagedTensor* dataset,
                            enum DistanceType metric,
                            float metric_arg,
                            cuvsBruteForceIndex_t index);
/**
 * @}
 */

/**
 * @defgroup bruteforce_c_index_search Bruteforce index search
 * @{
 */
/**
 * @brief Search a BRUTEFORCE index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        It is also important to note that the BRUTEFORCE Index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 * queries.dl_tensor.dtype.code` Types for input are:
 *        1. `queries`: kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/brute_force.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 * DLManagedTensor queries;
 * DLManagedTensor neighbors;
 *
 * // Search the `index` built using `bruteForceBuild`
 * cuvsError_t search_status = bruteForceSearch(res, index, queries, neighbors, distances);
 *
 * // de-allocate `params` and `res`
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index bruteForceIndex which has been returned by `bruteForceBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 */
cuvsError_t bruteForceSearch(cuvsResources_t res,
                             cuvsBruteForceIndex_t index,
                             DLManagedTensor* queries,
                             DLManagedTensor* neighbors,
                             DLManagedTensor* distances);
/**
 * @}
 */

#ifdef __cplusplus
}
#endif
