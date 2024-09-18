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

/*
 * NOTE: this file is generated by generate_mg.py
 *
 * Make changes there and run in this directory:
 *
 * > python generate_mg.py
 *
 */

#include "mg.cuh"

namespace cuvs::neighbors::mg {

#define CUVS_INST_MG_CAGRA(T, IdxT)                                                    \
  index<cagra::index<T, IdxT>, T, IdxT> build(                                         \
    const raft::device_resources& handle,                                              \
    const mg::index_params<cagra::index_params>& index_params,                         \
    raft::host_matrix_view<const T, int64_t, row_major> index_dataset)                 \
  {                                                                                    \
    const raft::comms::nccl_clique& clique = handle.get_nccl_clique_handle();          \
    index<cagra::index<T, IdxT>, T, IdxT> index(index_params.mode, clique.num_ranks_); \
    cuvs::neighbors::mg::detail::build(                                                \
      handle,                                                                          \
      index,                                                                           \
      static_cast<const cuvs::neighbors::index_params*>(&index_params),                \
      index_dataset);                                                                  \
    return index;                                                                      \
  }                                                                                    \
                                                                                       \
  void search(const raft::device_resources& handle,                                    \
              const index<cagra::index<T, IdxT>, T, IdxT>& index,                      \
              const mg::search_params<cagra::search_params>& search_params,            \
              raft::host_matrix_view<const T, int64_t, row_major> queries,             \
              raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,              \
              raft::host_matrix_view<float, int64_t, row_major> distances,             \
              int64_t n_rows_per_batch)                                                \
  {                                                                                    \
    cuvs::neighbors::mg::detail::search(                                               \
      handle,                                                                          \
      index,                                                                           \
      static_cast<const cuvs::neighbors::search_params*>(&search_params),              \
      queries,                                                                         \
      neighbors,                                                                       \
      distances,                                                                       \
      n_rows_per_batch);                                                               \
  }                                                                                    \
                                                                                       \
  void serialize(const raft::device_resources& handle,                                 \
                 const index<cagra::index<T, IdxT>, T, IdxT>& index,                   \
                 const std::string& filename)                                          \
  {                                                                                    \
    cuvs::neighbors::mg::detail::serialize(handle, index, filename);                   \
  }                                                                                    \
                                                                                       \
  template <>                                                                          \
  index<cagra::index<T, IdxT>, T, IdxT> deserialize_cagra<T, IdxT>(                    \
    const raft::device_resources& handle, const std::string& filename)                 \
  {                                                                                    \
    auto idx = index<cagra::index<T, IdxT>, T, IdxT>(handle, filename);                \
    return idx;                                                                        \
  }                                                                                    \
                                                                                       \
  template <>                                                                          \
  index<cagra::index<T, IdxT>, T, IdxT> distribute_cagra<T, IdxT>(                     \
    const raft::device_resources& handle, const std::string& filename)                 \
  {                                                                                    \
    const raft::comms::nccl_clique& clique = handle.get_nccl_clique_handle();          \
    auto idx = index<cagra::index<T, IdxT>, T, IdxT>(REPLICATED, clique.num_ranks_);   \
    cuvs::neighbors::mg::detail::deserialize_and_distribute(handle, idx, filename);    \
    return idx;                                                                        \
  }
CUVS_INST_MG_CAGRA(float, uint32_t);

#undef CUVS_INST_MG_CAGRA

}  // namespace cuvs::neighbors::mg
