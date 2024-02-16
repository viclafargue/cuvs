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

#include <cuvs/neighbors/ivf_flat.hpp>

namespace cuvs::neighbors::ivf_flat {

#define CUVS_INST_IVF_FLAT_BUILD(T, IdxT)                                                       \
  auto build(raft::resources const& handle,                                                     \
             const cuvs::neighbors::ivf_flat::index_params& params,                             \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset)                  \
    -> cuvs::neighbors::ivf_flat::index<T, IdxT>                                                \
  {                                                                                             \
    return cuvs::neighbors::ivf_flat::index<T, IdxT>(                                           \
      std::move(raft::runtime::neighbors::ivf_flat::build(handle, params, dataset)));           \
  }                                                                                             \
                                                                                                \
  void build(raft::resources const& handle,                                                     \
             const cuvs::neighbors::ivf_flat::index_params& params,                             \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset,                  \
             cuvs::neighbors::ivf_flat::index<T, IdxT>& idx)                                    \
  {                                                                                             \
    raft::runtime::neighbors::ivf_flat::build(handle, params, dataset, *idx.get_raft_index());  \
  }

CUVS_INST_IVF_FLAT_BUILD(float, int64_t);

#undef CUVS_INST_IVF_FLAT_BUILD

}   // namespace cuvs::neighbors::ivf_flat