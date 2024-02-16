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

#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq {

#define CUVS_INST_IVF_PQ_EXTEND(T, IdxT)                                                                                      \
  auto extend(raft::resources const& handle,                                                                                  \
              raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,                                           \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,                                          \
              const cuvs::neighbors::ivf_pq::index<IdxT>& orig_index)                                                         \
    -> cuvs::neighbors::ivf_pq::index<IdxT>                                                                                   \
  {                                                                                                                           \
    return cuvs::neighbors::ivf_pq::index<IdxT>(                                                                              \
      std::move(raft::runtime::neighbors::ivf_pq::extend(handle, new_vectors, new_indices, *orig_index.get_raft_index())));   \
  }                                                                                                                           \
                                                                                                                              \
  void extend(raft::resources const& handle,                                                                                  \
              raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,                                           \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,                                          \
              cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                                                      \
  {                                                                                                                           \
    raft::runtime::neighbors::ivf_pq::extend(handle, new_vectors, new_indices, idx->get_raft_index());                        \
  }

CUVS_INST_IVF_PQ_EXTEND(uint8_t, int64_t);

#undef CUVS_INST_IVF_PQ_EXTEND

}   // namespace cuvs::neighbors::ivf_pq