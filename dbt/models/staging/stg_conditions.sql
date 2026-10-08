select distinct c.nct_id, c.condition, m.term as mesh_term
from {{ source('raw', 'conditions') }} c
left join {{ source('raw', 'condition_mesh') }} m on m.nct_id = c.nct_id and m.kind = 'meshes'
