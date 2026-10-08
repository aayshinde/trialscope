select nct_id, upper(type) as intervention_type, lower(trim(name)) as intervention_name
from {{ source('raw', 'interventions') }} where name is not null
