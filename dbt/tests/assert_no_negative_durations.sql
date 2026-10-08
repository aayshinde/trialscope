{{ config(severity='warn') }}
-- Registry data has date-entry errors; surfaced as a warning, and such rows are excluded from modelling.
select nct_id, start_date, primary_completion_date, duration_days
from {{ ref('fct_trials') }}
where duration_days < 0
