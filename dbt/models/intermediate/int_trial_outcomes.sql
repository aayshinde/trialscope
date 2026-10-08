-- Outcome labels and survival times. Durations run from start to primary completion.
-- completion_event = 1 when the trial reached primary completion; terminated trials are censored at
-- their stop date and active trials at the data snapshot date (competing-risk caveat noted in README).
with snap as (select max(last_update_date) as snapshot_date from {{ ref('stg_studies') }}),
s as (select * from {{ ref('stg_studies') }} where start_date is not null)
select
    s.nct_id,
    snap.snapshot_date,
    s.status_group in ('completed', 'terminated'{% if var('withdrawn_as_failure') %}, 'withdrawn'{% endif %}) as is_resolved,
    case
        when s.status_group = 'terminated' then 1
        {% if var('withdrawn_as_failure') %}when s.status_group = 'withdrawn' then 1{% endif %}
        when s.status_group = 'completed' then 0
    end                                                                         as terminated_flag,
    case
        when s.status_group in ('completed', 'terminated') and s.primary_completion_type = 'ACTUAL'
            then datediff('day', s.start_date, s.primary_completion_date)
        when s.status_group = 'active' then datediff('day', s.start_date, snap.snapshot_date)
    end                                                                         as duration_days,
    (s.status_group = 'completed' and s.primary_completion_type = 'ACTUAL')::int as completion_event,
    case when s.primary_completion_type = 'ESTIMATED' then s.primary_completion_date end as planned_primary_completion,
    case when s.primary_completion_type = 'ESTIMATED'
         then datediff('day', s.start_date, s.primary_completion_date) end     as planned_duration_days,
    s.status_group = 'active' and s.primary_completion_type = 'ESTIMATED'
        and s.primary_completion_date < snap.snapshot_date                     as is_overdue
from s cross join snap
