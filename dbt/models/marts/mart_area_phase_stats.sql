-- Historical termination rate and median duration by therapeutic area x phase (resolved trials only)
select
    therapeutic_area, phase_group,
    count(*)                                                        as n_resolved,
    avg(terminated_flag)                                            as termination_rate,
    median(duration_days) filter (where completion_event = 1)       as median_days_to_completion
from {{ ref('fct_trials') }}
where is_resolved
group by 1, 2
