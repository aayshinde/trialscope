select
    sponsor_key,
    any_value(sponsor_name)                                          as sponsor_name,
    mode(sponsor_class)                                              as sponsor_class,
    count(*)                                                         as n_trials,
    count(*) filter (where status_group = 'terminated')              as n_terminated,
    count(*) filter (where status_group = 'completed')               as n_completed,
    count(*) filter (where status_group = 'terminated')::double
        / nullif(count(*) filter (where status_group in ('completed', 'terminated')), 0) as termination_rate,
    min(start_date)                                                  as first_trial_start,
    max(start_date)                                                  as last_trial_start
from {{ ref('fct_trials') }}
group by 1
