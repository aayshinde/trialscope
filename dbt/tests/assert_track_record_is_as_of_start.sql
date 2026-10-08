-- Leakage guard: recompute each trial's prior closures (5-year window before start) by brute force and compare
-- with the ASOF-join result. Any row returned = the feature counted a trial that closed on/after this trial's
-- start date, or outside the window, or missed one.
with brute as (
    select t.nct_id, count(c.nct_id) as n_prior
    from {{ ref('fct_trials') }} t
    left join {{ ref('fct_trials') }} c
      on c.sponsor_key = t.sponsor_key
     and c.status_group in ('completed', 'terminated') and c.primary_completion_type = 'ACTUAL'
     and c.primary_completion_date < t.start_date
     and c.primary_completion_date >= t.start_date - interval 5 year
    where t.start_date is not null
    group by 1
)
select f.nct_id, f.sponsor_prior_closed, b.n_prior
from {{ ref('fct_trials') }} f join brute b using (nct_id)
where f.sponsor_prior_closed != b.n_prior
