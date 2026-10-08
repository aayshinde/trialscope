-- Sponsor history in the 5 YEARS BEFORE each trial's start date (leakage-safe and stationary):
-- only the same sponsor's trials that had already CLOSED before this trial started are counted.
-- A trailing window (instead of all-time totals) keeps the feature's distribution stable over
-- time — all-time counts grow every year and showed up as severe drift in the PSI monitor.
-- Implemented with DuckDB ASOF joins on running totals (now minus 5 years ago), so it scales.
with t as (
    select s.nct_id, s.start_date, sp.sponsor_key
    from {{ ref('stg_studies') }} s join {{ ref('int_sponsors') }} sp using (nct_id)
    where s.start_date is not null
),
closures_daily as (
    select sp.sponsor_key, s.primary_completion_date as closed_on,
           count(*) as n, sum((s.status_group = 'terminated')::int) as n_term
    from {{ ref('stg_studies') }} s join {{ ref('int_sponsors') }} sp using (nct_id)
    where s.status_group in ('completed', 'terminated') and s.primary_completion_type = 'ACTUAL'
      and s.primary_completion_date is not null
    group by 1, 2
),
closures_cum as (
    select sponsor_key, closed_on, sum(n) over w as cum_closed, sum(n_term) over w as cum_term
    from closures_daily
    window w as (partition by sponsor_key order by closed_on rows between unbounded preceding and current row)
),
starts_cum as (
    select sponsor_key, start_date,
           sum(n) over (partition by sponsor_key order by start_date rows between unbounded preceding and current row) as cum_started
    from (select sponsor_key, start_date, count(*) as n from t group by 1, 2) d
),
w as (select *, start_date - interval 5 year as window_start from t)
select
    w.nct_id,
    w.sponsor_key,
    coalesce(c1.cum_closed, 0) - coalesce(c0.cum_closed, 0)                          as sponsor_prior_closed,
    coalesce(c1.cum_term, 0) - coalesce(c0.cum_term, 0)                              as sponsor_prior_terminated,
    coalesce(s1.cum_started, 0) - coalesce(s0.cum_started, 0)                        as sponsor_prior_started,
    -- smoothed toward a 15% prior so sponsors with little history don't get extreme rates
    (coalesce(c1.cum_term, 0) - coalesce(c0.cum_term, 0) + 0.15 * 10)
        / (coalesce(c1.cum_closed, 0) - coalesce(c0.cum_closed, 0) + 10)             as sponsor_prior_term_rate
from w
asof left join closures_cum c1 on c1.sponsor_key = w.sponsor_key and w.start_date > c1.closed_on
asof left join closures_cum c0 on c0.sponsor_key = w.sponsor_key and w.window_start > c0.closed_on
asof left join starts_cum s1 on s1.sponsor_key = w.sponsor_key and w.start_date > s1.start_date
asof left join starts_cum s0 on s0.sponsor_key = w.sponsor_key and w.window_start > s0.start_date
