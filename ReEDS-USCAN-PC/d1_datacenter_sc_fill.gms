* Fill datacenter_sc_exog_inv(r,datacenter_sc_bin,"%cur_year%") one model year
* at a time, in ascending-cost bin order (dc1..dc61), using each bin's
* capacity actually still available after realized (solved) occupancy from
* all strictly prior years -- datacenter_sc_occupied_total, updated at the
* end of each year's solve in d2_post_solve_adjustments.gms. This must run
* before this year's solve so eq_datacenter_sc_limit sees a claim consistent
* with what has truly already happened, rather than a bin assignment
* precomputed before any solve took place that has no way to know how much
* of a bin's capacity endogenous investment will separately claim in earlier
* years. dc61 has an effectively uncapped capacity (set in
* input_processing/datacenter.py), so the fill below always has somewhere to
* put whatever remains once the priced bins (dc1-dc60) are exhausted.

datacenter_sc_exog_inv(r,datacenter_sc_bin,"%cur_year%") = 0 ;

datacenter_sc_exog_remaining(r)$datacenter_r(r) =
    datacenter_sc_exog_req(r,"%cur_year%")
    - sum{tt$tprev("%cur_year%",tt), datacenter_sc_exog_req(r,tt) } ;

loop(datacenter_sc_bin,
    datacenter_sc_exog_inv(r,datacenter_sc_bin,"%cur_year%")$[
        datacenter_r(r)$(datacenter_sc_exog_remaining(r)>1e-9)] =
        min(
            datacenter_sc_exog_remaining(r),
            max(0, datacenter_sc_capacity(r,datacenter_sc_bin)
                   - datacenter_sc_occupied_total(r,datacenter_sc_bin))
        ) ;
    datacenter_sc_exog_remaining(r)$datacenter_r(r) =
        datacenter_sc_exog_remaining(r) - datacenter_sc_exog_inv(r,datacenter_sc_bin,"%cur_year%") ;
) ;
