# Electricity and compute mobility reduce the infrastructure burden of AI data-centre growth in North America

This repository contains the source code accompanying the manuscript:

> Siyuan Wang, Fengqi You. **Electricity and compute mobility reduce the infrastructure burden of AI data-centre growth in North America.**

## Overview

Rapid AI data-centre (AIDC) growth is creating concentrated electricity loads, yet power-system planning typically treats computing demand as fixed in place and time. This work develops a coupled US–Canada framework that co-optimizes power-system expansion, workload flexibility and AI-server siting over 2026–2035, showing that AI's infrastructure burden depends not only on demand growth but on how freely electricity and computation can move across regions and time.

The framework couples three mobility dimensions:
- **Electricity mobility** — cross-border US–Canada grid coordination and reserve sharing.
- **Operational compute mobility** — spatio-temporal flexibility of AI workloads across existing data centres.
- **Investment-stage compute mobility** — joint planning of AIDC siting and power-system capacity expansion.

## Repository Structure

All simulations were performed using **Python 3.11** and **Julia 1.8**. This repository contains three projects, corresponding to the three components of the modelling framework described in the Methods:

| Folder | Description |
|---|---|
| [`ReEDS-USCAN-PC/`](ReEDS-USCAN-PC/) | The **ReEDS-USCAN-PC** power-system capacity-expansion model: an extension of the NREL Regional Energy Deployment System (ReEDS) coupling US and Canadian grids for coordinated power-compute (PC) planning. |
| [`cerf_data_centers/`](cerf_data_centers/) | The **extended CERF-DC** siting module: an extension of the PNNL Capacity Expansion Regional Feasibility (CERF) model adapted to site AI data centres based on locational cost and grid-proximity ("gravity") scoring. |
| [`employment&economy/`](employment&economy/) | The **extended JEDI** economic-impact module: an extension of the NREL Jobs and Economic Development Impact (JEDI) framework, covering both power-sector employment and AIDC-campus economic impacts (construction and operating), including a Canadian provincial input–output extension. |
| [`source_data/`](source_data/) | Source data underlying the figures in the manuscript. |
