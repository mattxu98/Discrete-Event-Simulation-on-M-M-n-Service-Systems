### Discrete Event Simulation on M/M/n Service Systems
- Determined the optimal # of desks so that lucky clients queueing <= 8 min are >= 90%, without simpy package
- Computed the shortest queue for new clients given uncertain desks, by (nested) list comprehension for arrival/departure
- Divided and computed the performance of each batch: % of lucky clients and average queuing times. The optimal desks is 8, with polarized performance in simulations. Having 9 desks is optimal and robust: >= 99.31%, <= 0.29 min