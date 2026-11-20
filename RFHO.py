class RectifiedFHO(Optimizer):
    """
    Rectified Fire Hawk Optimizer (RectifiedFHO)
    1. Invariant Territory Allocation: n = max(2, int(0.10 * pop_size))
    2. Vectorized Voronoi Territory Partitioning
    3. Strict (mu + lambda) Elitist Truncation Selection
    """
    OPT_INFO = OptInfo(
        name="Rectified Fire Hawk Optimizer (RectifiedFHO)",
        year=2026,
        difficulty="medium",
        kind="variant",
        scientific_status="questionable"
    )

    def __init__(self, epoch: int = 1000, pop_size: int = 50, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch: int) -> None:
        n = max(2, int(0.20 * self.pop_size)) #Fire Hawk 20%, change to n = max(2, int(0.10 * self.pop_size)) for 10% Fire Hawk
        m = self.pop_size - n

        _, sorted_indices = self.get_sorted_population(self.pop, self.problem.minmax)
        pos_list = np.array([agent.solution for agent in self.pop])
        FH = pos_list[sorted_indices[:n]]
        PR = pos_list[sorted_indices[n:]]
        g_best_pos = self.g_best.solution

        dist = np.linalg.norm(FH[:, np.newaxis, :] - PR[np.newaxis, :, :], axis=2)
        territory_assignments = np.argmin(dist, axis=0)

        r1 = self.generator.random((n, 1))
        r2 = self.generator.random((n, 1))
        rand_offsets = self.generator.integers(1, n, size=n)
        near_indices = (np.arange(n) + rand_offsets) % n
        new_FH = FH + (r1 * g_best_pos - r2 * FH[near_indices])

        SP_global = np.mean(PR, axis=0)
        SP_local_matrix = np.zeros_like(PR)
        for l_idx in range(n):
            mask = (territory_assignments == l_idx)
            if np.any(mask):
                SP_local_matrix[mask] = np.mean(PR[mask], axis=0)

        FH_assigned = FH[territory_assignments]
        r3 = self.generator.random((m, 1))
        r4 = self.generator.random((m, 1))
        new_PR1 = PR + (r3 * FH_assigned - r4 * SP_local_matrix)

        alt_offsets = self.generator.integers(1, n, size=m)
        alter_indices = (territory_assignments + alt_offsets) % n
        FH_alter = FH[alter_indices]

        r5 = self.generator.random((m, 1))
        r6 = self.generator.random((m, 1))
        new_PR2 = PR + (r5 * FH_alter - r6 * SP_global)

        candidates_matrix = np.vstack((new_FH, new_PR1, new_PR2))
        candidates_matrix = np.clip(candidates_matrix, self.problem.lb, self.problem.ub)

        pop_offspring = []
        for pos in candidates_matrix:
            agent = self.generate_empty_agent(pos)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos)
            pop_offspring.append(agent)

        if self.mode in self.AVAILABLE_MODES:
            pop_offspring = self.update_target_for_population(pop_offspring)

        combined_pool = self.pop + pop_offspring
        _, sorted_pool_indices = self.get_sorted_population(combined_pool, self.problem.minmax)
        self.pop = [combined_pool[idx] for idx in sorted_pool_indices[:self.pop_size]]