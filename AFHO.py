class AziziFHO(Optimizer):
    """
    AziziFHO: Konversi Eksak Kode MATLAB Asli (Azizi et al., 2023) https://doi.org/10.1007/s10462-022-10173-w
    https://www.mathworks.com/matlabcentral/fileexchange/114325-fire-hawk-optimizer-fho-a-novel-metaheuristic-algorithm
    """
    OPT_INFO = OptInfo(
        name="Azizi FHO (Original MATLAB Logic)",
        year=2023,
        difficulty="medium",
        kind="original",
        scientific_status="questionable"
    )

    def __init__(self, epoch: int = 1000, pop_size: int = 50, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch: int) -> None:
        _, sorted_indices = self.get_sorted_population(self.pop, self.problem.minmax)
        pos_list = np.array([agent.solution for agent in self.pop])
        sorted_pos = pos_list[sorted_indices]

        max_fh = max(1, int(np.ceil(0.20 * self.pop_size)))
        n = int(self.generator.integers(1, max_fh + 1))
        m = self.pop_size - n

        FH = sorted_pos[:n].copy()
        rem_PR = sorted_pos[n:].copy()

        SP_global = np.mean(rem_PR, axis=0) if m > 0 else np.zeros(self.problem.n_dims)

        territories = []
        for i in range(n):
            if len(rem_PR) == 0:
                territories.append(np.empty((0, self.problem.n_dims)))
                continue

            dists = np.linalg.norm(rem_PR - FH[i], axis=1)
            near_idx = np.argsort(dists)

            if i == n - 1:
                alfa = len(rem_PR)
            else:
                alfa = int(self.generator.integers(1, len(rem_PR) + 1))

            selected_idx = near_idx[:alfa]
            territories.append(rem_PR[selected_idx].copy())
            rem_PR = np.delete(rem_PR, selected_idx, axis=0)

        new_POSITIONS = []

        for i in range(n):
            r1, r2 = self.generator.random(2)
            jdx = int(self.generator.integers(0, n))
            new_FH = FH[i] + (r1 * self.g_best.solution - r2 * FH[jdx])
            new_POSITIONS.append(new_FH)

        for i in range(n):
            PR_sub = territories[i]
            if len(PR_sub) == 0:
                continue

            SP_local = np.mean(PR_sub, axis=0)

            for q in range(len(PR_sub)):
                r3, r4 = self.generator.random(2)
                new_PR1 = PR_sub[q] + (r3 * FH[i] - r4 * SP_local)
                new_POSITIONS.append(new_PR1)

                r5, r6 = self.generator.random(2)
                alter_idx = int(self.generator.integers(0, n))
                new_PR2 = PR_sub[q] + (r5 * FH[alter_idx] - r6 * SP_global)
                new_POSITIONS.append(new_PR2)

        candidates_matrix = np.clip(np.array(new_POSITIONS), self.problem.lb, self.problem.ub)

        pop_temp = []
        for pos in candidates_matrix:
            agent = self.generate_empty_agent(pos)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos)
            pop_temp.append(agent)

        if self.mode in self.AVAILABLE_MODES:
            pop_temp = self.update_target_for_population(pop_temp)

        _, sorted_temp_indices = self.get_sorted_population(pop_temp, self.problem.minmax)
        self.pop = [pop_temp[idx] for idx in sorted_temp_indices[:self.pop_size]]