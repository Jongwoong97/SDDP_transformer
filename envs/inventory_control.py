import numpy as np
import cvxpy as cp

__all__ = [
    "InventoryControl"
]

n_supply = 2
n_inventory = 2
n_customer = 4


class InventoryControl:
    def __init__(self, n_stages=7):
        self.num_stage = n_stages

    def get_init_param_set(self):

        procedure_mat = np.zeros((3 * n_inventory + n_customer + n_supply, n_supply * n_inventory))
        for idx_inventory in range(n_inventory):
            one_mat = np.ones((1, n_supply))
            procedure_mat[idx_inventory, idx_inventory * n_supply:n_supply + idx_inventory * n_supply] = -one_mat

        for idx_supply in range(n_supply):
            supplier_cap_mat = np.zeros((1, n_supply * n_inventory))
            supplier_cap_mat[
                0, [idx_inventory * n_supply + idx_supply for idx_inventory in range(n_inventory)]] = np.ones(
                (1, n_inventory))
            procedure_mat[2 * n_inventory + n_customer + idx_supply] = supplier_cap_mat

        sales_mat = np.zeros((3 * n_inventory + n_customer + n_supply, n_customer * n_inventory))
        for idx_inventory in range(n_inventory):
            one_mat = np.ones((1, n_customer))
            # sales_mat[idx_inventory, n_supply*n_inventory + idx_inventory*n_customer:n_supply*n_inventory + n_customer + idx_inventory*n_customer] = one_mat
            sales_mat[idx_inventory, idx_inventory * n_customer: n_customer + idx_inventory * n_customer] = one_mat

        for idx_inventory in range(n_inventory):
            one_mat = np.ones((1, n_customer))
            sales_mat[n_inventory + idx_inventory,
            idx_inventory * n_customer:n_customer + idx_inventory * n_customer] = one_mat

        for idx_customer in range(n_customer):
            demand_bound_mat = np.zeros((1, n_customer * n_inventory))
            demand_bound_mat[
                0, [idx_inventory * n_customer + idx_customer for idx_inventory in range(n_inventory)]] = np.ones(
                (1, n_inventory))
            sales_mat[2 * n_inventory + idx_customer] = demand_bound_mat

        inventory_mat = np.zeros((3 * n_inventory + n_customer + n_supply, n_inventory))
        inventory_mat[:n_inventory] = np.identity(n_inventory)
        inventory_mat[2 * n_inventory + n_customer + n_supply:] = np.identity(n_inventory)

        theta_mat = np.zeros((3 * n_inventory + n_customer + n_supply, 1))

        A = np.concatenate((procedure_mat, sales_mat, inventory_mat, theta_mat), axis=1)

        B = np.zeros(
            (3 * n_inventory + n_customer + n_supply, n_supply * n_inventory + n_customer * n_inventory + n_inventory))
        identity_inventory = np.identity(n_inventory)
        B[:n_inventory, n_supply * n_inventory + n_customer * n_inventory:] = -identity_inventory
        B[n_inventory:2 * n_inventory, n_supply * n_inventory + n_customer * n_inventory:] = -identity_inventory

        initial_inventory = 10
        inventory_capacity = 200
        # b = np.array([-10, -10, 0, 0, -200])
        b = np.array([-initial_inventory] * 2 * n_inventory + [-15] * n_customer + [-1000] * n_supply + [
            -inventory_capacity] * n_inventory)

        threshold = -20000

        norm_cut = np.array([[0] * (n_supply * n_inventory + n_customer * n_inventory + n_inventory) + [-1, threshold]])

        obj_procecure = 4 * np.ones(n_supply * n_inventory)
        obj_sales = (10 - 0.5) * np.ones(n_inventory * n_customer)
        obj_inventory = 0.01 * np.ones(n_inventory)

        c = np.concatenate((obj_procecure, -obj_sales, obj_inventory), axis=0)

        return A, B, b, c, norm_cut, 'IC'

    def get_param_val(self, param='demand', param_rv=True):
        if param == 'demand':
            mean_origin = 15
            std_origin = 2.5
            if param_rv == True:
                loc = np.random.uniform(low=11, high=20)  # (0.01, 100)
                scale = np.random.uniform(low=std_origin - 1, high=std_origin + 1)
            else:
                loc = mean_origin
                scale = std_origin
        elif param == 'supply_capacity':
            mean_origin = 1000
            std_origin = 0.001
            if param_rv == True:
                loc = np.random.uniform(low=mean_origin - 0.68 * std_origin,
                                        high=mean_origin + 0.68 * std_origin)  # (0.01, 1000)
                scale = np.random.uniform(low=std_origin - 10, high=std_origin + 10)
            else:
                loc = mean_origin
                scale = std_origin
        return loc, scale

    def create_scenarioTree(self, scenarioNode=3, mm=True):
        loc_d, scale_d = self.get_param_val(param='demand', param_rv=False)
        loc_sc, scale_sc = self.get_param_val(param='supply_capacity', param_rv=False)

        scenarioTree_sc = []
        scenarioTree_d = []
        for idx in range(self.num_stage - 1):
            # np.random.seed(1)
            batch_sample_sc = np.random.normal(loc=loc_sc, scale=scale_sc, size=scenarioNode)
            if mm == True:
                normalized_batch_sample = (batch_sample_sc - np.mean(batch_sample_sc))
                normalized_batch_sample = normalized_batch_sample / np.std(batch_sample_sc)
                rescaled = normalized_batch_sample * scale_sc
                rescaled = rescaled + loc_sc
                scenarioTree_sc.append(rescaled.tolist())
            else:
                scenarioTree_sc.append(batch_sample_sc.tolist())

            batch_sample_d = np.random.normal(loc=loc_d, scale=scale_d, size=scenarioNode)
            if mm == True:
                normalized_batch_sample = (batch_sample_d - np.mean(batch_sample_d))
                normalized_batch_sample = normalized_batch_sample / np.std(batch_sample_d)
                rescaled = normalized_batch_sample * scale_d
                rescaled = rescaled + loc_d
                scenarioTree_d.append(rescaled.tolist())
            else:
                scenarioTree_d.append(batch_sample_d.tolist())

        return (scenarioTree_d, scenarioTree_sc), np.array((loc_d, scale_d, loc_sc, scale_sc))

    class StageFirst():
        def __init__(self, A):
            self.num_input = A.shape[1]

            self.procecure = cp.Variable(shape=self.n_supply * self.n_inventory, nonneg=True,
                                         name='Procecure')  # rsvl = reservoir level
            self.sales = cp.Variable(shape=self.n_inventory * self.n_customer, nonneg=True, name='Sales')
            self.inventory = cp.Variable(shape=self.n_inventory, nonneg=True, name='Inventory')
            self.theta = cp.Variable(name='Theta')
            self.coeff_procecure = A[:, :self.n_supply * self.n_inventory]
            self.coeff_sales = A[:,
                               self.n_supply * self.n_inventory:self.n_supply * self.n_inventory + n_inventory * n_customer]
            self.coeff_inventory = A[:,
                                   n_supply * n_inventory + n_inventory * n_customer:n_supply * n_inventory + n_inventory * n_customer + n_inventory]
            self.coeff_theta = A[:, -2]
            self.constant = A[:, -1]

            self.num_equality = n_inventory
            self.constraints_lst = [
                self.coeff_procecure[i] @ self.procecure + self.coeff_sales[i] @ self.sales + self.coeff_inventory[
                    i] @ self.inventory + self.coeff_theta[i] * self.theta + self.constant[i] == 0 for i in
                range(self.num_equality)]
            self.constraints_lst += [
                self.coeff_procecure[i] @ self.procecure + self.coeff_sales[i] @ self.sales + self.coeff_inventory[
                    i] @ self.inventory + self.coeff_theta[i] * self.theta + self.constant[i] <= 0 for i in
                range(self.num_equality, len(A))]

        def problem(self):
            obj_procecure = 4 * np.ones(n_supply * n_inventory)
            obj_sales = (10 - 0.5) * np.ones(n_inventory * n_customer)
            obj_inventory = 0.01 * np.ones(n_inventory)
            obj = cp.Minimize(
                obj_procecure @ self.procecure - obj_sales @ self.sales + obj_inventory @ self.inventory + self.theta)
            prob = cp.Problem(obj, self.constraints_lst)

            return prob

        def solve(self):
            prob = self.problem()
            prob.solve(solver=cp.CPLEX)
            obj = prob.value

            return obj, self.procecure.value, self.sales.value, self.inventory.value, self.theta.value

    class StageInter:
        def __init__(self, A, inventory_prev, demand, supplier_capacity):
            self.demand = cp.Parameter(shape=(n_customer, 1), value=demand * np.ones((n_customer, 1)))
            self.supplier_capacity = cp.Parameter(shape=(n_supply, 1), value=supplier_capacity * np.ones((n_supply, 1)))
            self.num_input = A.shape[1]

            self.procecure = cp.Variable(shape=n_supply * n_inventory, nonneg=True,
                                         name='Procecure')  # rsvl = reservoir level
            self.sales = cp.Variable(shape=n_inventory * n_customer, nonneg=True, name='Sales')
            self.inventory = cp.Variable(shape=n_inventory, nonneg=True, name='Inventory')
            self.theta = cp.Variable(name='Theta')
            self.coeff_procecure = A[:, :n_supply * n_inventory]
            self.coeff_sales = A[:, n_supply * n_inventory:n_supply * n_inventory + n_inventory * n_customer]
            self.coeff_inventory = A[:,
                                   n_supply * n_inventory + n_inventory * n_customer:n_supply * n_inventory + n_inventory * n_customer + n_inventory]
            self.coeff_theta = A[:, -2]
            self.constant = A[:, -1]

            self.inventory_prev = inventory_prev
            self.coeff_inventory_prev = np.concatenate((-np.identity(n_inventory), -np.identity(n_inventory),
                                                        np.zeros((A.shape[0] - 2 * n_inventory, n_inventory))), axis=0)

            self.num_equality = n_inventory
            self.constraints_lst = [
                self.coeff_procecure[i] @ self.procecure + self.coeff_sales[i] @ self.sales + self.coeff_inventory[
                    i] @ self.inventory + self.coeff_inventory_prev[i] @ self.inventory_prev == 0 for i in
                range(self.num_equality)]
            self.constraints_lst += [
                self.coeff_procecure[i] @ self.procecure + self.coeff_sales[i] @ self.sales + self.coeff_inventory[
                    i] @ self.inventory + self.coeff_inventory_prev[i] @ self.inventory_prev <= 0 for i in
                range(self.num_equality, 2 * self.num_equality)]
            self.constraints_lst += [
                self.coeff_procecure[i] @ self.procecure + self.coeff_sales[i] @ self.sales + self.coeff_inventory[
                    i] @ self.inventory + self.coeff_inventory_prev[i] @ self.inventory_prev + self.coeff_theta[
                    i] * self.theta + self.constant[i] <= 0 for i in range(2 * self.num_equality, len(A))]

        def problem(self):
            for i_customer in range(2 * self.num_equality, 2 * self.num_equality + n_customer):
                self.constraints_lst[i_customer] = self.coeff_procecure[i_customer] @ self.procecure + self.coeff_sales[
                    i_customer] @ self.sales + self.coeff_inventory[i_customer] @ self.inventory + self.coeff_theta[
                                                       i_customer] * self.theta - self.demand[
                                                       i_customer - 2 * self.num_equality] <= 0
            for i_supplier in range(2 * self.num_equality + n_customer, 2 * self.num_equality + n_customer + n_supply):
                self.constraints_lst[i_supplier] = self.coeff_procecure[i_supplier] @ self.procecure + self.coeff_sales[
                    i_supplier] @ self.sales + self.coeff_inventory[i_supplier] @ self.inventory + self.coeff_theta[
                                                       i_supplier] * self.theta - self.supplier_capacity[
                                                       i_supplier - 2 * self.num_equality - n_customer] <= 0
            obj_procecure = 4 * np.ones(n_supply * n_inventory)
            obj_sales = (10 - 0.5) * np.ones(n_inventory * n_customer)
            obj_inventory = 0.01 * np.ones(n_inventory)
            obj = cp.Minimize(
                obj_procecure @ self.procecure - obj_sales @ self.sales + obj_inventory @ self.inventory + self.theta)
            prob = cp.Problem(obj, self.constraints_lst)

            return prob

        def solve(self):
            prob = self.problem()
            prob.solve(solver=cp.CPLEX)
            obj = prob.value

            return obj, self.procecure.value, self.sales.value, self.inventory.value, self.theta.value

        def add_cut(self, scenario_trees):
            n_samples_d = len(scenario_trees[0])
            n_samples_sc = len(scenario_trees[1])
            new_cut = np.zeros((n_samples_sc * n_samples_d, self.num_input))
            for i in range(n_samples_d):
                for j in range(n_samples_sc):
                    self.demand.value = scenario_trees[0][i] * np.ones((n_customer, 1))
                    self.supplier_capacity.value = scenario_trees[1][j] * np.ones((n_supply, 1))
                    prob = self.problem()
                    prob.solve(solver=cp.CPLEX)
                    dual = np.hstack(list(prob.solution.dual_vars.values()))
                    obj = prob.value

                    grad = dual @ self.coeff_inventory_prev
                    coeff_theta = -1
                    constant = (obj - grad @ self.inventory_prev)

                    cut = np.hstack((grad, coeff_theta))
                    cut = np.hstack((cut, constant))

                    new_cut[i * n_samples_d + j, -n_inventory - 2:] = cut

            a = np.mean(new_cut, axis=0)
            return np.mean(new_cut, axis=0)

    class StageFinal:
        def __init__(self, A, inventory_prev, demand, supplier_capacity):
            self.demand = cp.Parameter(shape=(n_customer, 1), value=demand * np.ones((n_customer, 1)))
            self.supplier_capacity = cp.Parameter(shape=(n_supply, 1), value=supplier_capacity * np.ones((n_supply, 1)))
            self.num_input = A.shape[1]

            self.procecure = cp.Variable(shape=n_supply * n_inventory, nonneg=True,
                                         name='Procecure')  # rsvl = reservoir level
            self.sales = cp.Variable(shape=n_inventory * n_customer, nonneg=True, name='Sales')
            self.inventory = cp.Variable(shape=n_inventory, nonneg=True, name='Inventory')
            self.coeff_procecure = A[:, :n_supply * n_inventory]
            self.coeff_sales = A[:, n_supply * n_inventory:n_supply * n_inventory + n_inventory * n_customer]
            self.coeff_inventory = A[:,
                                   n_supply * n_inventory + n_inventory * n_customer:n_supply * n_inventory + n_inventory * n_customer + n_inventory]
            self.constant = A[:, -1]

            self.inventory_prev = inventory_prev
            self.coeff_inventory_prev = np.concatenate((-np.identity(n_inventory), -np.identity(n_inventory),
                                                        np.zeros((A.shape[0] - 2 * n_inventory, n_inventory))), axis=0)

            self.num_equality = n_inventory
            self.constraints_lst = [
                self.coeff_procecure[i] @ self.procecure + self.coeff_sales[i] @ self.sales + self.coeff_inventory[
                    i] @ self.inventory + self.coeff_inventory_prev[i] @ self.inventory_prev == 0 for i in
                range(self.num_equality)]
            self.constraints_lst += [
                self.coeff_procecure[i] @ self.procecure + self.coeff_sales[i] @ self.sales + self.coeff_inventory[
                    i] @ self.inventory + self.coeff_inventory_prev[i] @ self.inventory_prev <= 0 for i in
                range(self.num_equality, 2 * self.num_equality)]
            self.constraints_lst += [
                self.coeff_procecure[i] @ self.procecure + self.coeff_sales[i] @ self.sales + self.coeff_inventory[
                    i] @ self.inventory + self.coeff_inventory_prev[i] @ self.inventory_prev + self.constant[i] <= 0 for
                i in range(2 * self.num_equality, len(A))]

        def problem(self):
            for i_customer in range(2 * self.num_equality, 2 * self.num_equality + n_customer):
                self.constraints_lst[i_customer] = self.coeff_procecure[i_customer] @ self.procecure + self.coeff_sales[
                    i_customer] @ self.sales + self.coeff_inventory[i_customer] @ self.inventory - self.demand[
                                                       i_customer - 2 * self.num_equality] <= 0
            for i_supplier in range(2 * self.num_equality + n_customer, 2 * self.num_equality + n_customer + n_supply):
                self.constraints_lst[i_supplier] = self.coeff_procecure[i_supplier] @ self.procecure + self.coeff_sales[
                    i_supplier] @ self.sales + self.coeff_inventory[i_supplier] @ self.inventory - \
                                                   self.supplier_capacity[
                                                       i_supplier - 2 * self.num_equality - n_customer] <= 0
            obj_procecure = 4 * np.ones(n_supply * n_inventory)
            obj_sales = (10 - 0.5) * np.ones(n_inventory * n_customer)
            obj_inventory = 0.01 * np.ones(n_inventory)
            obj = cp.Minimize(obj_procecure @ self.procecure - obj_sales @ self.sales + obj_inventory @ self.inventory)
            prob = cp.Problem(obj, self.constraints_lst)

            return prob

        def solve(self):
            prob = self.problem()
            prob.solve(solver=cp.CPLEX)
            obj = prob.value

            return obj, self.procecure.value, self.sales.value, self.inventory.value

        def add_cut(self, scenario_trees):
            n_samples_d = len(scenario_trees[0])
            n_samples_sc = len(scenario_trees[1])
            new_cut = np.zeros((n_samples_sc * n_samples_d, self.num_input))
            for i in range(n_samples_d):
                for j in range(n_samples_sc):
                    self.demand.value = scenario_trees[0][i] * np.ones((n_customer, 1))
                    self.supplier_capacity.value = scenario_trees[1][j] * np.ones((n_supply, 1))
                    prob = self.problem()
                    prob.solve(solver=cp.CPLEX)
                    dual = np.hstack(list(prob.solution.dual_vars.values()))
                    obj = prob.value

                    grad = dual @ self.coeff_inventory_prev
                    coeff_theta = -1
                    constant = (obj - grad @ self.inventory_prev)

                    cut = np.hstack((grad, coeff_theta))
                    cut = np.hstack((cut, constant))

                    new_cut[i * n_samples_d + j, -n_inventory - 2:] = cut

            a = np.mean(new_cut, axis=0)
            return np.mean(new_cut, axis=0)
